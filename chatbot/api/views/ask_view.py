from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from backend.utils.device import get_device_string
from backend.utils.qdrant_client import (
    get_qdrant_client, hybrid_search, hybrid_search_with_fallback,
    generate_sparse_vector, get_sparse_builder, HybridSearchResult
)
from qdrant_client.http.models import Distance, Filter, FieldCondition, MatchValue
from django.conf import settings
import logging
import os
import re
import numpy as np
from langdetect import detect
from google import generativeai
from sentence_transformers import CrossEncoder
from django.http import JsonResponse
from backend.utils.retrieval_metrics import recall_at_k, mrr_at_k, ndcg_at_k
# NEW: add robust error handling imports
from google.api_core.exceptions import GoogleAPICallError, RetryError, DeadlineExceeded, ServiceUnavailable
import time
import uuid
from typing import List, Optional
from datetime import datetime  # Add this import for datetime.utcnow()

from backend.utils.mongo_repository import MongoRepository
from backend.utils.mongo_models import MongoQueryLog
from api.utils.query_preprocessing import preprocess_query
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)

class AskView(APIView):
    # Phase 2: Cache CrossEncoder at class level to avoid loading per-request (1-3s overhead)
    _reranker = None
    _reranker_model_name = None
    
    @classmethod
    def get_reranker(cls):
        """
        Get cached CrossEncoder reranker instance (singleton pattern).
        Avoids reloading model on every request, saving 1-3 seconds per request.
        Uses GPU if available for faster reranking.
        """
        reranker_model = os.getenv("RERANKER_MODEL")
        
        # Reload if model changed or not loaded
        if cls._reranker is None or cls._reranker_model_name != reranker_model:
            logger.info(f"Loading CrossEncoder reranker: {reranker_model}")
            device = get_device_string()
            cls._reranker = CrossEncoder(reranker_model, device=device)
            cls._reranker_model_name = reranker_model
            logger.info(f"CrossEncoder reranker loaded successfully on device: {device}")
        
        return cls._reranker
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mongo_repo = MongoRepository()
        # Add connection check
        try:
            self.mongo_repo.db.command('ping')
            logger.info("MongoDB connection successful.")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
    
    # NEW: Retrieve chunks from MongoDB when Qdrant fails
    def _retrieve_from_mongodb(self, question_embedding, top_k=20):
        """
        Fallback retrieval using MongoDB vector search via cosine similarity.
        Returns list of results in same format as Qdrant for compatibility.
        """
        try:
            logger.info(f"Attempting MongoDB vector search with top_k={top_k}")
            
            # Get all chunks from MongoDB
            chunks = list(self.mongo_repo.db['chunks'].find({}))
            
            if not chunks:
                logger.warning("No chunks found in MongoDB")
                return []
            
            # Compute cosine similarity manually
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            embeddings = []
            chunk_ids = []
            
            for chunk in chunks:
                emb = chunk.get('embedding')
                if emb and isinstance(emb, list):
                    embeddings.append(emb)
                    chunk_ids.append(chunk.get('_id'))
            
            if not embeddings:
                logger.warning("No valid embeddings found in MongoDB chunks")
                return []
            
            # Convert to numpy arrays
            embeddings_array = np.array(embeddings)
            question_emb_array = np.array(question_embedding).reshape(1, -1)
            
            # Compute similarities
            similarities = cosine_similarity(question_emb_array, embeddings_array)[0]
            
            # Get top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results in Qdrant-compatible format
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.0:  # Filter out zero/negative scores
                    chunk = chunks[idx]
                    result = type('obj', (object,), {
                        'id': str(chunk.get('_id')),
                        'score': float(similarities[idx]),
                        'payload': {
                            'text': chunk.get('text', ''),
                            'document_id': chunk.get('document_id', ''),
                            'page': chunk.get('page'),
                            'chunk_id': chunk.get('chunk_id'),
                            'section': chunk.get('section'),
                            'metadata': chunk.get('metadata', {})
                        }
                    })()
                    results.append(result)
            
            logger.info(f"MongoDB returned {len(results)} results with similarity scores")
            return results
            
        except Exception as e:
            logger.exception(f"MongoDB fallback retrieval failed: {e}")
            return []
        
    def _extract_gemini_text(self, resp):
        """
        Safely extract text from a Gemini response without using resp.text,
        which raises when there are no Parts.
        """
        try:
            candidates = getattr(resp, "candidates", []) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content else []
                texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                if texts:
                    return " ".join(texts).strip(), getattr(cand, "finish_reason", None)
            # No parts in any candidate
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            return "", finish_reason
        except Exception:
            return "", None

    # NEW: sanitize prompt to avoid safety-filter triggers
    def _sanitize_prompt_for_safety(self, question, context_text):
        """
        Return a sanitized, neutral prompt to reduce chance of triggering safety filters.
        FIX: More aggressive sanitization and neutral system prompt.
        """
        # More comprehensive word replacements (Vietnamese + English)
        replacements = {
            "giết": "loại bỏ",
            "bắn": "sử dụng",
            "tấn công": "tiếp cận",
            "phá hủy": "thay đổi",
            "chết": "kết thúc",
            "hạ gục": "không hoạt động",
            "bẻ gãy": "sửa đổi",
            "đánh": "tác động",
            "chiến tranh": "xung đột",
            "vũ khí": "công cụ",
            # English
            "kill": "remove",
            "attack": "approach",
            "bomb": "device",
            "destroy": "modify",
            "harm": "affect",
            "violence": "impact",
            "weapon": "tool",
            "war": "conflict",
        }

        safe_context = context_text or ""
        for trg, rep in replacements.items():
            # Case-insensitive replacement
            import re as regex_module
            safe_context = regex_module.sub(regex_module.escape(trg), rep, safe_context, flags=regex_module.IGNORECASE)

        # Trim context
        try:
            SAFE_CONTEXT_CHARS = int(os.getenv("SAFE_CONTEXT_CHARS", "2500"))
        except Exception:
            SAFE_CONTEXT_CHARS = 2500

        if len(safe_context) > SAFE_CONTEXT_CHARS:
            safe_context = safe_context[:SAFE_CONTEXT_CHARS] + "..."

        # FIX: Much more neutral system prompt to avoid filter triggers
        neutral_system = (
            "Bạn là một trợ lý hỗ trợ thông tin. "
            "Hãy trả lời câu hỏi dựa trên tài liệu được cung cấp. "
            "Trả lời ngắn gọn, khách quan, và chính xác. "
            "Nếu không tìm thấy thông tin, hãy nói: 'Không có thông tin trong tài liệu.'"
        )

        sanitized_prompt = (
            f"{neutral_system}\n\n"
            f"Tài liệu:\n{safe_context}\n\n"
            f"Câu hỏi: {question}\n"
            f"Trả lời:"
        )
        return sanitized_prompt

    def _build_context_text(self, results):
        """
        Build context text from retrieved chunks with enhanced citations.
        Includes section/hierarchy information when available.
        """
        context_blocks = []
        for result in results:
            payload = result['payload']
            doc_id = payload.get('document_id', 'unknown')
            page = payload.get('page', '?')
            chunk_id = result['id']
            
            # Build enhanced citation with hierarchy
            section_title = payload.get('section_title') or payload.get('section')
            section_path = payload.get('section_path', [])
            
            if section_path and len(section_path) > 0:
                # Use section path for hierarchical reference
                path_str = " > ".join(section_path[-2:]) if len(section_path) > 2 else " > ".join(section_path)
                ref = f"[{doc_id} tr.{page} §{path_str} #{chunk_id}]"
            elif section_title:
                ref = f"[{doc_id} tr.{page} §{section_title} #{chunk_id}]"
            else:
                ref = f"[{doc_id} tr.{page} #{chunk_id}]"
            
            context_blocks.append(f"{ref}\n{payload['text']}\n")
        return "\n".join(context_blocks)

    def _expand_chunk_context(
        self,
        top_results: List[dict],
        expand_top_n: int = 3,
        radius: int = 1
    ) -> Tuple[List[dict], Dict[str, str]]:
        """
        Retrieve surrounding chunks for the top N results to provide complete context.
        
        Args:
            top_results: Reranked results (each has 'id', 'payload', 'score')
            expand_top_n: Number of top results to expand (default: 3)
            radius: Number of chunks before/after to retrieve (default: 1)
            
        Returns:
            Tuple of:
                - List of adjacent chunk dicts with 'chunk_id', 'text', 'position' ('prev'/'next'), 'parent_chunk_id'
                - Dict mapping chunk_id -> 'main'/'expanded' for tracking
        """
        if radius <= 0 or expand_top_n <= 0:
            return [], {r['id']: 'main' for r in top_results}
        
        # Track which chunks are already included
        included_ids: Set[str] = set()
        chunk_sources: Dict[str, str] = {}  # chunk_id -> 'main' or 'expanded'
        
        # Mark all main results
        for result in top_results:
            chunk_id = result.get('id') or result['payload'].get('chunk_id')
            if chunk_id:
                included_ids.add(chunk_id)
                chunk_sources[chunk_id] = 'main'
        
        expanded_chunks = []
        
        # Only expand top N results
        for result in top_results[:expand_top_n]:
            payload = result['payload']
            doc_id = payload.get('document_id')
            order_index = payload.get('order_index')
            parent_chunk_id = result.get('id') or payload.get('chunk_id')
            
            if doc_id is None or order_index is None:
                logger.debug(f"Skipping expansion for chunk {parent_chunk_id}: missing doc_id or order_index")
                continue
            
            # Query MongoDB for adjacent chunks
            try:
                # Build list of indices to fetch
                adjacent_indices = []
                for offset in range(-radius, radius + 1):
                    if offset != 0:  # Skip the main chunk itself
                        adjacent_indices.append(order_index + offset)
                
                # Filter out negative indices
                adjacent_indices = [idx for idx in adjacent_indices if idx >= 0]
                
                if not adjacent_indices:
                    continue
                
                # Query adjacent chunks from MongoDB
                adjacent_cursor = self.mongo_repo.db['chunks'].find({
                    "document_id": doc_id,
                    "order_index": {"$in": adjacent_indices}
                }).sort("order_index", 1)
                
                for adj_chunk in adjacent_cursor:
                    adj_chunk_id = adj_chunk.get('chunk_id') or adj_chunk.get('_id')
                    
                    # Skip if already included (deduplication)
                    if adj_chunk_id in included_ids:
                        continue
                    
                    adj_order = adj_chunk.get('order_index', 0)
                    position = 'prev' if adj_order < order_index else 'next'
                    
                    expanded_chunks.append({
                        'chunk_id': adj_chunk_id,
                        'text': adj_chunk.get('text', ''),
                        'document_id': doc_id,
                        'page': adj_chunk.get('page'),
                        'section_title': adj_chunk.get('section_title'),
                        'section_path': adj_chunk.get('section_path', []),
                        'order_index': adj_order,
                        'position': position,  # 'prev' or 'next'
                        'parent_chunk_id': parent_chunk_id,
                        'parent_order_index': order_index,
                    })
                    
                    included_ids.add(adj_chunk_id)
                    chunk_sources[adj_chunk_id] = 'expanded'
                    
            except Exception as e:
                logger.warning(f"Failed to expand context for chunk {parent_chunk_id}: {e}")
                continue
        
        logger.info(f"Context expansion: expanded {expand_top_n} chunks with radius={radius}, found {len(expanded_chunks)} adjacent chunks")
        return expanded_chunks, chunk_sources

    def _build_context_with_expansion(
        self,
        top_results: List[dict],
        expanded_chunks: List[dict],
        max_context_chars: int = 12000,
        prioritize_main: bool = True
    ) -> Tuple[str, int, int]:
        """
        Build context text with main chunks and expanded neighbors.
        Uses smart truncation that prioritizes main chunks over expanded ones.
        
        Args:
            top_results: Main reranked results
            expanded_chunks: Adjacent chunks from expansion
            max_context_chars: Maximum context length
            prioritize_main: If True, include all main chunks before any expanded
            
        Returns:
            Tuple of (context_text, main_chunks_included, expanded_chunks_included)
        """
        context_parts = []
        current_length = 0
        main_included = 0
        expanded_included = 0
        
        def format_chunk_block(chunk_id: str, text: str, doc_id: str, page, 
                               section_title: str = None, section_path: list = None,
                               marker: str = "") -> str:
            """Format a chunk with citation header."""
            # Build citation
            if section_path and len(section_path) > 0:
                path_str = " > ".join(section_path[-2:]) if len(section_path) > 2 else " > ".join(section_path)
                ref = f"[{doc_id} tr.{page} §{path_str} #{chunk_id}]"
            elif section_title:
                ref = f"[{doc_id} tr.{page} §{section_title} #{chunk_id}]"
            else:
                ref = f"[{doc_id} tr.{page} #{chunk_id}]"
            
            if marker:
                ref = f"{marker} {ref}"
            
            return f"{ref}\n{text}\n"
        
        # Step 1: Add main chunks first (prioritized)
        for result in top_results:
            payload = result['payload']
            chunk_id = result['id']
            doc_id = payload.get('document_id', 'unknown')
            page = payload.get('page', '?')
            section_title = payload.get('section_title') or payload.get('section')
            section_path = payload.get('section_path', [])
            text = payload.get('text', '')
            
            block = format_chunk_block(
                chunk_id, text, doc_id, page, section_title, section_path,
                marker="[MAIN]"
            )
            
            # Check if adding this would exceed limit
            if current_length + len(block) > max_context_chars:
                if prioritize_main and main_included < 3:
                    # Force include at least top 3 main chunks
                    context_parts.append(block)
                    current_length += len(block)
                    main_included += 1
                else:
                    logger.info(f"Context truncated at {main_included} main chunks due to size limit")
                    break
            else:
                context_parts.append(block)
                current_length += len(block)
                main_included += 1
        
        # Step 2: Group expanded chunks by their parent for coherent ordering
        # Sort expanded by parent order, then by their own order
        expanded_sorted = sorted(
            expanded_chunks,
            key=lambda x: (x.get('parent_order_index', 0), x.get('order_index', 0))
        )
        
        # Step 3: Add expanded chunks if space allows
        for exp_chunk in expanded_sorted:
            chunk_id = exp_chunk['chunk_id']
            text = exp_chunk.get('text', '')
            doc_id = exp_chunk.get('document_id', 'unknown')
            page = exp_chunk.get('page', '?')
            section_title = exp_chunk.get('section_title')
            section_path = exp_chunk.get('section_path', [])
            position = exp_chunk.get('position', 'adj')  # 'prev' or 'next'
            
            marker = "[PREV]" if position == 'prev' else "[NEXT]"
            
            block = format_chunk_block(
                chunk_id, text, doc_id, page, section_title, section_path,
                marker=marker
            )
            
            # Strict limit for expanded chunks
            if current_length + len(block) > max_context_chars:
                logger.info(f"Context expansion truncated at {expanded_included} expanded chunks")
                break
            
            context_parts.append(block)
            current_length += len(block)
            expanded_included += 1
        
        logger.info(f"Built context: {main_included} main + {expanded_included} expanded chunks, {current_length} chars")
        return "\n".join(context_parts), main_included, expanded_included

    # NEW: thin wrapper to call Gemini with a configurable timeout
    def _call_gemini_once(self, model_name, prompt, temperature=0.2, max_output_tokens=512, timeout_s=30):
        model = generativeai.GenerativeModel(model_name)
        # NOTE: google-generativeai SDK does not support request_options in generate_content().
        # Timeout is handled at the underlying gRPC/HTTP layer with default/environment settings.
        # For strict timeout control, consider using asyncio or threading wrapper (future enhancement).
        return model.generate_content(
            prompt,
            generation_config=generativeai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                candidate_count=1,
            ),
        )

    # UPDATED: retry logic with handling for safety filter finish_reason=2, progressive token increase,
    # and exponential backoff for transient Gemini/network errors
    def _gemini_generate_with_retry(self, prompt, attempts=3, base_timeout=30, backoff=2.0,
                                    initial_max_tokens=2048, max_tokens_cap=4096, context_text=None, question=None):
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        last_exc = None

        # Ensure sensible integers
        try:
            initial_max_tokens = int(os.getenv("GEMINI_INITIAL_MAX_TOKENS", str(initial_max_tokens)))
        except Exception:
            initial_max_tokens = initial_max_tokens
        try:
            max_tokens_cap = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", str(max_tokens_cap)))
        except Exception:
            max_tokens_cap = max_tokens_cap

        for i in range(attempts):
            timeout = int(base_timeout * (backoff ** i))
            # progressive token strategy: increase tokens on each retry (but cap)
            try:
                tokens = min(int(initial_max_tokens * (backoff ** i)), max_tokens_cap)
            except Exception:
                tokens = initial_max_tokens

            # lower temperature on retries to reduce unexpected outputs
            temperature = 0.2 if i == 0 else 0.05

            try:
                logger.info(f"Calling Gemini attempt {i+1}/{attempts} (timeout={timeout}s, tokens={tokens}, temp={temperature}) using model={model_name}")
                resp = self._call_gemini_once(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=tokens,
                    timeout_s=timeout,
                )
                text, finish_reason = self._extract_gemini_text(resp)

                # Interpret finish_reason numeric/string values
                try:
                    fr_val = int(finish_reason) if finish_reason is not None and str(finish_reason).isdigit() else None
                except Exception:
                    fr_val = None

                # If no text and finish_reason suggests token limit, try increasing tokens (if we haven't hit cap)
                if (not text) and (fr_val == 4 or (isinstance(finish_reason, str) and "max" in finish_reason.lower() or "token" in str(finish_reason).lower())):
                    logger.warning(f"Gemini likely hit max tokens (finish_reason={finish_reason}). Attempt {i+1}/{attempts}.")
                    # If we can increase tokens, continue to next attempt with larger tokens
                    # The loop already increases tokens by factor backoff, so just continue
                    continue

                # If safety filter triggered (finish_reason == 2), attempt sanitized prompt if available
                if (not text) and (fr_val == 2 or (isinstance(finish_reason, str) and "safety" in finish_reason.lower())):
                    logger.warning(f"Safety filters triggered (finish_reason={finish_reason}), attempt {i+1}/{attempts}.")
                    if context_text and question:
                        prompt = self._sanitize_prompt_for_safety(question, context_text)
                        # After sanitizing, try again (loop will continue)
                        continue
                    # if no context/question to sanitize, just continue retries
                    continue

                # If we have text or other finish reasons, return result
                return text, finish_reason, resp

            except (RetryError, DeadlineExceeded, ServiceUnavailable, GoogleAPICallError) as e:
                last_exc = e
                logger.warning(f"Gemini transient error (attempt {i+1}/{attempts}, timeout={timeout}s): {e}")
                time.sleep(min(1.5 * (i + 1), 5))
            except Exception as e:
                last_exc = e
                logger.exception(f"Unexpected error calling Gemini (attempt {i+1}/{attempts}): {e}")
                time.sleep(1)

        # Exhausted attempts
        # last_exc may be None or not an Exception subclass (e.g. a string). Raise a proper exception.
        if last_exc is None:
            raise RuntimeError("Gemini exhausted retries without an exception (likely repeated safety filters or empty responses).")
        if isinstance(last_exc, BaseException):
            raise last_exc
        # fallback for non-exception values
        raise RuntimeError(f"Gemini failed with non-exception value: {last_exc}")

    def _qdrant_search_with_fallback(self, client, collection_name, question_embedding, top_k, initial_threshold):
        """
        Try Qdrant search with descending thresholds until we get any hits.
        """
        thresholds = []
        if initial_threshold is not None:
            thresholds.append(initial_threshold)
        # IMPROVEMENT: Updated fallback cascade to avoid None (no threshold)
        # Prevents retrieving completely irrelevant chunks
        thresholds.extend([0.4, 0.35, 0.25])

        for t in thresholds:
            try:
                kwargs = {
                    "collection_name": collection_name,
                    "query_vector": ("default", question_embedding),
                    "limit": top_k,
                    "with_payload": True,
                }
                if t is not None:
                    kwargs["score_threshold"] = float(t)
                search_result = client.search(**kwargs)
                logger.info(f"Qdrant returned {len(search_result)} results with score_threshold={t}.")
                if search_result:
                    return search_result, t
            except Exception as e:
                logger.exception(f"Error during Qdrant search at threshold {t}: {e}")
                continue
        return [], None

    def _qdrant_search_with_fallback_filtered(self, client, collection_name, question_embedding, top_k, initial_threshold, search_filter=None):
        """
        Try Qdrant search with descending thresholds until we get any hits.
        Supports optional filter for document-specific searches.
        """
        thresholds = []
        if initial_threshold is not None:
            thresholds.append(initial_threshold)
        # IMPROVEMENT: Updated fallback cascade to avoid None (no threshold)
        # Prevents retrieving completely irrelevant chunks
        thresholds.extend([0.4, 0.35, 0.25])

        for t in thresholds:
            try:
                kwargs = {
                    "collection_name": collection_name,
                    "query_vector": ("default", question_embedding),
                    "limit": top_k,
                    "with_payload": True,
                }
                if t is not None:
                    kwargs["score_threshold"] = float(t)
                if search_filter is not None:
                    kwargs["query_filter"] = search_filter
                search_result = client.search(**kwargs)
                logger.info(f"Qdrant returned {len(search_result)} results with score_threshold={t}, filter={bool(search_filter)}.")
                if search_result:
                    return search_result, t
            except Exception as e:
                logger.exception(f"Error during Qdrant search at threshold {t}: {e}")
                continue
        return [], None

    def _hybrid_search_with_fallback(
        self,
        question: str,
        question_embedding: List[float],
        collection_name: str,
        top_k: int,
        initial_threshold: float,
        search_filter: Optional[Filter] = None
    ):
        """
        Perform hybrid search (dense + sparse) with fallback to dense-only.
        
        Args:
            question: Original question text for sparse vector generation
            question_embedding: Dense embedding vector
            collection_name: Qdrant collection name
            top_k: Number of results to retrieve
            initial_threshold: Initial score threshold
            search_filter: Optional document filter
            
        Returns:
            Tuple of (results, used_threshold, is_hybrid)
        """
        enable_hybrid = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
        
        if enable_hybrid:
            try:
                # Generate sparse vector for query
                sparse_vector = generate_sparse_vector(question)
                
                if sparse_vector is None:
                    logger.warning("Could not generate sparse vector, falling back to dense-only")
                else:
                    # Try hybrid search
                    results, used_threshold = hybrid_search_with_fallback(
                        dense_vector=question_embedding,
                        sparse_vector=sparse_vector,
                        collection_name=collection_name,
                        limit=top_k,
                        initial_threshold=initial_threshold,
                        search_filter=search_filter,
                    )
                    
                    if results:
                        # Convert HybridSearchResult to compatible format
                        converted_results = []
                        for r in results:
                            # Create a mock object with the expected attributes
                            result_obj = type('HybridResult', (), {
                                'id': r.id,
                                'score': r.score,
                                'payload': r.payload,
                                'dense_score': r.dense_score,
                                'sparse_score': r.sparse_score,
                                'fusion_method': r.fusion_method,
                            })()
                            converted_results.append(result_obj)
                        
                        logger.info(f"Hybrid search returned {len(converted_results)} results (threshold={used_threshold})")
                        return converted_results, used_threshold, True
                        
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}. Falling back to dense-only.")
        
        # Fallback to dense-only search
        client = get_qdrant_client()
        results, used_threshold = self._qdrant_search_with_fallback_filtered(
            client, collection_name, question_embedding, top_k, initial_threshold, search_filter
        )
        return results, used_threshold, False

    def _search_with_expanded_queries_hybrid(
        self,
        client,
        collection_name: str,
        question: str,
        embedding_model,
        top_k: int,
        initial_threshold: float,
        search_filter: Optional[Filter] = None
    ):
        """
        Perform hybrid retrieval using expanded queries for better recall.
        Combines results from multiple query variants and deduplicates.
        
        Returns:
            Tuple of (merged_results, used_threshold, expansion_stats, is_hybrid)
        """
        # Get query variants
        query_variants = self._expand_query(question)
        
        all_results = []
        seen_ids = set()
        used_threshold = initial_threshold
        is_hybrid = False
        
        for i, query in enumerate(query_variants):
            # Embed the query variant
            query_embedding = embedding_model.embed_texts([query])[0]
            
            # Search with this variant using hybrid search
            results, threshold, hybrid_used = self._hybrid_search_with_fallback(
                question=query,
                question_embedding=query_embedding,
                collection_name=collection_name,
                top_k=top_k,
                initial_threshold=initial_threshold,
                search_filter=search_filter,
            )
            
            if threshold is not None:
                used_threshold = threshold
            if hybrid_used:
                is_hybrid = True
            
            # Add unique results
            for result in results:
                if result.id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.id)
            
            logger.debug(f"Query variant {i+1}/{len(query_variants)}: '{query[:30]}...' returned {len(results)} results (hybrid={hybrid_used})")
        
        # Sort all results by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k unique results
        merged_results = all_results[:top_k]
        
        expansion_stats = {
            "num_variants": len(query_variants),
            "variants": query_variants[:3],
            "total_unique_results": len(all_results),
            "merged_to": len(merged_results),
            "hybrid_used": is_hybrid,
        }
        
        logger.info(f"Hybrid query expansion: {len(query_variants)} variants → {len(all_results)} unique → top {len(merged_results)} (hybrid={is_hybrid})")
        
        return merged_results, used_threshold, expansion_stats, is_hybrid

    def _expand_query(self, question, num_variants=5):
        """
        Generate query variants using Gemini for better recall.
        Returns list of query strings including the original question.
        Phase 1 improvement: increased from 3 to 5 variants for better vocabulary coverage.
        """
        # Check if query expansion is enabled
        if os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() != "true":
            return [question]
        
        try:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            model = generativeai.GenerativeModel(model_name)
            
            # FIX: Sanitize question before expansion to reduce safety triggers
            safe_question = question
            safety_replacements = {
                "giết": "xử lý", "bắn": "thao tác", "tấn công": "tiếp cận",
                "phá hủy": "thay đổi", "chết": "dừng", "vũ khí": "thiết bị",
                "hại": "ảnh hưởng", "đánh": "tác động",
                "kill": "handle", "attack": "access", "destroy": "modify",
                "harm": "affect", "weapon": "tool",
            }
            for trg, rep in safety_replacements.items():
                safe_question = re.sub(re.escape(trg), rep, safe_question, flags=re.IGNORECASE)
            
            # FIX: More neutral expansion prompt
            expansion_prompt = f"""Bạn là trợ lý tìm kiếm tài liệu. Viết lại câu hỏi sau thành {num_variants} cách diễn đạt khác nhau.

Câu hỏi: "{safe_question}"

Yêu cầu:
- Mỗi phiên bản trên một dòng, bắt đầu bằng số (1., 2., 3.)
- Giữ nguyên ý nghĩa, thay đổi từ ngữ hoặc cấu trúc câu
- Không thêm thông tin mới

Các phiên bản:"""

            resp = model.generate_content(
                expansion_prompt,
                generation_config=generativeai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=300,
                    candidate_count=1,
                ),
            )
            
            text, finish_reason = self._extract_gemini_text(resp)
            
            # FIX: Log the actual question that triggered the filter for debugging
            if not text or not text.strip():
                logger.warning(f"Query expansion blocked (finish_reason={finish_reason}) for: '{question[:80]}...'")
                return [question]

            # Parse variants with robust error handling
            variants = [question]  # Always include original
            lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
            
            parsed_count = 0
            for line in lines:
                # Match: "1. text" or "1) text" or "1: text" or just "1 text"
                match = re.match(r'^\d+[\.\)\:\-\s]+(.+)$', line)
                if match:
                    variant = match.group(1).strip()
                    # Clean quotes and validate
                    variant = variant.strip('"').strip("'").strip()
                    
                    # Sanity checks - adjusted length limits
                    if (variant and 
                        variant.lower() != question.lower() and  # Case-insensitive comparison
                        len(variant) >= 5 and 
                        len(variant) <= len(question) * 3):  # Allow longer variants
                        variants.append(variant)
                        parsed_count += 1
                        logger.debug(f"Parsed variant {parsed_count}: {variant[:50]}...")
                elif len(line) >= 10 and not line[0].isdigit():
                    # Fallback: treat entire line as variant if it looks like a question
                    if line.lower() != question.lower() and len(line) <= len(question) * 3:
                        variants.append(line)
                        parsed_count += 1
                        logger.debug(f"Parsed fallback variant: {line[:50]}...")
            
            if parsed_count == 0:
                logger.warning(f"Query expansion parsed 0 variants from response: {text[:100]}...")
                return [question]
            
            # Limit to requested number + original
            variants = variants[:num_variants + 1]
            logger.info(f"Query expansion: generated {len(variants)-1} variants for '{question[:50]}...'")
            
            return variants
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original question.")
            return [question]

    def post(self, request):
        try:
            start_time = time.time()
            
            # Step 1: Normalize question with preprocessing
            raw_question = request.data.get("question", "").strip()
            document_id = request.data.get("document_id")  # Document filter
            final_k = int(request.data.get("final_k", 10))
            # IMPROVEMENT 1: Increase initial retrieval to 8x final_k for better recall
            # Phase 1 "Expand, Then Filter": retrieve 80+ candidates, rerank, take top 10
            top_k = int(request.data.get("top_k", final_k * 8))  # 8x multiplier for high recall
            
            if not raw_question:
                logger.error("No question provided in the request.")
                return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Apply consistent preprocessing to match document preprocessing
            question = preprocess_query(raw_question)
            logger.info(f"Query preprocessing: '{raw_question}' → '{question}'")
            logger.info(f"Document filter: {document_id if document_id else 'None (search all)'}")

            # Generate a stable question_id for logging
            question_id = str(uuid.uuid4())
            correlation_id = str(uuid.uuid4())

            # Optional: quick language hint for debugging
            try:
                lang = detect(question)
                logger.info(f"Detected question language: {lang}")
            except Exception:
                lang = "unknown"

            # Step 2: Embed the question
            embedding_model = HuggingfaceEmbeddingsModel()
            question_embedding = embedding_model.embed_texts([question])[0]
            logger.debug("Question embedding generated successfully.")

            # Step 3: Retrieve from Qdrant with document filter
            client = get_qdrant_client()
            collection_name = settings.QDRANT_COLLECTION

            score_threshold_param = request.data.get("score_threshold", None)
            # IMPROVEMENT: Raised threshold to 0.5 to reduce false positives
            # 0.3 cosine similarity (≈72.5° angle) was too permissive
            initial_threshold = float(os.getenv("SCORE_THRESHOLD", "0.55"))
            
            use_expansion = request.data.get("expand_query", os.getenv("ENABLE_QUERY_EXPANSION", "true")).lower() == "true"

            logger.info(f"Retrieving up to {top_k} chunks from Qdrant (threshold={initial_threshold}, expand={use_expansion}, doc_filter={bool(document_id)}).")

            # Build document filter if provided
            search_filter = None
            if document_id:
                from qdrant_client import models
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
                logger.info(f"Applying document filter: document_id={document_id}")

            # Try Qdrant with optional query expansion and document filter (hybrid search)
            search_result = None
            used_threshold = None
            source = "qdrant"
            expansion_stats = None
            is_hybrid = False
            
            try:
                if use_expansion:
                    # Modified search with filter support and hybrid search
                    search_result, used_threshold, expansion_stats, is_hybrid = self._search_with_expanded_queries_hybrid(
                        client, collection_name, question, embedding_model, top_k, initial_threshold, search_filter
                    )
                else:
                    # Single-query hybrid search with filter
                    search_result, used_threshold, is_hybrid = self._hybrid_search_with_fallback(
                        question=question,
                        question_embedding=question_embedding,
                        collection_name=collection_name,
                        top_k=top_k,
                        initial_threshold=initial_threshold,
                        search_filter=search_filter,
                    )
                
                if is_hybrid:
                    source = "qdrant_hybrid"
                    
            except Exception as e:
                logger.warning(f"Qdrant search failed: {e}. Falling back to MongoDB.")
                source = "mongodb"
            
            # If Qdrant failed or returned no results, try MongoDB
            if not search_result:
                logger.info("Attempting MongoDB fallback retrieval...")
                search_result = self._retrieve_from_mongodb(question_embedding, top_k)
                source = "mongodb" if search_result else "none"
            
            if not search_result:
                logger.warning(f"No results found from either Qdrant or MongoDB. lang={lang}")
                return Response(
                    {
                        "answer": "Tôi không tìm thấy thông tin trong tài liệu.",
                        "citations": [],
                        "debug": {
                            "reason": "No retrieval results",
                            "lang": lang,
                            "used_threshold": used_threshold,
                            "retrieval_source": source,
                            "document_filter": document_id
                        }
                    },
                    status=status.HTTP_200_OK
                )

            # IMPROVEMENT 3: Strict document filtering validation
            initial_count = len(search_result)
            if document_id:
                # Filter out any chunks that don't match the specified document
                filtered_search_result = []
                wrong_doc_count = 0
                for result in search_result:
                    result_doc_id = result.payload.get('document_id', '')
                    if result_doc_id == document_id:
                        filtered_search_result.append(result)
                    else:
                        wrong_doc_count += 1
                        logger.warning(f"Filtered chunk {result.id} from wrong document: {result_doc_id} (expected: {document_id})")
                
                search_result = filtered_search_result
                
                # IMPROVEMENT 5: Enhanced logging with filtering metrics
                if wrong_doc_count > 0:
                    logger.warning(f"Document filtering: {initial_count} → {len(search_result)} chunks (removed {wrong_doc_count} from wrong documents)")
                else:
                    logger.info(f"Document filtering: All {len(search_result)} chunks matched document_id={document_id}")
            
            logger.info(f"Retrieved {len(search_result)} results from {source} (initial: {initial_count}, after filtering: {len(search_result)}).")
            
            # Check if filtering removed all results
            if not search_result:
                logger.warning(f"No results remaining after document filtering for document_id={document_id}")
                return Response(
                    {
                        "answer": "Không tìm thấy thông tin liên quan trong tài liệu được chỉ định.",
                        "citations": [],
                        "debug": {
                            "reason": "No results after document filtering",
                            "initial_count": initial_count,
                            "filtered_count": 0,
                            "document_id": document_id,
                            "retrieval_source": source
                        }
                    },
                    status=status.HTTP_200_OK
                )

            # Step 4: Rerank results using cached CrossEncoder (Phase 2 optimization)
            rerank_start = time.time()
            reranker = self.get_reranker()
            
            # FIX: Truncate long texts to avoid tokenizer warning
            max_chunk_chars = int(os.getenv("RERANK_MAX_CHUNK_CHARS", "1500"))  # ~400 tokens
            rerank_batch_size = int(os.getenv("RERANK_BATCH_SIZE", "32"))  # GPU memory optimization
            
            rerank_inputs = [
                [question, result.payload['text'][:max_chunk_chars]] 
                for result in search_result
            ]
            rerank_scores = reranker.predict(rerank_inputs, batch_size=rerank_batch_size, convert_to_numpy=True)
            rerank_latency_ms = int((time.time() - rerank_start) * 1000)
            
            # Phase 2: Compute reranking score distribution for monitoring
            rerank_score_stats = {
                "min": float(np.min(rerank_scores)) if len(rerank_scores) > 0 else None,
                "max": float(np.max(rerank_scores)) if len(rerank_scores) > 0 else None,
                "mean": float(np.mean(rerank_scores)) if len(rerank_scores) > 0 else None,
                "median": float(np.median(rerank_scores)) if len(rerank_scores) > 0 else None,
                "std": float(np.std(rerank_scores)) if len(rerank_scores) > 0 else None,
            }
            logger.info(f"Reranking completed: {len(rerank_inputs)} candidates in {rerank_latency_ms}ms, scores: min={rerank_score_stats['min']:.3f}, max={rerank_score_stats['max']:.3f}, mean={rerank_score_stats['mean']:.3f}")

            # Combine and sort - use canonical chunk_id from payload, not Qdrant UUID
            reranked_results = [
                {
                    "id": result.payload.get('chunk_id', result.id),  # Prefer canonical chunk_id
                    "qdrant_id": result.id,  # Keep Qdrant UUID for debugging
                    "score": float(score),
                    "payload": result.payload
                }
                for result, score in zip(search_result, rerank_scores)
            ]
            reranked_results.sort(key=lambda x: x['score'], reverse=True)

            # IMPROVED: Filter by minimum rerank score first, then take top 10
            # Step 4a: Get rerank score threshold from env or request
            min_rerank_score = float(request.data.get("min_rerank_score", os.getenv("MIN_RERANK_SCORE", "0.1")))
            
            # Step 4b: Filter chunks that pass minimum rerank score
            passed_threshold = [r for r in reranked_results if r['score'] >= min_rerank_score]
            failed_threshold_count = len(reranked_results) - len(passed_threshold)
            
            logger.info(f"Rerank filtering: {len(reranked_results)} total, {len(passed_threshold)} passed min_score={min_rerank_score}, {failed_threshold_count} filtered out")
            
            # Step 4c: Take top 10 from those that passed
            filtered_results = passed_threshold[:10]
            
            # Step 4d: Final top_k selection
            top_results = filtered_results[:final_k]
            logger.info(f"Reranking: {len(search_result)} → {len(passed_threshold)} (passed threshold) → top {len(filtered_results)} → final {len(top_results)}")

            # Log retrieval with hybrid search effectiveness tracking
            try:
                # Use canonical chunk_id from payload, not Qdrant UUID
                retrieved_chunk_ids = [result.payload.get('chunk_id', result.id) for result in search_result]
                retrieved_scores = [result.score for result in search_result]
                
                # Extract hybrid search metrics for effectiveness tracking
                hybrid_metrics = {
                    "is_hybrid": is_hybrid,
                    "source": source,
                    "used_threshold": used_threshold,
                    "query_expansion_enabled": use_expansion,
                }
                
                # Phase 2: Add reranking metrics for monitoring
                rerank_metrics = {
                    "model": os.getenv("RERANKER_MODEL", "itdainb/PhoRanker"),
                    "candidates_reranked": len(rerank_inputs),
                    "passed_threshold": len(passed_threshold),
                    "min_rerank_score_used": min_rerank_score,
                    "latency_ms": rerank_latency_ms,
                    "score_stats": rerank_score_stats,
                    "batch_size": rerank_batch_size,
                }
                
                # Extract dense/sparse score contributions if available (hybrid search)
                if is_hybrid and search_result:
                    dense_scores = []
                    sparse_scores = []
                    for result in search_result:
                        dense_scores.append(getattr(result, 'dense_score', None))
                        sparse_scores.append(getattr(result, 'sparse_score', None))
                    hybrid_metrics["dense_scores"] = [s for s in dense_scores if s is not None]
                    hybrid_metrics["sparse_scores"] = [s for s in sparse_scores if s is not None]
                    hybrid_metrics["fusion_method"] = getattr(search_result[0], 'fusion_method', None) if search_result else None
                
                # Add expansion stats if available
                if expansion_stats:
                    hybrid_metrics["expansion_stats"] = expansion_stats
                
                self.mongo_repo.db['retrieval_log'].insert_one({
                    "question_id": question_id,
                    "correlation_id": correlation_id,
                    "question": question,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "retrieved_scores": retrieved_scores,
                    "top_k": final_k,
                    "initial_top_k": top_k,  # Track expanded retrieval count
                    "document_id": document_id or (top_results[0]['payload'].get('document_id') if top_results else None),
                    "hybrid_metrics": hybrid_metrics,  # Phase 1 effectiveness tracking
                    "rerank_metrics": rerank_metrics,  # Phase 2 reranking metrics
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "ts": datetime.now(),
                    "user_id": None,
                })
                logger.info(f"Logged retrieval to MongoDB: {len(retrieved_chunk_ids)} chunks, hybrid={is_hybrid}, source={source}, rerank_latency={rerank_latency_ms}ms")
            except Exception as e:
                logger.exception(f"Failed to log retrieval to MongoDB: {e}")
            
            # Step 4.5: Context Expansion - retrieve surrounding chunks for top results
            enable_expansion = os.getenv("ENABLE_CONTEXT_EXPANSION", "true").lower() == "true"
            expand_top_n = int(os.getenv("CONTEXT_EXPANSION_TOP_N", "3"))
            expansion_radius = int(os.getenv("CONTEXT_EXPANSION_RADIUS", "1"))
            
            expanded_chunks = []
            chunk_sources = {}
            
            if enable_expansion and top_results:
                try:
                    expanded_chunks, chunk_sources = self._expand_chunk_context(
                        top_results,
                        expand_top_n=expand_top_n,
                        radius=expansion_radius
                    )
                    logger.info(f"Context expansion: {len(expanded_chunks)} adjacent chunks retrieved")
                except Exception as e:
                    logger.warning(f"Context expansion failed: {e}. Proceeding without expansion.")
                    expanded_chunks = []
                    chunk_sources = {r['id']: 'main' for r in top_results}
            else:
                chunk_sources = {r['id']: 'main' for r in top_results}
            
            # Step 5: Generate answer with Gemini
            google_api_key = os.getenv("GEMINI_API_KEY")
            if not google_api_key:
                logger.error("Google API key is not configured.")
                return Response({"error": "Google API key is not configured."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            generativeai.configure(api_key=google_api_key)

            # IMPROVEMENT 4: Enhanced prompt with better Vietnamese instructions and citation guidance
            system_message = (
                "Bạn là trợ lý AI thông minh chuyên phân tích tài liệu kỹ thuật.\n\n"
                "QUY TẮC TRẢ LỜI:\n"
                "1. Trả lời TRỰC TIẾP và SÚC TÍCH dựa vào ngữ cảnh được cung cấp\n"
                "2. CHỈ sử dụng thông tin có trong tài liệu tham khảo\n"
                "3. LUÔN trích dẫn nguồn theo định dạng: [document_id tr.X]\n"
                "4. Nếu cần kết hợp nhiều đoạn, hãy tổng hợp rõ ràng và trích dẫn TẤT CẢ nguồn\n"
                "5. Nếu KHÔNG tìm thấy thông tin, trả lời: 'Không tìm thấy thông tin trong tài liệu.'\n"
                "6. Ưu tiên ĐỘ CHÍNH XÁC hơn độ dài câu trả lời\n"
                "7. Trả lời bằng tiếng Việt, ngắn gọn nhưng đầy đủ ý"
            )

            # Build context with expansion if available
            MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
            
            if expanded_chunks:
                context_text, main_count, expanded_count = self._build_context_with_expansion(
                    top_results,
                    expanded_chunks,
                    max_context_chars=MAX_CONTEXT_CHARS,
                    prioritize_main=True
                )
            else:
                context_text = self._build_context_text(top_results)
                main_count = len(top_results)
                expanded_count = 0
                
                if len(context_text) > MAX_CONTEXT_CHARS:
                    logger.info(f"Trimming context from {len(context_text)} to {MAX_CONTEXT_CHARS} chars.")
                    context_text = context_text[:MAX_CONTEXT_CHARS]

            prompt = f"{system_message}\n\nNGỮ CẢNH:\n{context_text}\n\nCâu hỏi: {question}\nCâu trả lời:"

            try:
                llm_timeout = int(os.getenv("GEMINI_TIMEOUT", "30"))
                llm_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
                initial_max_tokens = int(os.getenv("GEMINI_INITIAL_MAX_TOKENS", "2048"))
                max_tokens_cap = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
            except Exception:
                llm_timeout, llm_attempts, initial_max_tokens, max_tokens_cap = 30, 3, 2048, 4096

            try:
                response_text, finish_reason, _ = self._gemini_generate_with_retry(
                    prompt,
                    attempts=llm_attempts,
                    base_timeout=llm_timeout,
                    backoff=2.0,
                    initial_max_tokens=initial_max_tokens,
                    max_tokens_cap=max_tokens_cap,
                    context_text=context_text,
                    question=question,
                )

                if not response_text:
                    logger.warning(f"Gemini returned no content. finish_reason={finish_reason}")
                    return Response(
                        {"error": "LLM did not return content.", "finish_reason": str(finish_reason)},
                        status=status.HTTP_502_BAD_GATEWAY
                    )

                response = response_text
                logger.info("Response generated successfully.")
                
                # ADD GENERATION LOGGING HERE
                try:
                    generation_latency = int((time.time() - start_time) * 1000)
                    
                    # Extract citations from top_results (already available)
                    # Use canonical chunk_id from payload, not Qdrant UUID
                    citation_list = [
                        {
                            "chunk_id": result['payload'].get('chunk_id', result['id']),  # Prefer payload chunk_id
                            "document_id": result['payload']['document_id'],
                            "page": result['payload'].get('page', 'N/A'),
                            "score": result['score'],
                            "qdrant_id": result['id']  # Keep Qdrant UUID for reference
                        }
                        for result in top_results
                    ]
                    
                    self.mongo_repo.db['generation_log'].insert_one({
                        "question_id": question_id,
                        "correlation_id": correlation_id,
                        "question_text": question,
                        "raw_question": raw_question,
                        "document_id": document_id or (top_results[0]['payload'].get('document_id') if top_results else None),
                        "generated_answer": response,
                        "citations": citation_list,
                        "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                        "reranker_model": self._reranker_model_name or os.getenv("RERANKER_MODEL"),
                        "finish_reason": str(finish_reason) if finish_reason else None,
                        "context_length": len(context_text),
                        "num_chunks_used": len(top_results),
                        "num_expanded_chunks": len(expanded_chunks),
                        "chunk_sources": chunk_sources,
                        "expansion_config": {
                            "enabled": enable_expansion,
                            "top_n": expand_top_n,
                            "radius": expansion_radius
                        },
                        "latency_ms": generation_latency,
                        "ts": datetime.now(),
                        "user_id": None,
                    })
                    logger.info(f"Logged generation to MongoDB: answer_length={len(response)}, citations={len(citation_list)}")
                    
                    # ADD: Log each citation separately to citation_log collection
                    if citation_list:
                        citation_docs = []
                        for rank, citation in enumerate(citation_list, 1):
                            citation_docs.append({
                                "question_id": question_id,
                                "correlation_id": correlation_id,
                                "question_text": question,  # Added for eval matching
                                "citation_rank": rank,
                                "chunk_id": citation['chunk_id'],  # Now uses canonical format
                                "qdrant_id": citation.get('qdrant_id'),  # UUID for debugging
                                "document_id": citation['document_id'],
                                "page": citation['page'],
                                "rerank_score": citation['score'],
                                "source": chunk_sources.get(citation['chunk_id'], 'main'),  # 'main' or 'expanded'
                                "ts": datetime.now(),
                                "used_in_generation": True
                            })
                        
                        # Also log expanded chunks used in context
                        for exp_chunk in expanded_chunks:
                            citation_docs.append({
                                "question_id": question_id,
                                "correlation_id": correlation_id,
                                "question_text": question,  # Added for eval matching
                                "citation_rank": None,  # Not ranked, supplementary
                                "chunk_id": exp_chunk['chunk_id'],
                                "qdrant_id": None,
                                "document_id": exp_chunk['document_id'],
                                "page": exp_chunk.get('page'),
                                "rerank_score": None,
                                "source": 'expanded',
                                "position": exp_chunk.get('position'),  # 'prev' or 'next'
                                "parent_chunk_id": exp_chunk.get('parent_chunk_id'),
                                "ts": datetime.now(),
                                "used_in_generation": True
                            })
                        
                        if citation_docs:
                            self.mongo_repo.db['citation_log'].insert_many(citation_docs)
                            logger.info(f"Logged {len(citation_docs)} citations to citation_log (including {len(expanded_chunks)} expanded)")
                    
                except Exception as e:
                    logger.exception(f"Failed to log generation to MongoDB: {e}")
                
            except Exception as e:
                logger.exception("Gemini call failed after retries.")
                
                # LOG FAILED GENERATION TOO
                try:
                    self.mongo_repo.db['generation_log'].insert_one({
                        "question_id": question_id,
                        "correlation_id": correlation_id,
                        "question_text": question,
                        "raw_question": raw_question,
                        "document_id": document_id,
                        "generated_answer": None,
                        "citations": [],
                        "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                        "error": str(e)[:500],
                        "status": "failed",
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "ts": datetime.now(),
                        "user_id": None,
                    })
                except Exception as log_err:
                    logger.error(f"Failed to log generation error: {log_err}")
                
                citations = [
                    {"document_id": r['payload']['document_id'], "page": r['payload'].get('page', 'N/A'), "chunk_id": r['id']}  # r['id'] is now canonical chunk_id
                    for r in top_results
                ]
                return Response(
                    {
                        "answer": "Xin lỗi, dịch vụ tạo câu trả lời đang tạm thời không khả dụng. Vui lòng thử lại sau.",
                        "citations": citations,
                        "debug": {"llm_error": str(e)[:200]}
                    },
                    status=status.HTTP_200_OK
                )

            # Step 6: Extract citations and return response
            citations = [
                {
                    "document_id": result['payload']['document_id'],
                    "page": result['payload'].get('page', 'N/A'),
                    "chunk_id": result['payload'].get('chunk_id', result['id'])  # Use canonical format
                }
                for result in top_results
            ]
            logger.debug(f"Mapped {len(citations)} citations successfully.")

            # IMPROVEMENT 5: Add comprehensive metrics in response
            total_latency = int((time.time() - start_time) * 1000)
            
            return Response({
                "answer": response,
                "citations": citations,
                "debug": {
                    "retrieval_source": source,
                    "retrieval_metrics": {
                        "initial_count": initial_count,
                        "after_filtering": len(search_result),
                        "after_reranking": len(reranked_results),
                        "passed_min_rerank_score": len(passed_threshold),
                        "min_rerank_score_used": min_rerank_score,
                        "failed_rerank_threshold": failed_threshold_count,
                        "top_10_selected": len(filtered_results),
                        "final_top_k": len(top_results),
                        "chunks_filtered_out": initial_count - len(search_result)
                    },
                    "context_expansion": {
                        "enabled": enable_expansion,
                        "top_n_expanded": expand_top_n,
                        "radius": expansion_radius,
                        "expanded_chunks_count": len(expanded_chunks),
                        "main_chunks_in_context": main_count if expanded_chunks else len(top_results),
                        "expanded_chunks_in_context": expanded_count if expanded_chunks else 0
                    },
                    "used_threshold": used_threshold,
                    "document_filter": document_id,
                    "query_expansion_stats": expansion_stats,
                    "total_latency_ms": total_latency
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception(f"Unexpected error in post method: {e}")
            return Response(
                {"error": f"An unexpected error occurred: {str(e)[:200]}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
