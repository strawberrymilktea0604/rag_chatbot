"""
Qdrant client with hybrid search support (dense + sparse vectors).

Features:
- Dense vector search using sentence-transformers embeddings
- Sparse vector search using BM25/TF-IDF with Vietnamese tokenization
- Hybrid search combining both with RRF or weighted fusion
- Vietnamese compound word tokenization via underthesea
"""

import os
import pickle
import logging
import time
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, Distance, SparseVectorParams, SparseIndexParams,
    PointStruct, ScoredPoint, Filter, SearchParams,
    NamedVector, NamedSparseVector, SparseVector
)

logger = logging.getLogger(__name__)

# Qdrant server URL
QDRANT_SERVER_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Initialize Qdrant client in server mode
client = QdrantClient(url=QDRANT_SERVER_URL)


def get_qdrant_client():
    """
    Returns the Qdrant client instance.
    """
    return client


def collection_exists(collection_name: str) -> bool:
    """
    Return True if collection exists, False otherwise.
    Non-destructive: does not create the collection.
    """
    try:
        collections = client.get_collections()
        return any(c.name == collection_name for c in collections.collections)
    except Exception:
        # Fallback to single-collection check if get_collections fails
        try:
            client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False


def get_collection_info(collection_name: str):
    """
    Return collection info dict or None if not found.
    """
    try:
        info = client.get_collection(collection_name=collection_name)
        return info.dict()
    except Exception:
        return None


def create_collection(collection_name, vector_size=384, distance=Distance.COSINE):
    """
    Creates a collection in Qdrant if it doesn't already exist.
    Legacy function - use create_hybrid_collection for new collections.
    """
    qdrant = get_qdrant_client()
    collections = qdrant.get_collections()
    exists = any(collection.name == collection_name for collection in collections.collections)

    if not exists:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")


def create_hybrid_collection(
    collection_name: str,
    dense_vector_size: int = 768,
    distance: Distance = Distance.COSINE,
    sparse_on_disk: bool = False
) -> bool:
    """
    Creates a collection with both dense and sparse vector support for hybrid search.
    
    Args:
        collection_name: Name of the collection
        dense_vector_size: Dimension of dense vectors (768 for Vietnamese embedding)
        distance: Distance metric for dense vectors (COSINE recommended)
        sparse_on_disk: Whether to store sparse index on disk (False = in memory)
        
    Returns:
        True if created successfully, False if already exists
    """
    qdrant = get_qdrant_client()
    
    # Check if collection exists
    try:
        collections = qdrant.get_collections()
        if any(c.name == collection_name for c in collections.collections):
            logger.info(f"Collection '{collection_name}' already exists")
            return False
    except Exception as e:
        logger.warning(f"Could not check collections: {e}")
    
    try:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=dense_vector_size,
                    distance=distance
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=sparse_on_disk
                    )
                )
            }
        )
        logger.info(f"Created hybrid collection '{collection_name}' with dense ({dense_vector_size}-dim) and sparse vectors")
        return True
    except Exception as e:
        logger.error(f"Failed to create hybrid collection: {e}")
        raise


# ============================================================================
# VIETNAMESE TOKENIZATION
# ============================================================================

def tokenize_vietnamese(text: str, use_underthesea: bool = True) -> List[str]:
    """
    Tokenize Vietnamese text for BM25/TF-IDF.
    
    Args:
        text: Input text
        use_underthesea: If True, use underthesea for compound word detection.
                        If False, use simple syllable-based tokenization.
                        
    Returns:
        List of tokens (lowercased)
    """
    if not text or not text.strip():
        return []
    
    if use_underthesea:
        try:
            from underthesea import word_tokenize
            # word_tokenize returns space-separated tokens with underscores for compounds
            tokenized = word_tokenize(text, format="text")
            tokens = tokenized.lower().split()
            return tokens
        except ImportError:
            logger.warning("underthesea not installed, falling back to syllable tokenization")
            use_underthesea = False
    
    if not use_underthesea:
        # Simple syllable-based tokenization
        return text.lower().split()


# ============================================================================
# SPARSE VECTOR BUILDER (BM25/TF-IDF)
# ============================================================================

@dataclass
class SparseVectorBuilder:
    """
    Builds sparse vectors using TF-IDF/BM25 scoring with Vietnamese tokenization.
    
    Usage:
        builder = SparseVectorBuilder()
        builder.fit(corpus)  # Build vocabulary from corpus
        sparse_vec = builder.transform("query text")  # Generate sparse vector
        builder.save_vocabulary("path.pkl")  # Persist vocabulary
        
    The sparse vector contains term IDs as indices and TF-IDF scores as values.
    """
    
    # BM25 parameters
    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Length normalization parameter
    
    # Configuration
    use_underthesea: bool = True
    top_k_terms: int = 150  # Limit sparse vector dimensions
    min_df: int = 1  # Minimum document frequency for a term
    
    # Vocabulary state
    vocabulary: Dict[str, int] = field(default_factory=dict)
    idf_scores: Dict[str, float] = field(default_factory=dict)
    avg_doc_length: float = 0.0
    total_docs: int = 0
    
    def fit(self, corpus: List[str]) -> "SparseVectorBuilder":
        """
        Build vocabulary and compute IDF scores from corpus.
        
        Args:
            corpus: List of document texts
            
        Returns:
            self for chaining
        """
        start_time = time.perf_counter()
        
        if not corpus:
            logger.warning("Empty corpus provided to SparseVectorBuilder.fit()")
            return self
        
        # Tokenize all documents
        tokenized_docs = []
        total_tokens = 0
        
        for doc in corpus:
            tokens = tokenize_vietnamese(doc, self.use_underthesea)
            tokenized_docs.append(tokens)
            total_tokens += len(tokens)
        
        self.total_docs = len(corpus)
        self.avg_doc_length = total_tokens / self.total_docs if self.total_docs > 0 else 0
        
        # Count document frequencies
        doc_freq: Counter = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Build vocabulary (filter by min_df)
        term_id = 0
        for term, df in doc_freq.items():
            if df >= self.min_df:
                self.vocabulary[term] = term_id
                term_id += 1
        
        # Compute IDF scores (BM25 variant)
        for term, df in doc_freq.items():
            if term in self.vocabulary:
                # BM25 IDF formula
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
                self.idf_scores[term] = max(idf, 0)  # Ensure non-negative
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"SparseVectorBuilder.fit(): {len(self.vocabulary)} terms, "
                   f"{self.total_docs} docs, avg_len={self.avg_doc_length:.1f}, "
                   f"took {elapsed:.1f}ms")
        
        return self
    
    def transform(self, text: str) -> Optional[SparseVector]:
        """
        Transform text into sparse vector using BM25 scoring.
        
        Args:
            text: Input text
            
        Returns:
            Qdrant SparseVector or None if no valid terms
        """
        if not self.vocabulary:
            logger.warning("SparseVectorBuilder not fitted, returning None")
            return None
        
        if not text or not text.strip():
            return None
        
        # Tokenize
        tokens = tokenize_vietnamese(text, self.use_underthesea)
        if not tokens:
            return None
        
        # Count term frequencies
        tf_counts = Counter(tokens)
        doc_length = len(tokens)
        
        # Compute BM25 scores for each term
        term_scores: List[Tuple[int, float]] = []
        
        for term, tf in tf_counts.items():
            if term not in self.vocabulary:
                continue
            
            term_id = self.vocabulary[term]
            idf = self.idf_scores.get(term, 0)
            
            if idf <= 0:
                continue
            
            # BM25 term frequency scoring
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1))
            )
            
            score = idf * tf_component
            if score > 0:
                term_scores.append((term_id, score))
        
        if not term_scores:
            return None
        
        # Sort by score and keep top_k
        term_scores.sort(key=lambda x: x[1], reverse=True)
        term_scores = term_scores[:self.top_k_terms]
        
        # Sort by term_id for Qdrant (indices must be sorted)
        term_scores.sort(key=lambda x: x[0])
        
        indices = [t[0] for t in term_scores]
        values = [t[1] for t in term_scores]
        
        return SparseVector(indices=indices, values=values)
    
    def transform_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[SparseVector]]:
        """
        Transform multiple texts into sparse vectors.
        
        Args:
            texts: List of input texts
            batch_size: Not used (for API compatibility)
            
        Returns:
            List of SparseVector or None for each text
        """
        return [self.transform(text) for text in texts]
    
    def save_vocabulary(self, path: str) -> None:
        """
        Save vocabulary and IDF scores to file.
        
        Args:
            path: File path (typically .pkl)
        """
        data = {
            "vocabulary": self.vocabulary,
            "idf_scores": self.idf_scores,
            "avg_doc_length": self.avg_doc_length,
            "total_docs": self.total_docs,
            "k1": self.k1,
            "b": self.b,
            "use_underthesea": self.use_underthesea,
            "top_k_terms": self.top_k_terms,
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved sparse vector vocabulary to {path} ({len(self.vocabulary)} terms)")
    
    @classmethod
    def load_vocabulary(cls, path: str) -> "SparseVectorBuilder":
        """
        Load vocabulary and IDF scores from file.
        
        Args:
            path: File path
            
        Returns:
            Initialized SparseVectorBuilder
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        builder = cls(
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
            use_underthesea=data.get("use_underthesea", True),
            top_k_terms=data.get("top_k_terms", 150),
        )
        builder.vocabulary = data["vocabulary"]
        builder.idf_scores = data["idf_scores"]
        builder.avg_doc_length = data["avg_doc_length"]
        builder.total_docs = data["total_docs"]
        
        logger.info(f"Loaded sparse vector vocabulary from {path} ({len(builder.vocabulary)} terms)")
        return builder


# Global sparse vector builder instance (lazy loaded)
_sparse_builder = None

def _init_sparse_builder():
    """Initialize sparse vector builder (lazy initialization)."""
    global _sparse_builder
    if _sparse_builder is None:
        try:
            # Use the custom SparseVectorBuilder class defined in this module
            _sparse_builder = SparseVectorBuilder()
            logger.info("SparseVectorBuilder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SparseVectorBuilder: {e}")
            _sparse_builder = False  # Mark as failed
    return _sparse_builder if _sparse_builder is not False else None

def get_sparse_builder():
    """Get or initialize the sparse vector builder."""
    return _init_sparse_builder()

def generate_sparse_vector(text: str):
    """
    Generate sparse vector from text using BM25-like approach.
    FIX: Better error handling and fallback.
    """
    try:
        builder = get_sparse_builder()
        if builder is None:
            logger.debug("Sparse builder unavailable, using fallback")
            return _generate_sparse_vector_bm25(text)
        
        # Use the transform method (not build_from_text)
        sparse_vector = builder.transform(text)
        if sparse_vector and hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
            if len(sparse_vector.indices) > 0:
                logger.debug(f"Generated sparse vector with {len(sparse_vector.indices)} indices")
                return sparse_vector
        
        logger.debug("SparseVectorBuilder produced empty result, using BM25 fallback")
        return _generate_sparse_vector_bm25(text)
        
    except Exception as e:
        logger.warning(f"Sparse vector generation failed: {e}. Using BM25 fallback.")
        return _generate_sparse_vector_bm25(text)

def _generate_sparse_vector_bm25(text: str):
    """
    Fallback: Generate simple BM25-like sparse vector.
    """
    try:
        from collections import Counter
        import re
        from qdrant_client.models import SparseVector
        
        # Simple tokenization
        tokens = re.findall(r'\w+', text.lower())
        token_counts = Counter(tokens)
        
        # Map tokens to indices (simple hash-based)
        indices = []
        values = []
        for token, count in token_counts.most_common(50):  # Limit to 50 terms
            token_idx = hash(token) % 10000  # Map to 0-10000 range
            indices.append(token_idx)
            values.append(float(count))
        
        if indices:
            sparse_vector = SparseVector(indices=indices, values=values)
            logger.debug(f"Generated BM25 sparse vector with {len(indices)} indices")
            return sparse_vector
        
        return None
        
    except Exception as e:
        logger.warning(f"BM25 fallback failed: {e}")
        return None


# ============================================================================
# HYBRID SEARCH
# ============================================================================

@dataclass
class HybridSearchResult:
    """Result from hybrid search with score breakdown."""
    id: str
    score: float  # Final fused score
    payload: Dict[str, Any]
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fusion_method: str = "rrf"


def hybrid_search(
    dense_vector: List[float],
    sparse_vector: Optional[SparseVector],
    collection_name: str,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    search_filter: Optional[Filter] = None,
    fusion_method: str = "rrf",  # "rrf" or "weighted"
    alpha: float = 0.7,  # Weight for dense scores (only for weighted fusion)
    prefetch_limit: int = 100,
) -> List[HybridSearchResult]:
    """
    Perform hybrid search combining dense and sparse retrieval.
    
    Args:
        dense_vector: Dense embedding vector (768-dim for Vietnamese)
        sparse_vector: BM25 sparse vector (or None for dense-only)
        collection_name: Qdrant collection name
        limit: Number of results to return
        score_threshold: Minimum score threshold (for dense search)
        search_filter: Optional Qdrant filter for document-specific search
        fusion_method: "rrf" (Reciprocal Rank Fusion) or "weighted"
        alpha: Dense weight for weighted fusion (sparse = 1-alpha)
        prefetch_limit: Number of candidates to fetch from each retrieval mode
        
    Returns:
        List of HybridSearchResult sorted by fused score
    """
    start_time = time.perf_counter()
    qdrant = get_qdrant_client()
    
    enable_hybrid = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
    
    # Check if collection supports hybrid search
    try:
        collection_info = qdrant.get_collection(collection_name)
        has_sparse = (
            hasattr(collection_info.config, 'sparse_vectors_config') and 
            collection_info.config.sparse_vectors_config is not None
        )
        # Check vector config type
        vectors_config = collection_info.config.params.vectors
        if isinstance(vectors_config, dict):
            has_dense_named = "dense" in vectors_config
        else:
            has_dense_named = False
    except Exception:
        has_sparse = False
        has_dense_named = False
    
    # Determine search strategy
    use_hybrid = enable_hybrid and has_sparse and sparse_vector is not None
    
    dense_results = []
    sparse_results = []
    
    # Dense search
    dense_start = time.perf_counter()
    try:
        # Use "dense" if hybrid collection, "default" for legacy
        dense_vector_name = "dense" if has_dense_named else "default"
        search_kwargs = {
            "collection_name": collection_name,
            "query_vector": (dense_vector_name, dense_vector),
            "limit": prefetch_limit if use_hybrid else limit,
            "with_payload": True,
        }
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
        if search_filter is not None:
            search_kwargs["query_filter"] = search_filter
        
        dense_results = qdrant.search(**search_kwargs)
        logger.debug(f"Dense search returned {len(dense_results)} results")
    except Exception as e:
        logger.error(f"Dense search failed: {e}")
    dense_ms = (time.perf_counter() - dense_start) * 1000
    
    # Sparse search (if hybrid enabled)
    sparse_ms = 0
    if use_hybrid:
        sparse_start = time.perf_counter()
        try:
            sparse_kwargs = {
                "collection_name": collection_name,
                "query_vector": NamedSparseVector(
                    name="sparse",
                    vector=sparse_vector
                ),
                "limit": prefetch_limit,
                "with_payload": True,
            }
            if search_filter is not None:
                sparse_kwargs["query_filter"] = search_filter
            
            sparse_results = qdrant.search(**sparse_kwargs)
            logger.debug(f"Sparse search returned {len(sparse_results)} results")
        except Exception as e:
            logger.warning(f"Sparse search failed, falling back to dense-only: {e}")
            use_hybrid = False
        sparse_ms = (time.perf_counter() - sparse_start) * 1000
    
    # Fusion
    fusion_start = time.perf_counter()
    
    if not use_hybrid or not sparse_results:
        # Dense-only results
        results = [
            HybridSearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload,
                dense_score=r.score,
                sparse_score=None,
                fusion_method="dense_only"
            )
            for r in dense_results[:limit]
        ]
    else:
        # Hybrid fusion
        if fusion_method == "rrf":
            results = _reciprocal_rank_fusion(
                dense_results, sparse_results, limit
            )
        else:  # weighted
            results = _weighted_fusion(
                dense_results, sparse_results, alpha, limit
            )
    
    fusion_ms = (time.perf_counter() - fusion_start) * 1000
    total_ms = (time.perf_counter() - start_time) * 1000
    
    # Log performance
    if total_ms > 500:
        logger.warning(f"Slow hybrid search: {total_ms:.1f}ms "
                      f"(dense={dense_ms:.1f}ms, sparse={sparse_ms:.1f}ms, fusion={fusion_ms:.1f}ms)")
    else:
        logger.debug(f"Hybrid search: {total_ms:.1f}ms "
                    f"(dense={dense_ms:.1f}ms, sparse={sparse_ms:.1f}ms, fusion={fusion_ms:.1f}ms)")
    
    return results


def _reciprocal_rank_fusion(
    dense_results: List[ScoredPoint],
    sparse_results: List[ScoredPoint],
    limit: int,
    k: int = 60
) -> List[HybridSearchResult]:
    """
    Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
    
    RRF score = sum(1 / (k + rank)) across all rankings
    
    Args:
        dense_results: Results from dense search
        sparse_results: Results from sparse search
        limit: Number of results to return
        k: RRF constant (typically 60)
        
    Returns:
        Fused results sorted by RRF score
    """
    # Build score dictionaries
    dense_scores = {str(r.id): r.score for r in dense_results}
    sparse_scores = {str(r.id): r.score for r in sparse_results}
    payloads = {str(r.id): r.payload for r in dense_results}
    payloads.update({str(r.id): r.payload for r in sparse_results})
    
    # Compute RRF scores
    rrf_scores: Dict[str, float] = {}
    
    # Dense contribution
    for rank, result in enumerate(dense_results, start=1):
        result_id = str(result.id)
        rrf_scores[result_id] = rrf_scores.get(result_id, 0) + 1 / (k + rank)
    
    # Sparse contribution
    for rank, result in enumerate(sparse_results, start=1):
        result_id = str(result.id)
        rrf_scores[result_id] = rrf_scores.get(result_id, 0) + 1 / (k + rank)
    
    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    results = []
    for result_id in sorted_ids[:limit]:
        results.append(HybridSearchResult(
            id=result_id,
            score=rrf_scores[result_id],
            payload=payloads.get(result_id, {}),
            dense_score=dense_scores.get(result_id),
            sparse_score=sparse_scores.get(result_id),
            fusion_method="rrf"
        ))
    
    return results


def _weighted_fusion(
    dense_results: List[ScoredPoint],
    sparse_results: List[ScoredPoint],
    alpha: float,
    limit: int
) -> List[HybridSearchResult]:
    """
    Weighted fusion of dense and sparse scores.
    
    Final score = alpha * dense_score_norm + (1-alpha) * sparse_score_norm
    
    Args:
        dense_results: Results from dense search
        sparse_results: Results from sparse search
        alpha: Weight for dense scores (0-1)
        limit: Number of results to return
        
    Returns:
        Fused results sorted by weighted score
    """
    # Build score dictionaries
    dense_scores = {str(r.id): r.score for r in dense_results}
    sparse_scores = {str(r.id): r.score for r in sparse_results}
    payloads = {str(r.id): r.payload for r in dense_results}
    payloads.update({str(r.id): r.payload for r in sparse_results})
    
    # Normalize scores to [0, 1]
    max_dense = max(dense_scores.values()) if dense_scores else 1
    max_sparse = max(sparse_scores.values()) if sparse_scores else 1
    
    dense_norm = {k: v / max_dense for k, v in dense_scores.items()}
    sparse_norm = {k: v / max_sparse for k, v in sparse_scores.items()}
    
    # Compute weighted scores
    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    weighted_scores: Dict[str, float] = {}
    
    for result_id in all_ids:
        d_score = dense_norm.get(result_id, 0)
        s_score = sparse_norm.get(result_id, 0)
        weighted_scores[result_id] = alpha * d_score + (1 - alpha) * s_score
    
    # Sort by weighted score
    sorted_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
    
    results = []
    for result_id in sorted_ids[:limit]:
        results.append(HybridSearchResult(
            id=result_id,
            score=weighted_scores[result_id],
            payload=payloads.get(result_id, {}),
            dense_score=dense_scores.get(result_id),
            sparse_score=sparse_scores.get(result_id),
            fusion_method="weighted"
        ))
    
    return results


def hybrid_search_with_fallback(
    dense_vector: List[float],
    sparse_vector: Optional[SparseVector],
    collection_name: str,
    limit: int = 10,
    initial_threshold: Optional[float] = 0.5,
    search_filter: Optional[Filter] = None,
) -> Tuple[List[HybridSearchResult], Optional[float]]:
    """
    Hybrid search with threshold fallback for low-result scenarios.
    
    Args:
        dense_vector: Dense embedding vector
        sparse_vector: BM25 sparse vector
        collection_name: Collection name
        limit: Number of results
        initial_threshold: Starting score threshold
        search_filter: Optional document filter
        
    Returns:
        Tuple of (results, used_threshold)
    """
    thresholds = []
    if initial_threshold is not None:
        thresholds.append(initial_threshold)
    thresholds.extend([0.4, 0.35, 0.25])
    
    fusion_method = os.getenv("HYBRID_FUSION", "rrf")
    alpha = float(os.getenv("HYBRID_ALPHA", "0.7"))
    
    for threshold in thresholds:
        try:
            results = hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                collection_name=collection_name,
                limit=limit,
                score_threshold=threshold,
                search_filter=search_filter,
                fusion_method=fusion_method,
                alpha=alpha,
            )
            
            if results:
                logger.info(f"Hybrid search returned {len(results)} results at threshold={threshold}")
                return results, threshold
                
        except Exception as e:
            logger.warning(f"Hybrid search failed at threshold {threshold}: {e}")
            continue
    
    return [], None

def init_sparse_builder(collection_name: str = "test_collection", vocab_path: str = "backend/data/sparse_vocabulary.pkl"):
    """
    Initialize and fit the sparse vector builder on all chunks from Qdrant collection.
    
    Args:
        collection_name: Qdrant collection to extract chunks from
        vocab_path: Path to save/load vocabulary
        
    Returns:
        Initialized SparseVectorBuilder instance
    """
    import os
    
    # Check if vocabulary already exists
    if os.path.exists(vocab_path):
        try:
            logger.info(f"Loading existing sparse vocabulary from {vocab_path}")
            builder = SparseVectorBuilder.load_vocabulary(vocab_path)
            return builder
        except Exception as e:
            logger.warning(f"Failed to load vocabulary: {e}. Will refit.")
    
    # Fetch all chunks from Qdrant collection
    try:
        qdrant = get_qdrant_client()
        corpus = []
        
        # Scroll through all points in collection
        points, _ = qdrant.scroll(collection_name=collection_name, limit=1000)
        
        while points:
            for point in points:
                if hasattr(point, 'payload') and 'text' in point.payload:
                    corpus.append(point.payload['text'])
            
            # Continue scrolling
            points, _ = qdrant.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=len(corpus)
            )
        
        if not corpus:
            logger.warning(f"No chunks found in collection '{collection_name}'")
            # Return unfitted builder as fallback
            return SparseVectorBuilder()
        
        logger.info(f"Fetched {len(corpus)} chunks from Qdrant for sparse builder fitting")
        
        # Fit builder on corpus
        builder = SparseVectorBuilder()
        builder.fit(corpus)
        
        # Save vocabulary
        os.makedirs(os.path.dirname(vocab_path) if os.path.dirname(vocab_path) else ".", exist_ok=True)
        builder.save_vocabulary(vocab_path)
        
        logger.info(f"✅ Sparse builder fitted on {len(corpus)} chunks, vocabulary saved to {vocab_path}")
        return builder
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize sparse builder: {e}")
        # Return unfitted builder as fallback
        return SparseVectorBuilder()