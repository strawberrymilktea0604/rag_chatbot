import os
import sys
import django
import difflib  # Add at the top with other imports

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.retrieval_metrics import RetrievalEvaluator
from backend.utils.generate_metrics import GenerationEvaluator
from backend.utils.mongo_repository import MongoRepository
from backend.utils.qdrant_client import QdrantClient
import pandas as pd
import logging

# New imports for Gemini rating
import google.generativeai as genai
import re
import time

from datetime import datetime
import argparse
import hashlib

from backend.utils.mongo_models import MongoRetrievalRun, MongoGenerationRun, MongoEvalRun
from backend.utils.retrieval_metrics import _parse_list_field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qa_gold_path = "backend/media/rag_chatbot.qa_gold.csv"

# Initialize clients
mongo_repo = MongoRepository()
qdrant_wrapper = QdrantClient()
qdrant = qdrant_wrapper._client

def load_gold_standard_mapping():
    """Load gold standard and create mappings."""
    try:
        gold_df = pd.read_csv(qa_gold_path)
        
        # Create mappings
        question_to_id = dict(zip(gold_df['question_text'], gold_df['question_id']))
        doc_questions = {}
        question_to_doc = {}
        question_norm_map = {}
        for _, row in gold_df.iterrows():
            doc_id = row['document_id']
            qtext = row['question_text']
            qtext_norm = qtext.strip().lower()
            question_to_doc[qtext] = doc_id
            question_norm_map[qtext_norm] = qtext
            
            if doc_id not in doc_questions:
                doc_questions[doc_id] = []
            doc_questions[doc_id].append(qtext)
        
        logger.info(f"Loaded {len(question_to_id)} gold standard Q&A pairs")
        
        return gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc
    except Exception as e:
        logger.error(f"Failed to load gold standard: {e}")
        return None, {}, {}, {}, {}

def _create_gold_mappings(gold_df):
    """Helper to create mappings from a gold standard DataFrame."""
    question_to_id = {}
    doc_questions = {}
    question_to_doc = {}
    question_norm_map = {}
    
    for _, row in gold_df.iterrows():
        qtext = row.get('question_text')
        
        # Skip rows with null or empty question_text
        if not qtext or qtext is None:
            logger.warning(f"Skipping row with null question_text: {row.get('question_id')}")
            continue
        
        qtext = str(qtext).strip()
        if not qtext:
            logger.warning(f"Skipping row with empty question_text: {row.get('question_id')}")
            continue
        
        doc_id = row.get('document_id')
        question_id = row.get('question_id')
        
        qtext_norm = normalize_question(qtext)
        question_to_id[qtext] = question_id
        question_to_doc[qtext] = doc_id
        question_norm_map[qtext_norm] = qtext
        
        if doc_id not in doc_questions:
            doc_questions[doc_id] = []
        doc_questions[doc_id].append(qtext)
    
    return question_to_id, doc_questions, question_to_doc, question_norm_map


def load_gold_standard_from_mongo():
    """
    Load gold standard Q&A data from MongoDB 'qa_gold' collection.
    Raises error if collection is empty (no CSV fallback).
    
    Returns:
        Tuple of (gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc)
    """
    try:
        # Load from MongoDB only
        qa_gold_collection = mongo_repo.db['qa_gold']
        qa_gold_docs = list(qa_gold_collection.find({}))
        
        if not qa_gold_docs:
            logger.error("MongoDB qa_gold collection is empty. Run --sync-gold first to populate from CSV.")
            return None, {}, {}, {}, {}
        
        # Convert MongoDB documents to DataFrame
        records = []
        for doc in qa_gold_docs:
            # Handle both 'question_text' and 'question' field names
            question_text = doc.get('question_text') or doc.get('question')
            
            # Skip documents with null or missing question_text
            if not question_text:
                logger.debug(f"Skipping document with null question: {doc.get('_id')}")
                continue
            
            record = {
                'question_id': doc.get('_id') or doc.get('question_id'),
                'document_id': doc.get('document_id') or doc.get('chunk_id', '').split(':')[0],
                'question_text': str(question_text).strip(),
                'gold_answer': doc.get('gold_answer') or (doc.get('answers')[0] if doc.get('answers') else ''),
                'gold_support_chunk_ids': doc.get('gold_support_chunk_ids') or '|'.join(doc.get('relevant_docs', [])),
                'page_hint': doc.get('page_hint', '')
            }
            records.append(record)
        
        if not records:
            logger.error("No valid records found in MongoDB qa_gold collection (all have null question_text)")
            return None, {}, {}, {}, {}
        
        gold_df = pd.DataFrame(records)
        
        # Create mappings
        question_to_id, doc_questions, question_to_doc, question_norm_map = _create_gold_mappings(gold_df)
        
        if not question_to_id:
            logger.error("Failed to create mappings from gold standard data")
            return None, {}, {}, {}, {}
        
        logger.info(f"Loaded {len(question_to_id)} gold standard Q&A pairs from MongoDB")
        
        # Debug: show sample questions
        if question_norm_map:
            sample_keys = list(question_norm_map.keys())[:2]
            logger.info(f"Sample normalized gold questions (first 100 chars):")
            for k in sample_keys:
                logger.info(f"  - '{k[:100]}'")
        
        return gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc
        
    except Exception as e:
        logger.error(f"Failed to load from MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return None, {}, {}, {}, {}


def sync_gold_standard_to_mongo():
    """
    Sync gold standard Q&A data from CSV to MongoDB 'qa_gold' collection.
    Uses question_id as document _id.
    
    Returns:
        Number of documents synced
    """
    try:
        # Read CSV
        gold_df = pd.read_csv(qa_gold_path)
        
        if gold_df.empty:
            logger.warning("CSV file is empty, nothing to sync")
            return 0
        
        logger.info(f"CSV columns: {gold_df.columns.tolist()}")
        
        qa_gold_collection = mongo_repo.db['qa_gold']
        
        # Clear existing collection
        delete_result = qa_gold_collection.delete_many({})
        logger.info(f"Cleared {delete_result.deleted_count} existing documents from MongoDB")
        
        # Prepare documents for insertion
        # Handle different column naming conventions
        documents = []
        for _, row in gold_df.iterrows():
            # Handle question_id - try different column names
            question_id = row.get('_id') or row.get('question_id') or f"q_{_}"
            
            # Handle document_id - extract from chunk_id if not present
            document_id = row.get('document_id')
            if pd.isna(document_id) or not document_id:
                chunk_id = row.get('chunk_id', '')
                if chunk_id and ':' in str(chunk_id):
                    document_id = str(chunk_id).split(':')[0]
                else:
                    document_id = ''
            
            # Handle question text - try different column names
            question_text = row.get('question_text') or row.get('question') or ''
            
            # Handle gold answer - try different column names
            gold_answer = row.get('gold_answer') or row.get('answers[0]') or ''
            
            # Handle gold support chunk IDs - try different column names
            gold_chunk_ids = row.get('gold_support_chunk_ids') or row.get('relevant_docs[0]') or row.get('chunk_id') or ''
            
            doc = {
                '_id': question_id,
                'question_id': question_id,
                'document_id': document_id,
                'question_text': str(question_text).strip() if question_text else '',
                'gold_answer': str(gold_answer).strip() if gold_answer else '',
                'gold_support_chunk_ids': str(gold_chunk_ids).strip() if gold_chunk_ids else '',
                'page_hint': row.get('page_hint', ''),
                'run_tag': row.get('run_tag', ''),
                'type': row.get('type', ''),
                'synced_at': datetime.utcnow()
            }
            documents.append(doc)
        
        # Insert all documents
        if documents:
            result = qa_gold_collection.insert_many(documents)
            logger.info(f"Synced {len(result.inserted_ids)} Q&A records to MongoDB")
            return len(result.inserted_ids)
        
        return 0
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {qa_gold_path}")
        return 0
    except Exception as e:
        logger.error(f"Failed to sync gold standard to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return 0



def get_chunk_metadata_map():
    """Build mapping from Qdrant point ID (UUID) to canonical chunk ID format."""
    try:
        chunk_map = {}
        collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')
        
        points, _ = qdrant.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        for point in points:
            if not point.payload:
                continue

            point_uuid = str(point.id)
            chunk_id_canonical = point.payload.get('chunk_id')
            
            if chunk_id_canonical:
                chunk_map[point_uuid] = chunk_id_canonical
        
        logger.info(f"Built chunk metadata map with {len(chunk_map)} mappings")
        return chunk_map
    except Exception as e:
        logger.warning(f"Could not build chunk metadata map: {e}")
        return {}

def convert_chunk_id_format(chunk_id_str):
    """
    Convert chunk ID to standardized format (doc_id:N).
    
    Both gold standard and Qdrant now use the same format (doc_id:N).
    This function maintains backwards compatibility for legacy ::chunk_ format.
    
    Examples:
        '1e8e6c622bfb571b0b783d8347182318:105' -> '1e8e6c622bfb571b0b783d8347182318:105' (no change)
        '1e8e6c622bfb571b0b783d8347182318::chunk_105' -> '1e8e6c622bfb571b0b783d8347182318:105' (legacy conversion)
    """
    if not chunk_id_str:
        return chunk_id_str
    
    # Handle legacy double colon format: 'doc_id::chunk_N' -> 'doc_id:N'
    if '::chunk_' in chunk_id_str:
        parts = chunk_id_str.rsplit('::chunk_', 1)
        if len(parts) == 2:
            doc_id, chunk_num = parts
            return f"{doc_id}:{chunk_num}"
    
    # Already in correct format (doc_id:N)
    return chunk_id_str

def convert_chunk_id_to_gold_format(chunk_id_str):
    """Alias for convert_chunk_id_format to maintain consistency."""
    return convert_chunk_id_format(chunk_id_str)

def debug_chunk_format_mismatch(retrieval_df, gold_df):
    """Debug helper to show chunk ID format differences."""
    print("\n" + "=" * 80)
    print("DEBUGGING CHUNK ID FORMAT MISMATCH")
    print("=" * 80)
    
    # Handle None cases
    if retrieval_df is None:
        print("No retrieval logs to debug")
        print("=" * 80)
        return
    
    # Sample from retrieval logs
    if not retrieval_df.empty:
        print(f"\nTotal retrieval logs matched: {len(retrieval_df)}")
        print(f"\nFirst retrieval log:")
        first_row = retrieval_df.iloc[0]
        print(f"  Question ID: {first_row['question_id']}")
        print(f"  Question: {first_row['question_text'][:80]}...")
        print(f"  Document: {first_row['document_id']}")
        
        sample_retrieved = first_row['retrieved_chunk_ids']
        retrieved_chunks = sample_retrieved.split('|')[:3]
        print(f"  Retrieved chunk IDs (first 3):")
        for chunk_id in retrieved_chunks:
            print(f"    '{chunk_id}'")
    
    # Sample from gold standard
    if gold_df is not None and not gold_df.empty and 'gold_support_chunk_ids' in gold_df.columns:
        # Find matching gold standard entry
        if retrieval_df is not None and not retrieval_df.empty:
            first_qid = retrieval_df.iloc[0]['question_id']
            gold_match = gold_df[gold_df['question_id'] == first_qid]
            
            if not gold_match.empty:
                print(f"\n  Matching gold standard entry:")
                gold_row = gold_match.iloc[0]
                sample_gold = gold_row['gold_support_chunk_ids']
                
                if isinstance(sample_gold, str):
                    gold_chunks = sample_gold.split('|')[:3]
                    print(f"  Gold chunk IDs (first 3):")
                    for chunk_id in gold_chunks:
                        print(f"    '{chunk_id}'")
                    
                    # Check for any overlap
                    retrieved_set = set(sample_retrieved.split('|'))
                    gold_set = set(sample_gold.split('|'))
                    overlap = retrieved_set & gold_set
                    print(f"\n  Overlap: {len(overlap)} chunks match")
                    if overlap:
                        print(f"  Matching chunks: {list(overlap)[:3]}")
            else:
                print(f"\n  WARNING: No gold standard entry found for question_id={first_qid}")
        else:
            # Show first gold entry if no retrieval logs
            sample_gold = gold_df.iloc[0]['gold_support_chunk_ids']
            if isinstance(sample_gold, str):
                gold_chunks = sample_gold.split('|')[:3]
                print(f"\nSample gold standard chunk IDs (first 3):")
                for chunk_id in gold_chunks:
                    print(f"  '{chunk_id}'")
    
    print("\n" + "=" * 80)

def normalize_question(text):
    """Normalize question text for matching."""
    return text.strip().lower().replace("  ", " ").replace("?", "").replace(".", "")

def find_best_match(question_text, question_norm_map, threshold=0.85, debug=False):
    """Find best matching gold question using fuzzy matching."""
    q_norm = normalize_question(question_text)
    
    if debug:
        print(f"DEBUG find_best_match:")
        print(f"  Input normalized (first 80): '{q_norm[:80]}'")
        print(f"  question_norm_map has {len(question_norm_map)} keys")
        if question_norm_map:
            first_key = list(question_norm_map.keys())[0]
            print(f"  First key in map (first 80): '{first_key[:80]}'")
    
    # Try exact match first
    if q_norm in question_norm_map:
        return question_norm_map[q_norm]
    
    # Try fuzzy match
    matches = difflib.get_close_matches(q_norm, question_norm_map.keys(), n=1, cutoff=threshold)
    if matches:
        return question_norm_map[matches[0]]
    
    return None

def get_retrieval_logs_from_mongo(question_to_id, question_norm_map, question_to_doc, chunk_map):
    """Get ALL retrieval logs from MongoDB as DataFrame, matching against gold standard."""
    try:
        retrieval_logs = list(mongo_repo.db['retrieval_log'].find({}))
        
        if not retrieval_logs:
            logger.warning("No retrieval logs found in MongoDB")
            return None, 0
        
        rows = []
        mapped_count = 0
        skipped_count = 0
        skipped_samples = []  # Track some skipped questions for debugging
        conversion_stats = {"converted": 0, "not_converted": 0}
        fuzzy_matches = 0
        doc_mismatch_count = 0
        wrong_doc_chunks_filtered = 0
        
        for log in retrieval_logs:
            doc_id = log.get("document_id") or ""  # Handle None values
            question_text = log.get("question", "").strip()
            
            # Skip template variables and empty questions
            if not question_text or "{{" in question_text:
                skipped_count += 1
                continue
            
            # Enable debug for first question to diagnose matching issues
            debug_this = (mapped_count == 0 and skipped_count == 0)
            
            # Try to find matching gold question (with fuzzy matching)
            gold_qtext = find_best_match(question_text, question_norm_map, threshold=0.80, debug=debug_this)
            
            if not gold_qtext:
                skipped_count += 1
                if len(skipped_samples) < 3:
                    q_norm = normalize_question(question_text)
                    skipped_samples.append(f"LOG: '{q_norm[:80]}...'")
                continue
            
            # Check if document ID matches expected gold standard document
            expected_doc_id = question_to_doc.get(gold_qtext) or ""
            if expected_doc_id and doc_id != expected_doc_id:
                doc_id_display = doc_id[:8] if doc_id else "(empty)"
                expected_display = expected_doc_id[:8] if expected_doc_id else "(empty)"
                logger.debug(f"Document mismatch for '{gold_qtext[:50]}...': got {doc_id_display}..., expected {expected_display}...")
                doc_mismatch_count += 1
                # Don't skip - just filter chunks from wrong documents below
            
            if normalize_question(question_text) != normalize_question(gold_qtext):
                fuzzy_matches += 1
            
            question_id = question_to_id[gold_qtext]
            mapped_count += 1
            
            retrieved_chunk_ids = log.get("retrieved_chunk_ids", [])
            converted_ids = []
            for chunk_uuid in retrieved_chunk_ids:
                # First try chunk_map lookup
                if chunk_uuid in chunk_map:
                    canonical_id = chunk_map[chunk_uuid]
                    conversion_stats["converted"] += 1
                else:
                    canonical_id = chunk_uuid
                    conversion_stats["not_converted"] += 1
                
                # Convert to gold standard format (single colon)
                gold_format_id = convert_chunk_id_to_gold_format(canonical_id)
                
                # RELAXED: If document ID doesn't match, try to align it to expected doc ID
                # This handles cases where document was re-uploaded (new ID) but content is same
                chunk_doc_id = gold_format_id.split(':')[0] if ':' in gold_format_id else ""
                if expected_doc_id and chunk_doc_id != expected_doc_id:
                    wrong_doc_chunks_filtered += 1
                    if ':' in gold_format_id:
                        _, suffix = gold_format_id.split(':', 1)
                        gold_format_id = f"{expected_doc_id}:{suffix}"
                
                converted_ids.append(gold_format_id)
            
            retrieved_scores = log.get("retrieved_scores", [])
            
            # If we filtered out chunks, adjust scores accordingly
            if len(converted_ids) < len(retrieved_chunk_ids):
                retrieved_scores = retrieved_scores[:len(converted_ids)]
            
            rows.append({
                "question_id": question_id,
                "document_id": expected_doc_id or doc_id,  # Use expected doc ID
                "question_text": gold_qtext,
                "retrieved_chunk_ids": "|".join(converted_ids),
                "retrieved_scores": "|".join([str(s) for s in retrieved_scores]),
                "top_k": len(converted_ids)  # Actual number after filtering
            })
        
        logger.info(f"Loaded {mapped_count} retrieval logs from MongoDB")
        logger.info(f"  Exact matches: {mapped_count - fuzzy_matches}")
        logger.info(f"  Fuzzy matches: {fuzzy_matches}")
        logger.info(f"  Skipped: {skipped_count} (template/unmapped questions)")
        if skipped_samples:
            logger.info(f"  Sample skipped questions:")
            for sq in skipped_samples:
                logger.info(f"    - {sq}...")
        if doc_mismatch_count > 0:
            logger.warning(f"  Document mismatches: {doc_mismatch_count} logs had wrong document_id")
        if wrong_doc_chunks_filtered > 0:
            logger.warning(f"  Aligned {wrong_doc_chunks_filtered} chunks from mismatched documents to expected IDs")
        logger.info(f"  Chunk conversions - Successful: {conversion_stats['converted']}, Fallback: {conversion_stats['not_converted']}")
        
        if mapped_count == 0:
            logger.warning("No matching retrieval logs found")
            return None, 0
        
        df = pd.DataFrame(rows)
        return df, mapped_count
    except Exception as e:
        logger.error(f"Failed to load retrieval logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def get_citation_logs_from_mongo(question_to_id, question_norm_map, question_to_doc):
    """Get ALL citation logs from MongoDB as DataFrame, matching against gold standard."""
    try:
        citation_logs = list(mongo_repo.db['citation_log'].find({}))
        
        if not citation_logs:
            logger.warning("No citation logs found in MongoDB")
            return None, 0
        
        rows = []
        mapped_count = 0
        skipped_count = 0
        fuzzy_matches = 0
        
        # Group citations by question_id for aggregation
        question_citations = {}
        
        for log in citation_logs:
            question_text = log.get("question_text", "") or log.get("question", "")
            
            # Handle None or non-string question_text
            if not question_text:
                skipped_count += 1
                continue
            
            question_text = str(question_text).strip()
            
            if not question_text or "{{" in question_text:
                skipped_count += 1
                continue
            
            # Try to find matching gold question (with fuzzy matching)
            gold_qtext = find_best_match(question_text, question_norm_map, threshold=0.80)
            
            if not gold_qtext:
                skipped_count += 1
                continue
            
            if normalize_question(question_text) != normalize_question(gold_qtext):
                fuzzy_matches += 1
            
            question_id = question_to_id[gold_qtext]
            expected_doc_id = question_to_doc.get(gold_qtext, "")
            
            # Extract citation data
            citation_data = {
                "chunk_id": log.get("chunk_id", ""),
                "document_id": log.get("document_id", ""),
                "page": log.get("page", ""),
                "citation_rank": log.get("citation_rank", 0),
                "rerank_score": log.get("rerank_score", 0.0),
                "used_in_generation": log.get("used_in_generation", False)
            }
            
            if question_id not in question_citations:
                question_citations[question_id] = {
                    "question_id": question_id,
                    "question_text": gold_qtext,
                    "document_id": expected_doc_id,
                    "citations": []
                }
            
            question_citations[question_id]["citations"].append(citation_data)
        
        # Convert to rows
        for qid, data in question_citations.items():
            citations = data["citations"]
            # Sort by citation_rank (handle None values by treating them as 0)
            citations.sort(key=lambda x: x.get("citation_rank") or 0)
            
            chunk_ids = [c["chunk_id"] for c in citations]
            scores = [c["rerank_score"] for c in citations]
            used_flags = [c["used_in_generation"] for c in citations]
            
            rows.append({
                "question_id": qid,
                "question_text": data["question_text"],
                "document_id": data["document_id"],
                "cited_chunk_ids": "|".join(chunk_ids),
                "citation_scores": "|".join([str(s) for s in scores]),
                "used_in_generation": "|".join([str(u) for u in used_flags]),
                "num_citations": len(citations)
            })
            mapped_count += 1
        
        logger.info(f"Loaded {mapped_count} citation log groups from MongoDB")
        logger.info(f"  Total citation entries processed: {len(citation_logs)}")
        logger.info(f"  Fuzzy matches: {fuzzy_matches}")
        logger.info(f"  Skipped: {skipped_count} (template/unmapped questions)")
        
        if mapped_count == 0:
            logger.warning("No matching citation logs found")
            return None, 0
        
        df = pd.DataFrame(rows)
        return df, mapped_count
    except Exception as e:
        logger.error(f"Failed to load citation logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def get_generation_logs_from_mongo(question_to_id, gold_df, question_norm_map):
    """Get ALL generation logs from MongoDB as DataFrame, matching against gold standard."""
    try:
        generation_logs = list(mongo_repo.db['generation_log'].find({}))
        
        if not generation_logs:
            logger.warning("No generation logs found in MongoDB")
            return None, 0
        
        gold_answers = {}
        for _, row in gold_df.iterrows():
            gold_answers[row['question_text']] = row['gold_answer']
        
        rows = []
        mapped_count = 0
        skipped_count = 0
        answer_stats = {"has_content": 0, "not_found": 0, "empty": 0}
        fuzzy_matches = 0
        
        for log in generation_logs:
            question_text = log.get("question_text", "")
            
            # Handle None or non-string question_text
            if not question_text:
                skipped_count += 1
                continue
            
            question_text = str(question_text).strip()
            
            if not question_text or "{{" in question_text:
                skipped_count += 1
                continue
            
            # Try to find matching gold question (with fuzzy matching)
            gold_qtext = find_best_match(question_text, question_norm_map, threshold=0.80)
            
            if not gold_qtext:
                logger.debug(f"No match found for: {question_text[:80]}...")
                skipped_count += 1
                continue
            
            if normalize_question(question_text) != normalize_question(gold_qtext):
                fuzzy_matches += 1
                logger.debug(f"Fuzzy match: '{question_text[:50]}...' -> '{gold_qtext[:50]}...'")
            
            question_id = question_to_id[gold_qtext]
            mapped_count += 1
            
            # Handle None or non-string generated_answer
            answer_text = log.get("generated_answer")
            if answer_text is None:
                answer_text = ""
            else:
                answer_text = str(answer_text).strip()
            
            if not answer_text:
                answer_stats["empty"] += 1
            elif "không tìm thấy" in answer_text.lower() or "không có" in answer_text.lower():
                answer_stats["not_found"] += 1
            else:
                answer_stats["has_content"] += 1
            
            rows.append({
                "question_id": question_id,
                "question_text": gold_qtext,
                "generated_answer": answer_text,
                "citations": ""
            })
        
        logger.info(f"Loaded {mapped_count} generation logs from MongoDB")
        logger.info(f"  Exact matches: {mapped_count - fuzzy_matches}")
        logger.info(f"  Fuzzy matches: {fuzzy_matches}")
        logger.info(f"  Skipped: {skipped_count} (template/unmapped questions)")
        logger.info(f"  Answer quality - Content: {answer_stats['has_content']}, Not found: {answer_stats['not_found']}, Empty: {answer_stats['empty']}")
        
        if mapped_count == 0:
            logger.warning("No matching generation logs found")
            return None, 0
        
        df = pd.DataFrame(rows)
        return df, mapped_count
    except Exception as e:
        logger.error(f"Failed to load generation logs: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def rate_generation_with_gemini(gen_df, gold_df, model_name=None, api_key=None, sleep_between_calls=0.15):
    """Rate generated answers with Gemini model (1-5 scale)."""
    try:
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - skipping Gemini rating")
            return None, {}
        
        genai.configure(api_key=api_key)
        
        if not model_name:
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        model = genai.GenerativeModel(model_name)
        
        df = gen_df.copy()
        
        gold_map = {}
        if 'question_text' in gold_df.columns and 'gold_answer' in gold_df.columns:
            gold_map = dict(zip(gold_df['question_text'], gold_df['gold_answer']))
        
        ratings = []
        raw_outputs = []
        
        for idx, row in df.iterrows():
            qtext = str(row.get('question_text', '') or '').strip()
            gen_ans = str(row.get('generated_answer', '') or '').strip()
            gold_ans = str(gold_map.get(qtext, '') or '').strip()
            
            if not gen_ans:
                ratings.append(None)
                raw_outputs.append("EMPTY_GENERATED_ANSWER")
                logger.debug(f"  [{idx}] Skipped: empty generated answer")
                continue
            
            if not gold_ans:
                ratings.append(None)
                raw_outputs.append("GOLD_ANSWER_NOT_FOUND")
                logger.debug(f"  [{idx}] Skipped: no matching gold answer for '{qtext[:50]}...'")
                continue
            
            prompt = (
                "You are an objective evaluator. Compare the generated answer to the reference (gold) answer "
                "and assign a single integer rating from 1 to 5 where:\n"
                "5 = Generated answer fully matches the reference in correctness, completeness and relevance\n"
                "4 = Generated answer is mostly correct with minor omissions\n"
                "3 = Generated answer has moderate issues or partial correctness\n"
                "2 = Generated answer is mostly incorrect or incomplete\n"
                "1 = Generated answer is incorrect or irrelevant\n\n"
                f"Reference (gold): \"{gold_ans}\"\n\n"
                f"Generated answer: \"{gen_ans}\"\n\n"
                "Respond with ONLY a single digit (1-5) and nothing else."
            )
            
            try:
                response = model.generate_content(prompt)
                text = response.text.strip() if response.text else ""
                
                raw_outputs.append(text)
                
                m = re.search(r'[1-5]', text)
                if m:
                    rating = int(m.group(0))
                    ratings.append(rating)
                    logger.debug(f"  [{idx}] Rating: {rating}")
                else:
                    ratings.append(None)
                    logger.warning(f"  [{idx}] No valid rating extracted from: {text}")
                    
            except Exception as e:
                logger.warning(f"  [{idx}] Gemini call failed: {e}")
                ratings.append(None)
                raw_outputs.append(f"ERROR: {str(e)}")
            
            time.sleep(sleep_between_calls)
        
        df['gemini_rating'] = ratings
        df['gemini_raw'] = raw_outputs
        
        valid_ratings = [r for r in ratings if isinstance(r, int)]
        rating_dist = {}
        for i in range(1, 6):
            rating_dist[str(i)] = valid_ratings.count(i)
        
        summary = {
            "count": len(ratings),
            "rated": len(valid_ratings),
            "mean_rating": float(sum(valid_ratings) / len(valid_ratings)) if valid_ratings else None,
            "rating_distribution": rating_dist
        }
        
        logger.info(f"Gemini rating complete")
        logger.info(f"  Rated: {summary['rated']}/{summary['count']}")
        if summary['mean_rating']:
            logger.info(f"  Mean rating: {summary['mean_rating']:.2f}")
        logger.info(f"  Distribution: {rating_dist}")
        
        return df, summary
        
    except Exception as e:
        logger.error(f"Failed to rate with Gemini: {e}")
        import traceback
        traceback.print_exc()
        return None, {}

# Main execution - evaluate ALL logs in database
print("=" * 80)
print("RAG CHATBOT EVALUATION - ALL DATABASE LOGS")
print("=" * 80)
print()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate RAG chatbot performance')
parser.add_argument('--sync-gold', action='store_true', 
                    help='Sync gold standard from CSV to MongoDB before loading')
parser.add_argument('--run-id', type=str, default=None,
                    help='Evaluation run ID (auto-generated if not provided)')
args = parser.parse_args()

# Sync gold standard to MongoDB if requested
if args.sync_gold:
    print("=" * 80)
    print("Syncing gold standard from CSV to MongoDB...")
    print("=" * 80)
    synced_count = sync_gold_standard_to_mongo()
    print(f"Synced {synced_count} records")
    print()

# Load gold standard from MongoDB
print("=" * 80)
print("Loading gold standard Q&A pairs from MongoDB...")
print("=" * 80)

gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc = load_gold_standard_from_mongo()

if gold_df is None:
    logger.error("Cannot proceed without gold standard data")
    sys.exit(1)

print(f"Gold standard contains {len(gold_df)} questions")
print(f"Question norm map has {len(question_norm_map)} entries")
if question_norm_map:
    sample_norms = list(question_norm_map.keys())[:2]
    print("Sample normalized gold questions:")
    for sn in sample_norms:
        print(f"  - '{sn[:100]}'")
print()

# Generate or use provided run ID
if args.run_id:
    run_id = args.run_id
else:
    run_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

print(f"Evaluation Run ID: {run_id}")
print()

# Build chunk map for all documents
print("Building chunk metadata map from Qdrant...")
chunk_map = get_chunk_metadata_map()
print()

def evaluate_retrieval_from_dataframe(gold_df, run_df):
    """Evaluate retrieval using DataFrames directly."""
    from backend.utils.retrieval_metrics import RetrievalEvaluator, _parse_list_field, recall_at_k, mrr_at_k, ndcg_at_k
    
    # Add relevant_chunk_ids column if not present
    if 'gold_support_chunk_ids' in gold_df.columns and 'relevant_chunk_ids' not in gold_df.columns:
        gold_df['relevant_chunk_ids'] = gold_df['gold_support_chunk_ids']
    
    # Merge retrieval results with gold standard
    merged_df = run_df.merge(gold_df, on='question_id', suffixes=('_run', '_gold'))
    
    # Calculate metrics
    metrics = {}
    for k in [5, 10]:
        merged_df[f'top_{k}'] = merged_df['retrieved_chunk_ids'].apply(lambda x: _parse_list_field(x)[:k])
        
        merged_df[f'recall@{k}'] = merged_df.apply(
            lambda row: recall_at_k(
                _parse_list_field(row['relevant_chunk_ids']),
                row[f'top_{k}'],
                k
            ),
            axis=1
        )
        
        merged_df[f'mrr@{k}'] = merged_df.apply(
            lambda row: mrr_at_k(
                _parse_list_field(row['relevant_chunk_ids']),
                row[f'top_{k}'],
                k
            ),
            axis=1
        )
        
        merged_df[f'ndcg@{k}'] = merged_df.apply(
            lambda row: ndcg_at_k(
                _parse_list_field(row['relevant_chunk_ids']),
                row[f'top_{k}'],
                k
            ),
            axis=1
        )
        
        metrics[f'recall@{k}'] = {
            'mean': float(merged_df[f'recall@{k}'].mean()),
            'std': float(merged_df[f'recall@{k}'].std())
        }
        metrics[f'mrr@{k}'] = {
            'mean': float(merged_df[f'mrr@{k}'].mean()),
            'std': float(merged_df[f'mrr@{k}'].std())
        }
        metrics[f'ndcg@{k}'] = {
            'mean': float(merged_df[f'ndcg@{k}'].mean()),
            'std': float(merged_df[f'ndcg@{k}'].std())
        }
    
    return {
        'summary': metrics,
        'per_question': merged_df.to_dict('records')
    }

def normalize_citations_in_answer(answer_text):
    """
    Normalize citation formats in answer text for fair comparison.
    
    Converts [cite: N] format to [doc_id: chunk_id] format.
    This allows fair comparison between gold answers (which use cite numbers)
    and generated answers (which use doc_id references).
    
    Examples:
        "[cite: 1]" -> "[doc_id: gold_1]"
        "text [cite: 2] more text" -> "text [doc_id: gold_2] more text"
    """
    if not answer_text:
        return answer_text
    
    import re
    
    def replace_cite(match):
        cite_num = match.group(1)
        return f"[doc_id tr.{cite_num}]"
    
    # Replace [cite: N] with [doc_id: gold_N]
    normalized = re.sub(r'\[cite:\s*(\d+)\]', replace_cite, answer_text)
    return normalized

def evaluate_citations_from_dataframe(gold_df, citation_df, retrieval_df=None):
    """Evaluate citation quality using DataFrames directly.
    
    Metrics:
    - Precision: How many cited chunks are in gold standard support chunks
    - Recall: How many gold standard support chunks were cited
    - Hallucination rate: Citations to chunks not in retrieval results
    """
    import numpy as np
    
    # Create gold chunk mappings
    gold_chunks_map = {}
    for _, row in gold_df.iterrows():
        qid = row['question_id']
        gold_chunks = row.get('gold_support_chunk_ids', '')
        if isinstance(gold_chunks, str) and gold_chunks:
            gold_chunks_map[qid] = set(gold_chunks.split('|'))
        else:
            gold_chunks_map[qid] = set()
    
    # Create retrieval chunks mapping (if available)
    retrieved_chunks_map = {}
    if retrieval_df is not None:
        for _, row in retrieval_df.iterrows():
            qid = row['question_id']
            retrieved = row.get('retrieved_chunk_ids', '')
            if isinstance(retrieved, str) and retrieved:
                retrieved_chunks_map[qid] = set(retrieved.split('|'))
            else:
                retrieved_chunks_map[qid] = set()
    
    metrics = {
        'precision': [],
        'recall': [],
        'hallucination_rate': [],
        'num_citations': []
    }
    
    question_scores = []
    
    for _, row in citation_df.iterrows():
        qid = row['question_id']
        cited_chunks_str = row.get('cited_chunk_ids', '')
        
        if not cited_chunks_str:
            continue
        
        cited_chunks = set(cited_chunks_str.split('|'))
        gold_chunks = gold_chunks_map.get(qid, set())
        retrieved_chunks = retrieved_chunks_map.get(qid, set())
        
        num_citations = len(cited_chunks)
        
        # Precision: cited chunks that are in gold / total cited
        if num_citations > 0:
            correct_citations = cited_chunks & gold_chunks
            precision = len(correct_citations) / num_citations
        else:
            precision = 0.0
        
        # Recall: cited chunks that are in gold / total gold chunks
        if len(gold_chunks) > 0:
            correct_citations = cited_chunks & gold_chunks
            recall = len(correct_citations) / len(gold_chunks)
        else:
            recall = 1.0 if num_citations == 0 else 0.0
        
        # Hallucination: citations to chunks not in retrieval results
        if num_citations > 0 and len(retrieved_chunks) > 0:
            hallucinated = cited_chunks - retrieved_chunks
            hallucination_rate = len(hallucinated) / num_citations
        else:
            hallucination_rate = 0.0
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['hallucination_rate'].append(hallucination_rate)
        metrics['num_citations'].append(num_citations)
        
        question_scores.append({
            'question_id': qid,
            'precision': precision,
            'recall': recall,
            'hallucination_rate': hallucination_rate,
            'num_citations': num_citations
        })
    
    # Compute summary
    summary = {}
    for metric_name, values in metrics.items():
        if not values:
            continue
        vals_arr = np.array(values, dtype=float)
        summary[metric_name] = {
            'mean': float(np.mean(vals_arr)),
            'std': float(np.std(vals_arr)),
            'min': float(np.min(vals_arr)),
            'max': float(np.max(vals_arr)),
            'count': int(len(vals_arr))
        }
    
    # Add avg_citations as a top-level summary field
    if metrics['num_citations']:
        summary['avg_citations'] = float(np.mean(metrics['num_citations']))
    else:
        summary['avg_citations'] = 0.0
    
    return {
        'summary': summary,
        'per_question': question_scores
    }


def evaluate_generation_from_dataframe(gold_df, gen_df):
    """Evaluate generation using DataFrames directly."""
    from backend.utils.generate_metrics import GenerationEvaluator
    from rouge_score import rouge_scorer
    import numpy as np
    
    # Create gold answer mapping with normalized citations
    gold_map = {}
    for _, row in gold_df.iterrows():
        qtext = str(row.get('question_text', '') or '').strip()
        gold_ans = str(row.get('gold_answer', '') or '').strip()
        # Normalize citations in gold answers
        gold_ans_normalized = normalize_citations_in_answer(gold_ans)
        gold_map[qtext] = gold_ans_normalized
    
    # Initialize metrics
    metrics = {
        'em': [],
        'token_f1': [],
        'rouge1': [],
        'rouge_l': [],
        'answer_length': []
    }
    
    question_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    for _, row in gen_df.iterrows():
        qtext = str(row.get('question_text', '') or '').strip()
        gen_answer = str(row.get('generated_answer', '') or '').strip()
        gold_answer = str(gold_map.get(qtext, '') or '').strip()
        
        if not gen_answer or not gold_answer:
            continue
        
        # Normalize
        gold_norm = GenerationEvaluator.normalize_text(gold_answer)
        gen_norm = GenerationEvaluator.normalize_text(gen_answer)
        
        # Compute metrics
        em = GenerationEvaluator.exact_match(gold_answer, gen_answer)
        f1 = GenerationEvaluator.token_f1(gold_answer, gen_answer)
        
        rouge_scores = scorer.score(gold_norm, gen_norm)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        ans_len = float(len(gen_norm))
        
        metrics['em'].append(em)
        metrics['token_f1'].append(f1)
        metrics['rouge1'].append(rouge1)
        metrics['rouge_l'].append(rouge_l)
        metrics['answer_length'].append(ans_len)
        
        question_scores.append({
            'question_id': row.get('question_id'),
            'em': em,
            'token_f1': f1,
            'rouge1': rouge1,
            'rouge_l': rouge_l,
            'answer_length': ans_len
        })
    
    # Compute summary
    summary = {}
    for metric_name, values in metrics.items():
        if not values:
            continue
        vals_arr = np.array(values, dtype=float)
        summary[metric_name] = {
            'mean': float(np.mean(vals_arr)),
            'std': float(np.std(vals_arr)),
            'min': float(np.min(vals_arr)),
            'max': float(np.max(vals_arr)),
            'count': int(len(vals_arr))
        }
    
    return {
        'summary': summary,
        'per_question': question_scores
    }

# Load logs from MongoDB
print("=" * 80)
print("Loading ALL logs from MongoDB...")
print("=" * 80)
retrieval_df, retrieval_count = get_retrieval_logs_from_mongo(
    question_to_id, question_norm_map, question_to_doc, chunk_map
)
gen_df, gen_count = get_generation_logs_from_mongo(
    question_to_id, gold_df, question_norm_map
)
citation_df, citation_count = get_citation_logs_from_mongo(
    question_to_id, question_norm_map, question_to_doc
)
print()

# DEBUG: Check chunk formats
debug_chunk_format_mismatch(retrieval_df, gold_df)
sys.stdout.flush()

results = {
    "run_id": run_id,
    "total_retrieval_logs": retrieval_count,
    "total_generation_logs": gen_count,
    "total_citation_logs": citation_count,
    "retrieval_evaluation": None,
    "generation_evaluation": None,
    "citation_evaluation": None,
    "gemini_rating": None
}

# Track individual eval runs for MongoDB
retrieval_runs_to_save = []
generation_runs_to_save = []
citation_runs_to_save = []

# Run retrieval evaluation
if retrieval_df is not None and not retrieval_df.empty:
    print("=" * 80)
    print("Running Retrieval Evaluation...")
    print("=" * 80)
    try:
        retrieval = evaluate_retrieval_from_dataframe(gold_df, retrieval_df)
        results['retrieval_evaluation'] = retrieval
        
        # Extract per-question results and save to MongoDB
        if retrieval and retrieval.get('per_question'):
            for i, per_q in enumerate(retrieval['per_question']):
                try:
                    run_entry = MongoRetrievalRun(
                        run_id=run_id,
                        query_id=per_q.get('question_id', f"q_{i}"),
                        gold_chunk_ids=_parse_list_field(per_q.get('relevant_chunk_ids', '[]')),
                        retrieved_chunk_ids=_parse_list_field(per_q.get('retrieved_chunk_ids', '[]')),
                        k=10,
                        recall_at_k=float(per_q.get('recall@10', 0)),
                        mrr_at_k=float(per_q.get('mrr@10', 0)),
                        ndcg_at_k=float(per_q.get('ndcg@10', 0))
                    )
                    retrieval_runs_to_save.append(run_entry)
                except Exception as e:
                    logger.warning(f"Could not save retrieval run for query {per_q.get('question_id')}: {e}")
        
        if retrieval and retrieval.get('summary'):
            print(f"Recall@10: {retrieval['summary']['recall@10']['mean']:.3f}")
            print(f"MRR@10: {retrieval['summary']['mrr@10']['mean']:.3f}")
            print(f"NDCG@10: {retrieval['summary']['ndcg@10']['mean']:.3f}")
        print("Retrieval evaluation complete")
    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No retrieval logs to evaluate")

print()

print(f"DEBUG: Retrieval DF: {len(retrieval_df) if retrieval_df is not None else 0} rows")
print(f"DEBUG: Generation DF: {len(gen_df) if gen_df is not None else 0} rows")
print(f"DEBUG: Citation DF: {len(citation_df) if citation_df is not None else 0} rows")
sys.stdout.flush()

# Run generation evaluation
if gen_df is not None and not gen_df.empty:
    print("=" * 80)
    print("Running Gemini rating for generated answers...")
    print("=" * 80)
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    try:
        rated_gen_df, gemini_summary = rate_generation_with_gemini(
            gen_df, gold_df, model_name=gemini_model, api_key=gemini_api_key
        )
        results['gemini_rating'] = gemini_summary
        if gemini_summary and gemini_summary.get('mean_rating') is not None:
            print(f"Gemini mean rating: {gemini_summary['mean_rating']:.3f} (rated {gemini_summary['rated']}/{gemini_summary['count']})")
    except Exception as e:
        logger.error(f"Gemini rating failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Running Generation Evaluation...")
    print("=" * 80)
    try:
        generation = evaluate_generation_from_dataframe(gold_df, gen_df)
        results['generation_evaluation'] = generation
        
        # Extract per-question results and save to MongoDB
        if generation and generation.get('per_question'):
            for i, per_q in enumerate(generation['per_question']):
                try:
                    run_entry = MongoGenerationRun(
                        run_id=run_id,
                        query_id=per_q.get('question_id', f"q_{i}"),
                        final_answer=per_q.get('final_answer_run', ''),
                        gold_answer=per_q.get('gold_answer', ''),
                        exact_match=float(per_q.get('em', 0)),
                        rouge_l=float(per_q.get('rouge_l', 0)),
                        bleu=float(per_q.get('bleu', 0)),
                        citation_ok=bool(per_q.get('citation_ok', False))
                    )
                    generation_runs_to_save.append(run_entry)
                except Exception as e:
                    logger.warning(f"Could not save generation run for query {per_q.get('question_id')}: {e}")
        
        if generation and generation.get('summary'):
            print(f"ROUGE-1: {generation['summary']['rouge1']['mean']:.3f}")
            print(f"Token F1: {generation['summary']['token_f1']['mean']:.3f}")
            print(f"EM: {generation['summary']['em']['mean']:.3f}")
        print("Generation evaluation complete")
    except Exception as e:
        logger.error(f"Generation evaluation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No generation logs to evaluate")

# Run citation evaluation
if citation_df is not None and not citation_df.empty:
    print("=" * 80)
    print("Running Citation Evaluation...")
    print("=" * 80)
    try:
        citation_eval = evaluate_citations_from_dataframe(gold_df, citation_df, retrieval_df)
        results['citation_evaluation'] = citation_eval
        if citation_eval and citation_eval.get('summary'):
            summary = citation_eval['summary']
            print(f"Citation Precision: {summary.get('precision', {}).get('mean', 0):.3f}")
            print(f"Citation Recall: {summary.get('recall', {}).get('mean', 0):.3f}")
            print(f"Avg Citations per Question: {summary.get('avg_citations', 0):.2f}")
        print("Citation evaluation complete")
    except Exception as e:
        logger.error(f"Citation evaluation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No citation logs to evaluate")

print()
print("=" * 80)
print("Saving evaluation results to MongoDB...")
print("=" * 80)

try:
    # Save individual retrieval runs
    if retrieval_runs_to_save:
        saved_count = mongo_repo.save_retrieval_runs_batch(retrieval_runs_to_save)
        print(f"✓ Saved {saved_count} retrieval run results")
    
    # Save individual generation runs
    if generation_runs_to_save:
        saved_count = mongo_repo.save_generation_runs_batch(generation_runs_to_save)
        print(f"✓ Saved {saved_count} generation run results")
    
    # Create aggregate evaluation run summary
    aggregate_eval = MongoEvalRun(
        run_id=run_id,
        dataset_hash=hashlib.md5(str(len(gold_df)).encode()).hexdigest(),
        model="rag-chatbot",
        backend="mongodb",
        num_queries=len(gold_df),
        aggregates={
            "retrieval_evaluation": results.get('retrieval_evaluation', {}).get('summary', {}),
            "generation_evaluation": results.get('generation_evaluation', {}).get('summary', {}),
            "citation_evaluation": results.get('citation_evaluation', {}).get('summary', {}),
            "gemini_rating": results.get('gemini_rating', {})
        },
        notes=f"Full evaluation run with {retrieval_count} retrieval logs, {gen_count} generation logs, {citation_count} citation logs"
    )
    mongo_repo.save_eval_run(aggregate_eval)
    print(f"✓ Saved aggregate evaluation run: {run_id}")
    
except Exception as e:
    logger.error(f"Failed to save evaluation results to MongoDB: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(f"Run ID: {run_id}")
print(f"Total Retrieval Logs Evaluated: {retrieval_count}")
print(f"Total Generation Logs Evaluated: {gen_count}")
print(f"Total Citation Logs Evaluated: {citation_count}")
print(f"Retrieval runs saved: {len(retrieval_runs_to_save)}")
print(f"Generation runs saved: {len(generation_runs_to_save)}")
print("=" * 80)