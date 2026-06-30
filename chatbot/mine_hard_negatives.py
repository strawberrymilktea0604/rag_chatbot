"""
Hard Negative Mining for Reranking Model Training.

This script mines hard negatives from document chunks using BM25 keyword retrieval:
1. Loads training queries from MongoDB 'training_set' collection
2. Builds BM25 corpus from all chunks
3. For each query, retrieves top-K chunks using BM25 (keyword matching)
4. Filters out the positive chunk to find hard negatives (similar keywords, wrong answer)
5. Falls back to random negatives if BM25 only returns the correct chunk
6. Saves triplets (query → positive → negative) to 'trainset_neg' collection

Usage:
    python mine_hard_negatives.py [options]

Options:
    --document-ids <id1,id2,...>  Process specific documents (comma-separated). If not provided, processes all documents
    --top-k <n>                   Number of BM25 results to retrieve (default: 10)
    --min-bm25-score <float>      Minimum BM25 score threshold (default: 0.0)
    --batch-size <n>              Queries per progress report (default: 50)
    --dry-run                     Preview without saving to MongoDB
    --resume                      Skip already processed queries
    --limit <n>                   Limit number of queries to process (for testing)
"""

import os
import sys
import django
import argparse
import logging
import time
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.mongo_repository import MongoRepository
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Mines hard negatives using BM25 keyword-based retrieval.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        min_bm25_score: float = 0.0,
        batch_size: int = 50
    ):
        self.mongo_repo = MongoRepository()
        self.top_k = top_k
        self.min_bm25_score = min_bm25_score
        self.batch_size = batch_size
        
        # BM25 components (initialized later)
        self.bm25 = None
        self.corpus_chunks = []  # List of chunk dicts
        self.chunk_id_to_idx = {}  # Map chunk_id -> corpus index
        self.tokenized_corpus = []  # Tokenized texts for BM25
        
        logger.info(f"HardNegativeMiner initialized")
        logger.info(f"Top-K: {self.top_k}, Min BM25 Score: {self.min_bm25_score}")
    
    def tokenize_vietnamese(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text using underthesea.
        Returns list of tokens (words).
        """
        try:
            tokens = word_tokenize(text, format="text").lower().split()
            return tokens
        except Exception as e:
            logger.warning(f"Tokenization error: {e}. Falling back to simple split.")
            return text.lower().split()
    
    def build_bm25_corpus(self, document_ids: Optional[List[str]] = None):
        """
        Build BM25 index from all chunks in MongoDB.
        Loads chunks, tokenizes texts, and initializes BM25Okapi.
        """
        logger.info("=" * 80)
        logger.info("Building BM25 Corpus")
        logger.info("=" * 80)
        
        # Load chunks from MongoDB
        if document_ids:
            query = {"document_id": {"$in": document_ids}}
            logger.info(f"Loading chunks for {len(document_ids)} documents...")
        else:
            query = {}
            logger.info("Loading all chunks from MongoDB...")
        
        start_time = time.time()
        chunks = list(self.mongo_repo.db["chunks"].find(query))
        
        if not chunks:
            raise ValueError("No chunks found in MongoDB!")
        
        logger.info(f"Loaded {len(chunks)} chunks in {time.time() - start_time:.2f}s")
        
        # Tokenize corpus
        logger.info("Tokenizing corpus for BM25...")
        start_time = time.time()
        
        self.corpus_chunks = chunks
        self.tokenized_corpus = []
        self.chunk_id_to_idx = {}
        
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk["_id"]
            text = chunk.get("text", "")
            
            # Tokenize
            tokens = self.tokenize_vietnamese(text)
            self.tokenized_corpus.append(tokens)
            self.chunk_id_to_idx[chunk_id] = idx
            
            # Progress
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Tokenized {idx + 1}/{len(chunks)} chunks...")
        
        logger.info(f"Tokenization complete in {time.time() - start_time:.2f}s")
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        start_time = time.time()
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built in {time.time() - start_time:.2f}s")
        logger.info("=" * 80)
    
    def retrieve_bm25(self, query: str, exclude_chunk_id: str) -> List[Tuple[str, float]]:
        """
        Retrieve top-K chunks using BM25, excluding the positive chunk.
        Returns list of (chunk_id, bm25_score) tuples sorted by score descending.
        """
        # Tokenize query
        query_tokens = self.tokenize_vietnamese(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Create list of (chunk_id, score) tuples
        results = []
        for idx, score in enumerate(scores):
            chunk_id = self.corpus_chunks[idx]["_id"]
            
            # Skip positive chunk
            if chunk_id == exclude_chunk_id:
                continue
            
            # Apply minimum score threshold
            if score >= self.min_bm25_score:
                results.append((chunk_id, float(score)))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-K
        return results[:self.top_k]
    
    def get_random_negative(self, positive_doc_id: str, positive_chunk_id: str) -> Optional[Dict]:
        """
        Get a random chunk from a different document as fallback negative.
        """
        try:
            # Get random chunk from different document
            candidates = list(
                self.mongo_repo.db["chunks"].aggregate([
                    {
                        "$match": {
                            "document_id": {"$ne": positive_doc_id},
                            "_id": {"$ne": positive_chunk_id}
                        }
                    },
                    {"$sample": {"size": 10}}
                ])
            )
            
            if candidates:
                # Filter out very short chunks
                valid_candidates = [c for c in candidates if len(c.get("text", "")) > 50]
                if valid_candidates:
                    return random.choice(valid_candidates)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting random negative: {e}")
            return None
    
    def mine_hard_negative(
        self,
        query: str,
        positive_chunk_id: str,
        positive_doc_id: str
    ) -> Optional[Tuple[Dict, str]]:
        """
        Mine a hard negative for the given query.
        Returns (negative_chunk_dict, strategy) or None if failed.
        Strategy is either 'bm25' or 'random'.
        """
        # Try BM25 retrieval first
        bm25_results = self.retrieve_bm25(query, exclude_chunk_id=positive_chunk_id)
        
        if bm25_results:
            # Found hard negative via BM25
            negative_chunk_id, bm25_score = bm25_results[0]
            negative_chunk = self.corpus_chunks[self.chunk_id_to_idx[negative_chunk_id]]
            
            logger.debug(f"  BM25 hard negative: {negative_chunk_id} (score: {bm25_score:.4f})")
            return (negative_chunk, "bm25")
        
        else:
            # Fallback to random negative
            logger.debug(f"  BM25 failed (no valid negatives), using random fallback")
            negative_chunk = self.get_random_negative(positive_doc_id, positive_chunk_id)
            
            if negative_chunk:
                return (negative_chunk, "random")
            else:
                logger.warning(f"  Failed to get random negative")
                return None
    
    def get_processed_query_ids(self, document_ids: Optional[List[str]] = None) -> set:
        """
        Get set of query IDs that have already been processed.
        Query ID format: "{training_entry_id}::{query_index}"
        """
        if document_ids:
            query = {"document_id": {"$in": document_ids}}
        else:
            query = {}
        
        existing = self.mongo_repo.db["trainset_neg"].find(query, {"query_id": 1})
        return {doc["query_id"] for doc in existing}
    
    def save_training_triplet(
        self,
        training_entry: Dict,
        query: str,
        query_index: int,
        negative_chunk: Dict,
        strategy: str
    ) -> bool:
        """
        Save one training triplet to MongoDB trainset_neg collection.
        Format: 1 query → 1 positive → 1 negative (1 row per query).
        """
        try:
            # Generate unique ID for this query
            query_id = f"{training_entry['_id']}::q{query_index}"
            
            # Get positive chunk info
            positive_chunk_id = training_entry["chunk_id"]
            positive_chunk_text = training_entry["chunk_text"]
            
            # Create triplet entry
            entry = {
                "_id": query_id,  # Unique ID per query
                "query_id": query_id,
                "training_entry_id": training_entry["_id"],
                "document_id": training_entry.get("document_id"),
                
                # Query
                "query": query,
                "query_index": query_index,
                
                # Positive
                "positive_chunk_id": positive_chunk_id,
                "positive": positive_chunk_text,
                
                # Negative
                "negative_chunk_id": negative_chunk["_id"],
                "negative": negative_chunk.get("text", ""),
                "negative_document_id": negative_chunk.get("document_id"),
                
                # Metadata
                "metadata": {
                    "positive_metadata": training_entry.get("metadata", {}),
                    "negative_metadata": {
                        "section_title": negative_chunk.get("metadata", {}).get("section_title"),
                        "document_title": negative_chunk.get("metadata", {}).get("document_title"),
                        "page": negative_chunk.get("page"),
                    }
                },
                
                # Mining info
                "mining_strategy": strategy,
                "created_at": datetime.now(),
            }
            
            # Upsert to handle re-runs
            self.mongo_repo.db["trainset_neg"].update_one(
                {"_id": entry["_id"]},
                {"$set": entry},
                upsert=True
            )
            
            logger.debug(f"    Saved triplet: {query_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to save triplet for query '{query}': {e}")
            return False
    
    def process_training_set(
        self,
        document_ids: Optional[List[str]] = None,
        resume: bool = False,
        limit: Optional[int] = None,
        dry_run: bool = False
    ):
        """
        Main processing loop: iterate training_set, mine hard negatives, save triplets.
        """
        logger.info("=" * 80)
        logger.info("Starting Hard Negative Mining")
        logger.info("=" * 80)
        
        # Load training entries
        if document_ids:
            query = {"document_id": {"$in": document_ids}}
            logger.info(f"Loading training entries for {len(document_ids)} documents...")
        else:
            query = {}
            logger.info("Loading all training entries...")
        
        training_entries = list(self.mongo_repo.db["training_set"].find(query))
        
        if not training_entries:
            logger.warning("No training entries found!")
            return
        
        logger.info(f"Loaded {len(training_entries)} training entries")
        
        # Build BM25 corpus
        self.build_bm25_corpus(document_ids=document_ids)
        
        # Resume: skip already processed queries
        if resume:
            processed_ids = self.get_processed_query_ids(document_ids=document_ids)
            logger.info(f"Resume mode: {len(processed_ids)} queries already processed")
        else:
            processed_ids = set()
        
        # Count total queries to process
        total_queries = 0
        for entry in training_entries:
            queries = entry.get("queries", [])
            for idx in range(len(queries)):
                query_id = f"{entry['_id']}::q{idx}"
                if query_id not in processed_ids:
                    total_queries += 1
        
        if limit:
            total_queries = min(total_queries, limit)
            logger.info(f"Limiting to {limit} queries for testing")
        
        logger.info(f"Total queries to process: {total_queries}")
        
        if total_queries == 0:
            logger.warning("No queries to process!")
            return
        
        # Dry run preview
        if dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("DRY RUN - Preview of queries to process:")
            logger.info("=" * 80)
            
            preview_count = 0
            for entry in training_entries[:5]:  # Show first 5 entries
                chunk_id = entry["chunk_id"]
                queries = entry.get("queries", [])
                
                logger.info(f"\nTraining Entry: {entry['_id']}")
                logger.info(f"  Document: {entry.get('document_id')}")
                logger.info(f"  Positive Chunk: {chunk_id}")
                logger.info(f"  Queries:")
                
                for idx, query in enumerate(queries):
                    query_id = f"{entry['_id']}::q{idx}"
                    if query_id not in processed_ids:
                        logger.info(f"    [{idx}] {query}")
                        preview_count += 1
                
                if preview_count >= 10:
                    break
            
            logger.info(f"\n... Total: {total_queries} queries to process")
            logger.info("=" * 80)
            return
        
        # Process queries
        logger.info("\n" + "=" * 80)
        logger.info(f"Processing {total_queries} queries")
        logger.info("=" * 80)
        
        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "bm25": 0,
            "random": 0,
        }
        
        start_time = time.time()
        processed_count = 0
        
        for entry_idx, entry in enumerate(training_entries):
            chunk_id = entry["chunk_id"]
            document_id = entry.get("document_id")
            queries = entry.get("queries", [])
            
            # Process each query in this training entry
            for query_idx, query in enumerate(queries):
                query_id = f"{entry['_id']}::q{query_idx}"
                
                # Skip if already processed (resume mode)
                if query_id in processed_ids:
                    continue
                
                # Check limit
                if limit and processed_count >= limit:
                    logger.info(f"\nReached limit of {limit} queries")
                    break
                
                stats["total"] += 1
                processed_count += 1
                
                logger.info(f"[{processed_count}/{total_queries}] Processing query: {query_id}")
                logger.debug(f"  Query: {query}")
                logger.debug(f"  Positive: {chunk_id}")
                
                # Mine hard negative
                result = self.mine_hard_negative(
                    query=query,
                    positive_chunk_id=chunk_id,
                    positive_doc_id=document_id
                )
                
                if result:
                    negative_chunk, strategy = result
                    
                    # Save triplet
                    success = self.save_training_triplet(
                        training_entry=entry,
                        query=query,
                        query_index=query_idx,
                        negative_chunk=negative_chunk,
                        strategy=strategy
                    )
                    
                    if success:
                        stats["success"] += 1
                        stats[strategy] += 1
                        logger.info(f"  ✓ Negative: {negative_chunk['_id']} (strategy: {strategy})")
                    else:
                        stats["failed"] += 1
                else:
                    stats["failed"] += 1
                    logger.warning(f"  ✗ Failed to find negative")
                
                # Progress report
                if processed_count % self.batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    remaining = (total_queries - processed_count) / rate if rate > 0 else 0
                    
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"Progress: {processed_count}/{total_queries} queries processed")
                    logger.info(f"Success: {stats['success']}, Failed: {stats['failed']}")
                    logger.info(f"Strategies: BM25={stats['bm25']}, Random={stats['random']}")
                    logger.info(f"Rate: {rate:.2f} queries/sec")
                    logger.info(f"Estimated time remaining: {remaining/60:.1f} minutes")
                    logger.info(f"{'=' * 80}\n")
            
            # Check limit (outer loop)
            if limit and processed_count >= limit:
                break
        
        # Final summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("Hard Negative Mining Complete")
        logger.info("=" * 80)
        logger.info(f"Total queries processed: {stats['total']}")
        logger.info(f"Successfully mined: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Mining strategies:")
        logger.info(f"  BM25 (hard negatives): {stats['bm25']}")
        logger.info(f"  Random (fallback): {stats['random']}")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        if stats['total'] > 0:
            logger.info(f"Average rate: {stats['total']/elapsed:.2f} queries/sec")
            logger.info(f"Success rate: {stats['success']/stats['total']*100:.1f}%")
        logger.info("=" * 80)
        
        # Show samples
        logger.info("\nSample training triplets:")
        samples = list(self.mongo_repo.db["trainset_neg"].find().limit(3))
        for sample in samples:
            logger.info(f"\nQuery ID: {sample['query_id']}")
            logger.info(f"  Query: {sample['query']}")
            logger.info(f"  Positive: {sample['positive_chunk_id']}")
            logger.info(f"  Negative: {sample['negative_chunk_id']} (strategy: {sample['mining_strategy']})")
            logger.info(f"  Positive text: {sample['positive'][:100]}...")
            logger.info(f"  Negative text: {sample['negative'][:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Mine hard negatives using BM25 keyword retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--document-ids",
        type=str,
        help="Process specific documents (comma-separated). If not provided, processes all documents"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of BM25 results to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--min-bm25-score",
        type=float,
        default=0.0,
        help="Minimum BM25 score threshold (default: 0.0)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Queries per progress report (default: 50)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without saving to MongoDB"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed queries"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of queries to process (for testing)"
    )
    
    args = parser.parse_args()
    
    try:
        miner = HardNegativeMiner(
            top_k=args.top_k,
            min_bm25_score=args.min_bm25_score,
            batch_size=args.batch_size
        )
        
        # Parse document IDs if provided
        document_ids = None
        if args.document_ids:
            document_ids = [doc_id.strip() for doc_id in args.document_ids.split(",")]
            logger.info(f"Processing {len(document_ids)} specified documents: {document_ids}")
        
        miner.process_training_set(
            document_ids=document_ids,
            resume=args.resume,
            limit=args.limit,
            dry_run=args.dry_run
        )
        
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user. Progress saved to MongoDB.")
        logger.info("Use --resume flag to continue from checkpoint.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
