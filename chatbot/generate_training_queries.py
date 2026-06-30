"""
Training Data Generator for Reranking Model Fine-tuning.

This script generates training query sets from document chunks by:
1. Loading all chunks from MongoDB
2. Filtering out short/header-only chunks
3. Sending each chunk to Gemini with structured prompt
4. Generating 3 types of queries: keyword, natural language, scenario-based
5. Storing results in MongoDB 'training_set' collection

Usage:
    python generate_training_queries.py [options]

Options:
    --document-ids <id1,id2,...>  Process specific documents (comma-separated). If not provided, processes all documents
    --batch-size <n>              Number of chunks to process per batch (default: 10)
    --delay <seconds>             Delay between API calls (default: 2.0)
    --min-length <n>              Minimum chunk text length (default: 100)
    --temperature <float>         Gemini temperature for diversity (default: 0.8)
    --dry-run                     Preview chunks without generating queries
    --resume                      Continue from last checkpoint (per document)
    --limit <n>                   Limit number of chunks to process (for testing)
"""

import os
import sys
import django
import argparse
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.mongo_repository import MongoRepository
from google import generativeai
from google.api_core.exceptions import GoogleAPICallError, RetryError, DeadlineExceeded, ServiceUnavailable
from django.conf import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingQueryGenerator:
    """
    Generates training queries from document chunks using Gemini API.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        delay: float = 2.0,
        min_chunk_length: int = 100,
        temperature: float = 0.8,
        max_retries: int = 3
    ):
        self.mongo_repo = MongoRepository()
        self.batch_size = batch_size
        self.delay = delay
        self.min_chunk_length = min_chunk_length
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        generativeai.configure(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = generativeai.GenerativeModel(self.model_name)
        
        logger.info(f"TrainingQueryGenerator initialized with model: {self.model_name}")
        logger.info(f"Temperature: {self.temperature}, Min chunk length: {self.min_chunk_length}")
    
    def get_prompt_template(self, chunk_text: str) -> str:
        """
        Create prompt for Gemini using system prompt + user template.
        System prompt sets the role and rules, user template provides the content.
        """
        # System prompt - defines the AI's role and rules
        system_prompt = """Bạn là một giảng viên đại học uy tín tại Việt Nam, chuyên ngành Công nghệ thông tin.
Nhiệm vụ của bạn là tạo ra các câu hỏi ôn tập (Queries) dựa trên nội dung bài giảng được cung cấp.

QUY TẮC QUAN TRỌNG (Bắt buộc tuân thủ):
1. **Ngôn ngữ:** Sử dụng tiếng Việt tự nhiên, văn phong học thuật nhưng dễ hiểu.
2. **Thuật ngữ:** TUYỆT ĐÓI GIỮ NGUYÊN các thuật ngữ chuyên ngành tiếng Anh (ví dụ: 'Backpropagation', 'Docker Container', 'Gradient Descent'). Không dịch sang tiếng Việt nếu từ đó giới IT thường dùng nguyên bản.
3. **Đa dạng hóa câu hỏi:**
   - Một câu hỏi ngắn gọn theo từ khóa (Keyword-based).
   - Một câu hỏi dạng "Làm thế nào" hoặc "Tại sao" (How-to/Reasoning).
   - Một câu hỏi mô tả tình huống lỗi hoặc kịch bản thực tế (Scenario).
4. **Định dạng:** Chỉ trả về kết quả dưới dạng JSON Object chứa danh sách câu hỏi.
"""
        
        # User template - provides the actual content
        user_template = f"""Dưới đây là một đoạn nội dung từ slide bài giảng:
---
{chunk_text}
---

Hãy đóng vai một sinh viên Việt Nam đang ôn thi. Hãy đặt **3 câu hỏi tìm kiếm** mà sinh viên này có thể gõ vào thanh tìm kiếm để tìm ra đoạn nội dung trên.
Output format JSON: {{{{ "questions": ["Câu hỏi 1?", "Câu hỏi 2?", "Câu hỏi 3?"] }}}}
"""
        
        # Combine system prompt and user template
        return system_prompt + "\n\n" + user_template
    
    def get_simple_prompt(self, chunk_text: str) -> str:
        """
        Simplified fallback prompt for retry attempts.
        Uses a more concise format to reduce API errors.
        """
        return f"""Đoạn văn từ slide bài giảng:
{chunk_text}

Đặt 3 câu hỏi tìm kiếm (search query) mà sinh viên có thể gõ để tìm đoạn văn này.
JSON format: {{{{ "questions": ["câu 1", "câu 2", "câu 3"] }}}}
"""


    
    def clean_json_response(self, text: str) -> str:
        """
        Clean and extract JSON from Gemini response.
        Handles markdown code blocks and extracts complete JSON objects/arrays.
        """
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:].strip()
        
        # Remove trailing ```
        if text.endswith("```"):
            text = text[:-3].strip()
        
        # Try to extract JSON object first (our expected format: {"questions": [...]})
        if "{" in text and "}" in text:
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            # Only extract if we have a complete object
            if first_brace < last_brace:
                text = text[first_brace:last_brace + 1]
                return text.strip()
        
        # Fallback: try to extract JSON array (for backward compatibility)
        if "[" in text and "]" in text:
            first_bracket = text.find("[")
            last_bracket = text.rfind("]")
            if first_bracket < last_bracket:
                text = text[first_bracket:last_bracket + 1]
        
        return text.strip()
    
    def parse_queries_response(self, text: str) -> Optional[List[str]]:
        """
        Parse and validate queries from cleaned JSON text.
        Returns list of 3 valid queries or None if invalid.
        """
        try:
            queries = json.loads(text)
            
            # Handle case where response is a dict with questions/queries key
            if isinstance(queries, dict):
                if "questions" in queries:
                    queries = queries["questions"]
                elif "queries" in queries:
                    queries = queries["queries"]
                elif "results" in queries:
                    queries = queries["results"]
                else:
                    logger.warning(f"Response is a dict but no questions/queries key found. Keys: {list(queries.keys())[:5]}")
                    return None
            
            # Validate structure
            if not isinstance(queries, list):
                logger.warning(f"Response is not a list: {type(queries)}. Response: {text[:100]}")
                return None
            
            # Filter out non-string or empty items
            valid_queries = [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 0]
            
            # Handle different query counts
            if len(valid_queries) < 2:
                logger.warning(f"Too few valid queries: got {len(valid_queries)}. Response: {text[:100]}")
                return None
            
            if len(valid_queries) == 2:
                # Duplicate the second query as scenario-based
                logger.info("Got 2 queries, duplicating last one to make 3")
                valid_queries.append(valid_queries[1])
            elif len(valid_queries) > 3:
                # Take first 3
                logger.info(f"Got {len(valid_queries)} queries, using first 3")
                valid_queries = valid_queries[:3]
            elif len(valid_queries) != 3:
                logger.warning(f"Invalid query count: got {len(valid_queries)} valid queries. Response: {text[:100]}")
                return None
            
            return valid_queries
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}. Text: {text[:100]}")
            return None
    
    def generate_queries_for_chunk(self, chunk: Dict) -> Optional[List[str]]:
        """
        Send chunk to Gemini and parse response to get 3 queries.
        Returns list of 3 queries or None if failed.
        """
        chunk_text = chunk.get("text", "")
        
        # Start with full prompt, use simple prompt on last attempt
        prompt = self.get_prompt_template(chunk_text)
        
        for attempt in range(self.max_retries):
            # Use simpler prompt on last attempt as fallback
            if attempt == self.max_retries - 1:
                logger.info(f"Using simple fallback prompt for chunk {chunk['_id']} (last attempt)")
                current_prompt = self.get_simple_prompt(chunk_text)
            else:
                current_prompt = prompt
            
            try:
                logger.debug(f"Generating queries for chunk {chunk['_id']} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.model.generate_content(
                    current_prompt,
                    generation_config=generativeai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=4096,
                        candidate_count=1,
                    ),
                )
                
                # Extract text from response
                if not response or not response.candidates:
                    logger.warning(f"Empty response from Gemini for chunk {chunk['_id']}")
                    time.sleep(self.delay * (attempt + 1))
                    continue
                
                text = response.text.strip()
                
                # Clean and parse JSON response
                cleaned_text = self.clean_json_response(text)
                
                queries = self.parse_queries_response(cleaned_text)
                
                if queries is None:
                    logger.warning(f"Failed to parse valid queries for chunk {chunk['_id']} (attempt {attempt + 1})")
                    logger.warning(f"Raw response: {text[:300]}")
                    logger.warning(f"Cleaned: {cleaned_text[:300]}")
                    time.sleep(self.delay * (attempt + 1))
                    continue
                
                logger.info(f"✓ Successfully generated 3 queries for chunk {chunk['_id']}")
                return queries
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for chunk {chunk['_id']} (attempt {attempt + 1}): {e}")
                logger.debug(f"Raw response: {text if 'text' in locals() else 'N/A'}")
                time.sleep(self.delay * (attempt + 1))
                
            except (GoogleAPICallError, RetryError, DeadlineExceeded, ServiceUnavailable) as e:
                logger.warning(f"Gemini API error for chunk {chunk['_id']} (attempt {attempt + 1}): {e}")
                time.sleep(self.delay * (attempt + 1) * 2)  # Longer delay for API errors
                
            except Exception as e:
                logger.exception(f"Unexpected error for chunk {chunk['_id']}: {e}")
                time.sleep(self.delay * (attempt + 1))
        
        logger.error(f"✗ Failed to generate queries for chunk {chunk['_id']} after {self.max_retries} attempts")
        return None
    
    def filter_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Filter out very short chunks and header-only chunks.
        """
        filtered = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            text_len = len(text.strip())
            
            # Skip very short chunks
            if text_len < self.min_chunk_length:
                logger.debug(f"Skipping short chunk {chunk['_id']} ({text_len} chars)")
                continue
            
            # Skip header-only chunks (heuristic: all uppercase or very short lines)
            lines = text.strip().split("\n")
            non_empty_lines = [l for l in lines if l.strip()]
            
            if len(non_empty_lines) <= 2 and all(
                len(l.strip()) < 50 or l.strip().isupper() 
                for l in non_empty_lines
            ):
                logger.debug(f"Skipping header-only chunk {chunk['_id']}")
                continue
            
            filtered.append(chunk)
        
        logger.info(f"Filtered {len(chunks)} chunks -> {len(filtered)} chunks (removed {len(chunks) - len(filtered)})")
        return filtered
    
    def get_all_documents(self) -> List[str]:
        """
        Get list of all unique document IDs in the chunks collection.
        """
        docs = self.mongo_repo.db["chunks"].aggregate([
            {"$group": {"_id": "$document_id"}},
            {"$match": {"_id": {"$ne": None}}}
        ])
        return [doc["_id"] for doc in docs]
    
    def get_processed_chunk_ids(self) -> set:
        """
        Get set of chunk IDs that have already been processed.
        """
        existing = self.mongo_repo.db["training_set"].find({}, {"_id": 1})
        return {doc["_id"] for doc in existing}
    
    def get_processed_chunk_ids_by_document(self, document_id: str) -> set:
        """
        Get set of chunk IDs already processed for a specific document.
        """
        existing = self.mongo_repo.db["training_set"].find(
            {"document_id": document_id},
            {"_id": 1}
        )
        return {doc["_id"] for doc in existing}
    
    def save_training_entry(self, chunk: Dict, queries: List[str]) -> bool:
        """
        Save training entry to MongoDB training_set collection.
        """
        try:
            entry = {
                "_id": chunk["_id"],  # Use chunk_id as primary key
                "chunk_id": chunk["_id"],
                "document_id": chunk.get("document_id"),
                "chunk_text": chunk.get("text", ""),
                "queries": queries,
                "metadata": {
                    "section_title": chunk.get("metadata", {}).get("section_title"),
                    "section_path": chunk.get("metadata", {}).get("section_path"),
                    "document_title": chunk.get("metadata", {}).get("document_title"),
                    "page": chunk.get("page"),
                    "order_index": chunk.get("order_index"),
                    "hierarchy_level": chunk.get("metadata", {}).get("hierarchy_level"),
                },
                "generated_at": datetime.now(),
                "model": self.model_name,
                "temperature": self.temperature,
            }
            
            # Upsert to handle re-runs
            self.mongo_repo.db["training_set"].update_one(
                {"_id": entry["_id"]},
                {"$set": entry},
                upsert=True
            )
            
            logger.debug(f"Saved training entry for chunk {chunk['_id']}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to save training entry for chunk {chunk['_id']}: {e}")
            return False
    
    def process_chunks(
        self,
        document_ids: Optional[List[str]] = None,
        resume: bool = False,
        limit: Optional[int] = None,
        dry_run: bool = False
    ):
        """
        Main processing loop: load chunks, filter, generate queries, save to MongoDB.
        Supports multiple documents with per-document checkpoint tracking.
        """
        logger.info("=" * 80)
        logger.info("Starting Training Query Generation")
        logger.info("=" * 80)
        
        # Determine which documents to process
        if document_ids:
            docs_to_process = document_ids
            logger.info(f"Processing {len(docs_to_process)} specified documents")
        else:
            docs_to_process = self.get_all_documents()
            logger.info(f"Found {len(docs_to_process)} documents in MongoDB. Processing all...")
        
        if not docs_to_process:
            logger.warning("No documents found to process!")
            return
        
        # Overall stats across all documents
        overall_stats = {
            "total_documents": len(docs_to_process),
            "total_chunks": 0,
            "success": 0,
            "failed": 0,
        }
        
        overall_start_time = time.time()
        
        # Process each document
        for doc_idx, document_id in enumerate(docs_to_process, 1):
            logger.info("\n" + "=" * 80)
            logger.info(f"[{doc_idx}/{len(docs_to_process)}] Processing document: {document_id}")
            logger.info("=" * 80)
            
            # Load chunks for this document
            logger.info(f"Loading chunks for document: {document_id}")
            chunks = list(self.mongo_repo.db["chunks"].find({"document_id": document_id}))
            
            if not chunks:
                logger.warning(f"No chunks found for document: {document_id}")
                continue
            
            logger.info(f"Loaded {len(chunks)} chunks for this document")
            
            # Filter chunks
            chunks = self.filter_chunks(chunks)
            
            # Resume from checkpoint (per document)
            if resume:
                processed_ids = self.get_processed_chunk_ids_by_document(document_id)
                chunks = [c for c in chunks if c["_id"] not in processed_ids]
                logger.info(f"Resume mode: {len(chunks)} chunks remaining for this document (already processed {len(processed_ids)})")
            
            # Limit for testing
            if limit:
                chunks = chunks[:limit]
                logger.info(f"Limiting to {limit} chunks for testing")
            
            if not chunks:
                logger.warning(f"No chunks to process for document: {document_id}")
                continue
            
            # Dry run: just show what would be processed
            if dry_run:
                logger.info("DRY RUN - Preview of chunks to process for this document:")
                for i, chunk in enumerate(chunks[:10]):  # Show first 10
                    logger.info(f"  [{i+1}] Chunk ID: {chunk['_id']}")
                    logger.info(f"      Page: {chunk.get('page')}")
                    logger.info(f"      Length: {len(chunk.get('text', ''))} chars")
                    logger.info(f"      Preview: {chunk.get('text', '')[:100]}...")
                
                if len(chunks) > 10:
                    logger.info(f"  ... and {len(chunks) - 10} more chunks")
                
                logger.info(f"  Total for this document: {len(chunks)} chunks")
                continue
            
            # Process chunks for this document in batches
            logger.info(f"Processing {len(chunks)} chunks in batches of {self.batch_size}")
            logger.info(f"Delay between requests: {self.delay}s")
            
            doc_stats = {
                "total": len(chunks),
                "success": 0,
                "failed": 0,
            }
            
            doc_start_time = time.time()
            
            for i, chunk in enumerate(chunks):
                logger.info(f"  [{i+1}/{len(chunks)}] Processing chunk: {chunk['_id']}")
                
                # Generate queries
                queries = self.generate_queries_for_chunk(chunk)
                
                if queries:
                    # Save to MongoDB
                    if self.save_training_entry(chunk, queries):
                        doc_stats["success"] += 1
                        overall_stats["success"] += 1
                        logger.info(f"    ✓ Queries: {queries}")
                    else:
                        doc_stats["failed"] += 1
                        overall_stats["failed"] += 1
                else:
                    doc_stats["failed"] += 1
                    overall_stats["failed"] += 1
                
                # Delay between requests (except last one)
                if i < len(chunks) - 1:
                    time.sleep(self.delay)
                
                # Progress report every batch
                if (i + 1) % self.batch_size == 0:
                    elapsed = time.time() - doc_start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(chunks) - i - 1) / rate if rate > 0 else 0
                    
                    logger.info(f"  Progress: {i+1}/{len(chunks)} chunks processed")
                    logger.info(f"  Success: {doc_stats['success']}, Failed: {doc_stats['failed']}")
                    logger.info(f"  Rate: {rate:.2f} chunks/sec")
                    logger.info(f"  Estimated time remaining: {remaining/60:.1f} minutes")
            
            # Document summary
            doc_elapsed = time.time() - doc_start_time
            logger.info(f"\nDocument complete: {document_id}")
            logger.info(f"  Chunks processed: {doc_stats['total']}")
            logger.info(f"  Successfully generated: {doc_stats['success']}")
            logger.info(f"  Failed: {doc_stats['failed']}")
            logger.info(f"  Time: {doc_elapsed/60:.1f} minutes")
            
            overall_stats["total_chunks"] += doc_stats["total"]
        
        # Overall summary
        overall_elapsed = time.time() - overall_start_time
        logger.info("\n" + "=" * 80)
        logger.info("Training Query Generation Complete")
        logger.info("=" * 80)
        logger.info(f"Documents processed: {overall_stats['total_documents']}")
        logger.info(f"Total chunks processed: {overall_stats['total_chunks']}")
        logger.info(f"Successfully generated: {overall_stats['success']}")
        logger.info(f"Failed: {overall_stats['failed']}")
        logger.info(f"Total time: {overall_elapsed/60:.1f} minutes")
        if overall_stats['total_chunks'] > 0:
            logger.info(f"Average rate: {overall_stats['total_chunks']/overall_elapsed:.2f} chunks/sec")
        logger.info("=" * 80)
        
        # Show sample from training_set
        logger.info("\nSample training entries:")
        samples = list(self.mongo_repo.db["training_set"].find().limit(3))
        for sample in samples:
            logger.info(f"\nChunk: {sample['chunk_id']} (Doc: {sample.get('document_id', 'N/A')})")
            logger.info(f"  Queries: {sample['queries']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training queries from document chunks using Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--document-ids",
        type=str,
        help="Process specific documents (comma-separated). If not provided, processes all documents"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of chunks per progress report (default: 10)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between API calls in seconds (default: 2.0)"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum chunk text length to process (default: 100)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Gemini temperature for query diversity (default: 0.8)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview chunks without generating queries"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last checkpoint (skip already processed chunks)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of chunks to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate temperature range
    if not 0.0 <= args.temperature <= 1.0:
        parser.error("Temperature must be between 0.0 and 1.0")
    
    try:
        generator = TrainingQueryGenerator(
            batch_size=args.batch_size,
            delay=args.delay,
            min_chunk_length=args.min_length,
            temperature=args.temperature
        )
        
        # Parse document IDs if provided
        document_ids = None
        if args.document_ids:
            document_ids = [doc_id.strip() for doc_id in args.document_ids.split(",")]
            logger.info(f"Processing {len(document_ids)} specified documents: {document_ids}")
        
        generator.process_chunks(
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
