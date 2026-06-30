"""
MongoDB repository layer for CRUD operations.
Implements the repository pattern for clean data access.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from backend.utils.mongo_client import get_mongo_db
from backend.utils.mongo_models import (
    MongoDocument, MongoChunk, MongoQueryLog,
    MongoRetrievalRun, MongoGenerationRun, MongoEvalRun
)

logger = logging.getLogger(__name__)


class MongoRepository:
    """
    Repository pattern for MongoDB operations.
    Provides clean interface for all database operations.
    """
    
    def __init__(self):
        self.db = get_mongo_db()
    
    # ==================== DOCUMENTS ====================
    
    def save_document(self, doc: MongoDocument) -> str:
        """
        Save or update document metadata.
        Returns document_id.
        """
        try:
            result = self.db["documents"].insert_one(doc.to_dict())
            logger.info(f"Document saved: {doc.document_id} ({doc.filename})")
            return str(result.inserted_id)
        except DuplicateKeyError:
            # Update existing document
            self.db["documents"].update_one(
                {"_id": doc.document_id},
                {"$set": doc.to_dict()}
            )
            logger.info(f"Document updated: {doc.document_id}")
            return doc.document_id
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Retrieve document by ID."""
        return self.db["documents"].find_one({"_id": document_id})
    
    def get_document_by_hash(self, content_hash: str) -> Optional[Dict]:
        """Retrieve document by content hash."""
        return self.db["documents"].find_one({"content_hash": content_hash})
    
    def document_exists(self, content_hash: str) -> bool:
        """Check if document with hash already exists."""
        return self.db["documents"].find_one({"content_hash": content_hash}) is not None
    
    def list_documents(self, limit: int = 100, skip: int = 0) -> List[Dict]:
        """List all documents with pagination."""
        return list(
            self.db["documents"]
            .find()
            .sort("uploaded_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )
    
    def update_document_status(self, document_id: str, status: str, **kwargs) -> bool:
        """Update document status and optional fields."""
        update_fields = {"status": status, **kwargs}
        result = self.db["documents"].update_one(
            {"_id": document_id},
            {"$set": update_fields}
        )
        return result.modified_count > 0
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks."""
        # Delete chunks first
        self.delete_chunks_by_document(document_id)
        # Delete document
        result = self.db["documents"].delete_one({"_id": document_id})
        if result.deleted_count > 0:
            logger.info(f"Document deleted: {document_id}")
            return True
        return False
    
    def count_documents(self) -> int:
        """Count total documents."""
        return self.db["documents"].count_documents({})
    
    # ==================== CHUNKS ====================
    
    def save_chunks_batch(self, chunks: List[MongoChunk]) -> int:
        """
        Bulk insert/upsert chunks.
        Returns number of chunks inserted/updated.
        """
        if not chunks:
            return 0
        
        # Convert all chunks to dicts
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        
        try:
            # Try insert_many first (faster for new inserts)
            result = self.db["chunks"].insert_many(chunk_dicts, ordered=False)
            count = len(result.inserted_ids)
            logger.info(f"Inserted {count} chunks for document {chunks[0].document_id}")
            return count
        except Exception as e:
            # If duplicates exist, fall back to bulk upsert
            logger.warning(f"insert_many failed (may have duplicates): {e}")
            
            operations = []
            for chunk_dict in chunk_dicts:
                operations.append({
                    "updateOne": {
                        "filter": {"_id": chunk_dict["_id"]},
                        "update": {"$set": chunk_dict},
                        "upsert": True
                    }
                })
            
            try:
                result = self.db["chunks"].bulk_write(operations, ordered=False)
                count = result.upserted_count + result.modified_count
                logger.info(f"Upserted {count} chunks for document {chunks[0].document_id}")
                return count
            except Exception as bulk_err:
                logger.exception(f"bulk_write also failed: {bulk_err}")
                return 0
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict]:
        """Retrieve all chunks for a document, ordered by index."""
        return list(
            self.db["chunks"]
            .find({"document_id": document_id})
            .sort("order_index", ASCENDING)
        )
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve single chunk by ID."""
        return self.db["chunks"].find_one({"_id": chunk_id})
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """Retrieve multiple chunks by IDs."""
        return list(self.db["chunks"].find({"_id": {"$in": chunk_ids}}))
    
    def delete_chunks_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        result = self.db["chunks"].delete_many({"document_id": document_id})
        if result.deleted_count > 0:
            logger.info(f"Deleted {result.deleted_count} chunks for {document_id}")
        return result.deleted_count
    
    def count_chunks(self, document_id: Optional[str] = None) -> int:
        """Count total or per-document chunks."""
        query = {"document_id": document_id} if document_id else {}
        return self.db["chunks"].count_documents(query)
    
    def search_chunks_text(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Full-text search on chunk text.
        Note: Requires text index on 'text' field.
        """
        return list(
            self.db["chunks"]
            .find({"$text": {"$search": query}})
            .limit(limit)
        )
    
    # ==================== QUERY LOGS ====================
    
    def save_query_log(self, log: MongoQueryLog) -> str:
        """
        Save query log entry.
        Returns question_id.
        """
        try:
            result = self.db["query_logs"].insert_one(log.to_dict())
            logger.debug(f"Query log saved: {log.question_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.exception(f"Failed to save query log: {str(e)}")
            raise
    
    def update_query_log(self, question_id: str, updates: Dict[str, Any]) -> bool:
        """Update query log with retrieved/final answer."""
        result = self.db["query_logs"].update_one(
            {"_id": question_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    def get_query_log(self, question_id: str) -> Optional[Dict]:
        """Retrieve single query log by ID."""
        return self.db["query_logs"].find_one({"_id": question_id})
    
    def get_query_logs(self, days: int = 7, limit: int = 1000) -> List[Dict]:
        """Retrieve recent query logs."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return list(
            self.db["query_logs"]
            .find({"ts": {"$gte": cutoff}})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
    
    def get_query_logs_by_user(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Retrieve query logs for a specific user."""
        return list(
            self.db["query_logs"]
            .find({"user_id": user_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
    
    # ==================== RETRIEVAL RUNS ====================
    
    def save_retrieval_run(self, run: MongoRetrievalRun) -> str:
        """
        Save retrieval evaluation results.
        Returns inserted document ID.
        """
        try:
            result = self.db["retrieval_runs"].insert_one(run.to_dict())
            logger.debug(f"Retrieval run saved: {run.run_id} / {run.query_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.exception(f"Failed to save retrieval run: {str(e)}")
            raise
    
    def save_retrieval_runs_batch(self, runs: List[MongoRetrievalRun]) -> int:
        """Bulk insert retrieval runs."""
        if not runs:
            return 0
        docs = [run.to_dict() for run in runs]
        result = self.db["retrieval_runs"].insert_many(docs)
        return len(result.inserted_ids)
    
    def get_retrieval_runs(self, run_id: str) -> List[Dict]:
        """Get all retrieval runs for an evaluation run."""
        return list(self.db["retrieval_runs"].find({"run_id": run_id}))
    
    def get_retrieval_runs_by_query(self, query_id: str) -> List[Dict]:
        """Get all retrieval runs for a specific query."""
        return list(self.db["retrieval_runs"].find({"query_id": query_id}))
    
    # ==================== GENERATION RUNS ====================
    
    def save_generation_run(self, run: MongoGenerationRun) -> str:
        """
        Save generation evaluation results.
        Returns inserted document ID.
        """
        try:
            result = self.db["generation_runs"].insert_one(run.to_dict())
            logger.debug(f"Generation run saved: {run.run_id} / {run.query_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.exception(f"Failed to save generation run: {str(e)}")
            raise
    
    def save_generation_runs_batch(self, runs: List[MongoGenerationRun]) -> int:
        """Bulk insert generation runs."""
        if not runs:
            return 0
        docs = [run.to_dict() for run in runs]
        result = self.db["generation_runs"].insert_many(docs)
        return len(result.inserted_ids)
    
    def get_generation_runs(self, run_id: str) -> List[Dict]:
        """Get all generation runs for an evaluation run."""
        return list(self.db["generation_runs"].find({"run_id": run_id}))
    
    def get_generation_runs_by_query(self, query_id: str) -> List[Dict]:
        """Get all generation runs for a specific query."""
        return list(self.db["generation_runs"].find({"query_id": query_id}))
    
    # ==================== EVAL RUNS ====================
    
    def save_eval_run(self, run: MongoEvalRun) -> str:
        """
        Save aggregate evaluation run.
        Returns run_id.
        """
        try:
            result = self.db["eval_runs"].insert_one(run.to_dict())
            logger.info(f"Eval run saved: {run.run_id}")
            return str(result.inserted_id)
        except DuplicateKeyError:
            # Update existing run
            self.db["eval_runs"].update_one(
                {"_id": run.run_id},
                {"$set": run.to_dict()}
            )
            logger.info(f"Eval run updated: {run.run_id}")
            return run.run_id
    
    def get_eval_run(self, run_id: str) -> Optional[Dict]:
        """Retrieve eval run by ID."""
        return self.db["eval_runs"].find_one({"_id": run_id})
    
    def list_eval_runs(self, limit: int = 50) -> List[Dict]:
        """List recent eval runs."""
        return list(
            self.db["eval_runs"]
            .find()
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
    
    def get_eval_runs_by_backend(self, backend: str, limit: int = 50) -> List[Dict]:
        """List eval runs for specific backend (mongodb/qdrant)."""
        return list(
            self.db["eval_runs"]
            .find({"backend": backend})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
    
    # ==================== CLEANUP ====================
    
    def cleanup_old_logs(self, days: int = 90) -> int:
        """
        Delete query logs older than N days (manual TTL).
        Note: If TTL index is set, MongoDB handles this automatically.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = self.db["query_logs"].delete_many({"ts": {"$lt": cutoff}})
        if result.deleted_count > 0:
            logger.info(f"Cleaned up {result.deleted_count} old query logs")
        return result.deleted_count
    
    def cleanup_eval_runs(self, days: int = 365) -> int:
        """Delete old evaluation runs."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Get run IDs to delete
        old_runs = list(
            self.db["eval_runs"]
            .find({"created_at": {"$lt": cutoff}}, {"_id": 1})
        )
        run_ids = [r["_id"] for r in old_runs]
        
        if not run_ids:
            return 0
        
        # Delete eval runs and related data
        eval_count = self.db["eval_runs"].delete_many({"_id": {"$in": run_ids}}).deleted_count
        retr_count = self.db["retrieval_runs"].delete_many({"run_id": {"$in": run_ids}}).deleted_count
        gen_count = self.db["generation_runs"].delete_many({"run_id": {"$in": run_ids}}).deleted_count
        
        total = eval_count + retr_count + gen_count
        logger.info(f"Cleaned up {total} old evaluation records")
        return total
    
    # ==================== STATISTICS ====================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "documents": self.db["documents"].count_documents({}),
            "chunks": self.db["chunks"].count_documents({}),
            "query_logs": self.db["query_logs"].count_documents({}),
            "retrieval_runs": self.db["retrieval_runs"].count_documents({}),
            "generation_runs": self.db["generation_runs"].count_documents({}),
            "eval_runs": self.db["eval_runs"].count_documents({})
        }
