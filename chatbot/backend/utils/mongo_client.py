"""
MongoDB client and connection management.
Handles connection pooling, database access, and index initialization.
"""

import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ServerSelectionTimeoutError
import logging

logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "rag_chatbot")

# Global singleton instances
_client = None
_db = None


def get_mongo_client():
    """
    Returns MongoDB client singleton.
    Creates connection on first call and reuses for subsequent calls.
    """
    global _client
    if _client is None:
        try:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            _client.server_info()
            logger.info(f"✓ Connected to MongoDB at {MONGO_URI}")
        except ServerSelectionTimeoutError as e:
            logger.error(f"✗ Failed to connect to MongoDB at {MONGO_URI}: {str(e)}")
            raise
    return _client


def get_mongo_db():
    """
    Returns MongoDB database instance.
    """
    global _db
    if _db is None:
        mongo_client = get_mongo_client()
        _db = mongo_client[MONGO_DB_NAME]
        logger.info(f"✓ Using MongoDB database: {MONGO_DB_NAME}")
    return _db


def init_indexes():
    """
    Initialize all required indexes for collections.
    Safe to call multiple times - MongoDB won't recreate existing indexes.
    """
    database = get_mongo_db()
    
    try:
        # ===== DOCUMENTS COLLECTION =====
        logger.info("Creating indexes for 'documents' collection...")
        database["documents"].create_index("content_hash", unique=True, name="idx_content_hash")
        database["documents"].create_index([("uploaded_at", DESCENDING)], name="idx_uploaded_at")
        database["documents"].create_index("status", name="idx_status")
        logger.info("documents indexes created")
        
        # ===== CHUNKS COLLECTION =====
        logger.info("Creating indexes for 'chunks' collection...")
        database["chunks"].create_index("document_id", name="idx_document_id")
        database["chunks"].create_index(
            [("document_id", ASCENDING), ("order_index", ASCENDING)],
            name="idx_doc_order"
        )
        database["chunks"].create_index([("text", "text")], name="idx_text_search")
        logger.info("  ✓ chunks indexes created")
        
        # ===== QUERY LOGS COLLECTION =====
        logger.info("Creating indexes for 'query_logs' collection...")
        database["query_logs"].create_index([("ts", DESCENDING)], name="idx_ts")
        database["query_logs"].create_index("question_id", name="idx_question_id")
        database["query_logs"].create_index("correlation_id", name="idx_correlation_id")
        database["query_logs"].create_index("user_id", name="idx_user_id")
        
        # TTL index for auto-cleanup (90 days)
        database["query_logs"].create_index(
            "ts",
            expireAfterSeconds=90*24*60*60,  # 90 days in seconds
            name="idx_ts_ttl"
        )
        logger.info("query_logs indexes created (including TTL)")
        
        # ===== RETRIEVAL RUNS COLLECTION =====
        logger.info("Creating indexes for 'retrieval_runs' collection...")
        database["retrieval_runs"].create_index("run_id", name="idx_run_id")
        database["retrieval_runs"].create_index("query_id", name="idx_query_id")
        database["retrieval_runs"].create_index([("ts", DESCENDING)], name="idx_ts")
        logger.info("retrieval_runs indexes created")
        
        # ===== GENERATION RUNS COLLECTION =====
        logger.info("Creating indexes for 'generation_runs' collection...")
        database["generation_runs"].create_index("run_id", name="idx_run_id")
        database["generation_runs"].create_index("query_id", name="idx_query_id")
        database["generation_runs"].create_index([("ts", DESCENDING)], name="idx_ts")
        logger.info("generation_runs indexes created")
        
        # ===== EVAL RUNS COLLECTION =====
        logger.info("Creating indexes for 'eval_runs' collection...")
        database["eval_runs"].create_index("run_id", unique=True, name="idx_run_id_unique")
        database["eval_runs"].create_index([("created_at", DESCENDING)], name="idx_created_at")
        database["eval_runs"].create_index("backend", name="idx_backend")
        logger.info("eval_runs indexes created")
        
        logger.info("=" * 50)
        logger.info("All MongoDB indexes initialized successfully")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.exception(f"Error creating indexes: {str(e)}")
        raise


def close_mongo_connection():
    """
    Close MongoDB connection. Useful for cleanup.
    """
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")


def check_mongo_health():
    """
    Check MongoDB connection health.
    Returns dict with status information.
    """
    try:
        client = get_mongo_client()
        db = get_mongo_db()
        
        # Get server info
        server_info = client.server_info()
        
        # Get collection counts
        collections_info = {}
        for coll_name in ["documents", "chunks", "query_logs", "retrieval_runs", "generation_runs", "eval_runs"]:
            count = db[coll_name].count_documents({})
            collections_info[coll_name] = count
        
        return {
            "status": "healthy",
            "mongodb_version": server_info.get("version"),
            "database": MONGO_DB_NAME,
            "collections": collections_info
        }
    except Exception as e:
        logger.exception(f"MongoDB health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
