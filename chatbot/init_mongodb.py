"""
MongoDB initialization script.
Run this to create collections and indexes before starting the application.

Usage:
    python init_mongodb.py
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.mongo_client import get_mongo_client, get_mongo_db, init_indexes, check_mongo_health
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Initialize MongoDB collections and indexes."""
    print("=" * 60)
    print("MongoDB Initialization for RAG Chatbot")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Test connection
        print("Step 1: Testing MongoDB connection...")
        get_mongo_client()  # Test connection
        db = get_mongo_db()
        print("Connected to MongoDB")
        print(f"  Database: {db.name}")
        print()
        
        # Step 2: List existing collections
        print("Step 2: Checking existing collections...")
        existing_collections = db.list_collection_names()
        print(f"  Found {len(existing_collections)} existing collections:")
        for coll in existing_collections:
            count = db[coll].count_documents({})
            print(f"    - {coll}: {count} documents")
        print()
        
        # Step 3: Create indexes
        print("Step 3: Creating indexes...")
        init_indexes()
        print()
        
        # Step 4: Verify collections
        print("Step 4: Verifying collections after index creation...")
        required_collections = [
            "documents",
            "chunks", 
            "query_logs",
            "retrieval_runs",
            "generation_runs",
            "eval_runs"
        ]
        
        final_collections = db.list_collection_names()
        for coll_name in required_collections:
            if coll_name in final_collections:
                count = db[coll_name].count_documents({})
                indexes = list(db[coll_name].list_indexes())
                print(f"  {coll_name}: {count} documents, {len(indexes)} indexes")
            else:
                print(f"  ⚠ {coll_name}: Not found (will be created on first insert)")
        print()
        
        # Step 5: Health check
        print("Step 5: Running health check...")
        health = check_mongo_health()
        if health["status"] == "healthy":
            print("  MongoDB is healthy")
            print(f"  Version: {health.get('mongodb_version')}")
            print(f"  Database: {health.get('database')}")
            print(f"  Collections: {health.get('collections')}")
        else:
            print(f"  ✗ MongoDB health check failed: {health.get('error')}")
        print()
        
        # Summary
        print("=" * 60)
        print("MongoDB initialization complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Start MongoDB server: mongod --dbpath ./mongodb_data")
        print("  2. Update .env with MongoDB connection settings")
        print("  3. Start Django server: python manage.py runserver")
        print()
        
    except Exception as e:
        logger.exception("MongoDB initialization failed")
        print()
        print("=" * 60)
        print("✗ MongoDB initialization failed!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure MongoDB server is running")
        print("  2. Check MONGO_URI in .env file")
        print("  3. Verify network connectivity to MongoDB")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
