import os
import sys
import django
import requests
import time

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

import pandas as pd
from backend.utils.mongo_repository import MongoRepository
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qa_gold_path = 'qa_gold.csv'

def regenerate_logs_for_document(doc_id, questions):
    """Call API to regenerate logs for all questions of a document."""
    
    api_url = "http://127.0.0.1:8000/api/ask/"
    
    logger.info(f"\nRegenerating logs for document: {doc_id}")
    logger.info(f"Questions to process: {len(questions)}")
    
    successful = 0
    failed = 0
    
    for idx, question in enumerate(questions, 1):
        try:
            response = requests.post(
                api_url,
                json={
                    "question": question,
                    "document_id": doc_id,
                    "top_k": 10
                },
                timeout=30
            )
            
            if response.status_code == 200:
                successful += 1
                if idx % 5 == 0:
                    print(f"  [{idx}/{len(questions)}] Generated", end="\r")
            else:
                logger.error(f"  Failed: {response.status_code} - {question[:50]}...")
                failed += 1
                
        except Exception as e:
            logger.error(f"  Error: {str(e)} - {question[:50]}...")
            failed += 1
        
        time.sleep(0.5)  # Rate limiting
    
    print(f"  [{len(questions)}/{len(questions)}] Complete" + " " * 30)
    return successful, failed

def main():
    print("=" * 80)
    print("REGENERATING MONGODB LOGS FROM QDRANT")
    print("=" * 80)
    print()
    
    # Check if API is running
    try:
        response = requests.get("http://127.0.0.1:8000/api/", timeout=5)
        print("✓ API is running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"✗ API is not running: {str(e)}")
        print("Start backend with: python manage.py runserver")
        sys.exit(1)
    
    # Load gold standard
    gold_df = pd.read_csv(qa_gold_path)
    
    # Get only current document
    from backend.utils.qdrant_client import QdrantClient
    qdrant_wrapper = QdrantClient()
    qdrant = qdrant_wrapper._client
    collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')
    
    points, _ = qdrant.scroll(collection_name=collection_name, limit=1, with_payload=True, with_vectors=False)
    if not points:
        print("✗ No documents in Qdrant!")
        sys.exit(1)
    
    uploaded_doc_id = points[0].payload.get('document_id')
    print(f"Regenerating logs for uploaded document: {uploaded_doc_id}")
    print()
    
    # Get questions for this document
    doc_questions = gold_df[gold_df['document_id'] == uploaded_doc_id]['question_text'].tolist()
    
    if not doc_questions:
        print(f"✗ No questions found for document {uploaded_doc_id}")
        sys.exit(1)
    
    print(f"Found {len(doc_questions)} questions to regenerate")
    print()
    
    # Clear old logs for this document
    mongo_repo = MongoRepository()
    db = mongo_repo.db
    
    old_retrieval = db['retrieval_log'].delete_many({'document_id': uploaded_doc_id})
    old_generation = db['generation_log'].delete_many({'document_id': uploaded_doc_id})
    
    logger.info(f"Cleared {old_retrieval.deleted_count} old retrieval logs")
    logger.info(f"Cleared {old_generation.deleted_count} old generation logs")
    print()
    
    # Regenerate logs
    successful, failed = regenerate_logs_for_document(uploaded_doc_id, doc_questions)
    
    print()
    print("=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    print(f"Successful: {successful}/{len(doc_questions)}")
    print(f"Failed: {failed}/{len(doc_questions)}")
    print()
    
    if successful > 0:
        print("✓ Logs regenerated successfully!")
        print()
        print("Next steps:")
        print("  1. python eval_run.py")
        print()
    else:
        print("✗ All requests failed - check API logs")
        sys.exit(1)

if __name__ == "__main__":
    main()