"""
Reindex all documents with new multilingual embeddings.
This script:
1. Recreates the Qdrant collection
2. Re-uploads all documents from media/uploads/ with new embeddings
3. Verifies the collection after completion
"""
import os
import sys
import django
import glob
from pathlib import Path

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from django.conf import settings
from backend.utils.qdrant_client import get_qdrant_client
from qdrant_client.models import VectorParams, Distance
from documents.models import Document
from backend.utils.mongo_repository import MongoRepository
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recreate_qdrant_collection():
    """Delete and recreate the Qdrant collection."""
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION
    
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"✓ Deleted existing collection: {collection_name}")
    except Exception as e:
        logger.info(f"Collection did not exist or could not be deleted: {e}")
    
    # Get vector dimension from env (768 for dangvantuan/vietnamese-embedding)
    vector_size = int(os.getenv("EMBEDDING_DIM", "768"))
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "default": VectorParams(
                    size=vector_size,  # 768 for dangvantuan/vietnamese-embedding
                    distance=Distance.COSINE
                )
            }
        )
        logger.info(f"✓ Created new collection '{collection_name}' with {vector_size}-dim vectors")
    except Exception as e:
        logger.error(f"✗ Error creating collection: {e}")
        sys.exit(1)

def clear_mongodb_chunks():
    """Clear all chunks from MongoDB to prepare for reindexing."""
    mongo_repo = MongoRepository()
    try:
        result = mongo_repo.db['chunks'].delete_many({})
        logger.info(f"✓ Deleted {result.deleted_count} chunks from MongoDB")
    except Exception as e:
        logger.error(f"✗ Error clearing MongoDB chunks: {e}")

def clear_django_documents():
    """Clear all document records from Django DB."""
    try:
        count = Document.objects.count()
        Document.objects.all().delete()
        logger.info(f"✓ Deleted {count} documents from Django DB")
    except Exception as e:
        logger.error(f"✗ Error clearing Django documents: {e}")

def get_documents_to_reindex():
    """Find all documents in media/uploads directory."""
    uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    if not os.path.exists(uploads_dir):
        logger.error(f"✗ Uploads directory not found: {uploads_dir}")
        return []
    
    files = []
    for ext in ['*.pdf', '*.docx', '*.txt']:
        files.extend(glob.glob(os.path.join(uploads_dir, ext)))
    
    logger.info(f"Found {len(files)} documents to reindex")
    return files

def reindex_document(file_path):
    """Reindex a single document using the file upload API logic."""
    from django.core.files.uploadedfile import SimpleUploadedFile
    from api.views.file_upload_view import FileUploadView
    from rest_framework.test import APIRequestFactory
    
    factory = APIRequestFactory()
    
    # Read the file
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    # Determine content type
    file_name = os.path.basename(file_path)
    if file_name.endswith('.pdf'):
        content_type = 'application/pdf'
    elif file_name.endswith('.docx'):
        content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif file_name.endswith('.txt'):
        content_type = 'text/plain'
    else:
        logger.warning(f"✗ Unsupported file type: {file_name}")
        return False
    
    # Create upload request with force=true to allow reprocessing
    uploaded_file = SimpleUploadedFile(
        file_name,
        file_content,
        content_type=content_type
    )
    
    request = factory.post('/api/upload/', {'file': uploaded_file}, format='multipart')
    request.FILES['file'] = uploaded_file
    request.query_params = {'force': 'true'}  # Force reprocessing
    
    view = FileUploadView()
    
    try:
        response = view.post(request)
        if response.status_code in [200, 201]:
            logger.info(f"✓ Successfully reindexed: {file_name}")
            return True
        else:
            logger.error(f"✗ Failed to reindex {file_name}: {response.data}")
            return False
    except Exception as e:
        logger.error(f"✗ Error reindexing {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_collection():
    """Verify the collection has been properly created."""
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION
    vector_size = int(os.getenv("EMBEDDING_DIM", "768"))
    
    try:
        info = client.get_collection(collection_name)
        points_count = info.points_count
        logger.info("✓ Collection verification:")
        logger.info(f"  - Collection name: {collection_name}")
        logger.info(f"  - Points count: {points_count}")
        logger.info(f"  - Vector size: {vector_size} (vietnamese-embedding)")
        return points_count > 0
    except Exception as e:
        logger.error(f"✗ Collection verification failed: {e}")
        return False

def main():
    """Main reindexing workflow."""
    print("=" * 80)
    print("REINDEXING WITH VIETNAMESE EMBEDDINGS (768-dim)")
    print("=" * 80)
    print()
    
    embedding_model = os.getenv("EMBEDDING_MODEL", "dangvantuan/vietnamese-embedding")
    embedding_dim = os.getenv("EMBEDDING_DIM", "768")
    print(f"Using embedding model: {embedding_model}")
    print(f"Vector dimensions: {embedding_dim}")
    print()
    
    # Step 1: Recreate Qdrant collection
    print("Step 1: Recreating Qdrant collection...")
    recreate_qdrant_collection()
    print()
    
    # Step 2: Clear MongoDB chunks
    print("Step 2: Clearing MongoDB chunks...")
    clear_mongodb_chunks()
    print()
    
    # Step 3: Clear Django documents
    print("Step 3: Clearing Django document records...")
    clear_django_documents()
    print()
    
    # Step 4: Find documents to reindex
    print("Step 4: Finding documents to reindex...")
    files = get_documents_to_reindex()
    if not files:
        print("✗ No documents found to reindex!")
        sys.exit(1)
    print()
    
    # Step 5: Reindex each document
    print(f"Step 5: Reindexing {len(files)} documents...")
    print("-" * 80)
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {os.path.basename(file_path)}")
        if reindex_document(file_path):
            success_count += 1
        else:
            fail_count += 1
        print()
    
    print("-" * 80)
    print(f"Reindexing complete: {success_count} successful, {fail_count} failed")
    print()
    
    # Step 6: Verify collection
    print("Step 6: Verifying collection...")
    if verify_collection():
        print("✓ Collection verification passed!")
    else:
        print("✗ Collection verification failed!")
    print()
    
    print("=" * 80)
    print("REINDEXING COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Test queries with: curl -X POST http://localhost:8000/api/ask/ -H 'Content-Type: application/json' -d '{\"question\": \"your Vietnamese question\"}'")
    print("2. Run evaluation: python eval_run.py")
    print("3. Check improvements in Recall, MRR, and NDCG metrics")

if __name__ == "__main__":
    main()
