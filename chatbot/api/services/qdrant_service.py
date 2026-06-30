import os
from backend.utils.qdrant_client import get_qdrant_client
from django.conf import settings
from qdrant_client.http.models import VectorParams, Distance

def initialize_qdrant_collection():
    client = get_qdrant_client()
    # Get vector dimension from env (768 for dangvantuan/vietnamese-embedding)
    vector_size = int(os.getenv("EMBEDDING_DIM", "768"))
    try:
        client.get_collection(settings.QDRANT_COLLECTION)
        return {"message": f"Collection '{settings.QDRANT_COLLECTION}' already exists."}
    except Exception:
        client.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={
                "default": VectorParams(size=vector_size, distance=Distance.COSINE)
            }
        )
        return {"message": f"Collection '{settings.QDRANT_COLLECTION}' created successfully with {vector_size}-dim vectors."}

def delete_qdrant_collection():
    client = get_qdrant_client()
    try:
        client.delete_collection(settings.QDRANT_COLLECTION)
        return {"message": "Collection deleted."}
    except Exception as e:
        return {"error": str(e)}