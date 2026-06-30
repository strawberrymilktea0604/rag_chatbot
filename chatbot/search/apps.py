from django.apps import AppConfig
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from django.conf import settings
from backend.utils.qdrant_client import get_qdrant_client

class SearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'search'

    def ready(self):
        # Initialize Qdrant Embedded using the singleton
        client = get_qdrant_client()
        # Avoid initializing QdrantClient here to prevent concurrent access issues
        print("Search app is ready.")
        # Create a collection if it doesn't exist
        try:
            client.get_collection(settings.QDRANT_COLLECTION)
            print(f"Qdrant collection '{settings.QDRANT_COLLECTION}' already exists.")
        except Exception as e:
            print(f"Collection '{settings.QDRANT_COLLECTION}' not found. Creating it...")
            client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "default": VectorParams(
                        size=384,  # Adjust based on your embedding model
                        distance=Distance.COSINE  # Use "COSINE", "EUCLID", or "DOT" based on your use case
                    )
                }
            )
            print(f"Qdrant collection '{settings.QDRANT_COLLECTION}' created successfully.")