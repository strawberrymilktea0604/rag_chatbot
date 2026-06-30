from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
print(client.scroll(collection_name="test_collection", limit=5, with_payload=True)[0])