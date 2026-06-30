import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Specify the collection name (can be overridden via env)
collection_name = os.getenv("QDRANT_COLLECTION", "test_collection")
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
max_unique = int(os.getenv("MAX_UNIQUE_DOCS", "10"))

try:
    # Initialize Qdrant client in server mode
    client = QdrantClient(url=qdrant_url)
    print(f"Connected to Qdrant server at {qdrant_url}.")

    # Check if the collection exists
    collections = client.get_collections()
    collection_exists = any(collection.name == collection_name for collection in collections.collections)

    if not collection_exists:
        print(f"Collection '{collection_name}' does not exist.")
    else:
        print(f"Collection '{collection_name}' already exists.")

    # List all collections
    print("Available collections:")
    for collection in collections.collections:
        print(f"- {collection.name}")

    # Fetch points from the collection (increase limit if needed)
    resp = client.scroll(
        collection_name=collection_name,
        with_payload=True,  # Include payload in the response
        limit=1000          # Try to fetch up to 1000 points in one call
    )

    # resp may be a tuple (points, next_page) depending on client version
    points = resp[0] if isinstance(resp, (list, tuple)) and len(resp) > 0 and not hasattr(resp[0], "id") else resp
    if isinstance(resp, tuple) and len(resp) > 0 and hasattr(resp[0], "id"):
        points = resp[0]

    # Collect and print one point per distinct document_id
    seen_docs = set()
    printed = 0

    if not points:
        print("No points found in the collection.")
    else:
        for point in points:
            # point.payload is expected to be a dict
            payload = getattr(point, "payload", None) or {}
            # Try common payload keys for document id
            doc_id = None
            if isinstance(payload, dict):
                doc_id = payload.get("document_id") or payload.get("doc_id") or payload.get("file_id")
            # Fallback to point.id if no doc_id present (will still be unique)
            if doc_id is None:
                doc_id = f"__point_{point.id}"

            if doc_id in seen_docs:
                continue

            seen_docs.add(doc_id)
            printed += 1

            print("-" * 50)
            print(f"Document ID: {doc_id}")
            print(f"Point ID: {point.id}")
            print(f"Payload keys: {list(payload.keys()) if isinstance(payload, dict) else payload}")
            # If you want to see the full payload uncomment the next line
            # print(f"Full payload: {payload}")
            print("-" * 50)

            if printed >= max_unique:
                break

        if printed == 0:
            print("No points with recognizable document identifiers found.")
        else:
            print(f"Displayed {printed} unique point(s) from different document_id(s).")

except Exception as e:
    print(f"Error: {str(e)}")