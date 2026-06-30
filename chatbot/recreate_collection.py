"""
Recreate Qdrant collection with hybrid search support (dense + sparse vectors).

Usage:
    python recreate_collection.py [--hybrid] [--dense-only] [--build-vocabulary]
    
Options:
    --hybrid: Create hybrid collection with dense + sparse vectors (default)
    --dense-only: Create collection with dense vectors only (legacy mode)
    --build-vocabulary: Build BM25 vocabulary from existing chunks in MongoDB
"""

import os
import sys
import argparse
import django
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
from django.conf import settings

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.qdrant_client import (
    get_qdrant_client, create_hybrid_collection, 
    get_sparse_builder, SparseVectorBuilder
)


def get_corpus_from_mongodb():
    """
    Retrieve all chunk texts from MongoDB for vocabulary building.
    """
    try:
        from backend.utils.mongo_client import get_mongo_client
        client = get_mongo_client()
        db = client[settings.MONGO_DB_NAME]
        chunks_collection = db["chunks"]
        
        # Get all chunk texts
        chunks = list(chunks_collection.find({}, {"text": 1}))
        corpus = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]
        
        print(f"Retrieved {len(corpus)} chunks from MongoDB")
        return corpus
    except Exception as e:
        print(f"Error retrieving corpus from MongoDB: {e}")
        return []


def build_vocabulary(corpus=None):
    """
    Build BM25 vocabulary from corpus and save to cache.
    """
    if corpus is None or len(corpus) == 0:
        corpus = get_corpus_from_mongodb()
    
    if not corpus:
        print("No corpus available for vocabulary building.")
        return None
    
    cache_path = os.getenv("SPARSE_VOCAB_CACHE", "media/bm25_cache.pkl")
    
    # Build vocabulary
    builder = SparseVectorBuilder(
        k1=float(os.getenv("BM25_K1", "1.5")),
        b=float(os.getenv("BM25_B", "0.75")),
        use_underthesea=os.getenv("USE_UNDERTHESEA_TOKENIZATION", "true").lower() == "true",
        top_k_terms=int(os.getenv("SPARSE_VECTOR_TOP_K", "150")),
    )
    
    builder.fit(corpus)
    builder.save_vocabulary(cache_path)
    
    print(f"Built vocabulary with {len(builder.vocabulary)} terms, saved to {cache_path}")
    return builder


def recreate_collection(hybrid=True, build_vocab=False):
    """
    Recreate the Qdrant collection.
    
    Args:
        hybrid: If True, create hybrid collection with dense + sparse vectors.
               If False, create dense-only collection (legacy mode).
        build_vocab: If True, build BM25 vocabulary from MongoDB chunks.
    """
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION
    vector_size = int(os.getenv("EMBEDDING_DIM", "768"))

    # Delete the existing collection if it exists
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection didn't exist or error deleting: {e}")

    if hybrid:
        # Create hybrid collection with both dense and sparse vectors
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False  # Keep in memory for speed
                        )
                    )
                }
            )
            print(f"Created HYBRID collection '{collection_name}' with:")
            print(f"  - Dense vectors: {vector_size}-dim (field: 'dense')")
            print(f"  - Sparse vectors: BM25/TF-IDF (field: 'sparse')")
        except Exception as e:
            print(f"Error creating hybrid collection: {e}")
            raise
    else:
        # Create dense-only collection (legacy mode)
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "default": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                }
            )
            print(f"Created DENSE-ONLY collection '{collection_name}' with {vector_size}-dim vectors")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    # Build vocabulary if requested
    if build_vocab:
        print("\nBuilding BM25 vocabulary...")
        build_vocabulary()


def main():
    parser = argparse.ArgumentParser(description="Recreate Qdrant collection")
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Create dense-only collection (legacy mode)"
    )
    parser.add_argument(
        "--build-vocabulary",
        action="store_true",
        help="Build BM25 vocabulary from MongoDB chunks after creation"
    )
    
    args = parser.parse_args()
    
    hybrid = not args.dense_only
    recreate_collection(hybrid=hybrid, build_vocab=args.build_vocabulary)


if __name__ == "__main__":
    main()