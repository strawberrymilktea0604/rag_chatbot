"""
Migrate existing Qdrant collection to hybrid search (dense + sparse vectors).

This script:
1. Backs up the existing collection
2. Creates a new hybrid collection
3. Reads all points from the backup
4. Generates sparse vectors for each chunk
5. Upserts points with both dense and sparse vectors
6. Validates migration

Usage:
    python migrate_to_hybrid_collection.py [--dry-run] [--batch-size N]
    
Options:
    --dry-run: Preview migration without making changes
    --batch-size: Number of points to process at once (default: 100)
"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
import django
django.setup()

from django.conf import settings
from backend.utils.qdrant_client import (
    get_qdrant_client, SparseVectorBuilder, get_sparse_builder,
    create_hybrid_collection
)
from qdrant_client.http.models import (
    VectorParams, Distance, SparseVectorParams, SparseIndexParams,
    PointStruct, Filter, NamedSparseVector, SparseVector
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backup_collection(client, collection_name: str, backup_name: str) -> bool:
    """
    Create a backup of the existing collection.
    """
    try:
        # Check if backup already exists
        collections = client.get_collections()
        if any(c.name == backup_name for c in collections.collections):
            logger.warning(f"Backup collection {backup_name} already exists")
            return True
        
        # Get source collection info
        source_info = client.get_collection(collection_name)
        vectors_config = source_info.config.params.vectors
        
        # Create backup collection with same config
        if isinstance(vectors_config, dict):
            client.create_collection(
                collection_name=backup_name,
                vectors_config=vectors_config
            )
        else:
            client.create_collection(
                collection_name=backup_name,
                vectors_config=VectorParams(
                    size=vectors_config.size,
                    distance=vectors_config.distance
                )
            )
        
        # Copy all points
        offset = None
        batch_size = 100
        total_copied = 0
        
        while True:
            # Scroll through source collection
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not records:
                break
            
            # Convert to PointStruct
            points = []
            for record in records:
                points.append(PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload
                ))
            
            # Upsert to backup
            client.upsert(collection_name=backup_name, points=points)
            total_copied += len(points)
            logger.info(f"Backed up {total_copied} points...")
            
            if next_offset is None:
                break
            offset = next_offset
        
        logger.info(f"Backup complete: {total_copied} points copied to {backup_name}")
        return True
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False


def build_vocabulary_from_collection(client, collection_name: str) -> SparseVectorBuilder:
    """
    Build BM25 vocabulary from all texts in the collection.
    """
    logger.info("Building BM25 vocabulary from existing collection...")
    
    corpus = []
    offset = None
    batch_size = 500
    
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            break
        
        for record in records:
            text = record.payload.get("text", "")
            if text:
                corpus.append(text)
        
        logger.info(f"Collected {len(corpus)} texts for vocabulary...")
        
        if next_offset is None:
            break
        offset = next_offset
    
    # Build vocabulary
    builder = SparseVectorBuilder(
        k1=float(os.getenv("BM25_K1", "1.5")),
        b=float(os.getenv("BM25_B", "0.75")),
        use_underthesea=os.getenv("USE_UNDERTHESEA_TOKENIZATION", "true").lower() == "true",
        top_k_terms=int(os.getenv("SPARSE_VECTOR_TOP_K", "150")),
    )
    builder.fit(corpus)
    
    # Save vocabulary
    cache_path = os.getenv("SPARSE_VOCAB_CACHE", "media/bm25_cache.pkl")
    builder.save_vocabulary(cache_path)
    
    logger.info(f"Built vocabulary with {len(builder.vocabulary)} terms")
    return builder


def migrate_to_hybrid(
    client,
    source_name: str,
    target_name: str,
    sparse_builder: SparseVectorBuilder,
    batch_size: int = 100,
    dry_run: bool = False
) -> dict:
    """
    Migrate points from source collection to target hybrid collection.
    """
    stats = {
        "total_points": 0,
        "migrated": 0,
        "failed": 0,
        "sparse_generated": 0,
        "start_time": datetime.now(),
    }
    
    # Get source collection info
    source_info = client.get_collection(source_name)
    vectors_config = source_info.config.params.vectors
    
    # Determine vector size and field name
    if isinstance(vectors_config, dict):
        # Named vectors - find the dense vector config
        if "default" in vectors_config:
            vector_size = vectors_config["default"].size
            source_vector_name = "default"
        elif "dense" in vectors_config:
            vector_size = vectors_config["dense"].size
            source_vector_name = "dense"
        else:
            raise ValueError(f"Unknown vector config: {vectors_config.keys()}")
    else:
        vector_size = vectors_config.size
        source_vector_name = None
    
    logger.info(f"Source collection: vector_size={vector_size}, vector_name={source_vector_name}")
    
    # Create target hybrid collection if not exists
    if not dry_run:
        try:
            client.get_collection(target_name)
            logger.info(f"Target collection {target_name} already exists")
        except Exception:
            create_hybrid_collection(
                collection_name=target_name,
                dense_vector_size=vector_size,
                distance=Distance.COSINE,
                sparse_on_disk=False
            )
            logger.info(f"Created hybrid collection {target_name}")
    
    # Migrate points in batches
    offset = None
    
    while True:
        records, next_offset = client.scroll(
            collection_name=source_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        
        if not records:
            break
        
        points = []
        for record in records:
            stats["total_points"] += 1
            
            try:
                # Get dense vector
                if source_vector_name:
                    dense_vector = record.vector.get(source_vector_name)
                else:
                    dense_vector = record.vector
                
                if not dense_vector:
                    logger.warning(f"Point {record.id} has no dense vector, skipping")
                    stats["failed"] += 1
                    continue
                
                # Generate sparse vector
                text = record.payload.get("text", "")
                sparse_vector = sparse_builder.transform(text) if text else None
                
                if sparse_vector:
                    stats["sparse_generated"] += 1
                
                # Build vectors dict
                vectors = {"dense": dense_vector}
                if sparse_vector:
                    vectors["sparse"] = {
                        "indices": sparse_vector.indices,
                        "values": sparse_vector.values
                    }
                
                points.append(PointStruct(
                    id=record.id,
                    vector=vectors,
                    payload=record.payload
                ))
                stats["migrated"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process point {record.id}: {e}")
                stats["failed"] += 1
        
        # Upsert batch
        if points and not dry_run:
            try:
                client.upsert(collection_name=target_name, points=points)
                logger.info(f"Migrated {stats['migrated']}/{stats['total_points']} points (sparse: {stats['sparse_generated']})")
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
                stats["failed"] += len(points)
        elif dry_run:
            logger.info(f"[DRY RUN] Would migrate {len(points)} points")
        
        if next_offset is None:
            break
        offset = next_offset
    
    stats["end_time"] = datetime.now()
    stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()
    
    return stats


def validate_migration(client, source_name: str, target_name: str) -> dict:
    """
    Validate that migration was successful.
    """
    validation = {
        "source_count": 0,
        "target_count": 0,
        "has_sparse_vectors": False,
        "sample_check_passed": True,
    }
    
    try:
        source_info = client.get_collection(source_name)
        target_info = client.get_collection(target_name)
        
        validation["source_count"] = source_info.points_count
        validation["target_count"] = target_info.points_count
        
        # Check if target has sparse vector config
        validation["has_sparse_vectors"] = (
            hasattr(target_info.config, 'sparse_vectors_config') and
            target_info.config.sparse_vectors_config is not None
        )
        
        # Sample check - verify a few points have sparse vectors
        sample, _ = client.scroll(
            collection_name=target_name,
            limit=5,
            with_vectors=True
        )
        
        for record in sample:
            if isinstance(record.vector, dict):
                if "sparse" not in record.vector:
                    validation["sample_check_passed"] = False
                    break
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        validation["error"] = str(e)
    
    return validation


def main():
    parser = argparse.ArgumentParser(description="Migrate Qdrant collection to hybrid search")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for migration")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup step")
    parser.add_argument("--skip-vocabulary", action="store_true", help="Use existing vocabulary")
    args = parser.parse_args()
    
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION
    backup_name = f"{collection_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target_name = f"{collection_name}_hybrid"
    
    print(f"\n{'='*60}")
    print(f"HYBRID SEARCH MIGRATION")
    print(f"{'='*60}")
    print(f"Source collection: {collection_name}")
    print(f"Backup collection: {backup_name}")
    print(f"Target collection: {target_name}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}\n")
    
    # Step 1: Backup
    if not args.skip_backup and not args.dry_run:
        print("\n[1/4] Creating backup...")
        if not backup_collection(client, collection_name, backup_name):
            print("ERROR: Backup failed. Aborting migration.")
            sys.exit(1)
    else:
        print("\n[1/4] Skipping backup...")
    
    # Step 2: Build vocabulary
    if not args.skip_vocabulary:
        print("\n[2/4] Building BM25 vocabulary...")
        sparse_builder = build_vocabulary_from_collection(client, collection_name)
    else:
        print("\n[2/4] Loading existing vocabulary...")
        sparse_builder = get_sparse_builder()
    
    # Step 3: Migrate
    print("\n[3/4] Migrating to hybrid collection...")
    stats = migrate_to_hybrid(
        client,
        source_name=collection_name,
        target_name=target_name,
        sparse_builder=sparse_builder,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    
    print(f"\nMigration Statistics:")
    print(f"  Total points: {stats['total_points']}")
    print(f"  Migrated: {stats['migrated']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Sparse vectors: {stats['sparse_generated']}")
    print(f"  Duration: {stats['duration_seconds']:.1f}s")
    
    # Step 4: Validate
    if not args.dry_run:
        print("\n[4/4] Validating migration...")
        validation = validate_migration(client, collection_name, target_name)
        
        print(f"\nValidation Results:")
        print(f"  Source count: {validation['source_count']}")
        print(f"  Target count: {validation['target_count']}")
        print(f"  Has sparse vectors: {validation['has_sparse_vectors']}")
        print(f"  Sample check passed: {validation['sample_check_passed']}")
        
        if validation['target_count'] == validation['source_count'] and validation['sample_check_passed']:
            print("\n✓ Migration successful!")
            print(f"\nTo switch to hybrid collection, update QDRANT_COLLECTION in .env:")
            print(f"  QDRANT_COLLECTION={target_name}")
        else:
            print("\n✗ Migration may have issues. Please verify manually.")
    else:
        print("\n[4/4] Skipping validation (dry run)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
