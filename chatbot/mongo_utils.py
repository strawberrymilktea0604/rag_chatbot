"""
MongoDB utility script for common operations.
Provides helper commands for managing MongoDB data.

Usage:
    python mongo_utils.py <command> [options]

Commands:
    stats           - Show database statistics
    health          - Check MongoDB connection health
    cleanup         - Clean up old query logs (90+ days)
    list-docs       - List all documents
    list-queries    - List recent queries
    export-logs     - Export query logs to CSV
    reset           - Reset all collections (WARNING: deletes all data)
"""

import os
import sys
import django
import argparse
import csv
from datetime import datetime, timedelta

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.mongo_repository import MongoRepository
from backend.utils.mongo_client import check_mongo_health, get_mongo_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_stats():
    """Display database statistics."""
    repo = MongoRepository()
    stats = repo.get_database_stats()
    
    print("\n" + "=" * 60)
    print("MongoDB Database Statistics")
    print("=" * 60)
    for collection, count in stats.items():
        print(f"  {collection:20s}: {count:>10,} documents")
    print("=" * 60 + "\n")


def show_health():
    """Check MongoDB connection health."""
    health = check_mongo_health()
    
    print("\n" + "=" * 60)
    print("MongoDB Health Check")
    print("=" * 60)
    print(f"  Status: {health['status']}")
    if health['status'] == 'healthy':
        print(f"  Version: {health.get('mongodb_version')}")
        print(f"  Database: {health.get('database')}")
        print("\n  Collections:")
        for coll, count in health.get('collections', {}).items():
            print(f"    {coll:20s}: {count:>10,} documents")
    else:
        print(f"  Error: {health.get('error')}")
    print("=" * 60 + "\n")


def cleanup_old_logs(days=90):
    """Clean up old query logs."""
    repo = MongoRepository()
    deleted = repo.cleanup_old_logs(days=days)
    print(f"\n✓ Cleaned up {deleted} query logs older than {days} days\n")


def list_documents(limit=20):
    """List recent documents."""
    repo = MongoRepository()
    docs = repo.list_documents(limit=limit)
    
    print("\n" + "=" * 80)
    print(f"Recent Documents (showing {len(docs)} of {repo.count_documents()})")
    print("=" * 80)
    for doc in docs:
        print(f"\nDocument ID: {doc['_id']}")
        print(f"  Filename: {doc.get('filename')}")
        print(f"  Size: {doc.get('size_bytes', 0):,} bytes")
        print(f"  Status: {doc.get('status')}")
        print(f"  Chunks: {doc.get('num_chunks', 0)}")
        print(f"  Uploaded: {doc.get('uploaded_at')}")
    print("=" * 80 + "\n")


def list_queries(days=7, limit=20):
    """List recent queries."""
    repo = MongoRepository()
    queries = repo.get_query_logs(days=days, limit=limit)
    
    print("\n" + "=" * 80)
    print(f"Recent Queries (last {days} days, showing {len(queries)})")
    print("=" * 80)
    for q in queries:
        print(f"\nQuestion ID: {q['_id']}")
        print(f"  Question: {q.get('question', '')[:100]}...")
        print(f"  Answer: {q.get('final_answer', 'N/A')[:100]}...")
        print(f"  Retrieved: {len(q.get('retrieved_chunk_ids', []))} chunks")
        print(f"  Latency: {q.get('latency_ms', 'N/A')} ms")
        print(f"  Timestamp: {q.get('ts')}")
    print("=" * 80 + "\n")


def export_query_logs(output_file="query_logs_export.csv", days=30):
    """Export query logs to CSV."""
    repo = MongoRepository()
    queries = repo.get_query_logs(days=days, limit=10000)
    
    if not queries:
        print("\n⚠ No query logs found in the last {} days\n".format(days))
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'question_id', 'question', 'final_answer', 'latency_ms',
            'num_retrieved', 'document_id', 'timestamp'
        ])
        writer.writeheader()
        
        for q in queries:
            writer.writerow({
                'question_id': q['_id'],
                'question': q.get('question', ''),
                'final_answer': q.get('final_answer', ''),
                'latency_ms': q.get('latency_ms'),
                'num_retrieved': len(q.get('retrieved_chunk_ids', [])),
                'document_id': q.get('document_id', ''),
                'timestamp': q.get('ts')
            })
    
    print(f"\n✓ Exported {len(queries)} query logs to {output_file}\n")


def reset_collections():
    """Reset all collections (WARNING: deletes all data)."""
    confirm = input("\n⚠️  WARNING: This will delete ALL data from MongoDB!\nType 'DELETE ALL' to confirm: ")
    if confirm != "DELETE ALL":
        print("\n✗ Operation cancelled\n")
        return
    
    db = get_mongo_db()
    collections = ['documents', 'chunks', 'query_logs', 'retrieval_runs', 'generation_runs', 'eval_runs']
    
    for coll in collections:
        result = db[coll].delete_many({})
        print(f"  Deleted {result.deleted_count} documents from {coll}")
    
    print("\n✓ All collections reset\n")


def main():
    parser = argparse.ArgumentParser(description='MongoDB utility commands')
    parser.add_argument('command', choices=[
        'stats', 'health', 'cleanup', 'list-docs', 'list-queries', 'export-logs', 'reset'
    ], help='Command to execute')
    parser.add_argument('--days', type=int, default=90, help='Number of days (for cleanup, list-queries, export-logs)')
    parser.add_argument('--limit', type=int, default=20, help='Number of items to show (for list commands)')
    parser.add_argument('--output', type=str, default='query_logs_export.csv', help='Output file for export-logs')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'stats':
            show_stats()
        elif args.command == 'health':
            show_health()
        elif args.command == 'cleanup':
            cleanup_old_logs(days=args.days)
        elif args.command == 'list-docs':
            list_documents(limit=args.limit)
        elif args.command == 'list-queries':
            list_queries(days=args.days, limit=args.limit)
        elif args.command == 'export-logs':
            export_query_logs(output_file=args.output, days=args.days)
        elif args.command == 'reset':
            reset_collections()
    except Exception as e:
        logger.exception(f"Command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(0)
    main()
