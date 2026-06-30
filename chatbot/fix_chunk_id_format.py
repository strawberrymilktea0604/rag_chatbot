import os
import django
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

def normalize_chunk_id(chunk_id):
    """Convert any chunk ID format to Qdrant format: document_id:number"""
    if '::chunk_' in chunk_id:
        # Convert document_id::chunk_26 to document_id:26
        doc_id, chunk_num = chunk_id.split('::chunk_')
        return f"{doc_id}:{chunk_num}"
    return chunk_id

# Fix qa_gold.csv
gold_path = 'backend/media/qa_gold.csv'
if os.path.exists(gold_path):
    print("Fixing qa_gold.csv...")
    df = pd.read_csv(gold_path)
    
    # Normalize gold_support_chunk_ids
    df['gold_support_chunk_ids'] = df['gold_support_chunk_ids'].apply(
        lambda x: '|'.join([normalize_chunk_id(cid) for cid in x.split('|')])
    )
    
    df.to_csv(gold_path, index=False)
    print(f"  Updated {len(df)} questions")

# Fix retrieval_run.csv
ret_path = 'backend/media/retrieval_run.csv'
if os.path.exists(ret_path):
    print("\nFixing retrieval_run.csv...")
    df = pd.read_csv(ret_path)
    
    # Normalize retrieved_chunk_ids
    df['retrieved_chunk_ids'] = df['retrieved_chunk_ids'].apply(
        lambda x: '|'.join([normalize_chunk_id(cid) for cid in x.split('|')])
    )
    
    df.to_csv(ret_path, index=False)
    print(f"  Updated {len(df)} queries")

print("\n✓ Chunk ID format normalized to Qdrant standard (document_id:number)")
print("  Run debug_reranking.py again to verify")