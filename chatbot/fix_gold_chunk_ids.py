"""
Fix gold standard chunk IDs to match Qdrant chunk_id format.

Gold standard uses: document_id:order_index (e.g., 1e8e6c622bfb571b0b783d8347182318:105)
Qdrant uses: document_id::chunk_order_index (e.g., 1e8e6c622bfb571b0b783d8347182318::chunk_105)
"""
import os
import sys
import re
import pandas as pd
from pathlib import Path

def convert_chunk_id(old_id: str) -> str:
    """
    Convert gold standard chunk ID format to Qdrant format.
    
    Input:  document_id:order_index (e.g., 1e8e6c622bfb571b0b783d8347182318:105)
    Output: document_id::chunk_order_index (e.g., 1e8e6c622bfb571b0b783d8347182318::chunk_105)
    """
    if not old_id or not isinstance(old_id, str):
        return old_id
    
    old_id = old_id.strip()
    
    # Already in new format
    if "::chunk_" in old_id:
        return old_id
    
    # Match pattern: document_id:number
    match = re.match(r'^([a-f0-9]{32}):(\d+)$', old_id)
    if match:
        doc_id = match.group(1)
        order_index = match.group(2)
        return f"{doc_id}::chunk_{order_index}"
    
    # Return unchanged if doesn't match expected pattern
    return old_id


def convert_chunk_ids_field(field_value: str) -> str:
    """Convert pipe-separated chunk IDs."""
    if not field_value or not isinstance(field_value, str) or pd.isna(field_value):
        return field_value
    
    chunk_ids = field_value.split('|')
    converted = [convert_chunk_id(cid.strip()) for cid in chunk_ids]
    return '|'.join(converted)


def fix_gold_standard(input_path: str, output_path: str = None):
    """Fix chunk ID format in gold standard CSV."""
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    if output_path is None:
        # Create backup and overwrite original
        backup_path = input_path.replace('.csv', '_backup.csv')
        output_path = input_path
    else:
        backup_path = None
    
    # Read the gold standard
    df = pd.read_csv(input_path)
    
    print(f"Processing {len(df)} rows from {input_path}")
    print()
    
    # Show sample before conversion
    print("Sample BEFORE conversion:")
    for idx, row in df.head(3).iterrows():
        print(f"  {row['question_id']}: {row['gold_support_chunk_ids'][:80]}...")
    print()
    
    # Convert the gold_support_chunk_ids column
    if 'gold_support_chunk_ids' in df.columns:
        df['gold_support_chunk_ids'] = df['gold_support_chunk_ids'].apply(convert_chunk_ids_field)
    else:
        print("Warning: 'gold_support_chunk_ids' column not found")
        return False
    
    # Show sample after conversion
    print("Sample AFTER conversion:")
    for idx, row in df.head(3).iterrows():
        print(f"  {row['question_id']}: {row['gold_support_chunk_ids'][:80]}...")
    print()
    
    # Create backup if overwriting
    if backup_path:
        import shutil
        shutil.copy2(input_path, backup_path)
        print(f"Backup created: {backup_path}")
    
    # Save the fixed file
    df.to_csv(output_path, index=False)
    print(f"Saved fixed gold standard to: {output_path}")
    
    return True


if __name__ == "__main__":
    # Default paths
    gold_path = 'backend/media/qa_gold.csv'
    
    # Also check root-level qa_gold.csv
    if not os.path.exists(gold_path):
        gold_path = 'qa_gold.csv'
    
    if len(sys.argv) > 1:
        gold_path = sys.argv[1]
    
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("Fixing Gold Standard Chunk ID Format")
    print("=" * 60)
    print()
    
    success = fix_gold_standard(gold_path, output_path)
    
    if success:
        print()
        print("Done! You can now re-run eval_run.py")
    else:
        print()
        print("Failed to fix gold standard")
        sys.exit(1)