#!/usr/bin/env python3
"""
Update eval_run.py to load gold standard from MongoDB instead of CSV.
Run: python update_eval_mongodb.py
"""

import re
from pathlib import Path

def update_eval_run():
    """Update eval_run.py with MongoDB support."""
    
    eval_run_path = Path(__file__).parent / "eval_run.py"
    
    print(f"Updating {eval_run_path}...")
    
    with open(eval_run_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Step 1: Add imports
    print("✓ Adding imports...")
    import_replacement = '''import time

from datetime import datetime
import argparse'''
    
    content = content.replace('import time', import_replacement, 1)
    
    # Step 2: Add new functions after load_gold_standard_mapping()
    print("✓ Adding MongoDB functions...")
    
    new_functions = '''

def _create_gold_mappings(gold_df):
    """Helper to create mappings from a gold standard DataFrame."""
    question_to_id = dict(zip(gold_df['question_text'], gold_df['question_id']))
    doc_questions = {}
    question_to_doc = {}
    question_norm_map = {}
    
    for _, row in gold_df.iterrows():
        doc_id = row['document_id']
        qtext = row['question_text']
        qtext_norm = qtext.strip().lower()
        question_to_doc[qtext] = doc_id
        question_norm_map[qtext_norm] = qtext
        
        if doc_id not in doc_questions:
            doc_questions[doc_id] = []
        doc_questions[doc_id].append(qtext)
    
    return question_to_id, doc_questions, question_to_doc, question_norm_map


def load_gold_standard_from_mongo():
    """
    Load gold standard Q&A data from MongoDB 'qa_gold' collection.
    Falls back to CSV if collection is empty.
    
    Returns:
        Tuple of (gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc)
    """
    try:
        # Try to load from MongoDB
        qa_gold_collection = mongo_repo.db['qa_gold']
        qa_gold_docs = list(qa_gold_collection.find({}))
        
        if not qa_gold_docs:
            logger.info("MongoDB qa_gold collection is empty, falling back to CSV")
            return load_gold_standard_mapping()
        
        # Convert MongoDB documents to DataFrame
        records = []
        for doc in qa_gold_docs:
            record = {
                'question_id': doc.get('_id') or doc.get('question_id'),
                'document_id': doc.get('document_id'),
                'question_text': doc.get('question_text'),
                'gold_answer': doc.get('gold_answer'),
                'gold_support_chunk_ids': doc.get('gold_support_chunk_ids', ''),
                'page_hint': doc.get('page_hint', '')
            }
            records.append(record)
        
        gold_df = pd.DataFrame(records)
        
        # Create mappings
        question_to_id, doc_questions, question_to_doc, question_norm_map = _create_gold_mappings(gold_df)
        
        logger.info(f"Loaded {len(question_to_id)} gold standard Q&A pairs from MongoDB")
        
        return gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc
        
    except Exception as e:
        logger.error(f"Failed to load from MongoDB: {e}")
        logger.info("Falling back to CSV loading")
        return load_gold_standard_mapping()


def sync_gold_standard_to_mongo():
    """
    Sync gold standard Q&A data from CSV to MongoDB 'qa_gold' collection.
    Uses question_id as document _id.
    
    Returns:
        Number of documents synced
    """
    try:
        # Read CSV
        gold_df = pd.read_csv(qa_gold_path)
        
        if gold_df.empty:
            logger.warning("CSV file is empty, nothing to sync")
            return 0
        
        qa_gold_collection = mongo_repo.db['qa_gold']
        
        # Clear existing collection
        delete_result = qa_gold_collection.delete_many({})
        logger.info(f"Cleared {delete_result.deleted_count} existing documents from MongoDB")
        
        # Prepare documents for insertion
        documents = []
        for _, row in gold_df.iterrows():
            doc = {
                '_id': row['question_id'],
                'question_id': row['question_id'],
                'document_id': row['document_id'],
                'question_text': row['question_text'],
                'gold_answer': row['gold_answer'],
                'gold_support_chunk_ids': row.get('gold_support_chunk_ids', ''),
                'page_hint': row.get('page_hint', ''),
                'synced_at': datetime.utcnow()
            }
            documents.append(doc)
        
        # Insert all documents
        if documents:
            result = qa_gold_collection.insert_many(documents)
            logger.info(f"Synced {len(result.inserted_ids)} Q&A records to MongoDB")
            return len(result.inserted_ids)
        
        return 0
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {qa_gold_path}")
        return 0
    except Exception as e:
        logger.error(f"Failed to sync gold standard to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return 0
'''
    
    # Find insertion point - after load_gold_standard_mapping function
    # Look for the end of load_gold_standard_mapping
    pattern = r'(def load_gold_standard_mapping\(\):.*?return None, \{\}, \{\}, \{\}, \{\})\n\ndef get_chunk_metadata_map'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        insertion_point = match.end(1)
        content = content[:insertion_point] + new_functions + '\n' + content[insertion_point:]
        print("✓ Inserted new MongoDB functions")
    else:
        print("⚠ Warning: Could not find exact insertion point, trying alternative...")
        # Fallback: insert before get_chunk_metadata_map
        if 'def get_chunk_metadata_map' in content:
            pos = content.find('def get_chunk_metadata_map')
            content = content[:pos] + new_functions + '\n\n' + content[pos:]
            print("✓ Inserted new MongoDB functions (alternative location)")
    
    # Step 3: Update main execution section
    print("✓ Updating main execution section...")
    
    old_main_section = '''# Main execution - evaluate ALL logs in database
print("=" * 80)
print("RAG CHATBOT EVALUATION - ALL DATABASE LOGS")
print("=" * 80)
print()

# Load gold standard
print("=" * 80)
print("Loading gold standard Q&A pairs...")
print("=" * 80)
gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc = load_gold_standard_mapping()'''

    new_main_section = '''# Main execution - evaluate ALL logs in database
print("=" * 80)
print("RAG CHATBOT EVALUATION - ALL DATABASE LOGS")
print("=" * 80)
print()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate RAG chatbot performance')
parser.add_argument('--sync-gold', action='store_true', 
                    help='Sync gold standard from CSV to MongoDB before loading')
parser.add_argument('--source', choices=['mongo', 'csv'], default='mongo',
                    help='Load gold standard from MongoDB (default) or CSV')
args = parser.parse_args()

# Sync gold standard to MongoDB if requested
if args.sync_gold:
    print("=" * 80)
    print("Syncing gold standard from CSV to MongoDB...")
    print("=" * 80)
    synced_count = sync_gold_standard_to_mongo()
    print(f"Synced {synced_count} records")
    print()

# Load gold standard
print("=" * 80)
print(f"Loading gold standard Q&A pairs (source: {args.source})...")
print("=" * 80)

if args.source == 'mongo':
    gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc = load_gold_standard_from_mongo()
else:
    gold_df, question_to_id, doc_questions, question_norm_map, question_to_doc = load_gold_standard_mapping()'''
    
    if old_main_section in content:
        content = content.replace(old_main_section, new_main_section)
        print("✓ Updated main execution section")
    else:
        print("⚠ Warning: Could not find exact main section to replace")
    
    # Write back
    with open(eval_run_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "=" * 70)
    print("SUCCESS! eval_run.py has been updated with MongoDB support")
    print("=" * 70)
    print("\nUsage:")
    print("  python eval_run.py                    # Load from MongoDB (default)")
    print("  python eval_run.py --source csv       # Load from CSV")
    print("  python eval_run.py --sync-gold        # Sync CSV to MongoDB first")
    print("\nFeatures added:")
    print("  ✓ load_gold_standard_from_mongo() - loads Q&A from MongoDB with CSV fallback")
    print("  ✓ sync_gold_standard_to_mongo() - syncs CSV data to MongoDB collection 'qa_gold'")
    print("  ✓ --sync-gold flag to control sync behavior")
    print("  ✓ --source flag to choose data source (mongo or csv)")
    print("=" * 70)

if __name__ == "__main__":
    update_eval_run()