import os
import re
import shutil
from datetime import datetime
import django
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.qdrant_client import QdrantClient
from backend.utils.embeddings import HuggingfaceEmbeddingsModel
from sentence_transformers import CrossEncoder
from qdrant_client import models

# Page boost factor for chunks on cited pages
PAGE_BOOST_FACTOR = float(os.getenv('PAGE_BOOST_FACTOR', '1.5'))

# Cache for chunk content to avoid repeated queries
_chunk_cache = {}


# ============================================================================
# FORMAT CONVERSION UTILITIES
# ============================================================================

def convert_gold_to_qdrant_format(chunk_id: str) -> str:
    """
    Convert gold standard chunk ID to Qdrant format.
    
    Both formats now use doc_id:N (single-colon), so this is mostly an identity function.
    Maintains backwards compatibility for legacy ::chunk_ format data.
    
    Examples:
        '1e8e6c622bfb571b0b783d8347182318:105' -> '1e8e6c622bfb571b0b783d8347182318:105' (no change)
        '1e8e6c622bfb571b0b783d8347182318::chunk_105' -> '1e8e6c622bfb571b0b783d8347182318::chunk_105' (legacy, unchanged)
    """
    # Both gold and Qdrant now use the same format (doc_id:N)
    # No conversion needed - return as-is
    return chunk_id


def convert_qdrant_to_gold_format(chunk_id: str) -> str:
    """
    Convert Qdrant chunk ID to gold standard format.
    
    Both formats now use doc_id:N (single-colon), so this is mostly an identity function.
    Maintains backwards compatibility for legacy ::chunk_ format data.
    
    Examples:
        '1e8e6c622bfb571b0b783d8347182318:105' -> '1e8e6c622bfb571b0b783d8347182318:105' (no change)
        '1e8e6c622bfb571b0b783d8347182318::chunk_105' -> '1e8e6c622bfb571b0b783d8347182318:105' (legacy conversion)
    """
    # Handle legacy ::chunk_ format for backwards compatibility
    if '::chunk_' in chunk_id:
        parts = chunk_id.rsplit('::chunk_', 1)
        if len(parts) == 2:
            doc_id, chunk_num = parts
            return f"{doc_id}:{chunk_num}"
    
    # Already in correct format (doc_id:N) - return as-is
    return chunk_id


def extract_document_id(chunk_id: str) -> str:
    """Extract document_id from chunk_id (works with both formats)."""
    if '::' in chunk_id:
        return chunk_id.split('::')[0]
    elif ':' in chunk_id:
        return chunk_id.rsplit(':', 1)[0]
    return chunk_id


def extract_chunk_number(chunk_id: str) -> str:
    """Extract chunk number from chunk_id (works with both formats)."""
    if '::' in chunk_id:
        parts = chunk_id.rsplit('::', 1)
        if len(parts) == 2:
            return parts[1].replace('chunk_', '')
    elif ':' in chunk_id:
        parts = chunk_id.rsplit(':', 1)
        if len(parts) == 2:
            return parts[1]
    return ''


# ============================================================================
# CITATION AND PAGE HINT UTILITIES
# ============================================================================

def extract_citations_from_answer(gold_answer: str) -> list:
    """
    Extract page numbers from [cite: N] patterns in gold answer.
    
    Examples:
        "[cite: 1]" -> ['1']
        "[cite: 3, 4]" -> ['3', '4']
        "[cite: 56] ... [cite: 56, 57]" -> ['56', '57']
    """
    if not gold_answer or pd.isna(gold_answer):
        return []
    
    pattern = r'\[cite:\s*([0-9,\s]+)\]'
    matches = re.findall(pattern, gold_answer)
    
    pages = set()
    for match in matches:
        for num in match.split(','):
            num = num.strip()
            if num.isdigit():
                pages.add(num)
    
    return sorted(pages, key=lambda x: int(x))


def parse_page_hint(page_hint: str) -> list:
    """
    Parse pipe-separated page hint string into list of page numbers.
    
    Examples:
        '1' -> ['1']
        '3|3' -> ['3']
        '56|56|57' -> ['56', '57']
    """
    if not page_hint or pd.isna(page_hint):
        return []
    
    pages = set()
    for p in str(page_hint).split('|'):
        p = p.strip()
        if p.isdigit():
            pages.add(p)
    
    return sorted(pages, key=lambda x: int(x))


def format_page_hint(pages: list) -> str:
    """Format list of pages back to pipe-separated string."""
    return '|'.join(pages) if pages else ''


# ============================================================================
# QDRANT QUERY FUNCTIONS
# ============================================================================

def get_chunk_content(qdrant_client, collection_name, chunk_id: str) -> tuple:
    """
    Retrieve chunk content from Qdrant.
    
    Args:
        chunk_id: Chunk ID in gold format (single-colon: doc_id:chunk_num)
    
    Returns:
        tuple: (content, found_flag, page_number, qdrant_chunk_id)
    """
    global _chunk_cache
    
    cache_key = f"{collection_name}:{chunk_id}"
    if cache_key in _chunk_cache:
        return _chunk_cache[cache_key]
    
    # Convert to Qdrant format for query
    qdrant_chunk_id = convert_gold_to_qdrant_format(chunk_id)
    
    search_filter = models.Filter(
        must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=qdrant_chunk_id))]
    )
    
    try:
        results = qdrant_client._client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=1,
            with_payload=True
        )
        
        if results[0]:
            payload = results[0][0].payload
            content = payload.get('text', payload.get('content', 'N/A'))
            page = payload.get('page', payload.get('start_page', 'N/A'))
            result = (content, True, str(page), qdrant_chunk_id)
            _chunk_cache[cache_key] = result
            return result
    except Exception as e:
        print(f"  ⚠ Error querying chunk {chunk_id}: {e}")
    
    result = ('NOT FOUND', False, 'N/A', qdrant_chunk_id)
    _chunk_cache[cache_key] = result
    return result


def verify_chunk_exists(qdrant_client, collection_name, chunk_id: str) -> bool:
    """Check if a chunk exists in Qdrant."""
    _, found, _, _ = get_chunk_content(qdrant_client, collection_name, chunk_id)
    return found


def _get_dense_vector_name(qdrant_client, collection_name: str) -> str:
    """
    Determine which dense vector name to use for this collection.

    - Hybrid collections: typically "dense" (+ "sparse")
    - Legacy dense-only collections: typically "default"
    """
    try:
        info = qdrant_client._client.get_collection(collection_name=collection_name)
        vectors_cfg = info.config.params.vectors

        # Newer Qdrant: dict of named vectors
        if isinstance(vectors_cfg, dict):
            if "dense" in vectors_cfg:
                return "dense"
            if "default" in vectors_cfg:
                return "default"
            # Fallback: pick any available vector name
            for k in vectors_cfg.keys():
                return str(k)

        # Older Qdrant: single unnamed vector => treated as "default"
        return "default"
    except Exception:
        # Safe fallback for older setups / transient failures
        return "default"


def find_best_chunks(question: str, document_id: str, qdrant_client, embeddings_model, 
                     reranker, collection_name: str, page_hints: list = None, top_k: int = 10) -> list:
    """
    Find most relevant chunks using retrieval + reranking with page-aware scoring.
    
    Args:
        question: The question text
        document_id: Document ID to filter by
        page_hints: List of cited page numbers for score boosting
        top_k: Number of results to return
    
    Returns:
        List of tuples: (result, base_score, boosted_score, is_on_page, gold_format_id)
    """
    query_embedding = embeddings_model.embed_texts([question])[0]

    search_filter = models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )

    dense_vector_name = _get_dense_vector_name(qdrant_client, collection_name)

    results = qdrant_client._client.search(
        collection_name=collection_name,
        query_vector=(dense_vector_name, query_embedding),
        limit=100,
        score_threshold=0.25,
        query_filter=search_filter
    )

    if not results:
        return []

    # Rerank
    pairs = [(question, r.payload.get('text', r.payload.get('content', ''))) for r in results]
    rerank_scores = reranker.predict(pairs)
    
    # Apply page boost
    page_hints_set = set(page_hints) if page_hints else set()
    reranked = []
    
    for i, result in enumerate(results):
        base_score = float(rerank_scores[i])
        chunk_page = str(result.payload.get('page', result.payload.get('start_page', '')))
        is_on_page = chunk_page in page_hints_set if page_hints_set else False
        
        # Apply boost for chunks on cited pages
        boosted_score = base_score * PAGE_BOOST_FACTOR if is_on_page else base_score
        
        # Convert chunk_id to gold format for display
        qdrant_chunk_id = result.payload.get('chunk_id', '')
        gold_format_id = convert_qdrant_to_gold_format(qdrant_chunk_id)
        
        reranked.append((result, base_score, boosted_score, is_on_page, gold_format_id))
    
    # Sort by boosted score
    reranked.sort(key=lambda x: x[2], reverse=True)
    
    return reranked[:top_k]


# ============================================================================
# REVIEW UI FUNCTIONS
# ============================================================================

def display_gold_answer_with_citations(gold_answer: str):
    """Display gold answer with citations highlighted."""
    if not gold_answer or pd.isna(gold_answer):
        print("  (No gold answer)")
        return
    
    # Highlight citations
    highlighted = re.sub(r'(\[cite:\s*[0-9,\s]+\])', r'【\1】', gold_answer)
    print(f"  {highlighted}")


def review_question(qa_row, qdrant_client, embeddings_model, reranker, collection_name) -> dict:
    """
    Display question review interface with citation tracking.
    
    Returns:
        dict with 'best_chunks', 'missing_chunks', 'page_hints', 'citations'
    """
    question_id = qa_row['question_id']
    question = qa_row['question_text']
    document_id = qa_row['document_id']
    gold_answer = qa_row.get('gold_answer', '')
    current_gold_str = qa_row['gold_support_chunk_ids']
    
    # Handle NaN values from pandas
    if pd.isna(current_gold_str):
        current_gold_str = ''
    else:
        current_gold_str = str(current_gold_str).strip()
    
    current_gold = current_gold_str.split('|') if current_gold_str else []
    page_hint_str = qa_row.get('page_hint', '')
    
    # Handle NaN values for page_hint
    if pd.isna(page_hint_str):
        page_hint_str = ''
    else:
        page_hint_str = str(page_hint_str).strip()
    
    # Extract citation info
    citations = extract_citations_from_answer(gold_answer)
    page_hints = parse_page_hint(page_hint_str)
    
    print("\n" + "=" * 100)
    print(f"QUESTION ID: {question_id}")
    print("=" * 100)
    print(f"Question: {question}")
    print(f"Document: {document_id}")
    print()
    
    # Show gold answer with citations
    print("-" * 100)
    print("GOLD ANSWER (citations highlighted):")
    print("-" * 100)
    display_gold_answer_with_citations(gold_answer)
    print()
    print(f"  📄 Citations in answer: {', '.join(citations) if citations else 'None'}")
    print(f"  📄 Page hints field: {page_hint_str if page_hint_str else 'Empty'}")
    
    # Check for citation vs page_hint mismatch
    if citations and page_hints and set(citations) != set(page_hints):
        print(f"  ⚠ WARNING: Citations [{', '.join(citations)}] differ from page_hint [{', '.join(page_hints)}]")
    
    print()
    
    # Show current gold chunks with missing detection
    print("-" * 100)
    print("CURRENT GOLD CHUNKS:")
    print("-" * 100)
    
    missing_chunks = []
    current_gold_qdrant = []  # Track Qdrant format IDs for comparison
    
    for i, chunk_id in enumerate(current_gold, 1):
        content, found, page, qdrant_id = get_chunk_content(qdrant_client, collection_name, chunk_id)
        current_gold_qdrant.append(qdrant_id)
        
        status = ""
        if not found:
            status = " ⚠ MISSING IN QDRANT"
            missing_chunks.append(chunk_id)
        else:
            # Check if chunk is on a cited page
            if page_hints and page in page_hints:
                status = f" ✓ ON PAGE {page}"
            elif page_hints:
                status = f" ⚠ OFF PAGE (page {page}, expected {', '.join(page_hints)})"
            else:
                status = f" (page {page})"
        
        print(f"\n[{i}] {chunk_id}{status}")
        if found:
            print(f"Content: {content[:300]}{'...' if len(content) > 300 else ''}")
        else:
            print("Content: NOT FOUND - chunk may have been deleted or ID format mismatch")
    
    if missing_chunks:
        print(f"\n⚠ {len(missing_chunks)} chunk(s) missing from Qdrant!")
    
    # Get and show top reranked chunks with page-aware scoring
    best_chunks = find_best_chunks(
        question, document_id, qdrant_client, embeddings_model, reranker, 
        collection_name, page_hints=page_hints or citations, top_k=10
    )
    
    print("\n" + "-" * 100)
    print(f"TOP 10 CHUNKS BY RERANKING (page boost: {PAGE_BOOST_FACTOR}x for cited pages):")
    print("-" * 100)
    
    on_page_count = 0
    for i, (result, base_score, boosted_score, is_on_page, gold_format_id) in enumerate(best_chunks, 1):
        qdrant_chunk_id = result.payload.get('chunk_id', '')
        content = result.payload.get('text', result.payload.get('content', ''))
        chunk_page = result.payload.get('page', result.payload.get('start_page', 'N/A'))
        
        # Status indicators
        is_current_gold = '✓ GOLD' if qdrant_chunk_id in current_gold_qdrant else ''
        page_status = '✓ ON PAGE' if is_on_page else '⚠ OFF PAGE'
        if is_on_page:
            on_page_count += 1
        
        # Show both scores if boosted
        score_display = f"rerank: {base_score:.3f}"
        if is_on_page:
            score_display = f"rerank: {base_score:.3f} → {boosted_score:.3f} (boosted)"
        
        print(f"\n[{i}] {gold_format_id} ({score_display}) [page {chunk_page}] {page_status} {is_current_gold}")
        print(f"Content: {content[:300]}{'...' if len(content) > 300 else ''}")
    
    # Summary stats
    print("\n" + "-" * 100)
    print("SUMMARY:")
    print(f"  • {on_page_count}/{len(best_chunks)} top chunks are on cited pages")
    print(f"  • {len(missing_chunks)} current gold chunk(s) missing from Qdrant")
    print("-" * 100)
    
    print("\n" + "=" * 100)
    print("REVIEW INSTRUCTIONS:")
    print("  - Compare CURRENT GOLD chunks with TOP RERANKED chunks")
    print("  - ✓ ON PAGE chunks match cited pages in the answer")
    print("  - ⚠ OFF PAGE chunks may still be relevant but aren't on cited pages")
    print("  - Consider keeping chunks that directly answer the question")
    print("=" * 100)
    
    return {
        'best_chunks': best_chunks,
        'missing_chunks': missing_chunks,
        'page_hints': page_hints,
        'citations': citations,
        'current_gold_qdrant': current_gold_qdrant
    }

def handle_missing_chunks(missing_chunks: list, question_id: str) -> dict:
    """
    Handle missing chunks - let user decide what to do with each.
    
    Returns:
        dict with 'keep' and 'remove' lists
    """
    if not missing_chunks:
        return {'keep': [], 'remove': []}
    
    print(f"\n⚠ MISSING CHUNKS FOR {question_id}:")
    print("-" * 50)
    
    keep = []
    remove = []
    
    for chunk_id in missing_chunks:
        print(f"\nChunk: {chunk_id}")
        print("  This chunk is in qa_gold but not found in Qdrant.")
        print("  Options:")
        print("    1. Keep (preserve in gold standard)")
        print("    2. Remove (delete from gold standard)")
        
        action = input("  Enter choice (1/2): ").strip()
        
        if action == '2':
            remove.append(chunk_id)
            print(f"  → Will REMOVE {chunk_id}")
        else:
            keep.append(chunk_id)
            print(f"  → Will KEEP {chunk_id}")
    
    return {'keep': keep, 'remove': remove}


def validate_corrections(corrections: list, qdrant_client, collection_name: str) -> dict:
    """
    Validate all corrections before applying.
    
    Returns:
        dict with validation results
    """
    print("\n" + "=" * 100)
    print("VALIDATION REPORT")
    print("=" * 100)
    
    validation_results = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'details': []
    }
    
    for corr in corrections:
        question_id = corr['question_id']
        
        # Handle NaN values for new_gold
        new_gold_str = corr['new_gold']
        if pd.isna(new_gold_str):
            new_gold_str = ''
        else:
            new_gold_str = str(new_gold_str).strip()
        
        new_chunks = new_gold_str.split('|') if new_gold_str else []
        
        # Handle NaN values for old_gold
        old_gold_str = corr['old_gold']
        if pd.isna(old_gold_str):
            old_gold_str = ''
        else:
            old_gold_str = str(old_gold_str).strip()
        
        old_chunks = old_gold_str.split('|') if old_gold_str else []
        
        print(f"\n{question_id}:")
        
        # Check 1: Verify all new chunks exist in Qdrant
        missing_new = []
        for chunk_id in new_chunks:
            if not verify_chunk_exists(qdrant_client, collection_name, chunk_id):
                missing_new.append(chunk_id)
        
        if missing_new:
            validation_results['warnings'].append(
                f"{question_id}: {len(missing_new)} new chunk(s) not found in Qdrant"
            )
            print(f"  ⚠ {len(missing_new)} new chunk(s) not in Qdrant: {', '.join(missing_new)}")
        else:
            print(f"  ✓ All {len(new_chunks)} new chunks verified in Qdrant")
        
        # Check 2: Chunk count comparison
        old_count = len(old_chunks)
        new_count = len(new_chunks)
        if abs(old_count - new_count) > 2:
            validation_results['warnings'].append(
                f"{question_id}: Chunk count changed significantly ({old_count} → {new_count})"
            )
            print(f"  ⚠ Significant chunk count change: {old_count} → {new_count}")
        else:
            print(f"  ✓ Chunk count: {old_count} → {new_count}")
        
        # Check 3: page_hint format validation
        if 'new_page_hint' in corr:
            page_hint = corr['new_page_hint']
            if pd.isna(page_hint):
                page_hint = ''
            else:
                page_hint = str(page_hint).strip()
            
            if page_hint and not all(p.strip().isdigit() for p in page_hint.split('|') if p.strip()):
                validation_results['errors'].append(
                    f"{question_id}: Invalid page_hint format: {page_hint}"
                )
                validation_results['passed'] = False
                print(f"  ✗ Invalid page_hint format: {page_hint}")
            else:
                print(f"  ✓ page_hint format valid: {page_hint if page_hint else '(empty)'}")
        
        validation_results['details'].append({
            'question_id': question_id,
            'old_count': old_count,
            'new_count': new_count,
            'missing_new': missing_new
        })
    
    # Summary
    print("\n" + "-" * 100)
    print("VALIDATION SUMMARY:")
    print(f"  • Total corrections: {len(corrections)}")
    print(f"  • Warnings: {len(validation_results['warnings'])}")
    print(f"  • Errors: {len(validation_results['errors'])}")
    print(f"  • Status: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    
    if validation_results['warnings']:
        print("\nWarnings:")
        for w in validation_results['warnings']:
            print(f"  ⚠ {w}")
    
    if validation_results['errors']:
        print("\nErrors:")
        for e in validation_results['errors']:
            print(f"  ✗ {e}")
    
    print("-" * 100)
    
    return validation_results


def create_backup(filepath: str) -> str:
    """Create timestamped backup of file before modifications."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    backup_filename = f"{name}_backup_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    shutil.copy2(filepath, backup_path)
    return backup_path


def main():
    # Initialize clients
    qdrant_client = QdrantClient()
    embeddings_model = HuggingfaceEmbeddingsModel()
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    collection_name = os.getenv('QDRANT_COLLECTION', 'test_collection')
    
    # Load gold standard
    qa_gold_path = 'backend/qa_gold.csv'
    qa_gold = pd.read_csv(qa_gold_path)
    
    # Ensure page_hint column exists
    if 'page_hint' not in qa_gold.columns:
        qa_gold['page_hint'] = ''
    
    print("=" * 100)
    print("MANUAL GOLD STANDARD REVIEW (v2.0)")
    print("=" * 100)
    print(f"Total questions: {len(qa_gold)}")
    print(f"Collection: {collection_name}")
    print(f"Page boost factor: {PAGE_BOOST_FACTOR}x")
    print()
    print("Features:")
    print("  • Single-colon chunk ID format (doc_id:chunk_num)")
    print("  • Page-aware reranking with score boost for cited pages")
    print("  • Citation tracking with [cite: N] extraction")
    print("  • Missing chunk detection and handling")
    print("  • Comprehensive validation before saving")
    print("  • Automatic backup creation")
    print("=" * 100)
    
    # Ask which questions to review
    print("\nReview options:")
    print("  1. Review all questions")
    print("  2. Review specific question by ID (e.g., q0002)")
    print("  3. Review by document ID")
    print("  4. Review questions 1-10 only")
    print("  5. Review questions with missing chunks only")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    questions_to_review = []
    
    if choice == '1':
        questions_to_review = qa_gold.to_dict('records')
    elif choice == '2':
        qid = input("Enter question ID (e.g., q0002): ").strip()
        questions_to_review = qa_gold[qa_gold['question_id'] == qid].to_dict('records')
    elif choice == '3':
        doc_id = input("Enter document ID: ").strip()
        questions_to_review = qa_gold[qa_gold['document_id'] == doc_id].to_dict('records')
    elif choice == '4':
        questions_to_review = qa_gold.head(10).to_dict('records')
    elif choice == '5':
        # Find questions with missing chunks
        print("\nScanning for questions with missing chunks...")
        missing_questions = []
        for _, row in qa_gold.iterrows():
            chunks = row['gold_support_chunk_ids'].split('|') if row['gold_support_chunk_ids'] else []
            for chunk_id in chunks:
                if not verify_chunk_exists(qdrant_client, collection_name, chunk_id):
                    missing_questions.append(row.to_dict())
                    break
        questions_to_review = missing_questions
        print(f"Found {len(questions_to_review)} question(s) with missing chunks")
    else:
        print("Invalid choice. Exiting.")
        return
    
    if not questions_to_review:
        print("No questions found matching criteria.")
        return
    
    print(f"\nReviewing {len(questions_to_review)} question(s)...")
    
    # Review each question
    corrections = []
    for qa_row in questions_to_review:
        review_result = review_question(qa_row, qdrant_client, embeddings_model, reranker, collection_name)
        best_chunks = review_result['best_chunks']
        missing_chunks = review_result['missing_chunks']
        
        # Handle missing chunks first
        missing_decision = {'keep': [], 'remove': []}
        if missing_chunks:
            missing_decision = handle_missing_chunks(missing_chunks, qa_row['question_id'])
        
        # Suggested chunks (in gold format)
        suggested = [bc[4] for bc in best_chunks[:3]]  # bc[4] is gold_format_id
        
        print("\nSUGGESTED CORRECTIONS:")
        print("  Based on reranking, suggested gold chunks (top 3):")
        print(f"  {' | '.join(suggested)}")
        
        # Show page info for suggestions
        print("\n  Page analysis for suggestions:")
        for i, (result, base_score, boosted_score, is_on_page, gold_id) in enumerate(best_chunks[:3], 1):
            page = result.payload.get('page', result.payload.get('start_page', 'N/A'))
            status = "✓ ON PAGE" if is_on_page else "⚠ OFF PAGE"
            print(f"    [{i}] {gold_id} - page {page} {status}")
        
        # Ask for manual annotation
        print("\nACTION OPTIONS:")
        print("  1. Keep current gold chunks (no change)")
        print("  2. Use suggested top 3 chunks")
        print("  3. Enter custom chunk IDs (comma-separated)")
        print("  4. Remove missing chunks only (keep rest)")
        print("  5. Skip this question")
        
        action = input("\nEnter choice (1-5): ")

        new_gold = None
        new_page_hint = None
        
        if action == '1':
            print("✓ Keeping current gold chunks")
            # Still apply missing chunk removals if any
            if missing_decision['remove']:
                current = qa_row['gold_support_chunk_ids'].split('|')
                updated = [c for c in current if c not in missing_decision['remove']]
                if updated != current:
                    new_gold = '|'.join(updated)
                    print(f"  (Removed {len(missing_decision['remove'])} missing chunk(s))")
        
        elif action == '2':
            new_gold = '|'.join(suggested)
            # Update page_hint based on suggested chunks
            suggested_pages = set()
            for result, _, _, _, _ in best_chunks[:3]:
                page = result.payload.get('page', result.payload.get('start_page', ''))
                if page:
                    suggested_pages.add(str(page))
            new_page_hint = '|'.join(sorted(suggested_pages, key=lambda x: int(x) if x.isdigit() else 0))
            print(f"✓ Updated to: {new_gold}")
            print(f"  New page_hint: {new_page_hint}")
        
        elif action == '3':
            custom = input("Enter chunk IDs (comma-separated, e.g., doc:1,doc:2): ").strip()
            if custom:
                new_gold = '|'.join([c.strip() for c in custom.split(',')])
                # Optionally update page_hint
                update_pages = input("Update page_hint? Enter pages (pipe-separated) or press Enter to keep: ").strip()
                if update_pages:
                    new_page_hint = update_pages
                print(f"✓ Updated to: {new_gold}")
                if new_page_hint:
                    print(f"  New page_hint: {new_page_hint}")
        
        elif action == '4':
            # Remove only missing chunks
            if missing_decision['remove']:
                current = qa_row['gold_support_chunk_ids'].split('|')
                updated = [c for c in current if c not in missing_decision['remove']]
                new_gold = '|'.join(updated)
                print(f"✓ Removed {len(missing_decision['remove'])} missing chunk(s)")
                print(f"  Updated to: {new_gold}")
            else:
                print("No missing chunks to remove")
        
        elif action == '5':
            print("⊘ Skipped")
        else:
            print("Invalid choice, keeping current")
        
        if new_gold is not None:
            correction = {
                'question_id': qa_row['question_id'],
                'old_gold': qa_row['gold_support_chunk_ids'],
                'new_gold': new_gold,
                'old_page_hint': qa_row.get('page_hint', ''),
            }
            if new_page_hint is not None:
                correction['new_page_hint'] = new_page_hint
            corrections.append(correction)
        
        # Ask to continue
        if len(questions_to_review) > 1:
            cont = input("\nContinue to next question? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    # Summary and validation
    if corrections:
        print("\n" + "=" * 100)
        print("SUMMARY OF CORRECTIONS")
        print("=" * 100)
        for corr in corrections:
            print(f"\n{corr['question_id']}:")
            print(f"  OLD chunks: {corr['old_gold']}")
            print(f"  NEW chunks: {corr['new_gold']}")
            if 'new_page_hint' in corr:
                print(f"  OLD page_hint: {corr['old_page_hint']}")
                print(f"  NEW page_hint: {corr['new_page_hint']}")
        
        # Validate corrections
        validation = validate_corrections(corrections, qdrant_client, collection_name)
        
        if not validation['passed']:
            print("\n⚠ Validation failed! Some corrections have errors.")
            proceed = input("Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("⊘ Corrections not applied due to validation errors")
                return
        
        # Ask to apply corrections
        apply = input("\nApply these corrections to qa_gold.csv? (y/n): ").strip().lower()
        if apply == 'y':
            # Create backup first
            backup_path = create_backup(qa_gold_path)
            print(f"✓ Backup created: {backup_path}")
            
            # Apply corrections
            for corr in corrections:
                qa_gold.loc[qa_gold['question_id'] == corr['question_id'], 'gold_support_chunk_ids'] = corr['new_gold']
                if 'new_page_hint' in corr:
                    qa_gold.loc[qa_gold['question_id'] == corr['question_id'], 'page_hint'] = corr['new_page_hint']
            
            qa_gold.to_csv(qa_gold_path, index=False)
            print(f"✓ Applied {len(corrections)} correction(s) to {qa_gold_path}")
            
            # Export session log
            session_log = {
                'timestamp': datetime.now().isoformat(),
                'corrections': corrections,
                'validation': validation
            }
            log_path = f"media/review_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(session_log, f, indent=2, ensure_ascii=False)
            print(f"✓ Session log saved: {log_path}")
        else:
            print("⊘ Corrections not applied")
    else:
        print("\n✓ No corrections made")
    
    # Clear cache
    global _chunk_cache
    _chunk_cache = {}
    
    print("\n" + "=" * 100)
    print("REVIEW COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()