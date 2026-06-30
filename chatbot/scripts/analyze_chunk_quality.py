import django
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.mongo_repository import MongoRepository
import nltk
import numpy as np
import pandas as pd

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

def analyze_chunks():
    print("=" * 60)
    print("CHUNK QUALITY ANALYSIS")
    print("=" * 60)
    
    repo = MongoRepository()
    
    try:
        chunks = list(repo.db['chunks'].find({}))
    except Exception as e:
        print(f"\n✗ Failed to fetch chunks from MongoDB: {e}")
        print("Make sure MongoDB is running and contains chunk data.")
        return
    
    if not chunks:
        print("\n⚠️  No chunks found in MongoDB!")
        print("Upload some documents first to analyze chunk quality.")
        return
    
    print(f"\n✓ Found {len(chunks)} chunks in MongoDB")
    print("\nAnalyzing chunk quality metrics...\n")
    
    stats = {
        'lengths': [],
        'oversized': 0,
        'undersized': 0,
        'incomplete_sentences': 0,
        'empty_chunks': 0,
        'has_embedding': 0,
        'missing_embedding': 0
    }
    
    chunk_details = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        chunk_id = chunk.get('chunk_id', 'unknown')
        
        # Check for empty chunks
        if not text or len(text.strip()) == 0:
            stats['empty_chunks'] += 1
            continue
        
        # Tokenize and count tokens
        try:
            tokens = nltk.word_tokenize(text)
            token_count = len(tokens)
        except Exception:
            token_count = len(text.split())  # Fallback to simple split
        
        stats['lengths'].append(token_count)
        
        # Check size categories
        if token_count > 512:
            stats['oversized'] += 1
        if token_count < 50:
            stats['undersized'] += 1
        
        # Check if chunk ends mid-sentence
        text_stripped = text.rstrip()
        if text_stripped and not text_stripped.endswith(('.', '!', '?', '"', "'", '。', '！', '？')):
            stats['incomplete_sentences'] += 1
        
        # Check for embedding
        if 'embedding' in chunk and chunk['embedding']:
            stats['has_embedding'] += 1
        else:
            stats['missing_embedding'] += 1
        
        # Store details for further analysis
        chunk_details.append({
            'chunk_id': chunk_id,
            'token_count': token_count,
            'text_length': len(text),
            'document_id': chunk.get('document_id', 'unknown'),
            'page': chunk.get('page', 'N/A'),
            'has_embedding': 'embedding' in chunk and chunk['embedding'] is not None
        })
    
    # Calculate statistics
    lengths = np.array(stats['lengths'])
    valid_chunks = len(chunks) - stats['empty_chunks']
    
    print("=" * 60)
    print("CHUNK SIZE STATISTICS")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Valid chunks: {valid_chunks}")
    print(f"Empty chunks: {stats['empty_chunks']}")
    
    if len(lengths) > 0:
        print(f"\nToken Count Distribution:")
        print(f"  Average: {lengths.mean():.1f} tokens")
        print(f"  Median: {np.median(lengths):.1f} tokens")
        print(f"  Min: {lengths.min()} tokens")
        print(f"  Max: {lengths.max()} tokens")
        print(f"  Std Dev: {lengths.std():.1f} tokens")
        
        print(f"\nPercentiles:")
        print(f"  25th: {np.percentile(lengths, 25):.1f} tokens")
        print(f"  50th: {np.percentile(lengths, 50):.1f} tokens")
        print(f"  75th: {np.percentile(lengths, 75):.1f} tokens")
        print(f"  95th: {np.percentile(lengths, 95):.1f} tokens")
    
    print("\n" + "=" * 60)
    print("CHUNK QUALITY ISSUES")
    print("=" * 60)
    
    print(f"\n📏 SIZE ISSUES:")
    print(f"  Oversized (>512 tokens): {stats['oversized']} ({stats['oversized']/valid_chunks*100:.1f}%)")
    print(f"  Undersized (<50 tokens): {stats['undersized']} ({stats['undersized']/valid_chunks*100:.1f}%)")
    
    print(f"\n✂️  TRUNCATION ISSUES:")
    print(f"  Incomplete sentences: {stats['incomplete_sentences']} ({stats['incomplete_sentences']/valid_chunks*100:.1f}%)")
    
    print(f"\n🔢 EMBEDDING COVERAGE:")
    print(f"  Has embedding: {stats['has_embedding']} ({stats['has_embedding']/valid_chunks*100:.1f}%)")
    print(f"  Missing embedding: {stats['missing_embedding']} ({stats['missing_embedding']/valid_chunks*100:.1f}%)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if stats['oversized'] / valid_chunks > 0.05:
        print("\n⚠️  High percentage of oversized chunks!")
        print("   → Consider reducing max chunk size in chunking.py")
        print("   → Current limit: 512 tokens")
    
    if stats['undersized'] / valid_chunks > 0.10:
        print("\n⚠️  High percentage of undersized chunks!")
        print("   → These may lack context for accurate retrieval")
        print("   → Consider merging small chunks or adjusting min size")
    
    if stats['incomplete_sentences'] / valid_chunks > 0.15:
        print("\n⚠️  Many chunks end mid-sentence!")
        print("   → Implement sentence-boundary aware chunking")
        print("   → See semantic_chunk_text() in api/tools/chunking.py")
    
    if stats['missing_embedding'] > 0:
        print(f"\n⚠️  {stats['missing_embedding']} chunks missing embeddings!")
        print("   → Run: python reindex_with_new_embeddings.py")
    
    if len(lengths) > 0 and lengths.std() > 150:
        print("\n⚠️  High variance in chunk sizes!")
        print(f"   → Standard deviation: {lengths.std():.1f} tokens")
        print("   → Consider more consistent chunking strategy")
    
    # Save detailed report
    df = pd.DataFrame(chunk_details)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'media', 'chunk_quality_analysis.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Detailed analysis saved to: {output_path}")
    
    # Show top 5 problematic chunks
    print("\n" + "=" * 60)
    print("TOP 5 OVERSIZED CHUNKS")
    print("=" * 60)
    oversized = df[df['token_count'] > 512].nlargest(5, 'token_count')
    if not oversized.empty:
        for idx, row in oversized.iterrows():
            print(f"\n  Chunk: {row['chunk_id']}")
            print(f"  Document: {row['document_id']}, Page: {row['page']}")
            print(f"  Tokens: {row['token_count']}")
    else:
        print("\n✓ No oversized chunks found!")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    analyze_chunks()
