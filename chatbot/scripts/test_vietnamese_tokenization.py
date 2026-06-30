#!/usr/bin/env python
"""
Test Vietnamese tokenization approaches for BM25 hybrid search.

Compares:
1. underthesea word segmentation (compound word aware)
2. Syllable-based tokenization (simple whitespace split)

Measures:
- Tokenization speed
- Compound word preservation
- BM25 relevance improvement
"""

import os
import sys
import time
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sample Vietnamese technical terms for testing
SAMPLE_TERMS = [
    "hệ điều hành",          # Operating system
    "máy chủ ảo",            # Virtual server  
    "cơ sở dữ liệu",         # Database
    "trí tuệ nhân tạo",      # Artificial intelligence
    "học máy",               # Machine learning
    "xử lý ngôn ngữ tự nhiên", # Natural language processing
    "mạng nơ-ron",           # Neural network
    "điện toán đám mây",     # Cloud computing
    "bảo mật thông tin",     # Information security
    "phân tích dữ liệu",     # Data analysis
]

SAMPLE_DOCUMENTS = [
    "Hệ điều hành Linux là một hệ điều hành mã nguồn mở được sử dụng rộng rãi trong các máy chủ ảo.",
    "Cơ sở dữ liệu MongoDB là một hệ quản trị cơ sở dữ liệu NoSQL phổ biến.",
    "Trí tuệ nhân tạo và học máy đang thay đổi cách chúng ta xử lý ngôn ngữ tự nhiên.",
    "Mạng nơ-ron sâu là nền tảng của nhiều ứng dụng trí tuệ nhân tạo hiện đại.",
    "Điện toán đám mây cho phép triển khai máy chủ ảo một cách linh hoạt và tiết kiệm chi phí.",
]

SAMPLE_QUERIES = [
    "hệ điều hành Linux",
    "máy chủ ảo là gì",
    "cơ sở dữ liệu NoSQL",
    "ứng dụng trí tuệ nhân tạo",
]


def syllable_tokenize(text: str) -> List[str]:
    """
    Simple syllable-based tokenization (whitespace split).
    Vietnamese words are multi-syllable, so this breaks compound words.
    """
    # Lowercase and split on whitespace
    return text.lower().split()


def underthesea_tokenize(text: str) -> List[str]:
    """
    Vietnamese word segmentation using underthesea.
    Preserves compound words like "hệ_điều_hành".
    """
    try:
        from underthesea import word_tokenize
        # word_tokenize returns space-separated tokens with underscores for compounds
        tokenized = word_tokenize(text, format="text")
        return tokenized.lower().split()
    except ImportError:
        print("WARNING: underthesea not installed. Run: pip install underthesea")
        return syllable_tokenize(text)


def measure_tokenization_speed(tokenizer_func, texts: List[str], iterations: int = 100) -> Tuple[float, float]:
    """
    Measure tokenization speed.
    Returns: (total_time_ms, avg_time_per_doc_ms)
    """
    start = time.perf_counter()
    for _ in range(iterations):
        for text in texts:
            tokenizer_func(text)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    
    total_docs = iterations * len(texts)
    avg_per_doc = elapsed / total_docs
    
    return elapsed, avg_per_doc


def compare_tokenization(text: str) -> dict:
    """
    Compare syllable vs underthesea tokenization for a single text.
    """
    syllable_tokens = syllable_tokenize(text)
    underthesea_tokens = underthesea_tokenize(text)
    
    return {
        "text": text,
        "syllable": {
            "tokens": syllable_tokens,
            "count": len(syllable_tokens),
        },
        "underthesea": {
            "tokens": underthesea_tokens,
            "count": len(underthesea_tokens),
        },
        "compound_words_preserved": len(underthesea_tokens) < len(syllable_tokens),
    }


def test_bm25_relevance():
    """
    Test BM25 retrieval relevance with both tokenization approaches.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("WARNING: rank-bm25 not installed. Run: pip install rank-bm25")
        return
    
    print("\n" + "="*60)
    print("BM25 RELEVANCE TEST")
    print("="*60)
    
    # Build BM25 indexes with both tokenization approaches
    syllable_corpus = [syllable_tokenize(doc) for doc in SAMPLE_DOCUMENTS]
    underthesea_corpus = [underthesea_tokenize(doc) for doc in SAMPLE_DOCUMENTS]
    
    bm25_syllable = BM25Okapi(syllable_corpus, k1=1.5, b=0.75)
    bm25_underthesea = BM25Okapi(underthesea_corpus, k1=1.5, b=0.75)
    
    for query in SAMPLE_QUERIES:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Syllable-based retrieval
        syllable_query_tokens = syllable_tokenize(query)
        syllable_scores = bm25_syllable.get_scores(syllable_query_tokens)
        syllable_top_idx = sorted(range(len(syllable_scores)), key=lambda i: syllable_scores[i], reverse=True)[0]
        
        # Underthesea-based retrieval
        underthesea_query_tokens = underthesea_tokenize(query)
        underthesea_scores = bm25_underthesea.get_scores(underthesea_query_tokens)
        underthesea_top_idx = sorted(range(len(underthesea_scores)), key=lambda i: underthesea_scores[i], reverse=True)[0]
        
        print(f"  Syllable tokens: {syllable_query_tokens}")
        print(f"  Underthesea tokens: {underthesea_query_tokens}")
        print(f"  Syllable top score: {syllable_scores[syllable_top_idx]:.4f} (doc {syllable_top_idx})")
        print(f"  Underthesea top score: {underthesea_scores[underthesea_top_idx]:.4f} (doc {underthesea_top_idx})")
        
        # Check if underthesea gives higher scores (better compound word matching)
        if max(underthesea_scores) > max(syllable_scores):
            print(f"  ✓ Underthesea has HIGHER relevance score")
        elif max(underthesea_scores) < max(syllable_scores):
            print(f"  ✗ Syllable has higher relevance score")
        else:
            print(f"  = Same relevance scores")


def main():
    print("="*60)
    print("VIETNAMESE TOKENIZATION COMPARISON")
    print("="*60)
    
    # Test compound word preservation
    print("\n1. COMPOUND WORD PRESERVATION TEST")
    print("-"*40)
    
    for term in SAMPLE_TERMS:
        result = compare_tokenization(term)
        print(f"\nTerm: '{term}'")
        print(f"  Syllable:     {result['syllable']['tokens']} ({result['syllable']['count']} tokens)")
        print(f"  Underthesea:  {result['underthesea']['tokens']} ({result['underthesea']['count']} tokens)")
        if result['compound_words_preserved']:
            print(f"  ✓ Compound word preserved!")
    
    # Speed comparison
    print("\n" + "="*60)
    print("2. TOKENIZATION SPEED TEST")
    print("="*60)
    
    print("\nMeasuring tokenization speed (100 iterations)...")
    
    syllable_total, syllable_avg = measure_tokenization_speed(syllable_tokenize, SAMPLE_DOCUMENTS)
    print(f"  Syllable-based:  {syllable_total:.2f}ms total, {syllable_avg:.4f}ms/doc")
    
    underthesea_total, underthesea_avg = measure_tokenization_speed(underthesea_tokenize, SAMPLE_DOCUMENTS)
    print(f"  Underthesea:     {underthesea_total:.2f}ms total, {underthesea_avg:.4f}ms/doc")
    
    speedup = underthesea_total / syllable_total if syllable_total > 0 else 0
    print(f"\n  Underthesea is {speedup:.1f}x slower than syllable-based")
    
    if underthesea_avg < 100:
        print(f"  ✓ Underthesea is acceptable (<100ms per doc): {underthesea_avg:.2f}ms")
    else:
        print(f"  ✗ Underthesea may be too slow: {underthesea_avg:.2f}ms > 100ms target")
    
    # BM25 relevance test
    test_bm25_relevance()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Recommendations:
1. Use underthesea for document indexing (one-time cost during upload)
2. Use underthesea for query tokenization (acceptable latency <100ms)
3. Compound word preservation significantly improves BM25 relevance
4. For Vietnamese technical terms, underthesea provides better matching

Configuration:
- Set USE_UNDERTHESEA_TOKENIZATION=true in .env for hybrid search
- Set BM25_K1=1.5 and BM25_B=0.75 for Vietnamese documents
""")


if __name__ == "__main__":
    main()
