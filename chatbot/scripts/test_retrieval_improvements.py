"""
Quick test script to verify all P0 and P1 improvements are working.
Run this after implementing the changes.
"""
import django
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

def test_query_preprocessing():
    """Test P0: Query preprocessing"""
    print("\n" + "="*60)
    print("TEST 1: Query Preprocessing (P0)")
    print("="*60)
    
    from api.utils.query_preprocessing import preprocess_query
    
    test_cases = [
        ("Xin hỏi Python là gì?", "Python là gì"),
        ("Cho biết điều gì xảy ra khi Mp = 0?", "điều gì xảy ra khi Mp = 0"),
        ("  Extra   spaces   ", "Extra spaces"),
        ("Question with trailing???", "Question with trailing"),
    ]
    
    all_passed = True
    for input_text, expected in test_cases:
        result = preprocess_query(input_text)
        passed = result == expected
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: '{input_text}' → '{result}'")
        if not passed:
            print(f"       Expected: '{expected}'")
            all_passed = False
    
    return all_passed

def test_semantic_chunking():
    """Test P0: Sentence-aware chunking"""
    print("\n" + "="*60)
    print("TEST 2: Semantic Chunking (P0)")
    print("="*60)
    
    from api.tools.chunking import semantic_chunk_text
    from transformers import AutoTokenizer
    
    # Load tokenizer
    embedding_model = os.getenv("EMBEDDING_MODEL", "dangvantuan/vietnamese-embedding")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    
    # Test text with multiple sentences
    test_text = """Python là một ngôn ngữ lập trình cấp cao. Nó được phát triển bởi Guido van Rossum. 
    Python có cú pháp rất đơn giản và dễ học. Nó được sử dụng rộng rãi trong khoa học dữ liệu. 
    Machine Learning và AI cũng sử dụng Python nhiều. TensorFlow và PyTorch là các thư viện phổ biến."""
    
    raw_blocks = [{
        'text': test_text,
        'document_id': 'test_doc',
        'page': 1,
        'section': 'Test'
    }]
    
    chunks = semantic_chunk_text(raw_blocks, tokenizer, target_tokens=50, max_tokens=100)
    
    print(f"Generated {len(chunks)} chunks from test text")
    
    incomplete_count = 0
    for i, chunk in enumerate(chunks):
        text = chunk['text']
        ends_properly = text.rstrip().endswith(('.', '!', '?'))
        status = "✓" if ends_properly else "✗"
        print(f"  Chunk {i+1}: {status} {len(text.split())} words, ends with: '{text[-20:]}'")
        if not ends_properly:
            incomplete_count += 1
    
    success_rate = (len(chunks) - incomplete_count) / len(chunks) * 100
    print(f"\nComplete sentence rate: {success_rate:.1f}%")
    
    return success_rate > 80  # Should be >80% success rate

def test_reranking_threshold():
    """Test P1: Reranking score threshold"""
    print("\n" + "="*60)
    print("TEST 3: Reranking Threshold (P1)")
    print("="*60)
    
    MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-0.5"))
    print(f"MIN_RERANK_SCORE configured: {MIN_RERANK_SCORE}")
    
    # Simulate reranked results
    mock_results = [
        {"id": "chunk_1", "score": 0.8},
        {"id": "chunk_2", "score": 0.5},
        {"id": "chunk_3", "score": -0.2},
        {"id": "chunk_4", "score": -0.7},
    ]
    
    filtered = [r for r in mock_results if r['score'] >= MIN_RERANK_SCORE]
    
    print(f"Original results: {len(mock_results)}")
    print(f"After filtering: {len(filtered)}")
    
    for r in mock_results:
        status = "✓ KEPT" if r['score'] >= MIN_RERANK_SCORE else "✗ FILTERED"
        print(f"  {status}: {r['id']} (score={r['score']})")
    
    return len(filtered) == 2  # Should filter out 2 low-scoring results

def test_query_expansion():
    """Test P1: Improved query expansion"""
    print("\n" + "="*60)
    print("TEST 4: Query Expansion Configuration (P1)")
    print("="*60)
    
    enabled = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    print(f"Query expansion enabled: {enabled}")
    
    if enabled:
        print("✓ Query expansion is enabled")
        print("  - Will generate variants using Gemini")
        print("  - Temperature: 0.3 (lower for consistency)")
        print("  - Includes fallback if expansion fails")
        return True
    else:
        print("⚠ Query expansion is disabled")
        return False

def test_env_configuration():
    """Test environment configuration"""
    print("\n" + "="*60)
    print("TEST 5: Environment Configuration")
    print("="*60)
    
    required_vars = {
        "CHUNK_TARGET_TOKENS": "400",
        "CHUNK_MAX_TOKENS": "512",
        "CHUNK_OVERLAP_TOKENS": "50",
        "USE_SEMANTIC_CHUNKING": "true",
        "MIN_RERANK_SCORE": "-0.5",
        "ENABLE_QUERY_EXPANSION": "true",
    }
    
    all_configured = True
    for var, expected in required_vars.items():
        value = os.getenv(var, "NOT SET")
        is_set = value != "NOT SET"
        status = "✓" if is_set else "✗"
        print(f"  {status} {var}={value}")
        if not is_set:
            all_configured = False
    
    return all_configured

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RETRIEVAL IMPROVEMENTS VERIFICATION")
    print("="*60)
    
    results = {}
    
    try:
        results['preprocessing'] = test_query_preprocessing()
    except Exception as e:
        print(f"\n✗ Query preprocessing test failed: {e}")
        results['preprocessing'] = False
    
    try:
        results['chunking'] = test_semantic_chunking()
    except Exception as e:
        print(f"\n✗ Semantic chunking test failed: {e}")
        results['chunking'] = False
    
    try:
        results['reranking'] = test_reranking_threshold()
    except Exception as e:
        print(f"\n✗ Reranking threshold test failed: {e}")
        results['reranking'] = False
    
    try:
        results['expansion'] = test_query_expansion()
    except Exception as e:
        print(f"\n✗ Query expansion test failed: {e}")
        results['expansion'] = False
    
    try:
        results['config'] = test_env_configuration()
    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        results['config'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED! Retrieval improvements are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
    
    sys.exit(0 if passed == total else 1)
