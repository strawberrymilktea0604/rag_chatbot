import django
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.utils.embeddings import HuggingfaceEmbeddingsModel
import numpy as np

def test_consistency():
    print("=" * 60)
    print("EMBEDDING CONSISTENCY TEST")
    print("=" * 60)
    
    try:
        model = HuggingfaceEmbeddingsModel()
        embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        print(f"\n✓ Loaded embedding model: {embedding_model_name}")
    except Exception as e:
        print(f"\n✗ Failed to load embedding model: {e}")
        return False
    
    # Test cases with Vietnamese and English text
    test_cases = [
        {
            'name': 'Vietnamese sentence',
            'text': 'Công nghệ trí tuệ nhân tạo đang phát triển nhanh chóng.'
        },
        {
            'name': 'English sentence',
            'text': 'Artificial intelligence technology is developing rapidly.'
        },
        {
            'name': 'Mixed language',
            'text': 'Machine Learning và Deep Learning là các nhánh của AI.'
        },
        {
            'name': 'Technical terms',
            'text': 'Python, TensorFlow, PyTorch, scikit-learn'
        },
        {
            'name': 'Short text',
            'text': 'AI'
        }
    ]
    
    print("\nTesting embedding consistency across 3 runs for each test case...\n")
    
    all_consistent = True
    results = []
    
    for test_case in test_cases:
        text = test_case['text']
        name = test_case['name']
        
        print(f"Test: {name}")
        print(f"Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        
        # Generate embedding 3 times
        embeddings = []
        for i in range(3):
            emb = model.embed_texts([text])[0]
            embeddings.append(emb)
        
        # Check if all embeddings are identical
        similarities = []
        for i in range(1, 3):
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[i]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[i])
            )
            similarities.append(similarity)
            print(f"  Similarity run 0 vs {i}: {similarity:.10f}")
        
        # Check for perfect consistency (similarity >= 0.99999)
        min_similarity = min(similarities)
        is_consistent = min_similarity >= 0.99999
        
        if is_consistent:
            print(f"  ✓ CONSISTENT (min similarity: {min_similarity:.10f})\n")
        else:
            print(f"  ✗ INCONSISTENT (min similarity: {min_similarity:.10f})")
            print(f"    ⚠️  Embeddings vary between runs!\n")
            all_consistent = False
        
        results.append({
            'test_case': name,
            'text': text[:50],
            'min_similarity': min_similarity,
            'consistent': is_consistent
        })
    
    # Test with same text but different preprocessing
    print("=" * 60)
    print("PREPROCESSING IMPACT TEST")
    print("=" * 60)
    
    base_text = "Công nghệ AI đang phát triển."
    preprocessing_tests = [
        ('Original', base_text),
        ('Lowercase', base_text.lower()),
        ('Extra spaces', f"  {base_text}  "),
        ('With newline', f"{base_text}\n"),
    ]
    
    print("\nTesting how preprocessing affects embeddings:\n")
    
    base_embedding = model.embed_texts([base_text])[0]
    
    for name, variant in preprocessing_tests:
        variant_embedding = model.embed_texts([variant])[0]
        similarity = np.dot(base_embedding, variant_embedding) / (
            np.linalg.norm(base_embedding) * np.linalg.norm(variant_embedding)
        )
        
        print(f"{name:20s}: similarity = {similarity:.10f}")
        
        if name != 'Original' and similarity < 0.999:
            print(f"  ⚠️  Preprocessing affects embedding! Consider normalizing inputs.")
    
    # Test embedding dimension
    print("\n" + "=" * 60)
    print("EMBEDDING DIMENSION CHECK")
    print("=" * 60)
    
    test_embedding = model.embed_texts(["test"])[0]
    dimension = len(test_embedding)
    print(f"\nEmbedding dimension: {dimension}")
    
    # Check if all test embeddings have same dimension
    dims_consistent = all(len(emb) == dimension for test_case in test_cases for emb in [model.embed_texts([test_case['text']])[0]])
    if dims_consistent:
        print(f"✓ All embeddings have consistent {dimension}-dimensional vectors")
    else:
        print(f"✗ Embedding dimensions vary across different inputs!")
        all_consistent = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if all_consistent:
        print("\n✅ ALL TESTS PASSED")
        print("✓ Embeddings are perfectly consistent across runs")
        print("✓ Model produces deterministic outputs")
        print("✓ Safe for production use")
    else:
        print("\n❌ CONSISTENCY ISSUES DETECTED")
        print("\nPotential causes:")
        print("  1. Model has randomness/dropout enabled during inference")
        print("  2. Non-deterministic operations in model architecture")
        print("  3. Different tokenization or preprocessing between runs")
        print("\nRecommendations:")
        print("  1. Check model.eval() mode is set")
        print("  2. Set random seeds: torch.manual_seed(42)")
        print("  3. Verify sentence-transformers version compatibility")
    
    # Print detailed results table
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    print(f"\n{'Test Case':<30} {'Min Similarity':<15} {'Status':<10}")
    print("-" * 60)
    for result in results:
        status = "✓ PASS" if result['consistent'] else "✗ FAIL"
        print(f"{result['test_case']:<30} {result['min_similarity']:<15.10f} {status:<10}")
    
    return all_consistent

if __name__ == "__main__":
    success = test_consistency()
    sys.exit(0 if success else 1)
