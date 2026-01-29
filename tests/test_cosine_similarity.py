"""
Test cosine similarity with same file
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.search.similarity_search import VoiceSimilaritySearch

# Initialize search
search = VoiceSimilaritySearch()

# Test với file chunks
test_file = "data/chunks/yt_9c-yi4vCqZg_chunk0000.wav"

print("="*60)
print("Testing Cosine Similarity")
print("="*60)
print(f"\nQuery file: {test_file}")
print("\nTop 5 Results:\n")

results = search.search_similar(test_file, top_k=5)

for i, (file_path, similarity, cosine) in enumerate(results, 1):
    filename = Path(file_path).name
    print(f"#{i} - {filename}")
    print(f"   Similarity: {similarity:.1f}%")
    print(f"   Cosine: {cosine:.4f}")
    print()

print("="*60)
print("Expected: #1 should be same file with ~99-100% similarity")
print("="*60)
