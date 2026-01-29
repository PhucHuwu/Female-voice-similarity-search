"""
Test feature extraction consistency
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.feature_extraction.extractor import AudioFeatureExtractor
from src.utils.audio_utils import preprocess_audio

def test_same_file_features():
    """Test if same file produces same features"""
    
    extractor = AudioFeatureExtractor()
    
    # Test với file processed trong database
    processed_file = "data/processed/yt_9c-yi4vCqZg_chunk0000_processed.wav"
    
    print("="*60)
    print("TEST 1: Extract features từ cùng file 2 lần")
    print("="*60)
    
    # Extract 2 lần
    features1 = extractor.extract_from_file(processed_file)
    features2 = extractor.extract_from_file(processed_file)
    
    # So sánh
    diff = np.abs(features1 - features2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(f"Are they identical? {np.allclose(features1, features2)}")
    
    print("\n" + "="*60)
    print("TEST 2: So sánh chunks vs processed")
    print("="*60)
    
    # Test với file chunks (chưa processed)
    chunks_file = "data/chunks/yt_9c-yi4vCqZg_chunk0000.wav"
    
    # Extract từ chunks file (có preprocessing)
    chunks_audio = preprocess_audio(chunks_file)
    features_chunks = extractor.extract_all_features(chunks_audio)
    
    # Extract từ processed file
    features_processed = extractor.extract_from_file(processed_file)
    
    # So sánh
    diff = np.abs(features_chunks - features_processed)
    l2_distance = np.linalg.norm(features_chunks - features_processed)
    
    print(f"L2 distance: {l2_distance:.4f}")
    print(f"Max feature difference: {np.max(diff):.4f}")
    print(f"Mean feature difference: {np.mean(diff):.4f}")
    
    print("\n" + "="*60)
    print("TEST 3: Feature statistics")
    print("="*60)
    
    print(f"\nChunks features (first 10):")
    print(features_chunks[:10])
    
    print(f"\nProcessed features (first 10):")
    print(features_processed[:10])
    
    print(f"\nDifference (first 10):")
    print(diff[:10])

if __name__ == "__main__":
    test_same_file_features()
