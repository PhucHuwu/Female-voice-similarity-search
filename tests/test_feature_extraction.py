"""
Tests for feature extraction
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.feature_extraction.extractor import AudioFeatureExtractor


def test_feature_extractor_initialization():
    """Test feature extractor can be initialized"""
    extractor = AudioFeatureExtractor()
    assert extractor.sr == 16000
    assert extractor.n_mfcc == 13


def test_feature_dimension():
    """Test feature dimension is correct"""
    extractor = AudioFeatureExtractor()
    expected_dim = 26 + 4 + 6 + 4 + 12  # 52
    assert extractor.get_feature_dimension() == expected_dim


def test_mfcc_extraction():
    """Test MFCC extraction returns correct shape"""
    extractor = AudioFeatureExtractor()
    # Create dummy audio
    audio = np.random.randn(16000 * 3)  # 3 seconds
    
    mfcc_features = extractor.extract_mfcc(audio)
    assert mfcc_features.shape == (26,)  # 13 mean + 13 std
    assert not np.isnan(mfcc_features).any()


def test_pitch_extraction():
    """Test pitch extraction"""
    extractor = AudioFeatureExtractor()
    audio = np.random.randn(16000 * 3)
    
    pitch_features = extractor.extract_pitch_features(audio)
    assert pitch_features.shape == (4,)  # mean, std, min, max


def test_complete_feature_extraction():
    """Test complete feature extraction pipeline"""
    extractor = AudioFeatureExtractor()
    audio = np.random.randn(16000 * 3)
    
    features = extractor.extract_all_features(audio)
    assert features.shape == (52,)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
