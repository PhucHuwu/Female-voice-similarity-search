"""
Similarity search pipeline for voice matching
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Tuple
from src.feature_extraction.extractor import AudioFeatureExtractor
from src.vector_database.faiss_manager import FAISSManager
from src.utils.audio_utils import preprocess_audio


class VoiceSimilaritySearch:
    """End-to-end voice similarity search system"""
    
    def __init__(
        self,
        faiss_index_path: str = "database/vectors/faiss_index.bin",
        mapping_path: str = "database/index_mapping.json",
        feature_dim: int = 52
    ):
        """
        Initialize similarity search system
        
        Args:
            faiss_index_path: Path to FAISS index file
            mapping_path: Path to index mapping JSON
            feature_dim: Feature vector dimension
        """
        self.feature_extractor = AudioFeatureExtractor()
        self.faiss_manager = FAISSManager(
            dimension=feature_dim,
            index_path=faiss_index_path,
            mapping_path=mapping_path
        )
        
        # Load existing index if available
        try:
            self.faiss_manager.load_index()
            print("✓ Loaded existing FAISS index")
        except FileNotFoundError:
            print("⚠ No existing index found. Build index first.")
    
    def search_similar(
        self,
        query_audio_path: str,
        top_k: int = 5,
        preprocess: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for similar voices
        
        Args:
            query_audio_path: Path to query audio file
            top_k: Number of results to return
            preprocess: Whether to preprocess audio before feature extraction
            
        Returns:
            List of (file_path, similarity_score) tuples, sorted by similarity
        """
        # Extract features from query audio
        if preprocess:
            query_audio = preprocess_audio(query_audio_path)
            query_features = self.feature_extractor.extract_all_features(query_audio)
        else:
            query_features = self.feature_extractor.extract_from_file(query_audio_path)
        
        # Search in FAISS index
        results = self.faiss_manager.search(query_features, k=top_k)
        
        # Convert L2 distance to similarity score (0-100%)
        normalized_results = []
        for file_path, distance in results:
            # Normalize distance to 0-100 range
            # Lower distance = higher similarity
            similarity_score = max(0, 100 - distance * 10)  # Simple normalization
            normalized_results.append((file_path, similarity_score, distance))
        
        return normalized_results
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        faiss_stats = self.faiss_manager.get_stats()
        return {
            **faiss_stats,
            "feature_dimension": self.feature_extractor.get_feature_dimension()
        }


def search_similar(
    query_audio_path: str,
    top_k: int = 5,
    faiss_index_path: str = "database/vectors/faiss_index.bin",
    mapping_path: str = "database/index_mapping.json"
) -> List[Tuple[str, float, float]]:
    """
    Convenience function for similarity search
    
    Args:
        query_audio_path: Path to query audio
        top_k: Number of results
        faiss_index_path: FAISS index path
        mapping_path: Mapping file path
        
    Returns:
        List of (file_path, similarity_score, distance) tuples
    """
    search_system = VoiceSimilaritySearch(
        faiss_index_path=faiss_index_path,
        mapping_path=mapping_path
    )
    
    return search_system.search_similar(query_audio_path, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    print("Voice Similarity Search System")
    print("="*50)
    
    search_system = VoiceSimilaritySearch()
    print(search_system.get_system_stats())
