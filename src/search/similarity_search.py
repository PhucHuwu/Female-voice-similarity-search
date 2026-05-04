"""
Similarity search pipeline for voice matching
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Tuple, Optional, Dict
from src.feature_extraction.extractor import AudioFeatureExtractor
from src.vector_database.metadata_db import MetadataDB
from src.search.sqlite_vector_search import SQLiteVectorSearch
from src.utils.audio_utils import preprocess_audio


class VoiceSimilaritySearch:
    """End-to-end voice similarity search system"""
    
    def __init__(
        self,
        metadata_db_path: str = "database/metadata.db",
        feature_dim: int = 52
    ):
        """
        Initialize similarity search system
        
        Args:
            metadata_db_path: Path to SQLite metadata DB
            feature_dim: Feature vector dimension
        """
        self.feature_extractor = AudioFeatureExtractor()
        self.metadata_db = MetadataDB(metadata_db_path)
        self.sqlite_vector_search = SQLiteVectorSearch(metadata_db_path)
    
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
        
        rows = self.sqlite_vector_search.search(query_features, top_k=top_k)
        return [
            (r["file_path"], r["similarity_percent"], r["cosine_similarity"])
            for r in rows
        ]

    def get_metadata(self, file_path: str) -> Optional[Dict]:
        """Get metadata for a file path from SQLite DB."""
        return self.metadata_db.get_by_file_path(file_path)

    def search_by_metadata(
        self,
        voice: Optional[str] = None,
        video_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Search audio records by metadata fields."""
        return self.metadata_db.search(voice=voice, video_id=video_id, limit=limit)
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        total = self.metadata_db.count()
        return {
            "total_vectors": total,
            "index_type": "SQLiteVectorSearch",
            "feature_dimension": self.feature_extractor.get_feature_dimension()
        }


def search_similar(
    query_audio_path: str,
    top_k: int = 5,
    metadata_db_path: str = "database/metadata.db"
) -> List[Tuple[str, float, float]]:
    """
    Convenience function for similarity search
    
    Args:
        query_audio_path: Path to query audio
        top_k: Number of results
        metadata_db_path: SQLite metadata DB path
        
    Returns:
        List of (file_path, similarity_score, distance) tuples
    """
    search_system = VoiceSimilaritySearch(
        metadata_db_path=metadata_db_path
    )
    
    return search_system.search_similar(query_audio_path, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    print("Voice Similarity Search System")
    print("="*50)
    
    search_system = VoiceSimilaritySearch()
    print(search_system.get_system_stats())
