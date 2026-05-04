"""
Similarity search pipeline for voice matching
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import librosa
from typing import List, Tuple, Optional, Dict
from src.feature_extraction.extractor import AudioFeatureExtractor
from src.vector_database.metadata_db import MetadataDB
from src.search.sqlite_vector_search import SQLiteVectorSearch
from src.search.feature_transform import FeatureTransform
from src.utils.audio_utils import preprocess_audio, trim_silence, normalize_audio


class VoiceSimilaritySearch:
    """End-to-end voice similarity search system"""
    
    def __init__(
        self,
        metadata_db_path: str = "database/metadata.db",
        scaler_path: str = "database/scaler.pkl",
        pca_path: str = "database/pca.pkl",
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
        self.feature_transform = FeatureTransform(scaler_path=scaler_path, pca_path=pca_path)
        self.feature_transform.load()

    def _segment_query_audio(
        self,
        query_audio_path: str,
        segment_duration: float = 5.0,
        overlap: float = 2.5,
        target_sr: int = 16000,
    ) -> List[np.ndarray]:
        """Split long query into multiple 5s segments with overlap."""
        duration = librosa.get_duration(path=query_audio_path)
        if duration < segment_duration:
            raise ValueError("Query audio must be at least 5 seconds.")

        if duration <= segment_duration:
            audio = preprocess_audio(query_audio_path, target_sr=target_sr, target_duration=segment_duration, trim=True)
            return [audio]

        audio, _ = librosa.load(query_audio_path, sr=target_sr)
        audio = trim_silence(audio)
        audio = normalize_audio(audio)

        seg_len = int(segment_duration * target_sr)
        hop_len = int((segment_duration - overlap) * target_sr)
        if hop_len <= 0:
            hop_len = seg_len

        segments = []
        start = 0
        while start + seg_len <= len(audio):
            segments.append(audio[start:start + seg_len])
            start += hop_len

        if not segments:
            segments.append(preprocess_audio(query_audio_path, target_sr=target_sr, target_duration=segment_duration, trim=True))

        return segments

    @staticmethod
    def _aggregate_segment_results(segment_results: List[List[Dict]], top_k: int) -> List[Tuple[str, float, float]]:
        """Aggregate multi-segment retrieval results into final top-k."""
        bucket: Dict[str, Dict] = {}
        for results in segment_results:
            for row in results:
                fp = row["file_path"]
                if fp not in bucket:
                    bucket[fp] = {"sims": [], "voice_name": row.get("voice_name"), "source_video_id": row.get("source_video_id")}
                bucket[fp]["sims"].append(float(row["cosine_similarity"]))

        scored = []
        for fp, info in bucket.items():
            sims = info["sims"]
            mean_sim = float(np.mean(sims))
            max_sim = float(np.max(sims))
            support = len(sims)
            final_sim = (0.7 * mean_sim) + (0.3 * max_sim)
            scored.append((fp, final_sim * 100.0, final_sim, support))

        scored.sort(key=lambda x: (x[2], x[3]), reverse=True)
        return [(fp, sim_pct, sim) for fp, sim_pct, sim, _ in scored[:top_k]]

    def extract_query_features_for_display(self, query_audio_path: str) -> np.ndarray:
        """Return display feature vector; mean of multi-segment features when >5s."""
        segments = self._segment_query_audio(query_audio_path)
        feats = [self.feature_extractor.extract_all_features(seg) for seg in segments]
        return np.mean(np.vstack(feats), axis=0)
    
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
        if not preprocess:
            query_features = self.feature_extractor.extract_from_file(query_audio_path)
            query_features = self.feature_transform.transform(query_features.reshape(1, -1))[0]
            rows = self.sqlite_vector_search.search(query_features, top_k=top_k)
            return [(r["file_path"], r["similarity_percent"], r["cosine_similarity"]) for r in rows]

        segments = self._segment_query_audio(query_audio_path)
        segment_results: List[List[Dict]] = []
        for seg in segments:
            seg_feat = self.feature_extractor.extract_all_features(seg)
            seg_feat = self.feature_transform.transform(seg_feat.reshape(1, -1))[0]
            rows = self.sqlite_vector_search.search(seg_feat, top_k=top_k)
            segment_results.append(rows)

        return self._aggregate_segment_results(segment_results, top_k)

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
    metadata_db_path: str = "database/metadata.db",
    scaler_path: str = "database/scaler.pkl",
    pca_path: str = "database/pca.pkl",
) -> List[Tuple[str, float, float]]:
    """
    Convenience function for similarity search
    
    Args:
        query_audio_path: Path to query audio
        top_k: Number of results
        metadata_db_path: SQLite metadata DB path
        scaler_path: StandardScaler path
        pca_path: PCA path
        
    Returns:
        List of (file_path, similarity_score, distance) tuples
    """
    search_system = VoiceSimilaritySearch(
        metadata_db_path=metadata_db_path,
        scaler_path=scaler_path,
        pca_path=pca_path,
    )
    
    return search_system.search_similar(query_audio_path, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    print("Voice Similarity Search System")
    print("="*50)
    
    search_system = VoiceSimilaritySearch()
    print(search_system.get_system_stats())
