"""Vector similarity search directly from SQLite feature vectors."""
from typing import Dict, List
import numpy as np

from src.vector_database.metadata_db import MetadataDB


class SQLiteVectorSearch:
    """Brute-force cosine similarity search over SQLite-stored vectors."""

    def __init__(self, metadata_db_path: str = "database/metadata.db"):
        self.metadata_db = MetadataDB(metadata_db_path)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        voice: str = None,
        video_id: str = None,
    ) -> List[Dict]:
        rows = self.metadata_db.load_all_vectors()
        if voice:
            rows = [r for r in rows if r.get("voice_name") and voice.lower() in r["voice_name"].lower()]
        if video_id:
            rows = [r for r in rows if r.get("source_video_id") == video_id]

        scored = []
        for row in rows:
            sim = self._cosine_similarity(query_vector.astype(np.float32), row["vector"].astype(np.float32))
            scored.append({
                "file_path": row["file_path"],
                "voice_name": row.get("voice_name"),
                "source_video_id": row.get("source_video_id"),
                "cosine_similarity": sim,
                "similarity_percent": sim * 100.0,
            })

        scored.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return scored[:top_k]
