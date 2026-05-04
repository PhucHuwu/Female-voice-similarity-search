"""SQLite metadata database for voice search."""
import csv
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract YouTube video id from URL."""
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)
    return None


def parse_processed_filename(file_path: str) -> Dict[str, Optional[str]]:
    """Parse voice/video_id/chunk_id from processed filename."""
    stem = Path(file_path).stem
    pattern = r"^yt_(?P<voice>.+)_(?P<video_id>[A-Za-z0-9_-]{11})_chunk(?P<chunk_id>\d+)_processed$"
    match = re.match(pattern, stem)
    if not match:
        return {"voice": None, "video_id": None, "chunk_id": None}
    return {
        "voice": match.group("voice"),
        "video_id": match.group("video_id"),
        "chunk_id": match.group("chunk_id"),
    }


class MetadataDB:
    """Manage metadata storage and lookup in SQLite."""

    def __init__(self, db_path: str = "database/metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_idx INTEGER UNIQUE,
                    file_path TEXT UNIQUE NOT NULL,
                    source_url TEXT,
                    source_video_id TEXT,
                    voice_name TEXT,
                    chunk_id TEXT,
                    sample_rate INTEGER,
                    duration REAL,
                    feature_dim INTEGER,
                    feature_vector BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_voice_name ON audio_metadata(voice_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON audio_metadata(source_video_id)")

    def clear_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM audio_metadata")

    def upsert_records(self, records: List[Dict]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO audio_metadata(
                    vector_idx, file_path, source_url, source_video_id, voice_name,
                    chunk_id, sample_rate, duration, feature_dim, feature_vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    vector_idx=excluded.vector_idx,
                    source_url=excluded.source_url,
                    source_video_id=excluded.source_video_id,
                    voice_name=excluded.voice_name,
                    chunk_id=excluded.chunk_id,
                    sample_rate=excluded.sample_rate,
                    duration=excluded.duration,
                    feature_dim=excluded.feature_dim,
                    feature_vector=excluded.feature_vector
                """,
                [
                    (
                        r.get("vector_idx"),
                        r.get("file_path"),
                        r.get("source_url"),
                        r.get("source_video_id"),
                        r.get("voice_name"),
                        r.get("chunk_id"),
                        r.get("sample_rate"),
                        r.get("duration"),
                        r.get("feature_dim"),
                        sqlite3.Binary(np.asarray(r.get("feature_vector"), dtype=np.float32).tobytes())
                        if r.get("feature_vector") is not None
                        else None,
                    )
                    for r in records
                ],
            )

    def get_by_file_path(self, file_path: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM audio_metadata WHERE file_path = ?",
                (file_path,),
            ).fetchone()
        return dict(row) if row else None

    def search(self, voice: Optional[str] = None, video_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        conditions = []
        params = []

        if voice:
            conditions.append("voice_name LIKE ?")
            params.append(f"%{voice}%")
        if video_id:
            conditions.append("source_video_id = ?")
            params.append(video_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM audio_metadata
            WHERE {where_clause}
            ORDER BY id ASC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM audio_metadata").fetchone()
        return int(row["c"])

    def load_all_vectors(self) -> List[Dict]:
        """Load all vectors and metadata rows from SQLite."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file_path, voice_name, source_video_id, feature_dim, feature_vector
                FROM audio_metadata
                WHERE feature_vector IS NOT NULL
                """
            ).fetchall()

        parsed = []
        for row in rows:
            dim = int(row["feature_dim"]) if row["feature_dim"] else 0
            vec = np.frombuffer(row["feature_vector"], dtype=np.float32)
            if dim > 0 and vec.size == dim:
                parsed.append({
                    "file_path": row["file_path"],
                    "voice_name": row["voice_name"],
                    "source_video_id": row["source_video_id"],
                    "vector": vec,
                })
        return parsed


def load_video_catalog(csv_path: str = "data/list_video.csv") -> Dict[str, Dict[str, str]]:
    """Load mapping {video_id: {url, voice}} from video list CSV."""
    result: Dict[str, Dict[str, str]] = {}
    path = Path(csv_path)
    if not path.exists():
        return result

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            url = (row[0] or "").strip().strip('"')
            voice = (row[1] if len(row) > 1 else "").strip().strip('"')
            video_id = extract_video_id(url)
            if video_id:
                result[video_id] = {"url": url, "voice": voice}

    return result
