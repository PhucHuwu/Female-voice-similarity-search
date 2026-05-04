"""CLI search over metadata SQLite database."""
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.vector_database.metadata_db import MetadataDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Search audio metadata")
    parser.add_argument("--voice", type=str, default=None, help="Voice name keyword")
    parser.add_argument("--video-id", type=str, default=None, help="YouTube video id")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows")
    parser.add_argument("--db", type=str, default="database/metadata.db", help="Metadata DB path")
    parser.add_argument("--show-vector", action="store_true", help="Show vector dimension availability")
    args = parser.parse_args()

    db = MetadataDB(args.db)
    rows = db.search(voice=args.voice, video_id=args.video_id, limit=args.limit)

    print(f"Found {len(rows)} rows")
    for row in rows:
        print(
            f"vector_idx={row.get('vector_idx')} | voice={row.get('voice_name')} | "
            f"video_id={row.get('source_video_id')} | file={row.get('file_path')}"
        )
        if args.show_vector:
            print(f"  feature_dim={row.get('feature_dim')} | has_vector={row.get('feature_vector') is not None}")


if __name__ == "__main__":
    main()
