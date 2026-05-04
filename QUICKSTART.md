# Quick Start

## Manual Pipeline

```bash
# 1) Setup env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2) Download audio from CSV
python src/data_collection/download_audio.py

# 3) Split to chunks
python src/data_collection/split_audio_chunks.py

# Split outputs:
# - data/chunks (base 5s)
# - data/query_short (test 5s)
# - data/query_long (test long 10-20s)

# 4) Preprocess
python src/data_collection/preprocess_audio.py

# 5) Build SQLite database (metadata + vectors)
python scripts/build_database.py

# 6) Optional metadata query
python scripts/search_metadata.py --voice minh --limit 10

# 7) Run demo app
streamlit run app/streamlit_app.py
```

## Notes

- Build step enforces minimum 500 processed files.
- Main DB is `database/metadata.db`.
- Similarity search reads vectors from SQLite and returns top-5 matches.
- UI and evaluation can use both `data/query_short` and `data/query_long`.
