# Data Pipeline Documentation

## Pipeline Steps

1. Download raw audio
   - Script: `src/data_collection/download_audio.py`
   - Input: `data/list_video.csv`
   - Output: `data/raw/yt_<voice>_<id>.wav`

2. Split into fixed chunks
   - Script: `src/data_collection/split_audio_chunks.py`
   - Input: `data/raw/*.wav`
   - Output: `data/chunks/*.wav` (3s/chunk)

3. Preprocess all chunks
   - Script: `src/data_collection/preprocess_audio.py`
   - Input: `data/chunks/*.wav`
   - Output: `data/processed/*.wav` (16kHz, mono, normalized, fixed 3s)

4. Build database
   - Script: `scripts/build_database.py`
   - Input: `data/processed/*.wav`
   - Output:
     - `database/metadata.db` (SQLite metadata + feature vectors)
     - `database/features.npy` (backup vectors)

## Requirement Enforcement

- `scripts/build_database.py` sẽ dừng nếu số file processed < 500.
- Toàn bộ file processed có cùng độ dài mục tiêu (3 giây) và sample rate 16kHz.

## Data Flow Diagram

```
list_video.csv
    -> download_audio.py
    -> data/raw/
    -> split_audio_chunks.py
    -> data/chunks/
    -> preprocess_audio.py
    -> data/processed/
    -> build_database.py
    -> database/metadata.db + database/features.npy
```

## Useful Commands

```bash
python src/data_collection/download_audio.py
python src/data_collection/split_audio_chunks.py
python src/data_collection/preprocess_audio.py
python scripts/build_database.py
python scripts/search_metadata.py --voice minh --limit 10
streamlit run app/streamlit_app.py
```
