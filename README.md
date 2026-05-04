# Female Voice Similarity Search

A voice similarity search system that finds similar female voices using 52-dimensional audio features, with SQLite as the database for metadata and feature vectors.

## Overview

This system allows users to upload a female voice audio file and find the top 5 most similar voices from a database of 500+ pre-processed audio samples. Similarity is computed with cosine similarity on feature vectors stored in SQLite.

## Features

- Audio feature extraction (52 features: MFCC, Pitch, Spectral, Temporal, Chroma)
- SQLite database for metadata and feature vectors (`database/metadata.db`)
- Metadata search (voice name, video id)
- Top-5 voice similarity search from uploaded query audio
- Interactive Streamlit web interface with analysis visualizations

## Tech Stack

- Python 3.10+
- Streamlit
- librosa
- NumPy/Pandas
- SQLite (builtin `sqlite3`)
- Matplotlib/Plotly

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```bash
# 1) Download raw audio from list_video.csv
python src/data_collection/download_audio.py

# 2) Split raw files into 3-second chunks
python src/data_collection/split_audio_chunks.py

# 3) Preprocess chunks (16kHz, trim, normalize, fixed length)
python src/data_collection/preprocess_audio.py

# 4) Build SQLite database and feature artifacts
python scripts/build_database.py

# 5) Run app
streamlit run app/streamlit_app.py
```

App URL: `http://localhost:8501`

## Database Outputs

- `database/metadata.db`: main SQLite database (metadata + feature vectors)
- `database/features.npy`: extracted feature matrix backup (N x 52)

## Notes for Assignment Requirements

- Dataset requirement is enforced in `scripts/build_database.py` with minimum 500 processed files.
- Audio is standardized to same duration and sample rate before feature extraction.
- Metadata and vectors are stored in a DBMS (SQLite), and similarity search is performed from DB vectors.
