# Female Voice Similarity Search

A voice similarity search system that finds similar female voices using audio feature extraction and vector similarity search with FAISS.

## Overview

This system allows users to upload a female voice audio file and find the top 5 most similar voices from a database of 500+ pre-processed audio samples. The similarity is calculated using advanced audio features including MFCC, pitch, spectral characteristics, and more.

## Features

- Audio feature extraction (52 features including MFCC, Pitch, Spectral, Temporal, and Chroma)
- FAISS-based vector similarity search
- Interactive Streamlit web interface
- Advanced audio analysis and visualization
- Feature comparison and insights

## Tech Stack

- **Python 3.10+** - Core programming language
- **Streamlit** - Web interface
- **librosa** - Audio processing and feature extraction
- **FAISS** - Vector similarity search
- **NumPy/Pandas** - Data manipulation
- **Matplotlib/Plotly** - Visualization

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate voice-search
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Setup Environment

Activate the conda environment:

```bash
conda activate voice-search
```

### 2. Build Database

Build the FAISS database from audio files by running these Python scripts in order:

```bash
# Step 1: Download and chunk audio files
python src/data_collection/download_audio.py

# Step 2: Preprocess audio (normalize, trim, resample to 16kHz)
python src/data_collection/preprocess_audio.py

# Step 3: Extract features and build FAISS index
python scripts/build_database.py
```

### 3. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
Female-voice-similarity-search/
├── app/                    # Streamlit web application
├── data/                   # Audio dataset
│   ├── raw/               # Original full-length audio files
│   ├── chunks/            # Segmented audio chunks
│   └── processed/         # Preprocessed audio ready for feature extraction
├── database/              # FAISS index and feature vectors
├── src/                   # Source code
│   ├── data_collection/   # Audio download and preprocessing
│   ├── feature_extraction/# Audio feature extraction
│   ├── search/            # Similarity search implementation
│   ├── utils/             # Utility functions
│   └── vector_database/   # FAISS database management
├── scripts/               # Build and utility scripts
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## How It Works

1. **Audio Preprocessing**: Audio files are normalized, trimmed of silence, and resampled to 16kHz
2. **Feature Extraction**: 52 audio features are extracted from each file:
    - MFCC (Mel-Frequency Cepstral Coefficients) - 26 features
    - Pitch (F0) - 4 features
    - Spectral features (Centroid, Rolloff, Bandwidth) - 6 features
    - Temporal features (ZCR, RMS Energy) - 4 features
    - Chroma features - 12 features
3. **Vector Database**: Features are stored in a FAISS index for fast similarity search
4. **Search**: When a query audio is uploaded, its features are extracted and compared against the database using L2 distance
5. **Results**: The top 5 most similar voices are returned with similarity scores

## Usage

1. Launch the Streamlit application
2. Upload a female voice audio file (WAV, MP3, or FLAC format)
3. View the top 5 similar voices with:
    - Audio playback
    - Waveform visualization
    - Similarity scores
    - Feature analysis and comparison

## Testing

Run tests:

```bash
pytest tests/
```

## Requirements

- Python 3.10 or higher
- At least 2GB of RAM
- Approximately 1GB of disk space for the database

## Dependencies

See `requirements.txt` for the complete list of dependencies.
