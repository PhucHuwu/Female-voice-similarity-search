# Data Pipeline Documentation

## Folder Structure

```
data/
├── raw/              # Original full-length audio (backup)
│   ├── yt_*.wav     # 6 YouTube videos (full length, ~7.8 GB)
│   └── [BACKUP ONLY - not processed]
│
├── chunks/           # 3-second audio chunks
│   ├── yt_*_chunk0000.wav  # 600 chunks (100 per video)
│   └── [Intermediate - input for preprocessing]
│
└── processed/        # Standardized audio ready for ML
    ├── yt_*_chunk0000_processed.wav  # 600 files
    └── [Final - used for feature extraction]
```

## Pipeline Steps

### 1. Download YouTube Audio

**Script:** `src/data_collection/download_real_audio.py`

```bash
python src/data_collection/download_real_audio.py
```

**Output:** `data/raw/yt_*.wav` (6 full videos)

**What it does:**

- Downloads audio from YouTube URLs
- Saves as WAV format
- Original sample rate and duration preserved

---

### 2. Split into Chunks

**Script:** `src/data_collection/split_audio_chunks.py`

```bash
python src/data_collection/split_audio_chunks.py
```

**Input:** `data/raw/*.wav` (6 files)  
**Output:** `data/chunks/*.wav` (600 files)

**What it does:**

- Splits each video into 100 × 3-second chunks
- No overlap between chunks
- Skips incomplete chunks at the end
- Saves metadata to `data/chunks_metadata.csv`

---

### 3. Preprocess Chunks

**Script:** `src/data_collection/preprocess_audio.py`

```bash
python src/data_collection/preprocess_audio.py
```

**Input:** `data/chunks/*.wav` (600 files)  
**Output:** `data/processed/*.wav` (600 files)

**What it does:**

- **Resample** to 16,000 Hz (speech recognition standard)
- **Convert** to mono
- **Trim silence** (threshold: 20dB)
- **Normalize volume** (peak normalization)
- **Fix duration** to exactly 3.0 seconds (crop or pad)
- Saves metadata to `data/processed_metadata.csv`

---

### 4. Build Database

**Script:** `scripts/build_database.py`

```bash
python scripts/build_database.py
```

**Input:** `data/processed/*.wav` (600 files)  
**Output:**

- `database/features.npy` - Feature vectors (600 × 52)
- `database/index_mapping.json` - Vector ID → file path
- `database/vectors/faiss_index.bin` - FAISS index

**What it does:**

- Extracts 52-dimensional features from each audio
- Creates FAISS IndexFlatL2 for L2 distance search
- Maps vector indices to original file paths

---

## Quick Start (Windows)

### One-Command Pipeline

```bash
build_database.bat
```

This runs all 3 steps automatically:

1. Split videos → chunks
2. Preprocess chunks → processed
3. Build FAISS database

### Manual Step-by-Step

```bash
# 1. Download (if needed)
python src/data_collection/download_real_audio.py

# 2. Chunk
python src/data_collection/split_audio_chunks.py

# 3. Preprocess
python src/data_collection/preprocess_audio.py

# 4. Build database
python scripts/build_database.py
```

---

## Data Flow Diagram

```
YouTube Videos
     ↓
[download_real_audio.py]
     ↓
data/raw/ (6 full videos, 7.8 GB)
     ↓
[split_audio_chunks.py]
     ↓
data/chunks/ (600 chunks, 3s each)
     ↓
[preprocess_audio.py]
     ↓
data/processed/ (600 standardized, 55 MB)
     ↓
[build_database.py]
     ↓
database/vectors/faiss_index.bin (600 vectors, 52D)
     ↓
[similarity_search.py]
     ↓
Top-5 Results
```

---

## File Specifications

### data/raw/

- **Format:** WAV, variable sample rate
- **Duration:** Variable (60s - 27 minutes)
- **Size:** 135 MB - 273 MB per file
- **Total:** ~7.8 GB

### data/chunks/

- **Format:** WAV, variable sample rate
- **Duration:** 3 seconds each
- **Size:** ~90 KB per file
- **Total:** ~54 MB

### data/processed/

- **Format:** WAV, 16kHz mono
- **Duration:** Exactly 3.000 seconds
- **Size:** ~96 KB per file (all uniform)
- **Total:** ~55 MB

---

## Troubleshooting

### "No audio files found"

- Check that `data/raw/` contains yt\_\*.wav files
- Run download script first

### "Chunking creates < 600 files"

- Some videos might be shorter
- Check `data/chunks_metadata.csv` for details

### "Database build fails"

- Ensure `data/processed/` has 600 files
- Run preprocessing step again
- Check for corrupted audio files

---

## Clean Start

To rebuild from scratch:

```bash
# Remove intermediate files
Remove-Item data\chunks\* -Force
Remove-Item data\processed\* -Force
Remove-Item database\features.npy
Remove-Item database\index_mapping.json
Remove-Item database\vectors\faiss_index.bin

# Rebuild
build_database.bat
```

**Note:** Keep `data/raw/` as backup!
