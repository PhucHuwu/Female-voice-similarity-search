# Kiến trúc Hệ thống Tìm kiếm Giọng nói

## Tổng quan

Hệ thống tìm kiếm giọng nói phụ nữ dựa trên similarity search với vector embeddings, sử dụng FAISS cho tìm kiếm nhanh.

## Sơ đồ Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                     OFFLINE PHASE                            │
│  (Xây dựng database - chạy 1 lần)                           │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Data        │   │  Preprocess  │   │  Feature     │
│  Collection  │──▶│  Audio       │──▶│  Extraction  │
│              │   │  16kHz, 3s   │   │  52-dim      │
└──────────────┘   └──────────────┘   └──────────────┘
                                              │
                                              ▼
                                     ┌──────────────┐
                                     │  Build       │
                                     │  FAISS Index │
                                     └──────────────┘
                                              │
                                              ▼
                                     ┌──────────────┐
                                     │  Save Index  │
                                     │  + Mapping   │
                                     └──────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     ONLINE PHASE                             │
│  (Similarity search - real-time)                            │
└─────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  User Upload │
    │  Audio File  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Preprocess  │
    │  Audio       │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Extract     │
    │  Features    │
    │  (52-dim)    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  FAISS       │
    │  Search      │
    │  (L2)        │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Top-5       │
    │  Results     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Streamlit   │
    │  Display     │
    └──────────────┘
```

## Components

### 1. Data Collection (`src/data_collection/`)

**download_audio.py**

- Download từ HuggingFace Datasets (Common Voice, VoxCeleb)
- Download từ YouTube với yt-dlp
- Lưu metadata (file path, duration, source)

**preprocess_audio.py**

- Resample tất cả audio về 16kHz
- Trim silence (20dB threshold)
- Normalize volume
- Fix length về 3 giây (crop hoặc pad)

### 2. Feature Extraction (`src/feature_extraction/`)

**extractor.py** - AudioFeatureExtractor class

Trích xuất 52 đặc trưng:

- **MFCC** (26): Mean + Std của 13 coefficients → Voice timbre
- **Pitch** (4): Mean, Std, Min, Max F0 → Cao độ giọng
- **Spectral** (6): Centroid, Rolloff, Bandwidth → Brightness, tone
- **Temporal** (4): ZCR, RMS Energy → Voiced/Unvoiced
- **Chroma** (12): 12 pitch classes → Harmonic content

**features_config.py**

- Cấu hình tham số (sample rate, FFT size, hop length, pitch range)

### 3. Vector Database (`src/vector_database/`)

**faiss_manager.py** - FAISSManager class

- Create index: IndexFlatL2 (exact L2 distance search)
- Add vectors: Thêm batch vectors + mapping
- Search: Tìm k-nearest neighbors
- Save/Load: Lưu index và mapping vào disk

Database files:

- `faiss_index.bin` - FAISS index (binary)
- `index_mapping.json` - {vector_id: audio_file_path}
- `features.npy` - Feature vectors array (optional backup)

### 4. Similarity Search (`src/search/`)

**similarity_search.py** - VoiceSimilaritySearch class

Pipeline:

1. Load audio file
2. Preprocess (normalize, trim, fix length)
3. Extract features (52-dim vector)
4. Query FAISS index (L2 distance)
5. Return top-k results with similarity scores

### 5. Streamlit App (`app/`)

**streamlit_app.py**

UI features:

- File uploader (WAV, MP3, FLAC)
- Audio player cho query và results
- Waveform visualization với librosa
- Similarity score display (0-100%)
- System statistics sidebar

### 6. Utilities (`src/utils/`)

**audio_utils.py**

- load_audio(): Load và resample
- trim_silence(): Remove silence
- normalize_audio(): Peak normalization
- fix_length(): Crop hoặc pad
- preprocess_audio(): Complete pipeline

## Data Flow

### Offline (Build Database)

```
Raw Audio Files (500+)
    │
    ├─▶ Preprocess
    │   ├─ Resample to 16kHz
    │   ├─ Trim silence
    │   ├─ Normalize
    │   └─ Fix length to 3s
    │
    ├─▶ Feature Extraction
    │   ├─ MFCC (26 dims)
    │   ├─ Pitch (4 dims)
    │   ├─ Spectral (6 dims)
    │   ├─ Temporal (4 dims)
    │   └─ Chroma (12 dims)
    │   = 52-dim vector
    │
    └─▶ FAISS Index
        ├─ Build L2 index
        ├─ Add all vectors
        └─ Save index + mapping
```

### Online (Search)

```
Query Audio
    │
    ├─▶ Preprocess (same as offline)
    │
    ├─▶ Feature Extraction (52-dim)
    │
    ├─▶ FAISS Search (L2 distance)
    │   └─ Find k=5 nearest neighbors
    │
    └─▶ Results
        ├─ Map vector IDs to file paths
        ├─ Calculate similarity scores
        └─ Display in Streamlit
```

## Performance

### Time Complexity

- **Feature Extraction**: O(n) với n = audio length (~0.5s cho 3s audio)
- **FAISS Search**: O(N) với N = database size (IndexFlatL2 exact search)
- **Total Query Time**: < 1s cho database ~1000 voices

### Space Complexity

- **Feature storage**: 52 dims × 4 bytes × N samples
    - 1000 samples = ~200KB
    - 10000 samples = ~2MB
- **FAISS Index**: Gần bằng feature storage (IndexFlatL2)
- **Audio files**: Tùy quality (16kHz, 3s ≈ 300KB/file)

## Scaling Options

### Tăng Dataset Size (>10K voices)

1. **FAISS IndexIVFFlat** (approximate search)
    - Train với clustering
    - Faster search, slight accuracy trade-off

2. **Pinecone** (cloud vector DB)
    - Managed service
    - Auto-scaling
    - Hybrid search capabilities

### Tăng Feature Dimensions

- Thêm mel-spectrogram features
- Thêm formant features
- Sử dụng pre-trained models (Wav2Vec 2.0, ECAPA-TDNN)

## Dependencies

### Core Libraries

- **librosa**: Audio feature extraction
- **soundfile**: Audio I/O
- **pydub**: Audio conversion
- **numpy**: Array operations
- **faiss-cpu**: Vector similarity search

### Web Interface

- **streamlit**: Web UI
- **matplotlib**: Waveform plotting
- **pandas**: Metadata management

### Data Collection

- **datasets**: HuggingFace datasets
- **yt-dlp**: YouTube download

## Configuration

Tất cả config constants trong `src/feature_extraction/features_config.py`:

```python
SAMPLE_RATE = 16000
TARGET_DURATION = 3.0
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
PITCH_FMIN = 85  # Female voice range
PITCH_FMAX = 300
```

## Error Handling

- **Invalid audio**: Skip và log trong metadata
- **Feature extraction fail**: NaN/Inf check, skip nếu invalid
- **FAISS search fail**: Return empty results với error message
- **Missing index**: Streamlit hiển thị hướng dẫn build database

## Future Improvements

1. **Model-based features**: Thay MFCC bằng Wav2Vec 2.0 embeddings
2. **Real-time recording**: Streamlit audio recorder widget
3. **Batch search**: Upload nhiều files cùng lúc
4. **Evaluation metrics**: Precision@K, Recall@K với ground truth
5. **Speaker diarization**: Phân biệt nhiều người trong 1 file
6. **Voice conversion**: Chuyển đổi giọng nói giữa các speakers

---

**Version**: 1.0  
**Last Updated**: 2026-01-29
