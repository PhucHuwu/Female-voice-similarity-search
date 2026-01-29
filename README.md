# Hệ thống Tìm kiếm Giọng nói Phụ nữ dựa trên Độ tương đồng

Hệ thống tìm kiếm âm thanh giọng nói phụ nữ sử dụng **similarity search** với vector embeddings. Hệ thống nhận đầu vào là file âm thanh giọng phụ nữ, trả về 5 file âm thanh tương đồng nhất theo thứ tự giảm dần.

## 🎯 Tính năng

- Thu thập dataset giọng nói phụ nữ (HuggingFace, YouTube)
- Trích xuất 52 đặc trưng âm thanh (MFCC, pitch, spectral, temporal, chroma)
- Lưu trữ vector embeddings với FAISS
- Tìm kiếm similarity với độ chính xác cao
- Giao diện web Streamlit đẹp mắt, dễ sử dụng
- Hiển thị dạng sóng âm thanh

## 🛠️ Tech Stack

- **Backend:** Python 3.10+
- **Frontend:** Streamlit
- **Audio Processing:** librosa, soundfile, pydub
- **Vector DB:** FAISS (local, miễn phí)
- **Feature Extraction:** MFCC, Pitch (F0), Spectral features, ZCR, RMS Energy, Chroma
- **Environment:** Conda

## 📦 Cài đặt

### 1. Tạo Conda Environment

```bash
conda env create -f environment.yml
conda activate voice-search
```

Hoặc sử dụng pip:

```bash
conda create -n voice-search python=3.10 -y
conda activate voice-search
pip install -r requirements.txt
```

### 2. Cấu trúc thư mục

Project sẽ tự động tạo các thư mục cần thiết khi chạy scripts.

```
Female-voice-similarity-search/
├── data/
│   ├── raw/              # Audio files gốc
│   └── processed/        # Audio đã xử lý
├── database/
│   ├── vectors/          # FAISS index
│   ├── features.npy      # Feature vectors
│   └── index_mapping.json
├── src/                  # Source code modules
├── app/                  # Streamlit app
└── scripts/              # Build scripts
```

## 🚀 Hướng dẫn sử dụng

### Bước 1: Thu thập dữ liệu

```bash
# Download 50 sample audio files từ Mozilla Common Voice
python src/data_collection/download_audio.py
```

**Hoặc** để download nhiều hơn, chỉnh sửa trong file:

```python
download_sample_dataset(num_samples=500)  # Tải 500 files
```

### Bước 2: Tiền xử lý audio

```bash
# Chuẩn hóa audio: 16kHz, 3 giây, trim silence
python src/data_collection/preprocess_audio.py
```

### Bước 3: Build database

```bash
# Trích xuất features và tạo FAISS index
python scripts/build_database.py
```

Output:

- `database/features.npy` - Feature vectors (N × 52)
- `database/index_mapping.json` - Mapping vector ID → file path
- `database/vectors/faiss_index.bin` - FAISS index

### Bước 4: Chạy Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Mở trình duyệt tại `http://localhost:8501`

## Đặc trưng âm thanh (52 dimensions)

| Feature      | Số chiều | Mô tả                                        |
| ------------ | -------- | -------------------------------------------- |
| **MFCC**     | 26       | Mel-Frequency Cepstral Coefficients (timbre) |
| **Pitch**    | 4        | Fundamental frequency (mean, std, min, max)  |
| **Spectral** | 6        | Centroid, Rolloff, Bandwidth (mean + std)    |
| **Temporal** | 4        | Zero Crossing Rate, RMS Energy (mean + std)  |
| **Chroma**   | 12       | 12 pitch class energy distribution           |

**Tổng:** 52 features

## 🎨 Sử dụng Streamlit App

1. **Tải lên file âm thanh** giọng phụ nữ (WAV, MP3, FLAC)
2. **Chọn số kết quả** (1-10, mặc định 5)
3. **Xem kết quả:**
    - Top-5 giọng nói tương đồng nhất
    - Độ tương đồng (0-100%)
    - Audio player cho từng kết quả
    - Dạng sóng âm thanh (waveform)

## 📖 Cấu trúc Code

### Core Modules

- **`src/utils/audio_utils.py`** - Audio I/O, preprocessing utilities
- **`src/feature_extraction/extractor.py`** - Feature extraction class
- **`src/vector_database/faiss_manager.py`** - FAISS index management
- **`src/search/similarity_search.py`** - Search pipeline
- **`app/streamlit_app.py`** - Web UI

### Scripts

- **`src/data_collection/download_audio.py`** - Download dataset
- **`src/data_collection/preprocess_audio.py`** - Audio preprocessing
- **`scripts/build_database.py`** - Build FAISS database

## Tùy chỉnh

### Thay đổi feature extraction

Chỉnh sửa `src/feature_extraction/features_config.py`:

```python
N_MFCC = 20  # Tăng số MFCC coefficients
SAMPLE_RATE = 22050  # Thay đổi sample rate
TARGET_DURATION = 5.0  # Audio dài hơn
```

### Thay đổi vector database

Thay FAISS bằng Pinecone (cloud):

- Uncomment code trong `src/vector_database/pinecone_manager.py`
- Thêm API key vào `.env`

## 📈 Đánh giá kết quả

Chạy manual evaluation:

```python
# Test với 10 query samples
from src.search.similarity_search import search_similar

results = search_similar("path/to/test_audio.wav", top_k=5)
for file_path, similarity, distance in results:
    print(f"{file_path}: {similarity:.1f}%")
```

## 🐛 Troubleshooting

**Lỗi: "Index file not found"**
→ Chạy `python scripts/build_database.py` trước

**Lỗi: "No audio files found"**
→ Chạy data collection và preprocessing trước

**Lỗi: librosa import error**
→ Cài đặt lại: `pip install librosa soundfile`

## 📚 Tài liệu tham khảo

- [Plan chi tiết](voice-similarity-search.md)
- [Yêu cầu đề bài](require.md)

## 🤝 Đóng góp

Dự án học thuật - Voice Similarity Search System

## 📄 License

MIT License
