# Kiến trúc Hệ thống Tìm kiếm Giọng nói

## Tổng quan

Hệ thống tìm kiếm giọng nói phụ nữ dựa trên vector đặc trưng 52 chiều và sử dụng SQLite làm hệ quản trị CSDL trung tâm để lưu metadata + feature vectors.

## Sơ đồ Kiến trúc

```
OFFLINE (Build Database)
Raw Audio -> Chunking -> Preprocess -> Feature Extraction (52D)
         -> Save SQLite (metadata + vectors) -> Ready

ONLINE (Search)
Query Audio -> Preprocess -> Feature Extraction (52D)
           -> Cosine Search over SQLite vectors -> Top-5
           -> Streamlit hiển thị kết quả
```

## Thành phần chính

1. `src/data_collection/download_audio.py`
   - Tải audio từ danh sách `data/list_video.csv`
   - Đặt tên theo mẫu `yt_<voice>_<id>.wav`

2. `src/data_collection/split_audio_chunks.py`
   - Chia audio thành chunk 3 giây
   - Tạo metadata chunk ở `data/chunks_metadata.csv`

3. `src/data_collection/preprocess_audio.py`
   - Chuẩn hóa audio về 16kHz, trim silence, normalize, fix length 3 giây
   - Kết quả ở `data/processed/`

4. `src/feature_extraction/extractor.py`
   - Trích xuất 52 features (MFCC, Pitch, Spectral, Temporal, Chroma)

5. `scripts/build_database.py`
   - Build DB từ `data/processed/`
   - Enforce điều kiện >= 500 files
   - Lưu:
     - `database/metadata.db` (SQLite, nguồn dữ liệu chính)
     - `database/features.npy` (backup matrix)

6. `src/vector_database/metadata_db.py`
   - Lớp quản lý SQLite schema + CRUD + metadata search

7. `src/search/sqlite_vector_search.py`
   - Tính cosine similarity trực tiếp trên vectors trong SQLite

8. `src/search/similarity_search.py`
   - Pipeline end-to-end query audio -> top-5 kết quả

## Lược đồ CSDL SQLite

Table: `audio_metadata`

- `vector_idx` (unique)
- `file_path` (unique)
- `source_url`
- `source_video_id`
- `voice_name`
- `chunk_id`
- `sample_rate`
- `duration`
- `feature_dim`
- `feature_vector` (BLOB float32)
- `created_at`

## Luồng dữ liệu

1. Build phase:
   - xử lý hàng loạt audio
   - trích xuất vector 52 chiều
   - lưu metadata + vector vào SQLite

2. Search phase:
   - extract vector cho query
   - duyệt vectors trong SQLite
   - tính cosine similarity
   - trả top-5 giảm dần theo similarity
