# Hệ thống Tìm kiếm Giọng nói Phụ nữ dựa trên Độ tương đồng

## Mục tiêu

- Dataset >= 500 file giọng nói nữ, cùng độ dài chuẩn hóa
- Trích xuất bộ đặc trưng 52 chiều
- Lưu metadata + vectors trong hệ QTCSDL (SQLite)
- Tìm kiếm top-5 file tương đồng nhất theo cosine similarity

## Kiến trúc hiện tại (SQLite-only)

1. Thu thập dữ liệu: `src/data_collection/download_audio.py`
2. Chia chunk 5s + tạo query short/long: `src/data_collection/split_audio_chunks.py`
3. Chuẩn hóa base chunk: `src/data_collection/preprocess_audio.py`
4. Build DB: `scripts/build_database.py`
   - Lưu `database/metadata.db` (metadata + vector BLOB)
   - Lưu `database/features.npy` (backup)
5. Search: `src/search/similarity_search.py` + `src/search/sqlite_vector_search.py`

## Đặc trưng

- MFCC: 26
- Pitch: 4
- Spectral: 6
- Temporal: 4
- Chroma: 12
- Tổng: 52 chiều

## Truy vấn

- Metadata query: `python scripts/search_metadata.py --voice <keyword>`
- Similarity query: upload qua `app/streamlit_app.py`, trả top-5 giảm dần

## Query audio types

- `data/query_short`: query ngắn 5s
- `data/query_long`: query dài 10-20s
- Query dài được xử lý online theo cơ chế multi-segment 5s + aggregate score
