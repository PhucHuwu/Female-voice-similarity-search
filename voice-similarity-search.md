# Hệ thống Tìm kiếm Giọng nói Phụ nữ dựa trên Độ tương đồng

## Mục tiêu

- Dataset >= 500 file giọng nói nữ, cùng độ dài chuẩn hóa
- Trích xuất bộ đặc trưng 52 chiều
- Lưu metadata + vectors trong hệ QTCSDL (SQLite)
- Tìm kiếm top-5 file tương đồng nhất theo cosine similarity

## Kiến trúc hiện tại (SQLite-only)

1. Thu thập dữ liệu: `src/data_collection/download_audio.py`
2. Chia chunk 3s: `src/data_collection/split_audio_chunks.py`
3. Chuẩn hóa: `src/data_collection/preprocess_audio.py`
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

## Ghi chú

- Không sử dụng FAISS trong pipeline hiện tại.
- Hệ thống dùng SQLite làm nguồn dữ liệu chính cho cả metadata và vector.
