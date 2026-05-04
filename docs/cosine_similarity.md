# Cosine Similarity - Technical Details

## Overview

Hệ thống sử dụng cosine similarity để đo độ tương đồng giữa vector đặc trưng giọng nói.

## Formula

```text
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

- Giá trị gần 1.0: rất giống nhau
- Giá trị thấp hơn: khác biệt nhiều hơn

## Current Implementation

- Query vector được trích xuất từ audio đầu vào (52 chiều)
- Vectors tham chiếu được đọc từ SQLite (`audio_metadata.feature_vector`)
- Tính cosine similarity cho từng bản ghi
- Sắp xếp giảm dần và trả top-k (mặc định top-5)

Code location:

- `src/search/sqlite_vector_search.py`
- `src/search/similarity_search.py`

## Interpretation

- 0.95 - 1.00: rất tương đồng
- 0.85 - 0.95: tương đồng cao
- 0.70 - 0.85: tương đồng vừa
- < 0.70: khác biệt đáng kể
