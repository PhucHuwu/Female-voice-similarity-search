# Cosine Similarity - Technical Details

## Overview

System sử dụng **Cosine Similarity** thay vì L2 distance để đo độ tương đồng giọng nói.

## Tại sao Cosine Similarity?

### Cosine vs L2 Distance

| Metric                | Range | Ý nghĩa               | Reliability         |
| --------------------- | ----- | --------------------- | ------------------- |
| **L2 Distance**       | 0-∞   | Khoảng cách Euclidean | Phụ thuộc magnitude |
| **Cosine Similarity** | 0-1   | Góc giữa 2 vectors    | Độc lập magnitude   |

### Ví dụ

```
Vector A: [1, 2, 3]     magnitude = √14
Vector B: [2, 4, 6]     magnitude = √56

L2 Distance(A, B) = 3.74  (khác nhau!)
Cosine Similarity(A, B) = 1.0  (giống hệt - cùng direction!)
```

## Implementation

### 1. **Normalization**

Tất cả vectors được L2-normalized trước khi thêm vào FAISS:

```python
# Normalize vector to unit length
faiss.normalize_L2(vectors)
# After: ||vector|| = 1
```

### 2. **FAISS Index**

```python
# Use Inner Product index (after normalization = cosine)
index = faiss.IndexFlatIP(dimension=52)
```

### 3. **Search**

```python
# Normalize query
faiss.normalize_L2(query_vector)

# Search returns inner products
similarities, indices = index.search(query, k=5)

# Inner product of normalized vectors = cosine similarity
# Range: [-1, 1], but audio features → [0, 1]
```

## Mathematical Formula

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

After L2 normalization (||A|| = ||B|| = 1):
cosine_similarity(A, B) = A · B  (inner product)
```

## Interpretation

| Cosine Value | Percentage | Meaning                           |
| ------------ | ---------- | --------------------------------- |
| 1.0          | 100%       | Identical voices                  |
| 0.99-0.95    | 99-95%     | Very similar (same speaker)       |
| 0.95-0.85    | 95-85%     | Similar (similar characteristics) |
| 0.85-0.70    | 85-70%     | Somewhat similar                  |
| < 0.70       | < 70%      | Different voices                  |

## Test Results

### Same File Test

```
Query: yt_9c-yi4vCqZg_chunk0000.wav
Result #1: yt_9c-yi4vCqZg_chunk0000_processed.wav
Similarity: 100.0% (cosine = 1.0000)
```

**Expected:** ~99-100% ✅  
**Result:** 100.0% ✅

### Why not exactly 1.0 sometimes?

Nhỏ sai số có thể do:

1. **Preprocessing khác nhau** giữa chunks và processed
2. **Floating point precision** (32-bit)
3. **Feature extraction** có slight variations

→ Cosine 0.9995-1.0000 vẫn được coi là "same voice"

## Advantages

✅ **Reliable:** Không phụ thuộc vào magnitude của features  
✅ **Interpretable:** 0-100% dễ hiểu  
✅ **Standard:** Được dùng rộng rãi trong speaker recognition  
✅ **Stable:** Không cần tuning constants  
✅ **Mathematical:** Có ý nghĩa hình học rõ ràng (góc giữa vectors)

## Code Locations

- **FAISS Index:** `src/vector_database/faiss_manager.py`
- **Search:** `src/search/similarity_search.py`
- **Build Database:** `scripts/build_database.py`

## References

- FAISS IndexFlatIP: https://github.com/facebookresearch/faiss/wiki
- Cosine Similarity in ML: https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
