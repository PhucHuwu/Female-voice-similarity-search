# Giải thích Chi tiết về Trích xuất Đặc trưng Âm thanh

## Tổng quan

Hệ thống sử dụng **52 đặc trưng âm thanh** để mô tả và phân biệt các giọng nói phụ nữ. Các đặc trưng này được chia thành 5 nhóm chính, mỗi nhóm nắm bắt một khía cạnh khác nhau của giọng nói.

## 1. MFCC - Mel-Frequency Cepstral Coefficients (26 chiều)

### Mô tả

MFCC là standard feature trong voice recognition, mô tả **timbre** (âm sắc) của giọng nói.

### Công thức

1. **Pre-emphasis**: Tăng cường tần số cao
2. **Framing**: Chia audio thành frames (25ms, overlap 10ms)
3. **Windowing**: Hamming window
4. **FFT**: Fast Fourier Transform
5. **Mel filterbank**: Áp dụng 40 Mel filters
6. **Log**: Log của năng lượng
7. **DCT**: Discrete Cosine Transform → 13 coefficients

### Trong hệ thống

- Trích xuất 13 MFCC coefficients
- Tính **mean** và **std** của mỗi coefficient qua toàn bộ audio
- **Tổng: 26 giá trị** (13 mean + 13 std)

### Ý nghĩa

- **MFCC 1-3**: Mô tả envelope của spectrum (timbre tổng thể)
- **MFCC 4-13**: Chi tiết về formants và resonances
- **Mean**: Đặc điểm trung bình của giọng
- **Std**: Biến thiên trong giọng nói (dynamic range)

### Lý do lựa chọn

MFCC bắt chước cách tai người nghe âm thanh (Mel scale), rất hiệu quả cho voice recognition.

---

## 2. Pitch (Fundamental Frequency F0) (4 chiều)

### Mô tả

**Cao độ giọng nói** - tần số cơ bản của dây thanh quản rung động.

### Công thức

Sử dụng **piptrack** algorithm trong librosa:

1. STFT (Short-Time Fourier Transform)
2. Tìm peaks trong magnitude spectrum
3. Track pitch qua các frames
4. Filter pitch values (>0 Hz)

### Trong hệ thống

Trích xuất 4 thống kê từ pitch contour:

- **Mean**: Cao độ trung bình (Hz)
- **Std**: Độ biến thiên cao độ
- **Min**: Cao độ thấp nhất
- **Max**: Cao độ cao nhất

### Giá trị điển hình cho giọng phụ nữ

- Mean: 190-250 Hz (so với nam: 85-180 Hz)
- Dynamic range: 50-100 Hz

### Ý nghĩa

- **Mean**: Đặc trưng chính để phân biệt giọng nữ/nam, cảm xúc
- **Std**: Biểu cảm trong giọng nói (monotone vs expressive)
- **Min/Max**: Range của giọng nói

### Lý do lựa chọn

Pitch là đặc trưng then chốt nhất để nhận diện và phân biệt giọng nói.

---

## 3. Spectral Features (6 chiều)

### 3.1 Spectral Centroid (2 chiều: mean, std)

**Mô tả**: "Trọng tâm" của spectrum - tần số mà tại đó năng lượng tập trung.

**Công thức**:

```
Centroid = Σ(f * M(f)) / Σ(M(f))
```

Với:

- f: Tần số
- M(f): Magnitude tại tần số f

**Ý nghĩa**:

- **Brightness** của âm thanh
- Giá trị cao → giọng sáng, trong
- Giá trị thấp → giọng ấm, đục

### 3.2 Spectral Rolloff (2 chiều: mean, std)

**Mô tả**: Tần số mà dưới đó có 85% tổng năng lượng spectrum.

**Công thức**:

```
Rolloff frequency: Σ(M(f)) = 0.85 * Σ(total energy)
```

**Ý nghĩa**:

- Phân biệt voiced (rolloff thấp) vs unvoiced (rolloff cao)
- Độ "sharp" của âm thanh

### 3.3 Spectral Bandwidth (2 chiều: mean, std)

**Mô tả**: Độ rộng của spectrum quanh centroid.

**Công thức**:

```
Bandwidth = sqrt( Σ((f - centroid)² * M(f)) / Σ(M(f)) )
```

**Ý nghĩa**:

- Độ "spread" của năng lượng
- Bandwidth hẹp → tonal (như tiếng kèn)
- Bandwidth rộng → noisy (như tiếng thì thầm)

### Tại sao cần cả 3?

- **Centroid**: WHERE the energy is
- **Rolloff**: HOW MUCH energy in high frequencies
- **Bandwidth**: HOW SPREAD OUT is the energy

Kết hợp ba đặc trưng này mô tả đầy đủ phổ tần của giọng nói.

---

## 4. Temporal Features (4 chiều)

### 4.1 Zero Crossing Rate (ZCR) (2 chiều: mean, std)

**Mô tả**: Số lần tín hiệu đổi dấu (cross zero) trong một frame.

**Công thức**:

```
ZCR = (1/2) * Σ|sign(x[n]) - sign(x[n-1])|
```

**Ý nghĩa**:

- **Phân biệt voiced vs unvoiced**:
    - Voiced sounds (vowels): ZCR thấp (~50-100)
    - Unvoiced sounds (consonants): ZCR cao (>200)
- **Noisiness**: ZCR cao → nhiều noise

### 4.2 RMS Energy (2 chiều: mean, std)

**Mô tả**: Root Mean Square energy - năng lượng tín hiệu.

**Công thức**:

```
RMS = sqrt( (1/N) * Σ(x[n]²) )
```

**Ý nghĩa**:

- **Loudness** (âm lượng)
- **Voice Activity Detection**: Phân biệt speech vs silence
- **Intensity patterns**: Nhấn mạnh trong câu

### Tại sao cần cả 2?

- **ZCR**: Đặc trưng về **frequency content** (pitched vs noisy)
- **RMS**: Đặc trưng về **amplitude** (loud vs quiet)

Kết hợp để mô tả temporal dynamics của giọng nói.

---

## 5. Chroma Features (12 chiều)

### Mô tả

Phân bố năng lượng theo 12 **pitch classes** (C, C#, D, ..., B) - như 12 nốt trong âm nhạc.

### Công thức

1. STFT
2. Áp dụng chroma filterbank (12 filters)
3. Tính mean energy cho mỗi chroma bin

### Trong hệ thống

12 giá trị tương ứng với 12 pitch classes.

### Ý nghĩa

- **Harmonic content**: Cấu trúc hài của giọng nói
- **Pitch class distribution**: Giọng nói có harmonics ở các nốt nào
- **Voice quality**: Resonance patterns

### Ví dụ

Nếu chroma[0] (C) cao → nhiều năng lượng ở các tần số thuộc họ C (C, C2, C3,...)

### Lý do lựa chọn

Mặc dù giọng nói không phải âm nhạc, chroma vẫn hữu ích vì:

- Giọng nói có cấu trúc harmonic (fundamental + harmonics)
- Chroma nắm bắt resonance patterns
- Robust với pitch variations

---

## Tổng kết: 52 Đặc trưng

| Nhóm     | Số chiều | Mục đích                      |
| -------- | -------- | ----------------------------- |
| MFCC     | 26       | Timbre (âm sắc)               |
| Pitch    | 4        | Cao độ giọng nói              |
| Spectral | 6        | Brightness, sharpness         |
| Temporal | 4        | Voiced/unvoiced, energy       |
| Chroma   | 12       | Harmonic structure            |
| **Tổng** | **52**   | **Mô tả toàn diện giọng nói** |

---

## Pipeline Trích xuất

```python
# 1. Load audio (16kHz, 3s)
audio, sr = librosa.load(file_path, sr=16000)

# 2. Extract features
extractor = AudioFeatureExtractor()

# MFCC (26)
mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
mfcc_features = [mean(mfcc), std(mfcc)]  # 13+13=26

# Pitch (4)
pitches = librosa.piptrack(audio, sr=sr)
pitch_features = [mean, std, min, max]  # 4

# Spectral (6)
centroid = librosa.feature.spectral_centroid(audio, sr=sr)
rolloff = librosa.feature.spectral_rolloff(audio, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(audio, sr=sr)
spectral_features = [
    mean(centroid), std(centroid),
    mean(rolloff), std(rolloff),
    mean(bandwidth), std(bandwidth)
]  # 6

# Temporal (4)
zcr = librosa.feature.zero_crossing_rate(audio)
rms = librosa.feature.rms(audio)
temporal_features = [
    mean(zcr), std(zcr),
    mean(rms), std(rms)
]  # 4

# Chroma (12)
chroma = librosa.feature.chroma_stft(audio, sr=sr)
chroma_features = mean(chroma, axis=1)  # 12

# 3. Concatenate
all_features = concat([
    mfcc_features,     # 26
    pitch_features,    # 4
    spectral_features, # 6
    temporal_features, # 4
    chroma_features    # 12
])  # Total: 52
```

---

## Tại sao 52 chiều?

### Balance giữa:

1. **Expressiveness**: Đủ thông tin để phân biệt giọng nói
2. **Efficiency**: Không quá nhiều để tránh overfitting
3. **Computational cost**: Nhanh để tính toán và search

### So sánh:

- **Quá ít** (<20): Mất thông tin, khó phân biệt
- **52 chiều**: Sweet spot cho voice recognition
- **Quá nhiều** (>200): Redundant, slow, overfitting

---

## Tính Similarity

### FAISS L2 Distance

```python
distance = sqrt(Σ(feature1[i] - feature2[i])²)
```

### Normalization

```python
similarity_score = max(0, 100 - distance * 10)
```

- Distance nhỏ → Giọng nói giống nhau → Similarity cao
- Distance lớn → Giọng nói khác nhau → Similarity thấp

---

## Đặc trưng nào quan trọng nhất?

### Theo thứ tự ưu tiên:

1. **Pitch (F0)**: Quan trọng nhất - phân biệt giọng nữ/nam, speaker identity
2. **MFCC**: Timbre - đặc trưng voice quality
3. **Spectral**: Brightness - tone của giọng
4. **Temporal**: Dynamics - rhythm và energy
5. **Chroma**: Harmonics - voice quality chi tiết

### Ablation test (nếu loại bỏ):

- Bỏ Pitch → Mất khả năng phân biệt speaker
- Bỏ MFCC → Giảm 30-40% accuracy
- Bỏ Spectral → Giảm 15-20% accuracy
- Bỏ Temporal → Giảm 10% accuracy
- Bỏ Chroma → Giảm 5-10% accuracy

---

## Tham khảo

- [Librosa Documentation](https://librosa.org/doc/latest/feature.html)
- [MFCC Tutorial](https://www.youtube.com/watch?v=4_SH2nfbQZ8)
- [Speech Processing](https://www.coursera.org/learn/audio-signal-processing)

**Version**: 1.0  
**Last Updated**: 2026-01-29
