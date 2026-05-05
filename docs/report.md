# BÁO CÁO BÀI TẬP LỚN

## Học phần: Hệ cơ sở dữ liệu đa phương tiện

## Đề tài: Xây dựng hệ CSDL lưu trữ và tìm kiếm giọng nói phụ nữ dựa trên độ tương đồng đặc trưng âm thanh

---

## PHẦN CHIA CÔNG VIỆC

Do repository hiện tại không đính kèm thông tin phân công theo thành viên, phần này được để ở dạng khung để nhóm bổ sung theo tình hình thực tế:

- Thành viên 1: Thu thập dữ liệu, xử lý âm thanh ban đầu.
- Thành viên 2: Trích xuất đặc trưng, xây dựng pipeline chuyển đổi đặc trưng.
- Thành viên 3: Thiết kế CSDL SQLite, lưu metadata và vector.
- Thành viên 4: Xây dựng module tìm kiếm, đánh giá hệ thống, giao diện demo.

---

## CHƯƠNG I: GIỚI THIỆU

### 1.1 Tổng quan về hệ thống tìm kiếm âm thanh

Sự phát triển mạnh của các nền tảng video, podcast và mạng xã hội đã tạo ra lượng lớn dữ liệu âm thanh phi cấu trúc. Khác với dữ liệu văn bản có thể tìm bằng từ khóa, dữ liệu giọng nói đòi hỏi cơ chế tìm kiếm theo nội dung âm học. Vì vậy, các hệ thống truy hồi âm thanh dựa trên đặc trưng (content-based audio retrieval) ngày càng quan trọng trong những bài toán như tìm giọng tương tự, lọc dữ liệu theo người nói, hỗ trợ gắn nhãn bán tự động và tiền xử lý cho các hệ nhận dạng nâng cao.

Trong bối cảnh đó, đề tài xây dựng một hệ thống tìm kiếm giọng nói phụ nữ với đầu vào là file truy vấn và đầu ra là danh sách top-5 mẫu giống nhất trong cơ sở dữ liệu. Trọng tâm của hệ thống không nằm ở nhận dạng ngôn ngữ hay phiên âm nội dung lời nói, mà tập trung vào mức độ tương đồng về đặc tính giọng: âm sắc, cao độ, phân bố phổ và động học tín hiệu. Cách tiếp cận này phù hợp với các tình huống cần so sánh chất giọng ngay cả khi nội dung phát âm khác nhau.

Về nguyên lý, hệ thống vận hành theo chuỗi xử lý: âm thanh đầu vào được chuẩn hóa, trích xuất thành vector đặc trưng 52 chiều, sau đó ánh xạ vào không gian truy hồi để so khớp với các vector đã lưu trong CSDL. Độ giống được đo bằng cosine similarity, kết hợp cơ chế xếp hạng giảm dần để trả về top-k ứng viên. Đối với query dài, hệ thống sử dụng phân đoạn nhiều cửa sổ và tổng hợp điểm nhằm tăng độ ổn định trước nhiễu cục bộ.

Về mặt kiến trúc dữ liệu, đề tài triển khai mô hình CSDL đa phương tiện theo hướng lai: dữ liệu âm thanh lưu trên hệ thống tệp, còn metadata và vector đặc trưng lưu trong SQLite. Thiết kế này bảo đảm đồng thời ba yêu cầu: (i) truy vết nguồn dữ liệu rõ ràng qua metadata, (ii) truy hồi nhanh theo vector nội dung, và (iii) thuận tiện tái lập thực nghiệm khi thay đổi thuật toán đặc trưng hoặc chiến lược tìm kiếm.

Từ góc độ học thuật, hệ thống là một minh họa cụ thể cho việc tích hợp giữa xử lý tín hiệu số và quản trị dữ liệu đa phương tiện: tín hiệu thô được biến đổi thành biểu diễn có cấu trúc, lưu trữ có tổ chức và truy vấn định lượng bằng độ đo hình học trong không gian đặc trưng.

### 1.2 Mục tiêu nghiên cứu

Mục tiêu của đề tài bao gồm:

- Xây dựng bộ dữ liệu giọng nói phụ nữ đạt ngưỡng tối thiểu 500 file.
- Chuẩn hóa dữ liệu âm thanh về cùng sample rate và độ dài cho tập nền.
- Trích xuất bộ đặc trưng 52 chiều phản ánh các khía cạnh âm sắc, cao độ, phổ tần và động học tín hiệu.
- Xây dựng hệ CSDL lưu metadata và vector đặc trưng phục vụ tìm kiếm.
- Thực hiện truy hồi top-5 dựa trên độ đo cosine similarity.
- Đánh giá hiệu năng hệ thống trên tập query ngắn và query dài.

### 1.3 Phạm vi đề tài

Phạm vi triển khai trong đồ án gồm:

- Đối tượng: giọng nói phụ nữ (nguồn audio trích xuất từ YouTube).
- Định dạng chủ đạo: WAV sau các bước xử lý.
- CSDL: SQLite đóng vai trò hệ quản trị metadata và vector.
- Phương pháp tìm kiếm: duyệt vector trong CSDL và tính cosine similarity.

Đề tài chưa tập trung vào nhận dạng nội dung ngôn ngữ, nhận dạng người nói đa ngữ cảnh lớn, hoặc tối ưu hóa cho quy mô dữ liệu rất lớn (hàng trăm nghìn đến hàng triệu mẫu).

---

## CHƯƠNG II: CƠ SỞ LÝ THUYẾT

### 2.1 Xử lý tín hiệu âm thanh

#### 2.1.1 Đặc trưng âm thanh là gì?

Đặc trưng âm thanh (audio features) là tập giá trị số học đại diện cho tính chất vật lý và cảm nhận của tín hiệu âm thanh. Thay vì so sánh trực tiếp dạng sóng theo từng mẫu (sample), hệ thống so sánh trong không gian đặc trưng để:

- Giảm nhiễu và biến động không cần thiết.
- Tăng khả năng biểu diễn bản chất giọng nói.
- Hỗ trợ tính toán tương đồng hiệu quả hơn.

#### 2.1.2 Các kỹ thuật trích xuất đặc trưng

Hệ thống sử dụng các nhóm đặc trưng kinh điển trong xử lý tiếng nói: MFCC, Pitch, Spectral, Temporal và Chroma. Các đặc trưng này được trích xuất bằng thư viện `librosa`, sau đó tổng hợp thành vector 52 chiều cho mỗi mẫu âm thanh.

### 2.2 Các thuộc tính nhận diện giọng nói

#### 2.2.1 MFCC (Mel-Frequency Cepstral Coefficients)

MFCC mô tả âm sắc (timbre), là nhóm đặc trưng cốt lõi trong nhận dạng tiếng nói. Trong hệ thống:

- Trích xuất 13 hệ số MFCC theo frame.
- Tính mean và std cho từng hệ số.
- Tổng cộng 26 chiều.

MFCC giúp phân biệt chất giọng, độ vang, độ dày và đặc tính cộng hưởng của đường phát âm.

#### 2.2.2 Pitch (Tần số cơ bản)

Pitch phản ánh cao độ cơ bản F0 của giọng nói. Hệ thống tính 4 thống kê: mean, std, min, max của contour pitch (từ `librosa.piptrack`). Nhóm này rất quan trọng trong bài toán giọng nói phụ nữ vì khác biệt cao độ giữa các cá nhân và cách biểu cảm giọng nói.

#### 2.2.3 Spectral Features

Bao gồm 6 chiều:

- Spectral centroid (mean, std): độ sáng phổ.
- Spectral rolloff (mean, std): giới hạn năng lượng tần số cao.
- Spectral bandwidth (mean, std): độ phân tán năng lượng quanh centroid.

Nhóm này bổ sung thông tin về màu sắc phổ tần, giúp phân biệt các giọng có âm sắc gần nhau nhưng chất phổ khác nhau.

#### 2.2.4 Zero Crossing Rate

Zero Crossing Rate (ZCR) nằm trong nhóm temporal, đo tần suất tín hiệu đổi dấu. Kèm theo RMS energy, nhóm temporal (4 chiều) mô tả tính hữu thanh/vô thanh, độ ồn và mức năng lượng trong phát âm.

#### 2.2.5 Lý do lựa chọn từng thuộc tính

Bộ thuộc tính 52 chiều được chọn theo nguyên tắc bổ sung lẫn nhau:

- MFCC: giữ vai trò mô tả âm sắc cốt lõi.
- Pitch: nhấn mạnh bản sắc cao độ của người nói.
- Spectral: bổ sung cấu trúc năng lượng theo tần số.
- Temporal: nắm bắt động học theo thời gian.
- Chroma: khai thác cấu trúc harmonic.

Sự kết hợp này cân bằng giữa tính biểu diễn và chi phí tính toán, phù hợp cho bài toán truy hồi top-k.

### 2.3 Độ đo tương đồng giọng nói

#### 2.3.1 Cosine Similarity

Độ đo chính của hệ thống là cosine similarity:

cos(a,b) = (a.b) / (||a|| ||b||)

Ưu điểm:

- Không quá nhạy với độ lớn tuyệt đối của vector.
- Phù hợp khi vector đã được chuẩn hóa bằng `StandardScaler`.
- Tính nhanh, dễ diễn giải.

Trong mã nguồn, hệ thống tính cosine cho mỗi vector trong CSDL và sắp xếp giảm dần để lấy top-5.

#### 2.3.2 Euclidean Distance

Khoảng cách Euclidean là phương án thay thế phổ biến, đo độ lệch tuyệt đối giữa hai vector. Tuy nhiên, với bài toán này, cosine thường ổn định hơn khi hình dạng vector quan trọng hơn biên độ, đặc biệt sau các bước chuẩn hóa.

#### 2.3.3 DTW (Dynamic Time Warping)

DTW phù hợp cho so khớp chuỗi thời gian không đồng bộ. Dù có ưu điểm trong so khớp tiết tấu, DTW có chi phí tính toán cao hơn đáng kể so với cosine trên vector tổng hợp. Trong phạm vi đề tài, DTW được xem là hướng mở rộng thay vì thành phần mặc định.

---

## CHƯƠNG III: DỮ LIỆU

### 3.1 Thu thập/Xây dựng bộ dữ liệu

#### 3.1.1 Nguồn dữ liệu (500+ files)

Theo thống kê trực tiếp trên `database/metadata.db`, tập nền đang có 828 bản ghi (dữ liệu đã preprocess và trích đặc trưng), vượt yêu cầu tối thiểu 500 file. Dữ liệu gốc gồm 44 file audio dài (nguồn YouTube), sau khi cắt đoạn 5 giây tạo thành nhiều mẫu.

#### 3.1.2 Định dạng file âm thanh

Pipeline sử dụng WAV là định dạng xử lý chính trong các thư mục `data/raw`, `data/chunks`, `data/processed`, `data/query_short`, `data/query_long`.

#### 3.1.3 Độ dài các file

- Tập nền (base/processed): có độ dài cố định 5.0 giây.
- Query ngắn: 5 giây.
- Query dài: tạo ngẫu nhiên trong khoảng 10-20 giây.

### 3.2 Tiền xử lý dữ liệu âm thanh

#### 3.2.1 Chuẩn hóa độ dài

Tiền xử lý được thực hiện theo hai tầng: (i) tầng tạo chunk từ audio gốc và (ii) tầng chuẩn hóa chunk trước khi trích đặc trưng.

Ở tầng tạo chunk (`src/data_collection/split_audio_chunks.py`), mỗi file gốc được cắt thành các đoạn ngắn với cấu hình:

- `chunk_duration = 5.0s`
- `max_chunks_per_file = 20`
- `overlap = 0.0s`

Sau khi có các chunk thô, tầng preprocess (`src/data_collection/preprocess_audio.py`) tiếp tục chuẩn hóa về độ dài mục tiêu `target_duration = 5.0s`. Về nguyên tắc xử lý tín hiệu:

- Nếu tín hiệu dài hơn 5 giây: cắt phần dư để đảm bảo cùng số mẫu.
- Nếu tín hiệu ngắn hơn 5 giây: đệm thêm (padding) để đạt đúng độ dài chuẩn.

Việc đồng nhất độ dài mang ý nghĩa quan trọng: mọi vector đặc trưng được trích trên cùng cửa sổ thời gian, giúp giảm sai lệch khi so sánh cosine giữa các mẫu.

#### 3.2.2 Chuẩn hóa sample rate

Hệ thống thống nhất tần số lấy mẫu ở `16 kHz` cho toàn bộ dữ liệu nền. Mức này đủ bao phủ dải tần thiết yếu của tiếng nói người (xấp xỉ đến 8 kHz theo định lý Nyquist), đồng thời giảm đáng kể chi phí tính toán so với các mức lấy mẫu cao hơn.

Trong quá trình xử lý:

- Các file đầu vào có sample rate khác nhau được resample về 16 kHz.
- Dữ liệu sau resample được chuyển về miền tham chiếu chung trước khi trích đặc trưng.

Kiểm tra trên CSDL `database/metadata.db` cho thấy toàn bộ bản ghi đều có `sample_rate = 16000` (tương ứng 828/828 bản ghi), chứng tỏ quy trình chuẩn hóa được áp dụng nhất quán.

#### 3.2.3 Lọc nhiễu

Trong ngữ cảnh bài toán giọng nói, nhiễu thường đến từ nhạc nền, khoảng lặng dài, đoạn mở đầu/kết thúc video và khác biệt mức âm lượng giữa các nguồn. Pipeline xử lý theo các bước:

1. **Cắt biên đầu/cuối ở file gốc**  
   Trước khi chia chunk, hệ thống loại bỏ `30 giây đầu` và `30 giây cuối` (khi độ dài file cho phép). Mục tiêu là loại các đoạn intro/outro, quảng cáo hoặc phần không chứa tiếng nói ổn định.

2. **Trim silence ở mức chunk/query**  
   Khi preprocess, hệ thống thực hiện trim các khoảng lặng ở đầu/cuối đoạn âm thanh. Bước này làm tăng tỷ lệ thông tin tiếng nói hữu ích trong mỗi mẫu 5 giây.

3. **Chuẩn hóa biên độ (normalize audio)**  
   Tín hiệu được đưa về mức biên độ nhất quán để giảm ảnh hưởng của khác biệt gain/microphone. Nhờ đó, các đặc trưng năng lượng (như RMS) và một phần đặc trưng phổ ổn định hơn giữa các mẫu.

4. **Ràng buộc chất lượng đầu vào truy vấn**  
   Ở pha online, hệ thống từ chối query ngắn hơn 5 giây. Điều này tránh trường hợp trích đặc trưng trên tín hiệu quá ít thông tin, vốn dễ gây nhiễu cho quá trình xếp hạng.

5. **Cơ chế phân đoạn cho query dài (10-20 giây)**  
   Query dài được tách thành nhiều cửa sổ 5 giây (có chồng lấn), sau đó gộp điểm theo trung bình và cực đại có trọng số. Cách làm này giúp giảm rủi ro do nhiễu cục bộ ở một vài đoạn ngắn.

Tổng thể, khối tiền xử lý không chỉ nhằm “làm sạch” tín hiệu mà còn đóng vai trò chuẩn hóa điều kiện đo, để các vector đặc trưng phản ánh bản chất giọng nói nhiều hơn là phản ánh điều kiện thu âm.

### 3.3 Trích xuất đặc trưng

#### 3.3.1 Xây dựng bộ thuộc tính

Khối trích xuất đặc trưng được triển khai trong `src/feature_extraction/extractor.py` thông qua lớp `AudioFeatureExtractor`. Mỗi mẫu âm thanh sau tiền xử lý được ánh xạ thành một vector số thực 52 chiều, theo cấu trúc:

- MFCC: 26 chiều (13 mean + 13 std)
- Pitch: 4 chiều (mean, std, min, max)
- Spectral: 6 chiều (centroid, rolloff, bandwidth; mỗi loại gồm mean, std)
- Temporal: 4 chiều (ZCR mean/std, RMS mean/std)
- Chroma: 12 chiều (trung bình năng lượng 12 pitch classes)

Cấu hình chính khi trích xuất:

- `sr = 16000`
- `n_mfcc = 13`
- `n_fft = 2048`
- `hop_length = 512`

Quy trình xử lý một mẫu:

1. Nạp tín hiệu audio chuẩn hóa.
2. Tính từng nhóm đặc trưng theo frame.
3. Tổng hợp thống kê (mean/std hoặc min/max) để thu được vector cố định chiều.
4. Ghép nối (concatenate) thành vector 52 chiều cuối cùng.

Lợi thế của thiết kế này là chuyển biểu diễn chuỗi thời gian biến độ dài thành biểu diễn véc-tơ có kích thước bất biến, phù hợp cho lưu trữ CSDL và truy hồi tương đồng.

#### 3.3.2 Giá trị thông tin của từng thuộc tính

Giá trị thông tin được phân tích theo từng nhóm như sau:

1. **MFCC (26 chiều) - mô tả âm sắc cốt lõi của giọng**  
   MFCC mô hình hóa bao phổ theo thang Mel, gần với cảm nhận thính giác người. Trong nhận diện người nói, MFCC thường là nhóm đóng góp lớn vì phản ánh đặc tính phát âm và cộng hưởng riêng của từng người.

2. **Pitch (4 chiều) - mô tả cao độ và biến thiên cao độ**  
   Bốn thống kê pitch (trung bình, độ lệch chuẩn, cực tiểu, cực đại) biểu diễn “dải hoạt động” cao độ của người nói. Nhóm này hữu ích khi phân biệt các giọng có âm sắc gần nhau nhưng khác thói quen lên/xuống giọng.

3. **Spectral (6 chiều) - mô tả phân bố năng lượng theo tần số**  
   - Centroid: xu hướng sáng/tối của âm thanh.
   - Rolloff: mức lan tỏa năng lượng về phía tần số cao.
   - Bandwidth: độ phân tán quanh trọng tâm phổ.  
   Bộ ba này hỗ trợ phân tách các giọng có cấu trúc phổ khác nhau dù cùng giới tính và cùng ngữ cảnh nội dung.

4. **Temporal (4 chiều) - mô tả động học tín hiệu**  
   - ZCR: mức dao động dấu tín hiệu, liên quan thành phần vô thanh/hữu thanh.
   - RMS: cường độ năng lượng theo thời gian.  
   Hai đặc trưng này tăng độ bền của hệ thống trước thay đổi về nhịp phát âm và mức độ nhấn giọng.

5. **Chroma (12 chiều) - mô tả cấu trúc hòa âm**  
   Dù xuất phát từ bài toán âm nhạc, chroma vẫn mang thông tin về phân bố harmonic của tiếng nói. Đây là tín hiệu bổ sung có ích trong các trường hợp khó, khi các nhóm đặc trưng chính cho kết quả gần nhau.

Tổng thể, bộ 52 chiều được chọn theo nguyên tắc “đa góc nhìn”: mỗi nhóm nắm bắt một khía cạnh khác nhau của giọng nói, giúp giảm phụ thuộc vào một loại đặc trưng đơn lẻ.

#### 3.3.3 Cách trích xuất cụ thể: thuật toán và công thức

Để bảo đảm tính tái lập khoa học, phần này mô tả rõ cách hệ thống tạo từng nhóm đặc trưng từ tín hiệu đầu vào `x[n]`.

**(1) Tiền xử lý theo frame trước khi trích đặc trưng**

- Tín hiệu được lấy mẫu ở `sr = 16000` Hz.
- Tín hiệu được chia thành các frame bằng cửa sổ FFT với `n_fft = 2048`, bước nhảy `hop_length = 512`.
- Nhiều đặc trưng được tính theo từng frame, sau đó tổng hợp thống kê toàn đoạn.

**(2) MFCC (26 chiều)**

Thuật toán MFCC thực hiện theo chuỗi chuẩn:

1. Tính phổ ngắn hạn bằng STFT.
2. Áp dụng Mel filterbank lên phổ biên độ/công suất.
3. Lấy log năng lượng trên thang Mel.
4. Dùng DCT để thu 13 hệ số cepstral.

Công thức cốt lõi:

- STFT:
  \[
  X(m,k)=\sum_{n=0}^{N-1} x[n+mH]w[n]e^{-j2\pi kn/N}
  \]
- Log-Mel năng lượng (với bộ lọc Mel thứ i):
  \[
  E_i=\log\left(\sum_k |X(m,k)|^2\,M_i(k)\right)
  \]
- Cepstrum (MFCC):
  \[
  c_r=\sum_{i=1}^{B} E_i\cos\left(\frac{\pi r}{B}(i-0.5)\right),\; r=1..13
  \]

Hệ thống lấy `mean` và `std` theo thời gian cho 13 hệ số: `13 + 13 = 26` chiều.

**(3) Pitch/F0 (4 chiều)**

Hệ thống dùng `librosa.piptrack` (pitch tracking dựa trên đỉnh phổ theo từng frame):

1. Từ phổ từng frame, tìm chỉ số tần số có biên độ lớn nhất.
2. Giữ các giá trị F0 dương (loại frame vô thanh/không xác định pitch).
3. Tính 4 thống kê: `mean`, `std`, `min`, `max`.

Nếu tập pitch rỗng, vector pitch được gán về `0` để tránh lỗi NaN.

**(4) Spectral features (6 chiều)**

Từ phổ biên độ `M(k)` và tần số `f_k`:

- Spectral centroid:
  \[
  Centroid = \frac{\sum_k f_k M(k)}{\sum_k M(k)}
  \]
- Spectral rolloff (ngưỡng 85% năng lượng):
  \[
  \sum_{k=1}^{k_r} M(k)=0.85\sum_k M(k)
  \]
  với `f_{k_r}` là rolloff.
- Spectral bandwidth:
  \[
  Bandwidth=\sqrt{\frac{\sum_k (f_k-Centroid)^2M(k)}{\sum_k M(k)}}
  \]

Mỗi đặc trưng lấy `mean` và `std` theo frame: `3 x 2 = 6` chiều.

**(5) Temporal features (4 chiều)**

- Zero Crossing Rate (ZCR):
  \[
  ZCR=\frac{1}{2N}\sum_{n=1}^{N}|sign(x[n])-sign(x[n-1])|
  \]
- RMS energy:
  \[
  RMS=\sqrt{\frac{1}{N}\sum_{n=1}^{N}x[n]^2}
  \]

Mỗi đại lượng lấy `mean` và `std`: `2 x 2 = 4` chiều.

**(6) Chroma (12 chiều)**

Từ phổ STFT, năng lượng được chiếu lên 12 lớp cao độ (pitch classes: C, C#, ..., B). Hệ thống lấy trung bình theo thời gian cho từng lớp:

\[
Chroma_i = mean_t\big(C_i(t)\big),\; i=1..12
\]

Kết quả là vector 12 chiều.

**(7) Ghép vector cuối cùng (52 chiều)**

Vector đặc trưng cuối được nối theo thứ tự:

\[
\mathbf{f}=[\mathbf{f}_{MFCC}(26),\mathbf{f}_{Pitch}(4),\mathbf{f}_{Spectral}(6),\mathbf{f}_{Temporal}(4),\mathbf{f}_{Chroma}(12)]
\]

Tổng số chiều:

\[
26+4+6+4+12=52
\]

Như vậy, thư viện chỉ đóng vai trò hiện thực hóa phép tính; bản chất trích xuất vẫn dựa trên các thuật toán DSP chuẩn với công thức xác định.

#### 3.3.4 Ma trận đặc trưng

Trong quá trình build CSDL (`scripts/build_database.py`), các vector được tổ chức thành ma trận đặc trưng và xử lý theo pipeline:

1. **Tạo ma trận gốc**  
   Với N mẫu âm thanh, hệ thống tạo ma trận `X` kích thước `N x 52`, mỗi hàng là một vector đặc trưng của một file.

2. **Kiểm tra tính hợp lệ**  
   Mỗi vector được kiểm tra giá trị bất thường (`NaN`, `Inf`). Những mẫu lỗi bị loại để tránh lan truyền sai số vào giai đoạn huấn luyện biến đổi và tìm kiếm.

3. **Lưu backup đặc trưng thô**  
   Ma trận `X` được lưu tại `database/features.npy` để phục vụ trực quan hóa, kiểm thử và tái sử dụng không phụ thuộc CSDL.

4. **Chuẩn hóa đặc trưng bằng StandardScaler**  
   Hệ thống học tham số chuẩn hóa trên toàn bộ tập nền (trung bình và độ lệch chuẩn theo từng chiều), sau đó biến đổi đặc trưng về cùng thang đo. Bước này quan trọng vì cosine similarity nhạy với hướng vector; nếu không chuẩn hóa, một số chiều biên độ lớn có thể chi phối kết quả.

5. **Giảm chiều tùy chọn bằng PCA**  
   Pipeline hỗ trợ PCA (mặc định tắt, có thể bật với `--use-pca` và `--pca-components`). Khi bật, vector sau chuẩn hóa được chiếu sang không gian thấp hơn nhưng vẫn giữ phần lớn phương sai.

6. **Lưu artifact biến đổi**  
   - `database/scaler.pkl`: mô hình StandardScaler.
   - `database/pca.pkl`: mô hình PCA (nếu dùng).  
   Các artifact này đảm bảo query online được biến đổi đúng cùng hệ quy chiếu với tập nền.

7. **Lưu ma trận đã biến đổi vào CSDL**  
   Vector sau biến đổi được ghi vào `audio_metadata.feature_vector` dưới dạng BLOB float32; `feature_dim` được cập nhật theo số chiều thực tế sau biến đổi (52 nếu không PCA, nhỏ hơn 52 nếu có PCA).

Nhờ pipeline này, hệ thống đạt được ba mục tiêu đồng thời: (i) biểu diễn ổn định cho tìm kiếm, (ii) tái lập được kết quả giữa offline-online, và (iii) thuận tiện cho mở rộng/đánh giá thực nghiệm.

---

## CHƯƠNG IV: HỆ CSDL

### 4.1 Thiết kế CSDL

#### 4.1.1 Lưu trữ file âm thanh

Hệ thống áp dụng mô hình lưu trữ lai giữa hệ quản trị CSDL và hệ thống tệp:

- **Dữ liệu audio gốc/chunk/processed** được lưu trên thư mục dữ liệu (`data/raw`, `data/chunks`, `data/processed`, `data/query_*`).
- **CSDL SQLite** chỉ lưu đường dẫn chuẩn hóa `file_path` và các thông tin định danh liên quan.

Thiết kế này có các ưu điểm:

1. **Giảm kích thước CSDL**: tránh lưu nhị phân audio trực tiếp (BLOB lớn), giúp file `.db` gọn hơn và thao tác nhanh hơn.
2. **Thuận tiện bảo trì**: có thể tái tạo vector đặc trưng từ file âm thanh khi thay đổi thuật toán mà không cần trích xuất dữ liệu nhị phân khỏi CSDL.
3. **Linh hoạt pipeline**: các bước tiền xử lý, đánh giá, trực quan hóa đều truy cập được file âm thanh gốc thông qua `file_path`.

Tuy nhiên, mô hình này đòi hỏi kỷ luật đồng bộ dữ liệu: nếu file âm thanh bị di chuyển/đổi tên ngoài pipeline, bản ghi trong CSDL có thể trở thành “đường dẫn mồ côi”. Vì vậy, dự án sử dụng cơ chế build lại CSDL có kiểm soát để đảm bảo tính nhất quán.

#### 4.1.2 Lưu trữ siêu dữ liệu (metadata)

Bảng trung tâm `audio_metadata` đóng vai trò lớp ngữ nghĩa cho dữ liệu âm thanh. Các trường chính gồm:

- `source_url`: URL nguồn video.
- `source_video_id`: định danh video nguồn (YouTube ID).
- `voice_name`: nhãn người nói (rút ra từ tên file hoặc danh mục video).
- `chunk_id`: chỉ số đoạn cắt trong cùng nguồn.
- `sample_rate`, `duration`: thông số kỹ thuật của mẫu âm thanh.
- `created_at`: thời điểm ghi nhận bản ghi.

Metadata phục vụ ba lớp nghiệp vụ:

1. **Truy vết dữ liệu**: biết rõ mỗi vector đến từ nguồn nào, đoạn nào.
2. **Truy vấn điều kiện**: lọc theo `voice_name`, `source_video_id` cho các nhu cầu kiểm thử/đánh giá.
3. **Diễn giải kết quả tìm kiếm**: trả kết quả không chỉ là điểm tương đồng mà còn có bối cảnh dữ liệu.

Việc tách metadata thành lớp độc lập giúp hệ thống đáp ứng đúng tinh thần CSDL đa phương tiện: kết hợp nội dung tín hiệu (feature vector) và thông tin mô tả (descriptive metadata).

#### 4.1.3 Lưu trữ vector đặc trưng

Vector đặc trưng sau biến đổi (chuẩn hóa, PCA tùy chọn) được lưu trong cột `feature_vector` dưới dạng BLOB `float32`, đồng thời lưu `feature_dim` để kiểm tra tính toàn vẹn khi đọc lại.

Lý do chọn `float32 + BLOB`:

- **Hiệu quả lưu trữ**: `float32` giảm 50% dung lượng so với `float64` nhưng vẫn đủ chính xác cho bài toán cosine retrieval.
- **Tương thích tính toán**: thao tác nạp bằng `numpy.frombuffer` nhanh, thuận lợi cho so sánh vector hàng loạt.
- **Đơn giản triển khai**: không cần mở rộng SQLite bằng extension vector chuyên dụng.

Trong lớp truy cập dữ liệu (`MetadataDB`):

- Khi ghi: vector được ép kiểu `np.float32`, chuyển sang bytes và đóng gói vào `sqlite3.Binary`.
- Khi đọc: bytes được giải mã về mảng `numpy`, sau đó đối chiếu kích thước với `feature_dim`; bản ghi sai lệch kích thước sẽ bị loại bỏ khỏi tập tìm kiếm.

Cơ chế này tạo một hàng rào kiểm soát chất lượng dữ liệu ở tầng CSDL, giảm nguy cơ lỗi lan sang tầng truy hồi.

### 4.2 Cấu trúc bảng dữ liệu

Schema được khởi tạo tự động trong `src/vector_database/metadata_db.py` với bảng `audio_metadata` và các ràng buộc toàn vẹn. Mô tả vai trò từng cột:

- `id INTEGER PRIMARY KEY AUTOINCREMENT`: định danh nội bộ của bản ghi.
- `vector_idx INTEGER UNIQUE`: ánh xạ vị trí logic của vector trong pipeline build.
- `file_path TEXT UNIQUE NOT NULL`: khóa tự nhiên, đảm bảo không trùng mẫu âm thanh.
- `source_url TEXT`: nguồn xuất xứ dữ liệu.
- `source_video_id TEXT`: khóa tham chiếu logic theo video.
- `voice_name TEXT`: nhãn người nói phục vụ truy vấn và đánh giá.
- `chunk_id TEXT`: chỉ số đoạn chunk trong video gốc.
- `sample_rate INTEGER`, `duration REAL`: thông tin kỹ thuật tín hiệu.
- `feature_dim INTEGER`: số chiều vector sau biến đổi.
- `feature_vector BLOB`: dữ liệu vector nhúng.
- `created_at TEXT DEFAULT CURRENT_TIMESTAMP`: nhật ký thời gian tạo bản ghi.

Về cơ chế cập nhật, hệ thống dùng **upsert** theo `file_path`:

- Nếu bản ghi chưa tồn tại: thực hiện `INSERT`.
- Nếu đã tồn tại: cập nhật các trường metadata và vector mới nhất.

Mô hình upsert đặc biệt phù hợp với pipeline build lặp: cùng một tập file có thể được xử lý lại khi thay đổi thuật toán đặc trưng, nhưng không tạo trùng lặp dữ liệu. Ngoài ra, script build còn gọi `clear_all()` trước khi nạp lại toàn bộ dữ liệu, giúp trạng thái CSDL phản ánh đúng tập processed hiện hành.

Ở góc độ chuẩn hóa dữ liệu, bảng hiện tại đạt chuẩn thực dụng cho bài toán retrieval. Nếu mở rộng hệ thống, có thể tách thêm các bảng như `videos`, `voices`, `chunks` để tăng mức chuẩn hóa quan hệ và hỗ trợ phân tích đa chiều sâu hơn.

### 4.3 Indexing và tối ưu tìm kiếm

Hệ thống triển khai tối ưu ở hai lớp: **(i) metadata indexing** và **(ii) vector retrieval strategy**.

**1) Metadata indexing**

Hai chỉ mục B-tree hiện có trong SQLite:

- `idx_voice_name` trên `voice_name`
- `idx_video_id` trên `source_video_id`

Hai index này tăng tốc đáng kể các truy vấn dạng:

- Tìm các mẫu theo tên người nói (`LIKE '%keyword%'`).
- Lọc theo video nguồn phục vụ kiểm thử/đối chiếu.

Trong thực tế vận hành, đây là lớp tối ưu cần thiết cho các thao tác phân tích dữ liệu và kiểm chứng nhãn.

**2) Chiến lược tìm kiếm vector hiện tại**

Đối với truy hồi tương đồng, hệ thống dùng phương pháp brute-force có kiểm soát:

1. Nạp tập vector hợp lệ từ SQLite (`load_all_vectors`).
2. Tính cosine similarity giữa query vector và từng vector nền.
3. Sắp xếp giảm dần theo điểm cosine.
4. Trả về top-k (mặc định 5).

Ưu điểm của brute-force ở quy mô hiện tại (khoảng 828 vector):

- **Độ chính xác tuyệt đối trong không gian hiện hành**: không có sai số xấp xỉ do ANN.
- **Dễ kiểm thử và diễn giải**: mỗi điểm similarity có thể tái hiện trực tiếp.
- **Chi phí triển khai thấp**: không cần hạ tầng index vector chuyên dụng.

Hạn chế:

- Độ phức tạp truy vấn tăng tuyến tính theo số lượng vector O(N).
- Khi quy mô tăng mạnh (hàng chục nghìn trở lên), độ trễ và bộ nhớ có thể trở thành nút thắt.

**3) Các tối ưu đã có trong pipeline**

- Chuẩn hóa đặc trưng bằng `StandardScaler` trước khi lưu giúp phân bố vector đồng nhất hơn, cải thiện ổn định của cosine.
- Hỗ trợ PCA tùy chọn để giảm chiều, từ đó giảm chi phí tính toán ở pha online.
- Lọc bản ghi hỏng vector qua kiểm tra `feature_dim` khi nạp dữ liệu.

**4) Định hướng tối ưu mở rộng**

Khi dữ liệu tăng lớn, có thể nâng cấp theo lộ trình:

- Dùng ANN index (FAISS/HNSW/ScaNN) để giảm thời gian truy vấn.
- Tách kho vector sang dịch vụ chuyên biệt, giữ SQLite cho metadata quan hệ.
- Áp dụng tiền lọc theo metadata (ví dụ domain, nguồn, ngôn ngữ) trước khi truy hồi vector để giảm không gian tìm kiếm.

Như vậy, kiến trúc CSDL hiện tại được xem là điểm cân bằng tốt giữa tính học thuật, tính khả thi triển khai và khả năng kiểm chứng thực nghiệm trong phạm vi bài tập lớn.

---

## CHƯƠNG V: HỆ THỐNG TÌM KIẾM

### 5.1 Sơ đồ khối hệ thống

Hệ thống tìm kiếm được tổ chức theo kiến trúc hai pha rõ ràng, bảo đảm tách biệt giữa xử lý nền và xử lý truy vấn thời gian thực.

**Pha offline (xây dựng cơ sở dữ liệu vector):**

1. Thu thập audio nguồn (`src/data_collection/download_audio.py`).
2. Chia đoạn 5 giây và tạo tập query ngắn/dài (`src/data_collection/split_audio_chunks.py`).
3. Tiền xử lý đồng nhất tín hiệu (`src/data_collection/preprocess_audio.py`).
4. Trích xuất đặc trưng 52 chiều (`src/feature_extraction/extractor.py`).
5. Chuẩn hóa đặc trưng và PCA tùy chọn (`src/search/feature_transform.py`).
6. Lưu metadata + vector vào SQLite (`scripts/build_database.py`, `src/vector_database/metadata_db.py`).

**Pha online (truy vấn tương đồng):**

1. Nhận file query từ giao diện Streamlit (`app/streamlit_app.py`).
2. Kiểm tra điều kiện đầu vào và tiền xử lý query.
3. Nếu query dài: phân đoạn thành các cửa sổ 5 giây có chồng lấn.
4. Trích xuất và biến đổi đặc trưng theo đúng hệ quy chiếu của tập nền.
5. Tính cosine similarity với các vector trong CSDL (`src/search/sqlite_vector_search.py`).
6. Tổng hợp điểm, xếp hạng top-k và trả kết quả kèm metadata (`src/search/similarity_search.py`).

Về bản chất, sơ đồ khối thể hiện một hệ thống CBVR (Content-Based Voice Retrieval), trong đó “nội dung” của âm thanh được mã hóa thành vector và tìm kiếm trong không gian đặc trưng.

### 5.2 Quy trình tìm kiếm

#### 5.2.1 Đầu vào: file âm thanh mới

Nguời dùng có hai chế độ đưa dữ liệu truy vấn:

- Chọn từ danh sách file test (`data/query_short`, `data/query_long`) để phục vụ đánh giá có kiểm soát.
- Upload file mới từ máy người dùng để mô phỏng tình huống vận hành thực tế.

Hệ thống hỗ trợ các định dạng phổ biến (WAV/MP3/FLAC), sau đó chuẩn hóa về cùng quy trình xử lý nội bộ. Điều kiện tối thiểu là query phải có thời lượng từ 5 giây trở lên; query ngắn hơn bị từ chối để tránh suy giảm độ tin cậy của vector đặc trưng.

#### 5.2.2 Trích xuất đặc trưng file mới

Sau khi qua bước kiểm tra đầu vào, query được đưa vào cùng pipeline đặc trưng như dữ liệu nền để đảm bảo tính tương thích hình học trong không gian vector.

Quy trình gồm:

1. Chuẩn hóa tín hiệu (trim silence, normalize biên độ, quy đổi sample rate nếu cần).
2. Trích xuất vector 52 chiều theo cùng cấu hình tham số.
3. Áp dụng `StandardScaler` đã huấn luyện từ tập nền.
4. Áp dụng PCA nếu hệ thống được build ở chế độ giảm chiều.

Điểm quan trọng là query **không** được fit lại bộ biến đổi, mà chỉ transform bằng artifact đã lưu (`scaler.pkl`, `pca.pkl`). Nhờ đó, cả vector nền và vector query cùng nằm trong một hệ tọa độ thống nhất, bảo đảm ý nghĩa của độ đo cosine.

#### 5.2.3 So sánh với CSDL

Module `SQLiteVectorSearch` thực hiện truy hồi theo cơ chế duyệt toàn bộ vector hợp lệ:

1. Nạp danh sách vector từ bảng `audio_metadata` (cột `feature_vector`).
2. Giải mã BLOB thành mảng `float32` và xác thực theo `feature_dim`.
3. Tính cosine similarity với từng vector nền:

   \[
   \text{cosine}(q,x_i)=\frac{q\cdot x_i}{\|q\|\,\|x_i\|}
   \]

4. Gắn kèm metadata (`voice_name`, `source_video_id`, `file_path`).

Đây là phép so sánh 1:N trong không gian vector. Với quy mô hiện tại, cách tiếp cận này đáp ứng tốt cả về tốc độ lẫn khả năng kiểm chứng kết quả.

#### 5.2.4 Xếp hạng Top 5 kết quả

Sau khi có danh sách điểm tương đồng, hệ thống:

- Sắp xếp giảm dần theo `cosine_similarity`.
- Chuyển đổi sang thang phần trăm (`similarity_percent = cosine * 100`).
- Lấy `top_k` kết quả đầu (mặc định 5, có thể thay đổi trên giao diện).

Trong trường hợp query dài nhiều đoạn, xếp hạng cuối cùng không lấy trực tiếp từ một đoạn đơn lẻ mà dựa trên điểm tổng hợp liên đoạn (trình bày ở mục 5.3).

#### 5.2.5 Đầu ra: 5 files giống nhất

Kết quả đầu ra bao gồm cả lớp kỹ thuật và lớp diễn giải:

- `file_path`: file ứng viên tương đồng.
- `similarity_percent`: điểm tương đồng theo %.
- `cosine_similarity`: điểm cosine gốc.
- Metadata đi kèm: `voice_name`, `source_video_id`, `source_url`.

Trên giao diện Streamlit, người dùng có thể phát trực tiếp audio kết quả, xem waveform, đối chiếu đặc trưng và quan sát vị trí tương đối trong không gian PCA.

### 5.3 Xử lý 2 trường hợp

#### 5.3.1 Người đã có trong CSDL

Nếu query thuộc người nói đã xuất hiện trong tập nền, mục tiêu truy hồi là đưa các mẫu cùng `voice_name` lên thứ hạng cao nhất. Cơ chế đạt được điều này dựa trên:

- Tính nhất quán của quy trình tiền xử lý và trích đặc trưng.
- Độ phân biệt của bộ đặc trưng 52 chiều.
- Chuẩn hóa vector trước khi tính cosine.

Trong đánh giá thực nghiệm, nhóm query dài cho thấy đa số trường hợp đạt top-1 đúng nhãn, phản ánh khả năng nhận diện theo tương đồng âm học ở mức tốt trong phạm vi dữ liệu hiện có.

#### 5.3.2 Người chưa có trong CSDL

Với người nói chưa có trong CSDL, hệ thống không thể trả định danh “đúng tuyệt đối”, mà thực hiện truy hồi theo nguyên tắc lân cận gần nhất trong không gian đặc trưng. Khi đó:

- Top-5 phản ánh các giọng có đặc tính âm học gần nhất (âm sắc, cao độ, phân bố năng lượng, harmonic).
- Kết quả có ý nghĩa tham khảo về mức độ giống giọng, không phải kết luận nhận dạng danh tính.

Đây là cách ứng xử phù hợp với các hệ thống tìm kiếm theo nội dung, đồng thời mở ra khả năng sử dụng trong các ứng dụng gợi ý giọng tương tự hoặc lọc trước cho tác vụ nhận dạng chuyên sâu.

#### Bổ sung: Cơ chế cho query dài nhiều phân đoạn

Đối với query dài (10-20 giây), hệ thống áp dụng chiến lược multi-segment retrieval:

1. Chia query thành nhiều đoạn 5 giây với chồng lấn (`overlap = 2.5s`).
2. Truy hồi top-k cho từng đoạn độc lập.
3. Gộp các ứng viên trùng nhau theo `file_path`.
4. Tính điểm cuối theo công thức trọng số:

\[
S_{final}=0.7\times mean(S_{seg}) + 0.3\times max(S_{seg})
\]

Trong đó `mean` phản ánh độ ổn định toàn cục còn `max` giữ lại tín hiệu khớp mạnh cục bộ. Cơ chế này giúp giảm ảnh hưởng của nhiễu cục bộ ở một số đoạn, đồng thời tận dụng thông tin dài hạn của query.

### 5.4 Kết quả trung gian của quá trình tìm kiếm

Một điểm mạnh của hệ thống là khả năng quan sát và diễn giải từng tầng xử lý thay vì chỉ hiển thị kết quả cuối.

Các đầu ra trung gian chính gồm:

1. **Đặc trưng query 52 chiều**  
   Hiển thị theo nhóm (MFCC, Pitch, Spectral, Temporal, Chroma), giúp kiểm tra cấu hình và đánh giá định tính đặc điểm giọng.

2. **So sánh query - top match theo từng nhóm đặc trưng**  
   Hệ thống vẽ biểu đồ và bảng chênh lệch, cho phép nhận diện nhóm đặc trưng nào đóng góp nhiều vào mức tương đồng.

3. **Trực quan không gian vector (PCA 2D)**  
   Query, top matches và toàn bộ tập nền được chiếu về không gian 2 chiều để quan sát cấu trúc lân cận trực quan.

4. **Kết quả xếp hạng chi tiết**  
   Mỗi kết quả có điểm cosine, điểm %, metadata nguồn và khả năng phát lại âm thanh để kiểm tra bằng thính giác.

5. **Báo cáo đánh giá định lượng**  
   Module đánh giá tạo các tệp `retrieval_details.csv`, `retrieval_per_query.csv`, `retrieval_hit_rate_by_voice.csv`, `confusion_matrix_*.csv`, `retrieval_summary.json`. Đây là căn cứ thực nghiệm để phân tích chất lượng mô hình ở Chương VI.

Nhờ các lớp đầu ra trung gian này, hệ thống vừa đáp ứng yêu cầu tìm kiếm, vừa hỗ trợ phân tích khoa học và cải tiến có định hướng.

---

## CHƯƠNG VI: KẾT QUẢ VÀ ĐÁNH GIÁ

### 6.1 Demo hệ thống

Ứng dụng demo được triển khai bằng Streamlit (`app/streamlit_app.py`) theo mô hình tương tác trực tiếp với CSDL vector. Quy trình demo thực hiện:

1. Chọn query từ tập test hoặc upload file mới.
2. Hệ thống tự động tiền xử lý, trích đặc trưng, chuẩn hóa và truy hồi top-k.
3. Hiển thị kết quả gồm điểm cosine, điểm phần trăm, metadata nguồn và phát lại audio.
4. Trực quan hóa đặc trưng và không gian vector để hỗ trợ diễn giải.

Ngoài giao diện demo, các đánh giá định lượng được chạy bằng script để bảo đảm tái lập kết quả thực nghiệm.

### 6.2 Kết quả tìm kiếm thực tế

Đánh giá chính thức được thực hiện trực tiếp bằng script:

`python scripts/evaluate_retrieval.py --query-dir data/query_short,data/query_long --db database/metadata.db --top-k 5 --output-dir reports/retrieval`

Kết quả thực nghiệm hiện tại là:

- Số query: 88
- Mean similarity: 74.50%
- Hit@5: 0.9318
- MRR: 0.9205

Nhận xét: hệ thống đạt hiệu quả truy hồi cao trên cả hai tập truy vấn ngắn và dài, đồng thời duy trì mức tương đồng trung bình ổn định.

**Bảng 6.1 - Chỉ số tổng hợp của hệ thống**

| Chỉ số | Giá trị |
| --- | ---: |
| Số query | 88 |
| Mean similarity (%) | 74.50 |
| Hit@5 | 0.9318 |
| MRR | 0.9205 |

**Bảng 6.2 - Chỉ số theo loại truy vấn**

| Loại query | Số lượng | Hit@1 | Hit@5 | MRR@5 | Mean Similarity (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| short (5s) | 44 | 0.9091 | 0.9545 | 0.9318 | 73.03 |
| long (10-20s) | 44 | 0.9091 | 0.9091 | 0.9091 | 75.97 |

Từ Bảng 6.2 có thể thấy truy vấn dài cho điểm tương đồng trung bình cao hơn, trong khi truy vấn ngắn đạt Hit@5 nhỉnh hơn nhẹ.

### 6.3 Đánh giá độ chính xác

Để tăng sức thuyết phục, báo cáo sử dụng thêm các chỉ số theo mức K và theo thứ hạng:

**Bảng 6.3 - Hit@K và MRR@K**

| K | Hit@K | MRR@K |
| ---: | ---: | ---: |
| 1 | 0.9091 | 0.9091 |
| 2 | 0.9318 | 0.9205 |
| 3 | 0.9318 | 0.9205 |
| 4 | 0.9318 | 0.9205 |
| 5 | 0.9318 | 0.9205 |

Diễn giải:

- Bước tăng lớn nhất diễn ra từ K=1 lên K=2, cho thấy các truy vấn “khó” thường vẫn xuất hiện đáp án đúng trong top-2.
- Từ K=2 đến K=5, chỉ số giữ nguyên; điều này gợi ý lỗi chủ yếu tập trung ở số ít truy vấn mà mô hình nhầm ngay từ top đầu.

Ngoài ra, dữ liệu nền cho thấy mean similarity theo rank giảm dần từ hạng 1 đến hạng 5 (xem `reports/retrieval/analysis/mean_similarity_by_rank.csv`), phù hợp với tính đúng đắn của cơ chế xếp hạng.

**Biểu đồ sử dụng trong báo cáo**

- Hình 6.1: Đường cong Hit@K và MRR@K - `reports/retrieval/analysis/hit_mrr_at_k.png`
- Hình 6.2: Mean similarity theo thứ hạng - `reports/retrieval/analysis/mean_similarity_by_rank.png`
- Hình 6.3: So sánh Hit@5 theo loại query - `reports/retrieval/analysis/hit5_by_query_type.png`
- Hình 6.4: Ma trận nhầm lẫn top-1 (chuẩn hóa, tập con) - `reports/retrieval/analysis/confusion_matrix_top1_fixed_subset.png`

Các biểu đồ trên cho phép quan sát định lượng một cách trực quan và củng cố kết luận về hiệu năng truy hồi.

Kết quả theo từng người nói (top-1 hit rate, sau hiệu chỉnh) tại `reports/retrieval/analysis/voice_top1_hit_rate_fixed.csv` cho thấy phần lớn nhãn đạt 1.0, trong khi một số nhãn đạt 0.5 (ví dụ `ara_won`, `iamhamy`, `jenny_huynh`, `tuwi_ng`, `tuyen_sinh_247`). Đây là nhóm cần ưu tiên phân tích lỗi.

### 6.4 Phân tích sai số

Phân tích sai số được thực hiện trên các tệp `retrieval_per_query.csv`, `retrieval_details.csv` và ma trận nhầm lẫn top-1.

Các trường hợp lỗi nổi bật:

- Một số người nói có top-1 hit rate = 0.5 (mỗi người có 2 query short/long, đúng 1 và sai 1).
- Nhầm lẫn thường xảy ra giữa các giọng có cao độ và âm sắc gần nhau, đặc biệt khi ngữ cảnh thu âm khác biệt.

Nguyên nhân kỹ thuật khả dĩ:

- Khác biệt chất lượng thu âm giữa query và tập nền.
- Ngữ cảnh âm thanh (nhạc nền, tạp âm) làm lệch đặc trưng.
- Mức độ đa dạng nội dung phát âm trong cùng một voice.
- Bộ đặc trưng handcrafted chưa mô hình hóa đầy đủ các vi đặc trưng speaker-level trong điều kiện khó.

Các sai số quan sát được chủ yếu đến từ khác biệt điều kiện thu âm và độ tương đồng cao giữa một số cặp giọng nói.

### 6.5 Hạn chế và đề xuất cải tiến

Hạn chế hiện tại (rút ra từ kết quả thực nghiệm và biểu đồ):

- Tìm kiếm vector dạng brute-force, chưa tối ưu cho quy mô lớn.
- Đặc trưng phụ thuộc thủ công (handcrafted features), chưa khai thác embedding học sâu.
- Kết quả còn nhạy với điều kiện thu âm và nhiễu nền ở một số speaker.

Đề xuất cải tiến theo mức ưu tiên:

1. Xây dựng bộ kiểm thử hồi quy cho pipeline đánh giá và truy hồi để đảm bảo tính ổn định của chỉ số qua các phiên bản.
2. Thử nghiệm speaker embedding (x-vector/ECAPA-TDNN) kết hợp cosine hoặc PLDA để tăng độ phân biệt.
3. Nâng cấp truy hồi ANN (FAISS/HNSW) khi mở rộng quy mô dữ liệu.
4. Bổ sung đánh giá nâng cao: Precision@K, Recall@K, nDCG@K, phân tích confidence theo ngưỡng similarity.
5. Chuẩn hóa chất lượng tập dữ liệu (lọc nhiễu, cân bằng nội dung phát âm giữa các speaker).

Tổng hợp lại, kết quả thực nghiệm hiện tại cho thấy hệ thống đạt chất lượng truy hồi cao sau khi hiệu chỉnh nhãn đánh giá, đồng thời vẫn còn dư địa cải tiến rõ ràng cho cả mô hình đặc trưng lẫn hạ tầng tìm kiếm.

---

## CHƯƠNG VII: KẾT LUẬN

### 7.1 Tổng kết

Đề tài đã hoàn thiện một quy trình đầy đủ cho bài toán tìm kiếm giọng nói phụ nữ theo hướng cơ sở dữ liệu đa phương tiện, bao gồm cả xử lý dữ liệu, lưu trữ, truy hồi và đánh giá thực nghiệm. Về mặt kỹ thuật, hệ thống đạt được các kết quả chính sau:

1. **Hoàn thiện pipeline dữ liệu đầu-cuối**  
   Dữ liệu âm thanh được thu thập, chia đoạn, tiền xử lý và chuẩn hóa nhất quán trước khi đưa vào CSDL. Tập nền sau xử lý đạt 828 mẫu, vượt yêu cầu tối thiểu 500 mẫu của đề bài.

2. **Xây dựng biểu diễn nội dung âm thanh có tính mô tả cao**  
   Mỗi mẫu được ánh xạ thành vector đặc trưng 52 chiều (MFCC, Pitch, Spectral, Temporal, Chroma), phản ánh đa khía cạnh của tín hiệu tiếng nói và đáp ứng tốt cho truy hồi theo độ tương đồng.

3. **Thiết kế được tầng CSDL phục vụ truy hồi đa phương tiện**  
   SQLite được sử dụng làm hệ quản trị trung tâm để lưu cả metadata và vector đặc trưng, cho phép vừa truy vấn theo thuộc tính mô tả, vừa truy vấn theo nội dung vector trong cùng kiến trúc.

4. **Triển khai thành công cơ chế tìm kiếm top-k theo cosine similarity**  
   Hệ thống trả top-5 kết quả có độ tương đồng cao nhất; đồng thời hỗ trợ query dài qua cơ chế phân đoạn và tổng hợp điểm, giúp tăng tính ổn định khi truy vấn thực tế.

5. **Đạt kết quả thực nghiệm thuyết phục**  
   Trên tập 88 query (short + long), hệ thống đạt `Hit@5 = 0.9318`, `MRR = 0.9205`, cho thấy khả năng truy hồi chính xác cao trong phạm vi dữ liệu thử nghiệm hiện hành.

Ở góc độ học thuật, đồ án đã thể hiện rõ sự kết hợp giữa lý thuyết xử lý tín hiệu âm thanh và mô hình lưu trữ/truy vấn của CSDL đa phương tiện, đồng thời cung cấp đầy đủ bằng chứng định lượng (bảng chỉ số, biểu đồ, ma trận nhầm lẫn) để kiểm chứng kết luận.

### 7.2 Hạn chế

Mặc dù đạt kết quả khả quan, hệ thống vẫn tồn tại các hạn chế cần nhìn nhận khách quan:

1. **Khả năng mở rộng còn giới hạn**  
   Cơ chế truy hồi hiện tại là brute-force trên toàn bộ vector nên độ phức tạp tăng tuyến tính theo số mẫu. Khi dữ liệu tăng lớn, độ trễ truy vấn sẽ là thách thức chính.

2. **Đặc trưng còn thiên về handcrafted features**  
   Bộ đặc trưng 52 chiều có ưu điểm đơn giản, dễ diễn giải, nhưng chưa tận dụng năng lực biểu diễn sâu của các mô hình speaker embedding hiện đại.

3. **Độ bền trước biến thiên thu âm chưa tối ưu tuyệt đối**  
   Trong một số trường hợp, nhiễu nền, thiết bị ghi âm khác nhau hoặc nội dung phát âm quá đa dạng vẫn có thể làm giảm độ chính xác top-1.

4. **Phụ thuộc vào quy trình dữ liệu đồng nhất**  
   Hiệu năng hệ thống gắn chặt với chất lượng khâu tiền xử lý và tính nhất quán quy ước dữ liệu; nếu nguồn dữ liệu đầu vào quá dị biệt, cần có cơ chế chuẩn hóa mạnh hơn.

### 7.3 Hướng phát triển

Trong giai đoạn tiếp theo, nhóm định hướng phát triển theo bốn trục chính:

1. **Nâng cấp biểu diễn đặc trưng người nói**  
   Tích hợp các mô hình embedding như x-vector, ECAPA-TDNN để cải thiện khả năng phân biệt speaker trong điều kiện nhiễu và dữ liệu không đồng nhất.

2. **Tối ưu hạ tầng truy hồi ở quy mô lớn**  
   Chuyển từ brute-force sang ANN index (FAISS/HNSW), đồng thời giữ SQLite hoặc hệ CSDL quan hệ cho lớp metadata để đạt cân bằng giữa tốc độ và khả năng quản trị.

3. **Mở rộng bộ tiêu chí đánh giá**  
   Bổ sung Precision@K, Recall@K, nDCG@K, cùng các chỉ số đặc thù speaker verification (ROC, EER) nhằm đánh giá toàn diện hơn cả độ đúng lẫn độ tin cậy của điểm similarity.

4. **Chuẩn hóa và đa dạng hóa dữ liệu**  
   Tăng số lượng speaker, đa dạng miền nội dung, điều kiện thu âm và vùng giọng; đồng thời xây dựng quy trình kiểm thử hồi quy tự động để theo dõi chất lượng hệ thống qua từng phiên bản.

Tóm lại, hệ thống hiện tại đã đạt mức hoàn thiện tốt cho phạm vi bài tập lớn và có nền tảng kỹ thuật rõ ràng để phát triển thành một hệ truy hồi giọng nói có khả năng mở rộng và ứng dụng thực tiễn cao hơn.

---

## TÀI LIỆU THAM KHẢO

1. Librosa Development Team, *Librosa Documentation*, https://librosa.org/doc/latest/feature.html.
2. Tài liệu kỹ thuật dự án:
   - `README.md`
   - `docs/system_architecture.md`
   - `docs/pipeline.md`
   - `docs/feature_extraction.md`
   - `docs/cosine_similarity.md`
3. Mã nguồn chính:
   - `scripts/build_database.py`
   - `src/feature_extraction/extractor.py`
   - `src/vector_database/metadata_db.py`
   - `src/search/similarity_search.py`
   - `src/search/sqlite_vector_search.py`
   - `src/evaluation/retrieval_evaluator.py`
