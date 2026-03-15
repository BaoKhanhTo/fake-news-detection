# Hệ Thống Giáo Dục & Nhận Diện Tin Giả Tiếng Việt (Vietnamese Fake News Detection & Education System)

Dự án này là một nền tảng tích hợp giữa công nghệ học máy (Machine Learning) tiên tiến và môi trường giáo dục trực quan, nhằm giúp người dùng không chỉ nhận biết tin giả mà còn hiểu rõ các nguyên lý khoa học đằng sau các thuật toán AI.

---

## 📑 Mục Lục
1. [Tổng Quan Dự Án](#-tổng-quan-dự-án)
2. [Hướng Dẫn Cài Đặt & Khởi Chạy](#-hướng-dẫn-cài-đặt--khởi-chạy)
3. [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
4. [Nền Tảng Lý Thuyết Chuyên Sâu](#-nền-tảng-lý-thuyết-chuyên-sâu)
    - 4.1. [Xử Lý Ngôn Ngữ Tiếng Việt (underthesea)](#41-xử-lý-ngôn-ngữ-tiếng-việt-underthesea)
    - 4.2. [Vectơ Hóa Văn Bản (TF-IDF vs Doc2Vec)](#42-vectơ-hóa-văn-bản-tf-idf-vs-doc2vec)
    - 4.3. [Phân Tích 5 Thuật Toán Phân Loại AI](#43-phân-tích-5-thuật-toán-phân-loại-ai)
5. [Hệ Thống Chỉ Số Đánh Giá (Metrics)](#-hệ-thống-chỉ-số-đánh-giá-metrics)
6. [Ma Trận Nhầm Lẫn (Confusion Matrix)](#-ma-trận-nhầm-lẫn-confusion-matrix)
7. [Giao Diện Người Dùng & Trải Nghiệm](#-giao-diện-người-dùng--trải-nghiệm)

---

## 🚀 Tổng Quan Dự Án

Hệ thống được thiết kế để giải quyết vấn đề tin giả (Fake News) đang ngày càng tinh vi trên mạng xã hội Việt Nam. Khác với các công cụ kiểm tra thông thường, dự án này tập trung vào khía cạnh **giáo dục**:
- **Đa mô hình:** Sử dụng đồng thời 5 thuật toán để người dùng so sánh độ chính xác.
- **Minh họa trực quan:** Hiển thị các chỉ số toán học, sơ đồ luồng dữ liệu và lý thuyết chi tiết.
- **Xử lý đặc thù:** Tối ưu hóa cho ngôn ngữ tiếng Việt với các thư viện chuyên dụng.

---

## 🛠 Hướng Dẫn Cài Đặt & Khởi Chạy

### Yêu cầu hệ thống:
- Python 3.8+
- Node.js 16+
- RAM tối thiểu 4GB (để nạp các mô hình .pkl)

### Các bước cài đặt:

1. **Cài đặt thư viện Python:**
```bash
pip install -r requirements.txt
```

2. **Cài đặt thư viện Frontend:**
```bash
cd frontend
npm install
```

3. **Khởi chạy hệ thống:**
Sử dụng file batch tích hợp:
```bash
run_all.bat
```
Hoặc chạy thủ công:
- Backend: `python main.py`
- Frontend: `cd frontend && npm run dev`

---

## 🏗 Kiến Trúc Hệ Thống

Dữ liệu di chuyển qua các tầng sau:
1. **Tầng Thu Thập:** Tiếp nhận văn bản thô từ người dùng.
2. **Tầng Tiền Xử Lý:** Sử dụng `underthesea` để tách từ và `regex` để làm sạch dữ liệu.
3. **Tầng Vectơ Hóa:** Chuyển văn bản thành các dãy số (TF-IDF cho tính thống kê, Doc2Vec cho tính ngữ nghĩa).
4. **Tầng Dự Đoán:** 5 mô hình (SVM, LR, RF, DT, NB) thực hiện phân loại song song.
5. **Tầng Hiển Thị:** React Vite hiển thị kết quả, xác suất và nội dung giáo dục.

---

## 📚 Nền Tảng Lý Thuyết Chuyên Sâu

### 4.1. Xử Lý Ngôn Ngữ Tiếng Việt (underthesea)

Tiếng Việt là ngôn ngữ đơn lập, không có hình thái từ, ranh giới giữa các từ không được xác định rõ ràng chỉ bằng khoảng trắng.

**A. Tokenization (Tách từ):**
Thư viện `underthesea` sử dụng mô hình CRF (Conditional Random Fields) để dự đoán ranh giới từ.
- *Ví dụ:* "Học sinh học sinh học" -> `["Học_sinh", "học", "sinh_học"]`.
- *Tầm quan trọng:* Nếu tách sai thành `["Học", "sinh"]`, máy sẽ không biết đây là đối tượng con người.

**B. Data Cleaning:**
- **Lowercase:** Chuyển "Sốc" thành "sốc" để giảm kích thước từ điển.
- **Punctuation Removal:** Loại bỏ các ký tự nhiễu như `#, @, $, ...` nhưng giữ lại `!` và `?` vì chúng thường mang sắc thái biểu cảm mạnh trong tin giả.

---

### 4.2. Vectơ Hóa Văn Bản (TF-IDF vs Doc2Vec)

#### A. TF-IDF (Thống kê trọng số từ)
TF-IDF đánh giá tầm quan trọng của một từ $t$ trong một văn bản $d$ thuộc tập hợp văn bản $D$.

**Công thức:**
1. **Term Frequency (TF):** 
   $$TF(t, d) = \frac{n_{t,d}}{\sum_{k} n_{k,d}}$$
   (Tần suất xuất hiện của từ trong bài báo).

2. **Inverse Document Frequency (IDF):**
   $$IDF(t) = \log\left(\frac{N}{df_t}\right)$$
   (N là tổng số bài, $df_t$ là số bài có chứa từ t. Chỉ số này giúp hạ thấp trọng số của các từ quá phổ biến).

3. **Trọng số cuối cùng:**
   $$W = TF \times IDF$$

#### B. Doc2Vec (Ngữ nghĩa sâu)
Dựa trên kiến trúc PV-DM (Distributed Memory version of Paragraph Vector).
- **Cơ chế:** Mỗi văn bản được gán một vector định danh (Paragraph ID). Trong quá trình huấn luyện mạng nơ-ron, Paragraph ID sẽ cùng với các từ xung quanh tham gia vào việc dự đoán từ tiếp theo.
- **Kết quả:** Các bài báo có chủ đề hoặc phong cách viết giống nhau (dù không dùng chung từ ngữ) sẽ có khoảng cách Cosine giữa hai vector rất ngắn.

---

### 4.3. Phân Tích 5 Thuật Toán Phân Loại AI

#### 1. Logistic Regression (Hồi quy Logistic)
- **Nguyên lý:** Sử dụng một hàm tuyến tính $z = w^Tx + b$ sau đó đưa qua hàm Sigmoid.
- **Công thức Sigmoid:** 
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Đặc điểm:** Phù hợp nhất cho các bài toán phân loại nhị phân (Thật/Giả). Nó cho biết xác suất (ví dụ: 0.85 là 85% khả năng là tin giả).

#### 2. Support Vector Machine (SVM)
- **Nguyên lý:** Tìm một siêu phẳng (Hyperplane) trong không gian n-chiều để phân tách hai lớp dữ liệu sao cho **Margin** (khoảng cách đến các điểm gần nhất) là lớn nhất.
- **Mục tiêu:** 
  $$\text{Maximize } \frac{2}{\|w\|}$$
- **Ưu điểm:** Cực kỳ mạnh mẽ khi số lượng đặc trưng (từ vựng) nhiều hơn số lượng mẫu dữ liệu.

#### 3. Random Forest (Rừng Ngẫu Nhiên)
- **Nguyên lý:** Là thuật toán **Ensemble Learning** kiểu Bagging. Nó xây dựng hàng trăm cây quyết định (Decision Trees) độc lập.
- **Cơ chế:** Mỗi cây sẽ được huấn luyện trên một tập con ngẫu nhiên của dữ liệu và đặc trưng. Khi dự đoán, hệ thống sẽ lấy "phiếu bầu" đa số từ tất cả các cây.
- **Lợi ích:** Tránh hiện tượng Overfitting (quá khớp) và rất ổn định với dữ liệu nhiễu.

#### 4. Decision Tree (Cây Quyết Định)
- **Nguyên lý:** Xây dựng một cấu trúc phân cấp các câu hỏi. Tại mỗi nút, nó chọn đặc trưng (từ khóa) có **Information Gain** (Độ lợi thông tin) cao nhất để phân loại.
- **Công thức Entropy:** 
  $$H(S) = -\sum p_i \log_2(p_i)$$
- **Đặc điểm:** Cực kỳ trực quan, giúp người dùng thấy rõ AI đã chọn từ khóa nào để "nghi ngờ" tin tức.

#### 5. Naive Bayes (Bayes Ngây Thơ)
- **Nguyên lý:** Dựa trên định lý Bayes với giả định rằng sự hiện diện của một từ trong văn bản là độc lập với các từ khác.
- **Định lý Bayes:**
  $$P(Fake|Text) = \frac{P(Text|Fake) \times P(Fake)}{P(Text)}$$
- **Đặc điểm:** Hoạt động rất tốt với văn bản ngắn và cần ít dữ liệu huấn luyện hơn các mô hình phức tạp.

---

## 📈 Hệ Thống Chỉ Số Đánh Giá (Metrics)

Trong phát hiện tin giả, chúng ta không thể chỉ dựa vào Accuracy (Độ chính xác tổng quát).

1. **Accuracy (Độ chính xác tổng):**
   $$Acc = \frac{TP + TN}{TP + TN + FP + FN}$$
   (Tỷ lệ dự đoán đúng trên toàn bộ tập dữ liệu).

2. **Precision (Độ chuẩn xác):**
   $$P = \frac{TP}{TP + FP}$$
   (Trả lời câu hỏi: "Trong số các bài AI báo là tin giả, có bao nhiêu bài thật sự là tin giả?"). Chỉ số này cao giúp tránh việc báo oan cho tin thật.

3. **Recall (Độ phủ/Độ nhạy):**
   $$R = \frac{TP}{TP + FN}$$
   (Trả lời câu hỏi: "Hệ thống bắt được bao nhiêu phần trăm tổng số tin giả thực tế?"). Chỉ số này cao giúp không bỏ lọt tin giả độc hại.

4. **F1-Score (Điểm cân bằng):**
   $$F1 = 2 \times \frac{P \times R}{P + R}$$
   (Trung bình điều hòa giữa Precision và Recall. Đây là chỉ số đáng tin cậy nhất khi tập dữ liệu bị mất cân bằng giữa tin thật và giả).

5. **False Alarm Rate (Tỷ lệ báo động giả):**
   $$FAR = \frac{FP}{FP + TN}$$
   (Tỷ lệ tin thật bị gán nhãn nhầm là tin giả. Đây là thông số sống còn trong uy tín của hệ thống).

---

## 🧱 Ma Trận Nhầm Lẫn (Confusion Matrix)

Đây là bảng thống kê chi tiết các trường hợp:

- **True Positive (TP):** Tin giả và AI đoán đúng là giả. (Thành công ✅)
- **True Negative (TN):** Tin thật và AI đoán đúng là thật. (Thành công ✅)
- **False Positive (FP):** Tin thật nhưng AI đoán là giả. (Sai lầm "Oan sai" ❌)
- **False Negative (FN):** Tin giả nhưng AI đoán là thật. (Sai lầm "Nguy hiểm" ❌)

---

## 🎨 Giao Diện Người Dùng & Trải Nghiệm

Hệ thống cung cấp 2 chế độ:
1. **Detector Mode:** Người dùng nhập tin, nhận kết quả từ 5 mô hình đồng thời, xem biểu đồ xác suất và vector Doc2Vec.
2. **Theory Academy:** Trang lý thuyết tương tác với Sidebar, cung cấp mọi định nghĩa và công thức toán học nêu trên một cách trực quan.

---

## 🛠 Công Nghệ Sử Dụng

- **Backend:** Python, FastAPI, Scikit-learn, Joblib, Gensim (Doc2Vec), Underthesea.
- **Frontend:** React.js, Vite, Tailwind CSS, Axios, Framer Motion (Animations).
- **Data:** Dataset tin tức tiếng Việt (~21,000 dòng).

---

© 2026 Vietnamese Fake News Detection Project. Đóng góp bởi cộng đồng nghiên cứu AI giáo dục.
