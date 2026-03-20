# Hệ Thống Giáo Dục & Nhận Diện Tin Giả Tiếng Việt (Vietnamese Fake News Detection & Education System)

Dự án này là một nền tảng tích hợp giữa công nghệ học máy (Machine Learning) tiên tiến và môi trường giáo dục trực quan, nhằm giúp người dùng không chỉ nhận biết tin giả mà còn hiểu rõ các nguyên lý khoa học đằng sau các thuật toán AI.

---

## 📑 Mục Lục
1. [Tổng Quan Dự Án](#-tổng-quan-dự-án)
2. [Hướng Dẫn Cài Đặt & Khởi Chạy](#-hướng-dẫn-cài-đặt--khởi-chạy)
3. [Quy Trình Huấn Luyện AI Chuyên Sâu](#-quy-trình-huấn-luyện-ai-chuyên-sâu)
4. [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
5. [Nền Tảng Lý Thuyết Chuyên Sâu](#-nền-tảng-lý-thuyết-chuyên-sâu)
6. [Hệ Thống Chỉ Số Đánh Giá (Metrics)](#-hệ-thống-chỉ-số-đánh-giá-metrics)
7. [Giao Diện Người Dùng & Trải Nghiệm](#-giao-diện-người-dùng--trải-nghiệm)

---

## 🚀 Tổng Quan Dự Án

Hệ thống được thiết kế để giải quyết vấn đề tin giả (Fake News) đang ngày càng tinh vi trên mạng xã hội Việt Nam. Khác với các công cụ kiểm tra thông thường, dự án này tập trung vào khía cạnh **giáo dục**:
- **Đa mô hình:** Sử dụng đồng thời 5 thuật toán để người dùng so sánh độ chính xác.
- **Minh họa trực quan:** Hiển thị các chỉ số toán học, sơ đồ luồng dữ liệu và lý thuyết chi tiết.
- **Xử lý đặc thù:** Tối ưu hóa cho ngôn ngữ tiếng Việt với các thư viện chuyên dụng.
- **Minh bạch quy trình:** Toàn bộ quá trình từ dữ liệu thô đến mô hình hoàn chỉnh được báo cáo chi tiết.

---

## 🛠 Hướng Dẫn Cài Đặt & Khởi Chạy

Hệ thống đã được tối ưu hóa với các file batch tự động kiểm tra và cài đặt môi trường.

### Yêu cầu hệ thống:
- **Windows OS** (Hỗ trợ tốt nhất cho các file .bat)
- **Python 3.8+**
- **Node.js 16+**

### Cách khởi chạy nhanh:

1. **Khởi chạy toàn bộ hệ thống (All-in-One):**
Chỉ cần chạy file sau, hệ thống sẽ tự động kiểm tra Python, NPM, cài đặt thư viện thiếu và mở trình duyệt:
```bash
run_all.bat
```

2. **Huấn luyện lại AI (Nếu có dữ liệu mới):**
Nếu bạn cập nhật thêm tin tức vào `data/fake_news.csv`, hãy chạy file này để cập nhật "bộ não" cho hệ thống:
```bash
train_ai.bat
```

---

## 🧠 Quy Trình Huấn Luyện AI Chuyên Sâu (7 Bước)

Hệ thống sử dụng một quy trình huấn luyện nghiêm ngặt gồm 7 bước, đảm bảo tính khoa học và minh bạch:

1. **Đọc dữ liệu nguồn:** Phân tích cấu trúc file CSV, thống kê số lượng mẫu tin thật/giả và kiểm tra tính toàn vẹn.
2. **Tiền xử lý (Preprocessing):** Sử dụng `underthesea` để tách từ tiếng Việt, làm sạch nhiễu và chuẩn hóa văn bản.
3. **Trích xuất đặc trưng (TF-IDF):** Phân tích tần suất từ và tầm quan trọng của chúng. Lưu trữ `tfidf_vectorizer.pkl` cho Backend.
4. **Vectơ hóa ngữ nghĩa (Doc2Vec):** Chuyển câu văn thành không gian 100 chiều để AI hiểu được ngữ cảnh sâu xa.
5. **Huấn luyện Đa mô hình (Classifiers):** 
    - Chạy song song 5 thuật toán: **Logistic Regression, SVM, Random Forest, Naive Bayes, Decision Tree**.
    - Báo cáo tiến độ (%) và thời gian thực hiện của từng mô hình.
6. **Lưu trữ báo cáo (Metrics):** Xuất toàn bộ chỉ số Precision, Recall, F1 vào `models_metrics.json`.
7. **Tổng kết hệ thống:** Báo cáo tổng thời gian và trạng thái sẵn sàng của các file mô hình `.pkl`.

---

## 🏗 Kiến Trúc Hệ Thống

Dữ liệu di chuyển qua các tầng sau:
1. **Tầng Thu Thập:** Tiếp nhận văn bản thô từ người dùng.
2. **Tầng Tiền Xử Lý:** Sử dụng `underthesea` để tách từ và Chuẩn hóa (Normalization).
3. **Tầng Vectơ Hóa:** TF-IDF (Tính toán trọng số từ) & Doc2Vec (Vectơ hóa ngữ nghĩa).
4. **Tầng Dự Đoán:** 5 mô hình thực hiện phân loại song song và trả về xác suất tin giả.
5. **Tầng Hiển Thị:** Frontend React hiển thị kết quả kèm theo các giải thích giáo dục tương ứng.

---

## 📚 Nền Tảng Lý Thuyết & Thuật Toán

### 5.1. Vectơ Hóa Văn Bản (TF-IDF vs Doc2Vec)
- **TF-IDF:** Tính toán dựa trên tần suất từ (Term Frequency) và nghịch đảo tần suất văn bản (Inverse Document Frequency). Giúp xác định các từ khóa "đắt giá" trong tin tức.
- **Doc2Vec:** Sử dụng mạng nơ-ron để gán cho mỗi bài báo một "tọa độ" trong không gian vector. Các bài báo có nội dung tương tự sẽ nằm gần nhau về mặt toán học.

### 5.2. Các Thuật Toán AI Chủ Đạo
- **SVM:** Tìm siêu phẳng tối ưu để chia tách dữ liệu.
- **Logistic Regression:** Dự đoán xác suất dựa trên hàm Sigmoid (0 đến 1).
- **Random Forest:** Kết hợp kết quả từ 100 cây quyết định để đưa ra lựa chọn chính xác nhất.
- **Naive Bayes:** Dựa trên xác suất thống kê Bayes (cực nhanh và hiệu quả với văn bản ngắn).
- **Decision Tree:** Chia nhỏ dữ liệu theo các câu hỏi từ khóa quan trọng.

---

## 📈 Hệ Thống Chỉ Số Đánh Giá (Metrics)

Để đảm bảo AI không bị "đoán mò", hệ thống sử dụng các chỉ số khắt khe:
- **Accuracy:** Độ chính xác tổng quát.
- **Precision:** Khả năng dự đoán đúng tin giả (tránh báo oan tin thật).
- **Recall:** Khả năng bắt trọn mọi tin giả (tránh bỏ lọt tin độc hại).
- **F1-Score:** Điểm trung bình hài hòa giữa Precision và Recall.
- **False Alarm Rate:** Tỷ lệ báo động giả (Chỉ số quan trọng để đánh giá độ tin cậy).

---

## 🎨 Giao Diện Người Dùng

- **Detector Mode:** Nhập văn bản -> AI phân tích -> Xem xác suất & giải thích lý thuyết.
- **Theory Academy:** Thư viện lý thuyết chuyên sâu về NLP và Machine Learning.
- **Metrics Dashboard:** Xem biểu đồ so sánh sức mạnh giữa các thuật toán AI.

---

## 🛠 Công Nghệ Sử Dụng

- **Backend:** FastAPI, Scikit-learn, Joblib, Gensim, Underthesea.
- **Frontend:** React, Vite, Tailwind CSS, Framer Motion.
- **Automation:** Batch Script (.bat) cho Windows.

---

© 2026 Vietnamese Fake News Detection Project. Đóng góp bởi cộng đồng nghiên cứu AI giáo dục.
