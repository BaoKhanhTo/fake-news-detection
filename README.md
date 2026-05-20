# Hệ thống nhận diện tin giả tiếng Việt

Dự án xây dựng một ứng dụng web đơn giản bằng Django để phân loại nội dung tin tức tiếng Việt thành **Real News** hoặc **Fake News**. Pipeline hiện tại sử dụng tiền xử lý tiếng Việt, trích xuất đặc trưng bằng TF-IDF và mô hình **Logistic Regression**.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [Chạy ứng dụng web](#chạy-ứng-dụng-web)
- [Luồng xử lý](#luồng-xử-lý)
- [Kết quả hiện tại](#kết-quả-hiện-tại)
- [Ghi chú dữ liệu](#ghi-chú-dữ-liệu)

## Tổng quan

Người dùng nhập nội dung tin tức tiếng Việt trên giao diện web. Hệ thống sẽ:

1. Làm sạch và tách từ nội dung đầu vào.
2. Biến đổi văn bản thành vector TF-IDF.
3. Dự đoán bằng mô hình Logistic Regression đã huấn luyện.
4. Trả về nhãn dự đoán và xác suất của từng lớp.

Nhãn dữ liệu đang dùng:

- `0`: Real News
- `1`: Fake News

## Công nghệ sử dụng

- Python
- Django
- Pandas
- Scikit-learn
- Joblib
- Underthesea
- NumPy
- TQDM
- SQLite cho cấu hình database mặc định của Django

## Cấu trúc thư mục

```text
Fake_News_Detect/
├── config/                         # Cấu hình Django project
├── detector/                       # Django app xử lý giao diện và dự đoán
│   ├── services.py                 # Load model và hàm predict_news
│   ├── views.py                    # View nhận form POST từ người dùng
│   └── templates/detector/
│       └── index.html              # Giao diện nhập tin tức và hiển thị kết quả
├── data/
│   ├── train/
│   │   ├── fake_news.csv
│   │   ├── train_data.csv
│   │   └── update_train_data.csv
│   ├── val/
│   │   ├── val_data.csv
│   │   └── update_val_data.csv
│   └── test/
│       └── fix_test_data.csv
├── models/
│   ├── logistic_regression.pkl     # Model đã huấn luyện
│   └── models_metrics.json         # Chỉ số đánh giá model
├── src/
│   ├── preprocess.py               # Làm sạch văn bản và tách từ
│   ├── split_data.py               # Script hỗ trợ chia dữ liệu
│   ├── train.py                    # Huấn luyện model
│   └── show_metrics.py             # Script xem metrics
├── manage.py
├── requirements.txt
└── run_all.bat                     # Chạy kiểm tra, migrate và server Django
```

## Cài đặt

Tạo và kích hoạt môi trường ảo:

```bash
python -m venv venv
venv\Scripts\activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

## Huấn luyện mô hình

Chạy lệnh:

```bash
python src/train.py
```

Script huấn luyện sẽ:

- Đọc dữ liệu từ `data/train`, `data/val` và `data/test`.
- Chuẩn hóa tên cột văn bản và nhãn.
- Làm sạch dữ liệu, loại bỏ dòng thiếu, dòng rỗng và bản ghi trùng trong từng split.
- Huấn luyện pipeline gồm `TfidfVectorizer(max_features=5000)` và `LogisticRegression(max_iter=1000)`.
- Lưu model vào `models/logistic_regression.pkl`.
- Lưu chỉ số đánh giá vào `models/models_metrics.json`.

## Chạy ứng dụng web

Cách nhanh nhất trên Windows:

```bash
run_all.bat
```

Hoặc chạy thủ công:

```bash
python manage.py check
python manage.py migrate
python manage.py runserver
```

Sau đó mở trình duyệt tại:

```text
http://127.0.0.1:8000/
```

## Luồng xử lý

1. Người dùng nhập nội dung tin tức trên trang chủ.
2. `detector/views.py` nhận nội dung từ form POST.
3. `detector/services.py` gọi `predict_news(text)`.
4. `src/preprocess.py` làm sạch văn bản:
   - chuyển về chữ thường;
   - loại bỏ ký tự đặc biệt không cần thiết;
   - giữ lại dấu `!` và `?`;
   - tách từ bằng `underthesea.word_tokenize`.
5. Model trong `models/logistic_regression.pkl` dự đoán nhãn và xác suất.
6. Giao diện hiển thị kết quả:
   - nhãn dự đoán;
   - xác suất Real News;
   - xác suất Fake News;
   - văn bản sau tiền xử lý.

## Kết quả hiện tại

Theo file `models/models_metrics.json`, model hiện tại đạt:

| Metric | Giá trị |
| --- | ---: |
| Accuracy | 0.9379 |
| Precision | 0.9067 |
| Recall | 0.9577 |
| F1-score | 0.9315 |
| Training duration | 0.5816 giây |

Confusion matrix:

```text
[[83, 7],
 [ 3, 68]]
```

## Ghi chú dữ liệu

`src/train.py` ưu tiên các file sau khi huấn luyện:

- Train: `data/train/fake_news.csv`, `data/train/update_train_data.csv`
- Validation: `data/val/val_data.csv`
- Test: `data/test/fix_test_data.csv`

Các cột văn bản được hỗ trợ:

- `post_message`
- `Maintext`
- `maintext`
- `text`
- `content`

Các cột nhãn được hỗ trợ:

- `Label`
- `label`

## Hướng phát triển

- Bổ sung dữ liệu mới để giảm lệch theo thời gian.
- Cải thiện giao diện tiếng Việt và thông báo lỗi thân thiện hơn.
- So sánh thêm các mô hình như SVM, Random Forest hoặc PhoBERT.
- Bổ sung kiểm thử tự động cho tiền xử lý, huấn luyện và dự đoán.
