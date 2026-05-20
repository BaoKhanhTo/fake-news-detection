# Hệ thống nhận diện tin giả tiếng Việt

Dự án xây dựng một ứng dụng web đơn giản bằng Django để phân loại nội dung tin tức tiếng Việt thành **Real News** hoặc **Fake News**. Pipeline hiện tại sử dụng tiền xử lý tiếng Việt, trích xuất đặc trưng bằng TF-IDF (1–2 grams, `sublinear_tf`) và mô hình **Logistic Regression** với `class_weight="balanced"` để xử lý mất cân bằng lớp.

> Các phiên bản trước có thử nghiệm với PhoBERT/ViBERT/SBERT nhưng đã được loại bỏ trong bản refactor `af93adc` để giữ stack tối thiểu (Django + scikit-learn). Tham khảo lịch sử git nếu cần khôi phục.

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
│   ├── test/
│   │   └── fix_test_data.csv
│   └── processed/
│       ├── train_clean.csv
│       ├── val_clean.csv
│       └── test_clean.csv
├── models/
│   ├── logistic_regression.pkl     # Model đã huấn luyện
│   ├── models_metrics.json         # Chỉ số đánh giá model
│   └── threshold.json              # Ngưỡng xác suất tối ưu cho lớp Fake
├── src/
│   ├── preprocess.py               # Làm sạch văn bản và tách từ
│   ├── train.py                    # Huấn luyện model + tuning threshold
│   └── show_metrics.py             # Script xem metrics
├── manage.py
├── requirements.txt
├── .env.example                    # Mẫu biến môi trường cho production
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

## Cấu hình môi trường

Khi chạy local có thể bỏ qua bước này (Django dùng `DEBUG=True` và `SECRET_KEY` mặc định). **Khi deploy production**, bắt buộc set các biến môi trường — xem `.env.example`:

| Biến | Mô tả | Bắt buộc |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | Khoá bí mật dài, ngẫu nhiên | Yes (khi `DJANGO_DEBUG=0`) |
| `DJANGO_DEBUG` | `1` để bật debug, `0` cho production | No (mặc định `1`) |
| `DJANGO_ALLOWED_HOSTS` | Danh sách host phân tách bằng dấu phẩy | Yes (khi `DJANGO_DEBUG=0`) |

Sinh `SECRET_KEY` mới:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Nếu chạy production mà vẫn để `SECRET_KEY` mặc định, server sẽ từ chối khởi động.

## Huấn luyện mô hình

Chạy lệnh:

```bash
python src/train.py
```

Script huấn luyện sẽ:

- Đọc dữ liệu từ `data/train`, `data/val` và `data/test`.
- Chuẩn hóa tên cột văn bản và nhãn.
- Làm sạch dữ liệu, loại bỏ dòng thiếu, dòng rỗng, nhãn không hợp lệ và bản ghi trùng.
- Tạo lại bộ chia dữ liệu sạch theo tỷ lệ 70% train, 15% validation và 15% test.
- Kiểm tra và loại bỏ trùng lặp `cleaned_text` giữa các split để tránh data leakage.
- Lưu split mới vào `data/processed/train_clean.csv`, `data/processed/val_clean.csv` và `data/processed/test_clean.csv`.
- Huấn luyện pipeline gồm `TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)` và `LogisticRegression(max_iter=2000, class_weight="balanced")`.
- Tự động tìm ngưỡng xác suất tối ưu F1 cho lớp Fake trên tập validation, lưu vào `models/threshold.json`.
- Lưu model vào `models/logistic_regression.pkl`.
- Lưu chỉ số đánh giá (cả threshold mặc định 0.5 và threshold đã tune) vào `models/models_metrics.json`.

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

Theo file `models/models_metrics.json`, model hiện tại đạt (819 mẫu test, lớp Fake = 186):

| Metric | Threshold mặc định (0.5) | Threshold đã tune (0.45) |
| --- | ---: | ---: |
| Accuracy | 0.8620 | 0.8462 |
| Precision (Fake) | 0.6872 | 0.6327 |
| Recall (Fake) | 0.7204 | 0.7688 |
| F1-score (Fake) | 0.7034 | 0.6942 |

So với phiên bản chưa cân bằng lớp, **Recall lớp Fake tăng từ 0.5161 lên 0.7204** (+20 điểm) — model không còn bỏ sót gần một nửa tin giả như trước. Hạ ngưỡng xuống 0.45 đẩy Recall lên 0.7688 nếu ưu tiên không bỏ sót tin giả hơn là tránh báo nhầm.

## Ghi chú dữ liệu

`src/train.py` giữ nguyên dữ liệu gốc và đọc các file sau để dựng bộ dữ liệu sạch:

- `data/train/fake_news.csv`
- `data/train/train_data.csv`
- `data/train/update_train_data.csv`
- `data/val/val_data.csv`
- `data/val/update_val_data.csv`
- `data/test/fix_test_data.csv`

Các file processed được tạo mới:

- Train: `data/processed/train_clean.csv`
- Validation: `data/processed/val_clean.csv`
- Test: `data/processed/test_clean.csv`

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
