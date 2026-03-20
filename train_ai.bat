@echo off
title Quy trinh Training AI - Fake News Detection
echo ======================================================
echo    QUY TRINH DUA DU LIEU VAO TRAINING MO HINH AI
echo ======================================================
echo.

:: 1. Kiem tra moi truong
echo [BUOC 1] Kiem tra moi truong Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python chua duoc cai dat.
    pause
    exit /b
)

echo [+] Dang kiem tra va cai dat thu vien can thiet...
call python -m pip install -r requirements.txt --quiet
echo.

:: 2. Kiem tra du lieu dau vao
echo [BUOC 2] Kiem tra du lieu dau vao...
if not exist "data\fake_news.csv" (
    echo [ERROR] Khong tim thay file du lieu data\fake_news.csv
    pause
    exit /b
)
echo [+] Da tim thay file du lieu: data\fake_news.csv
echo.

:: 3. Tien hanh Pre-processing (Lam sach du lieu)
echo [BUOC 3] Tien hanh lam sach du lieu (Preprocessing)...
echo Dang chay: python src\preprocess.py
call python src\preprocess.py
echo [+] Hoan tat buoc lam sach du lieu mau.
echo.

:: 4. Tien hanh Training
echo [BUOC 4] Bat dau qua trinh Training cac mo hinh...
echo (Luu y: Qua trinh nay co the mat vai phut tuy vao luong du lieu)
echo.
echo Dang thuc hien:
echo   - Phan chia du lieu (Train/Test split: 80/20)
echo   - Trich xuat dac trung (TF-IDF Vectorization)
echo   - Huan luyen 5 mo hinh: Logistic Regression, SVM, Decision Tree, Naive Bayes, Random Forest
echo   - Huan luyen mo hinh Doc2Vec cho vector hoa van ban
echo.
call python src\train.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Co loi xay ra trong qua trinh Training.
    pause
    exit /b
)
echo.

:: 5. Kiem tra ket qua sau Training
echo [BUOC 5] Kiem tra cac file mo hinh da duoc tao ra...
dir models\*.pkl /b
if exist "models\doc2vec.model" echo doc2vec.model
echo.

echo [BUOC 6] Ket qua danh gia mo hinh (Metrics):
if exist "models\models_metrics.json" (
    echo [+] Da tao file bao cao: models\models_metrics.json
    echo Noi dung tom tat:
    python -c "import json; m=json.load(open('models/models_metrics.json')); [print(f'- {k}: Accuracy={m[k][\"accuracy\"]:.4f}, F1={m[k][\"f1\"]:.4f}') for k in m]"
) else (
    echo [!] Khong tim thay file metrics.
)

echo.
echo ======================================================
echo    HOAN TAT QUA TRINH TRAINING AI
echo ======================================================
pause