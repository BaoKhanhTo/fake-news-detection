@echo off
title Quy trinh Training PhoBERT - Fake News Detection
echo ======================================================
echo    QUY TRINH HUAN LUYEN MO HINH PHOBERT (TRANSFORMER)
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

echo [+] Dang kiem tra va cai dat thu vien (Transformers, Torch)...
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

:: 3. Tien hanh Training PhoBERT
echo [BUOC 3] Bat dau qua trinh Training PhoBERT...
echo (Luu y: Qua trinh nay co the mat nhieu thoi gian neu khong co GPU)
echo.
echo Dang thuc hien:
echo   - Load Pre-trained PhoBERT (vinai/phobert-base-v2)
echo   - Fine-tuning cho bai toan Fake News Detection
echo   - Cap nhat metrics vao models/models_metrics.json
echo.
call python src\train_phobert.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Co loi xay ra trong qua trinh Training PhoBERT.
    pause
    exit /b
)
echo.

:: 4. Kiem tra ket qua sau Training
echo [BUOC 4] Kiem tra ket qua...
if exist "models\phobert_model" (
    echo [+] Da luu model tai: models\phobert_model
) else (
    echo [!] Khong tim thay thu muc phobert_model.
)

if exist "models\models_metrics.json" (
    echo [+] Da cap nhat file bao cao: models\models_metrics.json
    echo Noi dung tom tat moi nhat:
    python -c "import json; m=json.load(open('models/models_metrics.json')); k='phobert'; print(f'- {k}: Accuracy={m[k][\"accuracy\"]:.4f}, F1={m[k][\"f1\"]:.4f}') if k in m else print('PhoBERT metrics not found')"
)

echo.
echo ======================================================
echo    HOAN TAT QUA TRINH TRAINING PHOBERT
echo ======================================================
pause
