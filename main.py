from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
import sys
import json
import numpy as np
from gensim.models.doc2vec import Doc2Vec

# Import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocess import clean_text

app = FastAPI(title="Hệ thống Giáo dục Nhận biết Tin giả")

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models and Metrics
MODELS_DIR = "models"
models = {}
metrics = {}
d2v_model = None

def load_resources():
    global models, metrics, d2v_model
    print(f"DEBUG: Loading resources from directory: {os.path.abspath(MODELS_DIR)}")
    try:
        metrics_path = os.path.join(MODELS_DIR, "models_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            print(f"DEBUG: Metrics loaded for {list(metrics.keys())}")
        else:
            print(f"DEBUG: Metrics file NOT found: {metrics_path}")
        
        model_names = ['logistic_regression', 'svm', 'decision_tree', 'naive_bayes', 'random_forest']
        for name in model_names:
            path = os.path.join(MODELS_DIR, f"{name}.pkl")
            if os.path.exists(path):
                try:
                    print(f"DEBUG: Attempting to load {name}...")
                    models[name] = joblib.load(path)
                    print(f"DEBUG: Success loading {name}")
                except Exception as model_e:
                    print(f"DEBUG: Failed to load {name}: {model_e}")
            else:
                print(f"DEBUG: Model file NOT found: {path}")
        
        d2v_path = os.path.join(MODELS_DIR, "doc2vec.model")
        if os.path.exists(d2v_path):
            print(f"DEBUG: Loading Doc2Vec from {d2v_path}...")
            d2v_model = Doc2Vec.load(d2v_path)
        else:
            print(f"DEBUG: Doc2Vec file NOT found: {d2v_path}")
            
        print(f"DEBUG: Total models loaded: {list(models.keys())}")
    except Exception as e:
        print(f"DEBUG: Error loading resources: {e}")

load_resources()

class NewsItem(BaseModel):
    text: str

# Educational Content
EDUCATIONAL_CONTENT = {
    "logistic_regression": {
        "name": "Logistic Regression (Hồi quy Logistic)",
        "concept": "Hồi quy Logistic là một thuật toán phân loại được sử dụng để dự đoán xác suất của một biến mục tiêu nhị phân.",
        "principle": "Thuật toán sử dụng hàm Logistic (hàm Sigmoid) để chuyển đổi đầu ra thành xác suất từ 0 đến 1.",
        "flow": "Văn bản -> TF-IDF -> Tính tổng trọng số -> Hàm Sigmoid -> Xác suất -> Phân loại."
    },
    "svm": {
        "name": "Support Vector Machine (Máy vector hỗ trợ)",
        "concept": "SVM tìm kiếm một siêu phẳng tối ưu để phân tách các điểm dữ liệu thành hai lớp.",
        "principle": "Nó cố gắng tối đa hóa khoảng cách giữa các điểm dữ liệu gần nhất của hai lớp.",
        "flow": "Văn bản -> TF-IDF -> Ánh xạ không gian -> Tìm siêu phẳng -> Phân loại."
    },
    "decision_tree": {
        "name": "Decision Tree (Cây quyết định)",
        "concept": "Cây quyết định sử dụng cấu trúc cây để đưa ra các quyết định dựa trên đặc trưng.",
        "principle": "Chia nhỏ dữ liệu dựa trên các câu hỏi về từ khóa quan trọng nhất.",
        "flow": "Văn bản -> TF-IDF -> Kiểm tra nút -> Đi theo nhánh -> Kết luận tại lá."
    },
    "naive_bayes": {
        "name": "Naive Bayes (Bayes ngây thơ)",
        "concept": "Dựa trên định lý Bayes với giả định các đặc trưng độc lập.",
        "principle": "Tính xác suất hậu nghiệm dựa trên tần suất xuất hiện của từ.",
        "flow": "Văn bản -> TF-IDF -> Tính xác suất Bayes -> So sánh -> Chọn lớp."
    },
    "random_forest": {
        "name": "Random Forest (Rừng ngẫu nhiên)",
        "concept": "Xây dựng hàng trăm cây quyết định và lấy phiếu bầu đa số.",
        "principle": "Giảm quá khớp bằng cách kết hợp kết quả của nhiều cây độc lập.",
        "flow": "Văn bản -> TF-IDF -> Qua hàng trăm cây -> Voting -> Kết luận."
    },
    "doc2vec": {
        "name": "Doc2Vec (Vectơ hóa văn bản)",
        "concept": "Biểu diễn văn bản dưới dạng các vectơ số có độ dài cố định hiểu được ngữ nghĩa.",
        "principle": "Học ngữ cảnh bằng cách dự đoán từ xung quanh trong đoạn văn.",
        "flow": "Văn bản -> Tokenize -> Mạng nơ-ron -> Vectơ 100 chiều -> Ngữ nghĩa."
    }
}

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/predict")
def predict(item: NewsItem):
    if not models:
        load_resources()
        if not models:
            raise HTTPException(status_code=503, detail="Models not loaded on server")
    
    cleaned_text = clean_text(item.text)
    results = {}
    
    for name in ['logistic_regression', 'svm', 'decision_tree', 'naive_bayes', 'random_forest']:
        if name in models:
            try:
                pipeline = models[name]
                probs = pipeline.predict_proba([cleaned_text])[0]
                label = int(pipeline.predict([cleaned_text])[0])
                
                model_metrics = metrics.get(name, {
                    "accuracy": 0, "f1": 0, "recall": 0, "precision": 0, "false_alarm_rate": 0,
                    "confusion_matrix": [[0, 0], [0, 0]]
                })
                
                results[name] = {
                    "prediction": "Fake" if label == 1 else "Real",
                    "fake_probability": float(probs[1]),
                    "real_probability": float(probs[0]),
                    "metrics": model_metrics,
                    "education": EDUCATIONAL_CONTENT.get(name, {})
                }
            except Exception as e:
                print(f"DEBUG: Error predicting with {name}: {e}")
    
    if not results:
        raise HTTPException(status_code=500, detail="Prediction failed for all models.")
    
    d2v_vector = []
    if d2v_model:
        d2v_vector = d2v_model.infer_vector(cleaned_text.split()).tolist()
    
    return {
        "input_text": item.text,
        "cleaned_text": cleaned_text,
        "results": results,
        "doc2vec_vector": d2v_vector[:10],
        "education_doc2vec": EDUCATIONAL_CONTENT["doc2vec"]
    }

# Serve Frontend
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
