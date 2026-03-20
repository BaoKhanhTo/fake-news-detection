import pandas as pd
import joblib
import os
import sys
import numpy as np
import json
import time
import logging
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Chan cac canh bao va log he thong de hien thi sach se
warnings.filterwarnings("ignore")
logging.getLogger('gensim').setLevel(logging.ERROR)

# Add parent dir to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

def print_header(step, title):
    print(f"\n" + "="*80)
    print(f" [{step}/7] {title.upper()}")
    print("="*80)

def train_all_models(data_path, models_dir):
    start_total = time.time()
    
    # --- BUOC 1: DOC DU LIEU ---
    print_header(1, "Khoi tao va Doc du lieu nguon")
    if not os.path.exists(data_path):
        print(f" [ERROR] Khong tim thay file {data_path}")
        return
    
    df = pd.read_csv(data_path)
    text_col = 'post_message' if 'post_message' in df.columns else 'text'
    df.dropna(subset=[text_col, 'label'], inplace=True)
    
    print(f"  - Tep tin: {os.path.basename(data_path)}")
    print(f"  - Tong so mau du lieu tho: {len(df)} ban ghi")
    print(f"  - Trang thai: Da san sang de xu ly.")

    # --- BUOC 2: PREPROCESSING ---
    print_header(2, "Tien hanh lam sach va Phan tach du lieu")
    start_pre = time.time()
    tqdm.pandas(desc="    Tien trinh lam sach (%)")
    df['clean_text'] = df[text_col].progress_apply(clean_text)
    
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  - Thoi gian xu ly: {time.time() - start_pre:.2f}s")
    print(f"  - Ket qua phan tach:")
    print(f"    + Tap Huan luyen (Train): {len(X_train)} mau")
    print(f"    + Tap Kiem thu (Test): {len(X_test)} mau")

    # --- BUOC 3: TF-IDF VECTORIZATION (CHI TIET HON) ---
    print_header(3, "Trich xuat dac trung van ban (TF-IDF)")
    start_tfidf = time.time()
    
    print(f"  - [1/4] Khoi tao cau hinh TF-IDF (Max features: 5000)...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    
    print(f"  - [2/4] Dang phan tich tu vung tu {len(X_train)} mau van ban...")
    tfidf_vectorizer.fit(X_train)
    
    vocab_size = len(tfidf_vectorizer.vocabulary_)
    print(f"  - [3/4] Dang tinh toan trong so IDF cho {vocab_size} dac trung...")
    
    print(f"  - [4/4] Dang dong goi va luu tru Vectorizer...")
    joblib.dump(tfidf_vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    
    print(f"  - Hoan tat. Thoi gian thuc hien: {time.time() - start_tfidf:.2f}s")
    print(f"  - Trang thai: Da san sang {vocab_size} dac trung cho cac buoc sau.")

    # --- BUOC 4: DOC2VEC ---
    print_header(4, "Xay dung Vector dai dien (Doc2Vec)")
    start_d2v = time.time()
    tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(X)]
    
    d2v_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
    print(f"  - Dang khoi tao tu vung (Vocabulary size: {len(X)} docs)...")
    d2v_model.build_vocab(tagged_data)
    
    # Training voi thanh tien trinh Epoch
    print(f"  - Dang huan luyen Vector hoa:")
    pbar = tqdm(total=40, desc="    Tien do huan luyen (%)")
    for epoch in range(40):
        d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=1)
        pbar.update(1)
    pbar.close()
    
    d2v_model.save(os.path.join(models_dir, "doc2vec.model"))
    print(f"  - Hoan tat. Thoi gian thuc hien: {time.time() - start_d2v:.2f}s")

    # --- BUOC 5: CLASSIFIERS ---
    print_header(5, "Huan luyen cac thuat toan Phan loai")
    
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000), 'logistic_regression'),
        ('SVM (Support Vector)', SVC(probability=True), 'svm'),
        ('Random Forest', RandomForestClassifier(n_estimators=100), 'random_forest'),
        ('Naive Bayes', MultinomialNB(), 'naive_bayes'),
        ('Decision Tree', DecisionTreeClassifier(), 'decision_tree')
    ]

    models_info = {}
    for i, (full_name, clf, short_name) in enumerate(classifiers):
        print(f"\n  ({i+1}/{len(classifiers)}) --- {full_name.upper()} ---")
        start_step = time.time()
        
        # Thong tin du lieu vao cho model nay
        print(f"    [1] Chuan bi: Su dung {vocab_size} dac trung TF-IDF...")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, vocabulary=tfidf_vectorizer.vocabulary_)),
            ('clf', clf)
        ])
        
        print(f"    [2] Thuc thi: Dang huan luyen thuat toan tren {len(X_train)} mau...")
        pipeline.fit(X_train, y_train)
        
        print(f"    [3] Danh gia: Dang kiem thu tren {len(X_test)} mau...")
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        duration = time.time() - start_step
        print(f"    [4] Ket qua: Accuracy = {acc*100:.2f}% | F1 = {f1*100:.2f}%")
        print(f"    [5] Thoi gian: {duration:.2f} giay")
        
        models_info[short_name] = {
            "accuracy": float(acc),
            "f1": float(f1),
            "duration": float(duration)
        }
        joblib.dump(pipeline, os.path.join(models_dir, f"{short_name}.pkl"))

    # --- BUOC 6: LUU KET QUA ---
    print_header(6, "Luu tru bao cao Metrics")
    with open(os.path.join(models_dir, "models_metrics.json"), "w") as f:
        json.dump(models_info, f, indent=4)
    print(f"  - Da cap nhat tep tin: models/models_metrics.json")

    # --- BUOC 7: TONG KET ---
    print_header(7, "Tong ket toan bo quy trinh")
    print(f"  - Tong thoi gian chay: {time.time() - start_total:.2f} giay")
    print(f"  - Trang thai: Tat ca mo hinh da duoc toi uu va luu tru.")
    print("="*80 + "\n")

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "fake_news.csv")
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_all_models(DATA_PATH, MODELS_DIR)
