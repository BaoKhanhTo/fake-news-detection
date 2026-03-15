import pandas as pd
import joblib
import os
import sys
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Add parent dir to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

def train_all_models(data_path, models_dir):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    text_col = 'post_message' if 'post_message' in df.columns else 'text'
    df.dropna(subset=[text_col, 'label'], inplace=True)
    
    print("Preprocessing data...")
    df['clean_text'] = df[text_col].apply(clean_text)

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_info = {}

    # 1. TF-IDF + Logistic Regression
    print("Training TF-IDF + Logistic Regression...")
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    models_info['logistic_regression'] = evaluate_and_save(lr_pipeline, X_test, y_test, 'logistic_regression', models_dir)

    # 2. TF-IDF + SVM
    print("Training TF-IDF + SVM...")
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', SVC(probability=True))
    ])
    svm_pipeline.fit(X_train, y_train)
    models_info['svm'] = evaluate_and_save(svm_pipeline, X_test, y_test, 'svm', models_dir)

    # 3. TF-IDF + Decision Tree
    print("Training TF-IDF + Decision Tree...")
    dt_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', DecisionTreeClassifier())
    ])
    dt_pipeline.fit(X_train, y_train)
    models_info['decision_tree'] = evaluate_and_save(dt_pipeline, X_test, y_test, 'decision_tree', models_dir)

    # 4. TF-IDF + Naive Bayes
    print("Training TF-IDF + Naive Bayes...")
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)
    models_info['naive_bayes'] = evaluate_and_save(nb_pipeline, X_test, y_test, 'naive_bayes', models_dir)

    # 5. TF-IDF + Random Forest
    print("Training TF-IDF + Random Forest...")
    rf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])
    rf_pipeline.fit(X_train, y_train)
    models_info['random_forest'] = evaluate_and_save(rf_pipeline, X_test, y_test, 'random_forest', models_dir)

    # 6. Doc2Vec Training
    print("Training Doc2Vec model...")
    tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(X)]
    d2v_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
    d2v_model.build_vocab(tagged_data)
    d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
    d2v_model.save(os.path.join(models_dir, "doc2vec.model"))
    
    # Save all metrics
    with open(os.path.join(models_dir, "models_metrics.json"), "w") as f:
        json.dump(models_info, f, indent=4)
    
    print("All models trained and saved!")

def evaluate_and_save(pipeline, X_test, y_test, name, models_dir):
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    # cm format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Alarm Rate
    
    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "false_alarm_rate": float(far),
        "confusion_matrix": cm.tolist()
    }
    
    joblib.dump(pipeline, os.path.join(models_dir, f"{name}.pkl"))
    return metrics

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "fake_news.csv")
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_all_models(DATA_PATH, MODELS_DIR)
