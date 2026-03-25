import pandas as pd
import torch
import os
import sys
import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from tqdm import tqdm

# Add parent dir to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec
    }

def train_phobert(data_path, models_dir):
    start_total = time.time()
    print("="*80)
    print(" [PHOBERT] BAT DAU HUAN LUYEN PHOBERT")
    print("="*80)

    # 1. Doc du lieu
    df = pd.read_csv(data_path)
    text_col = 'post_message' if 'post_message' in df.columns else 'text'
    df.dropna(subset=[text_col, 'label'], inplace=True)
    
    # 2. Preprocessing
    print("  - Dang lam sach van ban...")
    df['clean_text'] = df[text_col].apply(clean_text)
    
    X = df['clean_text'].tolist()
    y = df['label'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Load Tokenizer & Model
    model_name = "vinai/phobert-base-v2"
    print(f"  - Dang tai model va tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. Tokenization
    print("  - Dang ma hoa (Tokenizing) du lieu...")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

    train_dataset = FakeNewsDataset(train_encodings, y_train)
    test_dataset = FakeNewsDataset(test_encodings, y_test)

    # 5. Training Arguments
    output_dir = os.path.join(models_dir, "phobert_checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none", # Tránh yêu cầu wandb hoặc tensorboard
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("  - Dang huan luyen model (co the mat thoi gian)...")
    trainer.train()

    # 7. Evaluate & Save
    print("  - Dang danh gia model...")
    eval_results = trainer.evaluate()
    
    # Predict to get confusion matrix
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Save model and tokenizer
    phobert_save_path = os.path.join(models_dir, "phobert_model")
    model.save_pretrained(phobert_save_path)
    tokenizer.save_pretrained(phobert_save_path)
    print(f"  - Da luu model tai: {phobert_save_path}")

    # Update metrics
    metrics_path = os.path.join(models_dir, "models_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            models_info = json.load(f)
    else:
        models_info = {}

    duration = time.time() - start_total
    models_info["phobert"] = {
        "accuracy": float(eval_results['eval_accuracy']),
        "f1": float(eval_results['eval_f1']),
        "precision": float(eval_results['eval_precision']),
        "recall": float(eval_results['eval_recall']),
        "false_alarm_rate": float(far),
        "duration": float(duration),
        "confusion_matrix": cm.tolist()
    }

    with open(metrics_path, "w") as f:
        json.dump(models_info, f, indent=4)
    
    print(f"  - Da cap nhat bao cao: {metrics_path}")
    print(f"  - Hoan tat PhoBERT trong {duration:.2f}s")
    print("="*80)

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "fake_news.csv")
    MODELS_DIR = "models"
    train_phobert(DATA_PATH, MODELS_DIR)
