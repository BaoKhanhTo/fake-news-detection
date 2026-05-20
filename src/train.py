import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import clean_text


TEXT_COLUMNS = ["post_message", "Maintext", "maintext", "text", "content"]
LABEL_COLUMNS = ["Label", "label"]

PREFERRED_FILES = {
    "train": [
        Path("train/fake_news.csv"),
        Path("train/update_train_data.csv"),
    ],
    "validation": [
        Path("val/val_data.csv"),
    ],
    "test": [
        Path("test/fix_test_data.csv"),
    ],
}


def find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def normalize_frame(df, source_name):
    text_col = find_column(df.columns, TEXT_COLUMNS)
    label_col = find_column(df.columns, LABEL_COLUMNS)

    if not text_col or not label_col:
        raise ValueError(f"{source_name} is missing a supported text or label column.")

    return df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})


def load_split(split_name, files):
    frames = []
    report = []

    for file_path in files:
        if not file_path.exists():
            print(f"[WARN] Missing {split_name} file: {file_path}")
            continue

        raw_df = pd.read_csv(file_path)
        normalized = normalize_frame(raw_df, str(file_path))
        frames.append(normalized)
        report.append({"file": str(file_path), "rows": len(normalized)})

    if not frames:
        raise FileNotFoundError(f"No valid CSV files found for {split_name}.")

    return pd.concat(frames, ignore_index=True), report


def clean_split(df):
    before = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    tqdm.pandas(desc="Cleaning text")
    df["text"] = df["text"].progress_apply(clean_text)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.strip() != ""]
    after_missing = len(df)
    df = df.drop_duplicates(subset=["text", "label"])
    duplicates_removed = after_missing - len(df)

    return df.reset_index(drop=True), {
        "rows_before": before,
        "rows_after": len(df),
        "duplicates_removed": duplicates_removed,
    }


def print_split_report(split_name, file_report, cleaning_report, df):
    print(f"\n{split_name.upper()} DATA")
    for item in file_report:
        print(f"  - {item['file']}: {item['rows']} rows")
    print(f"  - Rows before cleaning: {cleaning_report['rows_before']}")
    print(f"  - Rows after cleaning: {cleaning_report['rows_after']}")
    print(f"  - Duplicates removed: {cleaning_report['duplicates_removed']}")
    print("  - Label distribution:")
    print(df["label"].value_counts().sort_index().to_string())


def warn_cross_split_overlap(train_df, val_df, test_df):
    checks = [
        ("train vs validation", train_df, val_df),
        ("train vs test", train_df, test_df),
        ("validation vs test", val_df, test_df),
    ]

    for name, left, right in checks:
        overlap_count = len(set(left["text"]).intersection(set(right["text"])))
        if overlap_count:
            print(f"[WARN] {overlap_count} duplicate text values found across {name}.")


def train_logistic_regression(data_dir="data", models_dir="models"):
    started_at = time.time()
    data_path = Path(data_dir)
    os.makedirs(models_dir, exist_ok=True)

    print("Loading preferred dataset files...")
    train_df, train_files = load_split(
        "train", [data_path / path for path in PREFERRED_FILES["train"]]
    )
    val_df, val_files = load_split(
        "validation", [data_path / path for path in PREFERRED_FILES["validation"]]
    )
    test_df, test_files = load_split(
        "test", [data_path / path for path in PREFERRED_FILES["test"]]
    )

    total_before = len(train_df) + len(val_df) + len(test_df)

    train_df, train_cleaning = clean_split(train_df)
    val_df, val_cleaning = clean_split(val_df)
    test_df, test_cleaning = clean_split(test_df)

    total_after = len(train_df) + len(val_df) + len(test_df)
    duplicates_removed = (
        train_cleaning["duplicates_removed"]
        + val_cleaning["duplicates_removed"]
        + test_cleaning["duplicates_removed"]
    )

    print("\nDATASET REPORT")
    print(f"Total rows before cleaning: {total_before}")
    print(f"Total rows after cleaning: {total_after}")
    print(f"Total duplicates removed within splits: {duplicates_removed}")
    print_split_report("train", train_files, train_cleaning, train_df)
    print_split_report("validation", val_files, val_cleaning, val_df)
    print_split_report("test", test_files, test_cleaning, test_df)
    warn_cross_split_overlap(train_df, val_df, test_df)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    print("\nTraining Logistic Regression with TF-IDF...")
    train_started_at = time.time()
    pipeline.fit(train_df["text"], train_df["label"])
    training_duration = time.time() - train_started_at

    val_predictions = pipeline.predict(val_df["text"])
    test_predictions = pipeline.predict(test_df["text"])

    print(f"Validation accuracy: {accuracy_score(val_df['label'], val_predictions):.4f}")

    metrics = {
        "accuracy": float(accuracy_score(test_df["label"], test_predictions)),
        "precision": float(precision_score(test_df["label"], test_predictions, zero_division=0)),
        "recall": float(recall_score(test_df["label"], test_predictions, zero_division=0)),
        "f1_score": float(f1_score(test_df["label"], test_predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(test_df["label"], test_predictions).tolist(),
        "training_duration": float(training_duration),
    }

    model_path = Path(models_dir) / "logistic_regression.pkl"
    metrics_path = Path(models_dir) / "models_metrics.json"
    joblib.dump(pipeline, model_path)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4, ensure_ascii=False)

    print("\nSaved artifacts:")
    print(f"  - {model_path}")
    print(f"  - {metrics_path}")
    print(f"Total run duration: {time.time() - started_at:.2f}s")


if __name__ == "__main__":
    train_logistic_regression()
