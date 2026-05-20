import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import clean_text


logger = logging.getLogger("fnd.train")


TEXT_COLUMNS = ["post_message", "Maintext", "maintext", "text", "content"]
LABEL_COLUMNS = ["Label", "label"]

SOURCE_FILES = [
    Path("train/fake_news.csv"),
    Path("train/train_data.csv"),
    Path("train/update_train_data.csv"),
    Path("val/val_data.csv"),
    Path("val/update_val_data.csv"),
    Path("test/fix_test_data.csv"),
]

PROCESSED_FILES = {
    "train": Path("processed/train_clean.csv"),
    "validation": Path("processed/val_clean.csv"),
    "test": Path("processed/test_clean.csv"),
}

RANDOM_STATE = 42


def find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def normalize_frame(df, source_name):
    text_col = find_column(df.columns, TEXT_COLUMNS)
    label_col = find_column(df.columns, LABEL_COLUMNS)

    if not text_col or not label_col:
        logger.warning("Skipping %s: missing supported text or label column.", source_name)
        return None

    normalized = df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    normalized["source_file"] = source_name
    return normalized


def load_source_data(data_dir):
    frames = []
    report = []

    for relative_path in SOURCE_FILES:
        file_path = data_dir / relative_path
        if not file_path.exists():
            logger.warning("Missing source file: %s", file_path)
            continue

        raw_df = pd.read_csv(file_path)
        normalized = normalize_frame(raw_df, str(file_path))
        report.append({"file": str(file_path), "rows": len(raw_df)})

        if normalized is not None:
            frames.append(normalized)

    if not frames:
        raise FileNotFoundError("No valid source CSV files found.")

    return pd.concat(frames, ignore_index=True), report


def log_label_distribution(name, df):
    distribution = df["label"].value_counts().sort_index().to_dict()
    logger.info("%s label distribution: %s (total=%d)", name.upper(), distribution, len(df))


def clean_and_deduplicate(df):
    total_loaded = len(df)

    clean_df = df.copy()
    clean_df["text"] = clean_df["text"].astype("string")
    clean_df["label"] = pd.to_numeric(clean_df["label"], errors="coerce")

    clean_df = clean_df.dropna(subset=["text", "label"])
    clean_df["text"] = clean_df["text"].str.strip()
    clean_df = clean_df[clean_df["text"] != ""]
    clean_df = clean_df[clean_df["label"].isin([0, 1])]
    clean_df["label"] = clean_df["label"].astype(int)

    rows_after_invalid = len(clean_df)

    before_raw_dedupe = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["text", "label"]).copy()
    raw_duplicates_removed = before_raw_dedupe - len(clean_df)

    tqdm.pandas(desc="Cleaning text")
    clean_df["cleaned_text"] = clean_df["text"].progress_apply(clean_text)
    clean_df["cleaned_text"] = clean_df["cleaned_text"].astype("string").str.strip()
    clean_df = clean_df[clean_df["cleaned_text"] != ""]

    before_cleaned_dedupe = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["cleaned_text", "label"]).copy()
    cleaned_duplicates_removed = before_cleaned_dedupe - len(clean_df)

    before_conflict_dedupe = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["cleaned_text"], keep="first").copy()
    conflicting_cleaned_text_removed = before_conflict_dedupe - len(clean_df)

    clean_df = clean_df[["text", "cleaned_text", "label"]].reset_index(drop=True)

    report = {
        "total_rows_loaded_before_cleaning": total_loaded,
        "total_rows_after_dropping_invalid_rows": rows_after_invalid,
        "raw_duplicates_removed": raw_duplicates_removed,
        "cleaned_text_duplicates_removed": cleaned_duplicates_removed,
        "conflicting_cleaned_text_removed": conflicting_cleaned_text_removed,
        "final_clean_samples": len(clean_df),
    }
    return clean_df, report


def split_clean_dataset(clean_df):
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        stratify=clean_df["label"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def overlap_count(left, right):
    return len(set(left["cleaned_text"]).intersection(set(right["cleaned_text"])))


def remove_leakage(train_df, val_df, test_df):
    train_texts = set(train_df["cleaned_text"])

    train_val_overlap = overlap_count(train_df, val_df)
    if train_val_overlap:
        logger.warning(
            "Removing %d leaked validation rows also present in train.", train_val_overlap
        )
        val_df = val_df[~val_df["cleaned_text"].isin(train_texts)].copy()

    train_test_overlap = overlap_count(train_df, test_df)
    if train_test_overlap:
        logger.warning(
            "Removing %d leaked test rows also present in train.", train_test_overlap
        )
        test_df = test_df[~test_df["cleaned_text"].isin(train_texts)].copy()

    val_texts = set(val_df["cleaned_text"])
    val_test_overlap = overlap_count(val_df, test_df)
    if val_test_overlap:
        logger.warning(
            "Removing %d leaked test rows also present in validation.", val_test_overlap
        )
        test_df = test_df[~test_df["cleaned_text"].isin(val_texts)].copy()

    leakage_report = {
        "train_vs_validation_overlap": overlap_count(train_df, val_df),
        "train_vs_test_overlap": overlap_count(train_df, test_df),
        "validation_vs_test_overlap": overlap_count(val_df, test_df),
    }

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        leakage_report,
    )


def save_processed_splits(data_dir, train_df, val_df, test_df):
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    for split_name, split_df in splits.items():
        output_path = data_dir / PROCESSED_FILES[split_name]
        split_df[["text", "cleaned_text", "label"]].to_csv(
            output_path, index=False, encoding="utf-8"
        )
        logger.info("Saved %s split: %s (%d rows)", split_name, output_path, len(split_df))


def load_processed_splits(data_dir):
    train_df = pd.read_csv(data_dir / PROCESSED_FILES["train"])
    val_df = pd.read_csv(data_dir / PROCESSED_FILES["validation"])
    test_df = pd.read_csv(data_dir / PROCESSED_FILES["test"])
    return train_df, val_df, test_df


def build_processed_dataset(data_dir="data"):
    data_path = Path(data_dir)

    logger.info("Loading source CSV files...")
    loaded_df, source_report = load_source_data(data_path)

    for item in source_report:
        logger.info("Source: %s -> %d rows", item["file"], item["rows"])

    clean_df, cleaning_report = clean_and_deduplicate(loaded_df)

    logger.info("Cleaning report: %s", cleaning_report)
    log_label_distribution("full clean dataset", clean_df)

    train_df, val_df, test_df = split_clean_dataset(clean_df)
    train_df, val_df, test_df, leakage_report = remove_leakage(train_df, val_df, test_df)

    logger.info("Leakage report: %s", leakage_report)

    log_label_distribution("train split", train_df)
    log_label_distribution("validation split", val_df)
    log_label_distribution("test split", test_df)

    save_processed_splits(data_path, train_df, val_df, test_df)
    return train_df, val_df, test_df, cleaning_report, leakage_report


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def tune_threshold(y_true, probabilities):
    """Pick the probability cutoff that maximizes F1 on the positive (fake) class."""
    best_threshold = 0.5
    best_f1 = -1.0
    for candidate in np.linspace(0.05, 0.95, 19):
        preds = (probabilities >= candidate).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(candidate)
    return best_threshold, float(best_f1)


def train_logistic_regression(data_dir="data", models_dir="models"):
    started_at = time.time()
    os.makedirs(models_dir, exist_ok=True)

    build_processed_dataset(data_dir)
    train_df, val_df, test_df = load_processed_splits(Path(data_dir))

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logger.info("Training Logistic Regression with TF-IDF (balanced, 1-2 grams)...")
    train_started_at = time.time()
    pipeline.fit(train_df["cleaned_text"], train_df["label"])
    training_duration = time.time() - train_started_at

    fake_class_idx = list(pipeline.classes_).index(1)

    val_probs = pipeline.predict_proba(val_df["cleaned_text"])[:, fake_class_idx]
    test_probs = pipeline.predict_proba(test_df["cleaned_text"])[:, fake_class_idx]

    best_threshold, best_val_f1 = tune_threshold(val_df["label"].to_numpy(), val_probs)
    logger.info("Best validation threshold: %.2f (F1=%.4f)", best_threshold, best_val_f1)

    val_predictions_default = (val_probs >= 0.5).astype(int)
    test_predictions_default = (test_probs >= 0.5).astype(int)
    val_predictions_tuned = (val_probs >= best_threshold).astype(int)
    test_predictions_tuned = (test_probs >= best_threshold).astype(int)

    metrics = {
        "model": "logistic_regression",
        "features": "tfidf(ngram=1-2, max_features=30000, sublinear_tf=True)",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "tuned_threshold": best_threshold,
        "split": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "validation": compute_metrics(val_df["label"], val_predictions_default),
        "validation_tuned": compute_metrics(val_df["label"], val_predictions_tuned),
        "test": compute_metrics(test_df["label"], test_predictions_default),
        "test_tuned": compute_metrics(test_df["label"], test_predictions_tuned),
        "training_duration": float(training_duration),
    }

    for split_name in ("validation", "validation_tuned", "test", "test_tuned"):
        m = metrics[split_name]
        logger.info(
            "%s: acc=%.4f, P=%.4f, R=%.4f, F1=%.4f",
            split_name,
            m["accuracy"],
            m["precision"],
            m["recall"],
            m["f1_score"],
        )

    model_path = Path(models_dir) / "logistic_regression.pkl"
    metrics_path = Path(models_dir) / "models_metrics.json"
    threshold_path = Path(models_dir) / "threshold.json"
    joblib.dump(pipeline, model_path)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4, ensure_ascii=False)
    with threshold_path.open("w", encoding="utf-8") as threshold_file:
        json.dump({"fake_threshold": best_threshold}, threshold_file, indent=4)

    logger.info("Saved artifacts: %s, %s, %s", model_path, metrics_path, threshold_path)
    logger.info("Total run duration: %.2fs", time.time() - started_at)


def _configure_cli_logging():
    """When running this script directly (no Django), wire up a basic stream handler."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("DETECTOR_LOG_LEVEL", "INFO"))
    logger.propagate = False


if __name__ == "__main__":
    _configure_cli_logging()
    train_logistic_regression()
