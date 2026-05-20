"""Stay-in-scope hyperparameter search for the TF-IDF + Logistic Regression pipeline.

This script does NOT switch to other models — by design. It explores TF-IDF
and LR knobs only (C, ngram_range, min_df, max_df, max_features) using
GridSearchCV with the F1 score on the Fake class as the optimization target,
since that is the metric the project cares about most.

Run:
    python src/tune.py

It expects data/processed/{train,val}_clean.csv to already exist
(produced by `python src/train.py`).
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


logger = logging.getLogger("fnd.tune")


PARAM_GRID = {
    "tfidf__max_features": [10000, 30000, 50000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "tfidf__max_df": [0.9, 0.95, 1.0],
    "classifier__C": [0.5, 1.0, 2.0, 5.0],
}


def run(data_dir="data", models_dir="models"):
    data_path = Path(data_dir)
    train_df = pd.read_csv(data_path / "processed" / "train_clean.csv")
    val_df = pd.read_csv(data_path / "processed" / "val_clean.csv")

    # Use the union of train+val for cross-validated tuning, then re-train on it.
    pooled_text = pd.concat([train_df["cleaned_text"], val_df["cleaned_text"]], ignore_index=True)
    pooled_label = pd.concat([train_df["label"], val_df["label"]], ignore_index=True)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

    logger.info(
        "Starting grid search: %d combinations × 3-fold CV",
        len(PARAM_GRID["tfidf__max_features"])
        * len(PARAM_GRID["tfidf__ngram_range"])
        * len(PARAM_GRID["tfidf__min_df"])
        * len(PARAM_GRID["tfidf__max_df"])
        * len(PARAM_GRID["classifier__C"]),
    )
    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(pooled_text, pooled_label)

    logger.info("Best F1 (CV) = %.4f", search.best_score_)
    logger.info("Best params  = %s", search.best_params_)

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "best_score_f1_fake": float(search.best_score_),
        "best_params": {k: v for k, v in search.best_params_.items()},
    }
    with (Path(models_dir) / "tune_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)

    # Persist the best model under a separate name so train.py output stays untouched.
    joblib.dump(search.best_estimator_, Path(models_dir) / "logistic_regression_tuned.pkl")
    logger.info("Saved best model to models/logistic_regression_tuned.pkl")


def _configure_cli_logging():
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
    run()
