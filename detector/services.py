import json
import logging
from pathlib import Path

import joblib
import numpy as np

from src.preprocess import clean_text


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"
DEFAULT_FAKE_THRESHOLD = 0.5
MAX_INPUT_LENGTH = 50_000
EXPLAIN_TOP_K = 5

_model = None
_threshold = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Run `python src/train.py` before using prediction."
            )
        logger.info("Loading model from %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


def load_threshold():
    global _threshold
    if _threshold is None:
        if THRESHOLD_PATH.exists():
            try:
                with THRESHOLD_PATH.open("r", encoding="utf-8") as f:
                    _threshold = float(json.load(f).get("fake_threshold", DEFAULT_FAKE_THRESHOLD))
                logger.info("Loaded fake threshold %.2f from %s", _threshold, THRESHOLD_PATH)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Failed to read %s (%s); falling back to default threshold %.2f",
                    THRESHOLD_PATH,
                    exc,
                    DEFAULT_FAKE_THRESHOLD,
                )
                _threshold = DEFAULT_FAKE_THRESHOLD
        else:
            logger.info(
                "No threshold file at %s; using default threshold %.2f",
                THRESHOLD_PATH,
                DEFAULT_FAKE_THRESHOLD,
            )
            _threshold = DEFAULT_FAKE_THRESHOLD
    return _threshold


def explain_prediction(model, cleaned_text, top_k=EXPLAIN_TOP_K):
    """Return top features pushing toward Fake and Real for an input text.

    Works when the loaded model is a sklearn Pipeline with a TfidfVectorizer step
    named "tfidf" and a LogisticRegression step named "classifier". Returns
    {"fake": [(term, contribution), ...], "real": [...]}. If the model layout
    differs, returns empty lists rather than raising.
    """
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
    except (AttributeError, KeyError):
        return {"fake": [], "real": []}

    if not hasattr(classifier, "coef_") or not hasattr(vectorizer, "get_feature_names_out"):
        return {"fake": [], "real": []}

    vector = vectorizer.transform([cleaned_text])
    if vector.nnz == 0:
        return {"fake": [], "real": []}

    # For binary LR, coef_ has shape (1, n_features); a positive value pushes
    # the prediction toward class 1 (Fake) and a negative value toward class 0 (Real).
    coef = np.asarray(classifier.coef_).ravel()
    cols = vector.nonzero()[1]
    values = np.asarray(vector[0, cols].todense()).ravel()
    contributions = values * coef[cols]
    feature_names = vectorizer.get_feature_names_out()

    order = np.argsort(contributions)
    real_indices = [i for i in order[:top_k] if contributions[i] < 0]
    fake_indices = [i for i in order[::-1][:top_k] if contributions[i] > 0]

    def _format(idx_list):
        return [
            {"term": str(feature_names[cols[i]]), "weight": round(float(contributions[i]), 4)}
            for i in idx_list
        ]

    return {"fake": _format(fake_indices), "real": _format(real_indices)}


def predict_news(text):
    if not text or not text.strip():
        raise ValueError("Vui lòng nhập nội dung tin tức trước khi phân tích.")

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Nội dung quá dài ({len(text):,} ký tự). Tối đa {MAX_INPUT_LENGTH:,} ký tự."
        )

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("Nội dung rỗng sau khi tiền xử lý.")

    model = load_model()

    if not hasattr(model, "predict_proba"):
        raise ValueError("The loaded model does not support probability prediction.")

    class_indices = {int(label): index for index, label in enumerate(model.classes_)}
    if 0 not in class_indices or 1 not in class_indices:
        raise ValueError(f"Invalid model classes: {list(model.classes_)}")

    probabilities = model.predict_proba([cleaned_text])[0]
    fake_probability = float(probabilities[class_indices[1]])
    real_probability = float(probabilities[class_indices[0]])

    threshold = load_threshold()
    label = 1 if fake_probability >= threshold else 0
    explanation = explain_prediction(model, cleaned_text)

    logger.info(
        "predict: len=%d, fake_prob=%.4f, threshold=%.2f, label=%d",
        len(text),
        fake_probability,
        threshold,
        label,
    )

    return {
        "input_text": text,
        "cleaned_text": cleaned_text,
        "label": label,
        "prediction": "Fake News" if label == 1 else "Real News",
        "real_probability": round(real_probability * 100, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "threshold": round(threshold, 2),
        "top_fake_features": explanation["fake"],
        "top_real_features": explanation["real"],
    }
