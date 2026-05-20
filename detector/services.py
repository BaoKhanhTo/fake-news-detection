import json
from pathlib import Path

import joblib

from src.preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"
DEFAULT_FAKE_THRESHOLD = 0.5

_model = None
_threshold = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Run `python src/train.py` before using prediction."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def load_threshold():
    global _threshold
    if _threshold is None:
        if THRESHOLD_PATH.exists():
            try:
                with THRESHOLD_PATH.open("r", encoding="utf-8") as f:
                    _threshold = float(json.load(f).get("fake_threshold", DEFAULT_FAKE_THRESHOLD))
            except (ValueError, OSError, json.JSONDecodeError):
                _threshold = DEFAULT_FAKE_THRESHOLD
        else:
            _threshold = DEFAULT_FAKE_THRESHOLD
    return _threshold


def predict_news(text):
    if not text or not text.strip():
        raise ValueError("Please enter news content before prediction.")

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("Input text is empty after preprocessing.")

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

    return {
        "input_text": text,
        "cleaned_text": cleaned_text,
        "label": label,
        "prediction": "Fake News" if label == 1 else "Real News",
        "real_probability": round(real_probability * 100, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "threshold": round(threshold, 2),
    }
