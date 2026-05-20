from pathlib import Path

import joblib

from src.preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Run `python src/train.py` before using prediction."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_news(text):
    if not text or not text.strip():
        raise ValueError("Please enter news content before prediction.")

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("Input text is empty after preprocessing.")

    model = load_model()

    raw_prediction = model.predict([cleaned_text])[0]
    try:
        label = int(raw_prediction)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prediction result: {raw_prediction}") from exc

    if label not in (0, 1):
        raise ValueError(f"Invalid prediction label: {label}")

    if not hasattr(model, "predict_proba"):
        raise ValueError("The loaded model does not support probability prediction.")

    probabilities = model.predict_proba([cleaned_text])[0]
    class_indices = {int(label): index for index, label in enumerate(model.classes_)}

    if 0 not in class_indices or 1 not in class_indices:
        raise ValueError(f"Invalid model classes: {list(model.classes_)}")

    real_prob = round(float(probabilities[class_indices[0]]) * 100, 2)
    fake_prob = round(float(probabilities[class_indices[1]]) * 100, 2)

    return {
        "input_text": text,
        "cleaned_text": cleaned_text,
        "label": label,
        "prediction": "Fake News" if label == 1 else "Real News",
        "real_probability": real_prob,
        "fake_probability": fake_prob,
    }
