import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from django.test import SimpleTestCase

from detector import services


class FakeModel:
    """Minimal stand-in for a fitted scikit-learn classifier."""

    def __init__(self, fake_probability=0.8):
        self.classes_ = np.array([0, 1])
        self._fake_probability = fake_probability

    def predict_proba(self, _texts):
        return np.array([[1.0 - self._fake_probability, self._fake_probability]])


class PredictNewsValidationTests(SimpleTestCase):
    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            services.predict_news("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            services.predict_news("   \n\t  ")

    def test_text_that_cleans_to_empty_raises(self):
        # Only punctuation that the cleaner strips entirely.
        with patch("detector.services.clean_text", return_value=""):
            with self.assertRaises(ValueError):
                services.predict_news("@#$%")


class PredictNewsBehaviourTests(SimpleTestCase):
    def setUp(self):
        # Reset module-level caches so each test gets a clean slate.
        services._model = None
        services._threshold = None

    def tearDown(self):
        services._model = None
        services._threshold = None

    def _patch_model(self, fake_probability):
        return patch.object(
            services, "load_model", return_value=FakeModel(fake_probability)
        )

    def _patch_threshold(self, threshold):
        return patch.object(services, "load_threshold", return_value=threshold)

    def test_high_fake_probability_predicts_fake(self):
        with self._patch_model(0.9), self._patch_threshold(0.5):
            result = services.predict_news("Tin gia day la tin gia")
        self.assertEqual(result["label"], 1)
        self.assertEqual(result["prediction"], "Fake News")
        self.assertAlmostEqual(result["fake_probability"], 90.0, places=1)
        self.assertAlmostEqual(result["real_probability"], 10.0, places=1)

    def test_low_fake_probability_predicts_real(self):
        with self._patch_model(0.1), self._patch_threshold(0.5):
            result = services.predict_news("Tin chinh thong")
        self.assertEqual(result["label"], 0)
        self.assertEqual(result["prediction"], "Real News")

    def test_threshold_changes_decision(self):
        # Same probability, two different thresholds → different labels.
        with self._patch_model(0.4), self._patch_threshold(0.5):
            result_high = services.predict_news("xyz")
        with self._patch_model(0.4), self._patch_threshold(0.3):
            result_low = services.predict_news("xyz")
        self.assertEqual(result_high["label"], 0)
        self.assertEqual(result_low["label"], 1)

    def test_result_has_expected_keys(self):
        with self._patch_model(0.7), self._patch_threshold(0.5):
            result = services.predict_news("Tin tuc")
        expected = {
            "input_text",
            "cleaned_text",
            "label",
            "prediction",
            "real_probability",
            "fake_probability",
            "threshold",
            "top_fake_features",
            "top_real_features",
        }
        self.assertEqual(set(result.keys()), expected)

    def test_rejects_text_exceeding_max_length(self):
        too_long = "a" * (services.MAX_INPUT_LENGTH + 1)
        with self.assertRaises(ValueError):
            services.predict_news(too_long)

    def test_fake_model_returns_empty_explanation(self):
        # FakeModel is not a sklearn Pipeline so explanation returns empty lists.
        with self._patch_model(0.7), self._patch_threshold(0.5):
            result = services.predict_news("Tin tuc")
        self.assertEqual(result["top_fake_features"], [])
        self.assertEqual(result["top_real_features"], [])

    def test_probabilities_sum_to_about_100(self):
        with self._patch_model(0.37), self._patch_threshold(0.5):
            result = services.predict_news("Tin tuc")
        total = result["real_probability"] + result["fake_probability"]
        self.assertAlmostEqual(total, 100.0, places=1)


class LoadThresholdTests(SimpleTestCase):
    def setUp(self):
        services._threshold = None
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name) / "threshold.json"

    def tearDown(self):
        services._threshold = None
        self._tmp.cleanup()

    def test_falls_back_to_default_when_file_missing(self):
        # tmp_path does not exist yet — load_threshold should use the default.
        with patch.object(services, "THRESHOLD_PATH", self.tmp_path):
            self.assertEqual(services.load_threshold(), services.DEFAULT_FAKE_THRESHOLD)

    def test_reads_threshold_from_json(self):
        self.tmp_path.write_text(json.dumps({"fake_threshold": 0.42}), encoding="utf-8")
        with patch.object(services, "THRESHOLD_PATH", self.tmp_path):
            self.assertAlmostEqual(services.load_threshold(), 0.42)

    def test_invalid_json_falls_back_to_default(self):
        self.tmp_path.write_text("not-json", encoding="utf-8")
        with patch.object(services, "THRESHOLD_PATH", self.tmp_path):
            self.assertEqual(services.load_threshold(), services.DEFAULT_FAKE_THRESHOLD)
