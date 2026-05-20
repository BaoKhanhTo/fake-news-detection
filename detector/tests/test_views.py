import json
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from detector.models import Prediction


class IndexViewTests(TestCase):
    def test_get_renders_form(self):
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<form", status_code=200)
        self.assertContains(response, 'name="text"')

    def test_post_empty_text_shows_error(self):
        response = self.client.post(reverse("index"), {"text": "   "})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Vui lòng nhập nội dung")

    def test_post_renders_prediction_when_service_succeeds(self):
        fake_result = {
            "input_text": "Tin tuc test",
            "cleaned_text": "tin tuc test",
            "label": 1,
            "prediction": "Fake News",
            "real_probability": 12.34,
            "fake_probability": 87.66,
            "threshold": 0.5,
            "top_fake_features": [{"term": "lừa đảo", "weight": 0.42}],
            "top_real_features": [{"term": "chính thức", "weight": -0.31}],
        }
        with patch("detector.views.predict_news", return_value=fake_result):
            response = self.client.post(reverse("index"), {"text": "Tin tuc test"})
        self.assertEqual(response.status_code, 200)
        body = response.content.decode("utf-8")
        self.assertIn("Tin giả", body)
        # Django i18n may render the decimal separator as "." or "," depending on
        # the active locale, so normalize before asserting.
        normalized = body.replace(",", ".")
        self.assertIn("87.66", normalized)
        self.assertIn("12.34", normalized)
        self.assertIn("lừa đảo", body)
        self.assertIn("chính thức", body)

    def test_post_renders_error_when_service_raises(self):
        with patch(
            "detector.views.predict_news",
            side_effect=ValueError("Model file not found."),
        ):
            response = self.client.post(reverse("index"), {"text": "Tin tuc"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Model file not found.")

    def test_successful_prediction_persists_history(self):
        fake_result = {
            "input_text": "Tin tuc test",
            "cleaned_text": "tin tuc test",
            "label": 1,
            "prediction": "Fake News",
            "real_probability": 12.34,
            "fake_probability": 87.66,
            "threshold": 0.5,
            "top_fake_features": [],
            "top_real_features": [],
        }
        with patch("detector.views.predict_news", return_value=fake_result):
            self.client.post(reverse("index"), {"text": "Tin tuc test"})
        self.assertEqual(Prediction.objects.count(), 1)
        entry = Prediction.objects.first()
        self.assertEqual(entry.label, 1)
        self.assertEqual(entry.source, "web")


class HistoryViewTests(TestCase):
    def test_empty_history_renders_placeholder(self):
        response = self.client.get(reverse("history"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Chưa có lượt phân tích")

    def test_history_renders_entries(self):
        Prediction.objects.create(
            text_preview="Sample news preview",
            text_length=20,
            label=0,
            fake_probability=12.0,
            real_probability=88.0,
            threshold=0.5,
            source="web",
        )
        response = self.client.get(reverse("history"))
        self.assertContains(response, "Sample news preview")
        self.assertContains(response, "Tin thật")


class ApiPredictTests(TestCase):
    def test_rejects_get(self):
        response = self.client.get(reverse("api_predict"))
        self.assertEqual(response.status_code, 405)

    def test_rejects_invalid_json(self):
        response = self.client.post(
            reverse("api_predict"),
            data="not-json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_json")

    def test_rejects_missing_text(self):
        response = self.client.post(
            reverse("api_predict"),
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "missing_text")

    def test_returns_prediction_json(self):
        fake_result = {
            "input_text": "Tin tuc",
            "cleaned_text": "tin tuc",
            "label": 0,
            "prediction": "Real News",
            "real_probability": 80.0,
            "fake_probability": 20.0,
            "threshold": 0.5,
            "top_fake_features": [],
            "top_real_features": [],
        }
        with patch("detector.views.predict_news", return_value=fake_result):
            response = self.client.post(
                reverse("api_predict"),
                data=json.dumps({"text": "Tin tuc"}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["label"], 0)
        self.assertEqual(Prediction.objects.filter(source="api").count(), 1)

    def test_returns_503_when_model_missing(self):
        with patch(
            "detector.views.predict_news",
            side_effect=FileNotFoundError("model missing"),
        ):
            response = self.client.post(
                reverse("api_predict"),
                data=json.dumps({"text": "Tin tuc"}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "model_unavailable")
