from django.db import models


class Prediction(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    text_preview = models.CharField(max_length=240)
    text_length = models.PositiveIntegerField()
    label = models.PositiveSmallIntegerField()  # 0 = real, 1 = fake
    fake_probability = models.FloatField()
    real_probability = models.FloatField()
    threshold = models.FloatField()
    source = models.CharField(max_length=20, default="web")  # "web" or "api"

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.created_at:%Y-%m-%d %H:%M:%S} [{self.get_label_display()}] {self.text_preview[:40]}"

    def get_label_display(self):
        return "Fake" if self.label == 1 else "Real"
