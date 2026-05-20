import logging
import sys

from django.apps import AppConfig


logger = logging.getLogger(__name__)


def _should_preload():
    """Skip preload during tests, migrations, or check commands to keep them snappy."""
    skip_commands = {"test", "makemigrations", "migrate", "check", "collectstatic", "shell"}
    if len(sys.argv) > 1 and sys.argv[1] in skip_commands:
        return False
    return True


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detector"

    def ready(self):
        if not _should_preload():
            return
        try:
            from . import services

            services.load_model()
            services.load_threshold()
            logger.info("Model and threshold preloaded; first request will be fast.")
        except FileNotFoundError as exc:
            logger.warning(
                "Model preload skipped: %s (run `python src/train.py` to generate it).",
                exc,
            )
        except Exception:
            logger.exception("Unexpected error while preloading model")
