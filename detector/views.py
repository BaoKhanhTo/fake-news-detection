import json
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from . import ratelimit
from .models import Prediction
from .services import predict_news


logger = logging.getLogger(__name__)

# Allow up to 20 predictions per IP per minute on the HTML form.
WEB_RATE_LIMIT_MAX = 20
WEB_RATE_LIMIT_WINDOW = 60
# API limit is stricter — 30 requests per minute.
API_RATE_LIMIT_MAX = 30
API_RATE_LIMIT_WINDOW = 60


def _store_prediction(text, result, *, source):
    try:
        Prediction.objects.create(
            text_preview=text[:240],
            text_length=len(text),
            label=result["label"],
            fake_probability=result["fake_probability"],
            real_probability=result["real_probability"],
            threshold=result["threshold"],
            source=source,
        )
    except Exception:
        logger.exception("Failed to persist prediction history entry")


def index(request):
    context = {}

    if request.method == "POST":
        allowed, retry_after = ratelimit.check(
            request,
            scope="web_predict",
            max_requests=WEB_RATE_LIMIT_MAX,
            window_seconds=WEB_RATE_LIMIT_WINDOW,
        )
        text = request.POST.get("text", "").strip()
        context["text"] = text

        if not allowed:
            logger.warning("Rate limit hit on web predict from %s", request.META.get("REMOTE_ADDR"))
            context["error"] = (
                f"Bạn đã gửi quá nhiều yêu cầu. Vui lòng thử lại sau {retry_after} giây."
            )
        elif not text:
            context["error"] = "Vui lòng nhập nội dung tin tức trước khi phân tích."
        else:
            try:
                result = predict_news(text)
                context["result"] = result
                _store_prediction(text, result, source="web")
            except ValueError as exc:
                logger.warning("Prediction rejected: %s", exc)
                context["error"] = str(exc)
            except FileNotFoundError as exc:
                logger.error("Model artifact missing: %s", exc)
                context["error"] = str(exc)
            except Exception:
                logger.exception("Unexpected error during prediction")
                context["error"] = "Lỗi nội bộ khi phân tích. Vui lòng thử lại."

    return render(request, "detector/index.html", context)


def history(request):
    entries = Prediction.objects.all()[:50]
    return render(request, "detector/history.html", {"entries": entries})


@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """JSON endpoint: POST {"text": "..."} -> prediction dict.

    CSRF is exempted because this is meant for programmatic clients
    (curl, scripts, browser extensions). Pair with API keys / reverse proxy
    auth before exposing to untrusted networks.
    """
    allowed, retry_after = ratelimit.check(
        request,
        scope="api_predict",
        max_requests=API_RATE_LIMIT_MAX,
        window_seconds=API_RATE_LIMIT_WINDOW,
    )
    if not allowed:
        return JsonResponse(
            {"error": "rate_limited", "retry_after_seconds": retry_after},
            status=429,
        )

    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid_json"}, status=400)

    text = (payload.get("text") or "").strip()
    if not text:
        return JsonResponse({"error": "missing_text"}, status=400)

    try:
        result = predict_news(text)
    except ValueError as exc:
        return JsonResponse({"error": "invalid_input", "detail": str(exc)}, status=400)
    except FileNotFoundError as exc:
        logger.error("Model artifact missing: %s", exc)
        return JsonResponse({"error": "model_unavailable", "detail": str(exc)}, status=503)
    except Exception:
        logger.exception("Unexpected error during API prediction")
        return JsonResponse({"error": "internal"}, status=500)

    _store_prediction(text, result, source="api")
    return JsonResponse(result)
