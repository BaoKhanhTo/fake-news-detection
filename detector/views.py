from django.shortcuts import render

from .services import predict_news


def index(request):
    context = {}

    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        context["text"] = text

        if not text:
            context["error"] = "Please enter news content before prediction."
        else:
            try:
                context["result"] = predict_news(text)
            except Exception as exc:
                context["error"] = str(exc)

    return render(request, "detector/index.html", context)
