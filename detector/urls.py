from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("history/", views.history, name="history"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
