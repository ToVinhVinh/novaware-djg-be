"""URL configuration for the CBF recommender."""

from __future__ import annotations

from django.urls import re_path

from .views import RecommendCBFView, TrainCBFView

app_name = "recommendations-cbf"

urlpatterns = [
    re_path(r"^train/?$", TrainCBFView.as_view(), name="train"),
    re_path(r"^recommend/?$", RecommendCBFView.as_view(), name="recommend"),
]

