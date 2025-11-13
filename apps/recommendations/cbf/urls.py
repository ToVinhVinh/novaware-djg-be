"""URL configuration for the CBF recommender."""

from __future__ import annotations

from django.urls import path

from .views import RecommendCBFView, TrainCBFView

app_name = "recommendations-cbf"

urlpatterns = [
    path("train/", TrainCBFView.as_view(), name="train"),
    path("recommend/", RecommendCBFView.as_view(), name="recommend"),
]

