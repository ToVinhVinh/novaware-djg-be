"""URL configuration for the hybrid recommender."""

from __future__ import annotations

from django.urls import path

from .views import RecommendHybridView, TrainHybridView

app_name = "recommendations-hybrid"

urlpatterns = [
    path("train/", TrainHybridView.as_view(), name="train"),
    path("recommend/", RecommendHybridView.as_view(), name="recommend"),
]

