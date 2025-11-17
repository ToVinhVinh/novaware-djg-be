"""URL patterns for GNN recommendation endpoints."""

from __future__ import annotations

from django.urls import path

from .views import RecommendGNNView, TrainGNNView

app_name = "recommendations-gnn"

urlpatterns = [
    path("train", TrainGNNView.as_view(), name="train"),
    path("recommend", RecommendGNNView.as_view(), name="recommend"),
]

