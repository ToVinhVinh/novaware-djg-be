from __future__ import annotations

from django.urls import re_path

from .views import RecommendHybridView

app_name = "recommendations-hybrid"

urlpatterns = [
    re_path(r"^recommend/?$", RecommendHybridView.as_view(), name="recommend"),
]

