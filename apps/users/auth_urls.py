"""Các route xác thực tùy chỉnh."""

from django.urls import re_path
from rest_framework_simplejwt.views import TokenRefreshView

from .auth_views_mongo import (
    MongoEngineTokenObtainPairView,
    PasswordResetConfirmView,
    PasswordResetRequestView,
    RegisterView,
)


urlpatterns = [
    re_path(r"^register/?$", RegisterView.as_view(), name="auth-register"),
    re_path(r"^login/?$", MongoEngineTokenObtainPairView.as_view(), name="auth-login"),
    re_path(r"^refresh/?$", TokenRefreshView.as_view(), name="auth-refresh"),
    re_path(r"^password/reset/?$", PasswordResetRequestView.as_view(), name="auth-password-reset"),
    re_path(r"^password/reset/confirm/?$", PasswordResetConfirmView.as_view(), name="auth-password-reset-confirm"),
]

