"""Các route xác thực tùy chỉnh."""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .auth_views_mongo import (
    MongoEngineTokenObtainPairView,
    PasswordResetConfirmView,
    PasswordResetRequestView,
    RegisterView,
)


urlpatterns = [
    path("register/", RegisterView.as_view(), name="auth-register"),
    path("login/", MongoEngineTokenObtainPairView.as_view(), name="auth-login"),
    path("refresh/", TokenRefreshView.as_view(), name="auth-refresh"),
    path("password/reset/", PasswordResetRequestView.as_view(), name="auth-password-reset"),
    path("password/reset/confirm/", PasswordResetConfirmView.as_view(), name="auth-password-reset-confirm"),
]

