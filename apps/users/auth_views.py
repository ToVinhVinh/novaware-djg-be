"""Các API liên quan đến xác thực."""

from __future__ import annotations

import secrets

from django.contrib.auth import get_user_model
from rest_framework import permissions, status
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from apps.utils import api_error, api_success

from .serializers import (
    PasswordResetConfirmSerializer,
    PasswordResetRequestSerializer,
    RegisterSerializer,
    UserSerializer,
)
from .models import PasswordResetAudit

User = get_user_model()


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = User.objects.create_user(**serializer.validated_data)
        output = UserSerializer(user)
        return api_success(
            "Đăng ký tài khoản thành công.",
            {
                "user": output.data,
            },
            status_code=status.HTTP_201_CREATED,
        )


class PasswordResetRequestView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = PasswordResetRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = User.objects.filter(email=serializer.validated_data["email"]).first()
        if not user:
            return api_success(
                "Nếu email tồn tại, một liên kết đặt lại đã được gửi.",
                data=None,
            )
        token = secrets.token_urlsafe(32)
        user.set_reset_password_token(token)
        PasswordResetAudit.objects.create(
            user=user,
            ip_address=request.META.get("REMOTE_ADDR"),
            user_agent=request.META.get("HTTP_USER_AGENT", ""),
        )
        # TODO: gửi email reset bằng Celery + utils.send_email
        return api_success(
            "Nếu email tồn tại, một liên kết đặt lại đã được gửi.",
            data=None,
        )


class PasswordResetConfirmView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        token = serializer.validated_data["token"]
        user = User.objects.filter(reset_password_token=token).first()
        if not user:
            return api_error(
                "Token không hợp lệ.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        user.set_password(serializer.validated_data["new_password"])
        user.clear_reset_password_token()
        return api_success(
            "Mật khẩu đã được đặt lại.",
            data=None,
        )


__all__ = [
    "RegisterView",
    "PasswordResetRequestView",
    "PasswordResetConfirmView",
    "TokenObtainPairView",
    "TokenRefreshView",
]

