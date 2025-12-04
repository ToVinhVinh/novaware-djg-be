from __future__ import annotations

import secrets

from rest_framework import permissions, status
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenRefreshView

from apps.utils import api_error, api_success

from .authentication import MongoEngineTokenObtainPairSerializer
from .mongo_models import PasswordResetAudit, User
from .mongo_serializers import (
    PasswordResetConfirmSerializer,
    PasswordResetRequestSerializer,
    RegisterSerializer,
    UserSerializer,
)

class MongoEngineTokenObtainPairView(APIView):

    permission_classes = [permissions.AllowAny]
    authentication_classes: list = []

    def post(self, request, *args, **kwargs):
        serializer = MongoEngineTokenObtainPairSerializer()
        try:
            data = serializer.validate(request.data)
            user = serializer.user
            user_data = UserSerializer(user).data if user else None
            if user_data is not None:
                user_data["isAdmin"] = bool(getattr(user, "is_admin", False))
            return api_success(
                "Login successfully.",
                {
                    "tokens": data,
                    "user": user_data,
                },
            )
        except Exception as e:
            return api_error(
                str(e),
                data=None,
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes: list = []

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        validated_data = serializer.validated_data.copy()
        password = validated_data.pop("password")

        first_name = validated_data.get("first_name", "")
        last_name = validated_data.get("last_name", "")
        name = f"{first_name} {last_name}".strip() or validated_data.get("email", "").split("@")[0]

        user = User(
            name=name,
            email=validated_data["email"],
            username=validated_data.get("username"),
            first_name=validated_data.get("first_name", ""),
            last_name=validated_data.get("last_name", ""),
            height=validated_data.get("height"),
            weight=validated_data.get("weight"),
            gender=validated_data.get("gender"),
            age=validated_data.get("age"),
        )
        user.set_password(password)
        user.save()

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
            user_id=user.id,
            ip_address=request.META.get("REMOTE_ADDR"),
            user_agent=request.META.get("HTTP_USER_AGENT", ""),
        )
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
        user = User.objects.filter(unhashed_reset_password_token=token).first()
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
    "MongoEngineTokenObtainPairView",
    "TokenRefreshView",
]

