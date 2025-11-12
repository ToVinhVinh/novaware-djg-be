"""Custom authentication backend cho MongoEngine User."""

from __future__ import annotations

from typing import Optional

from rest_framework import authentication, exceptions
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

from apps.utils import api_success

from .mongo_models import User


class MongoEngineJWTAuthentication(JWTAuthentication):
    """JWT Authentication backend cho MongoEngine User model."""
    
    def get_user(self, validated_token):
        """
        Lấy user từ validated token.
        Override để sử dụng MongoEngine User thay vì Django User.
        """
        try:
            user_id = validated_token["user_id"]
        except KeyError:
            raise InvalidToken("Token không chứa user_id")
        
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed("User không tồn tại")
        
        if not user.is_active:
            raise exceptions.AuthenticationFailed("User đã bị vô hiệu hóa")
        
        return user


class MongoEngineTokenObtainPairSerializer:
    """Custom serializer cho JWT token với MongoEngine User."""
    
    username_field = "email"
    
    def __init__(self, *args, **kwargs):
        pass
    
    def validate(self, attrs):
        """
        Validate credentials và trả về token.
        Override để sử dụng MongoEngine User.
        """
        email = attrs.get("email") or attrs.get("username")
        password = attrs.get("password")
        
        if not email or not password:
            raise exceptions.ValidationError("Email và mật khẩu là bắt buộc.")
        
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed("Email hoặc mật khẩu không chính xác.")
        
        if not user.check_password(password):
            raise exceptions.AuthenticationFailed("Email hoặc mật khẩu không chính xác.")
        
        if not user.is_active:
            raise exceptions.AuthenticationFailed("Tài khoản đã bị vô hiệu hóa.")
        
        refresh = self.get_token(user)
        
        return {
            "refresh": str(refresh),
            "access": str(refresh.access_token),
        }
    
    def get_token(self, user):
        """Tạo JWT token cho user."""
        from rest_framework_simplejwt.tokens import RefreshToken
        
        token = RefreshToken()
        token["user_id"] = str(user.id)
        token["email"] = user.email
        token["is_admin"] = user.is_admin
        
        return token


class MongoEngineTokenObtainPairView:
    """Custom view cho JWT token với MongoEngine User."""
    
    serializer_class = MongoEngineTokenObtainPairSerializer
    
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class()
        data = serializer.validate(request.data)
        return api_success(
            "Đăng nhập thành công.",
            {
                "tokens": data,
            },
        )

