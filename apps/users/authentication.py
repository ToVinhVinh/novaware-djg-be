"""Custom authentication backend cho MongoEngine User."""

from __future__ import annotations

from typing import Optional, Tuple

from rest_framework import authentication, exceptions
from rest_framework.request import Request
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

from apps.utils import api_success

from .mongo_models import User


class MongoEngineJWTAuthentication(JWTAuthentication):
    """JWT Authentication backend cho MongoEngine User model."""
    
    def authenticate(self, request: Request) -> Optional[Tuple[User, dict]]:
        """
        Override authenticate để hỗ trợ lấy token từ nhiều nguồn hơn.
        
        Ưu tiên header chuẩn theo SimpleJWT. Nếu không có, thử đọc từ query params
        hoặc cookies để tương thích với các client cũ.
        """
        # Thử xác thực với cơ chế mặc định (Authorization header)
        auth_result = super().authenticate(request)
        if auth_result is not None:
            return auth_result
        
        # Fallback: query parameter ?token=... hoặc ?access_token=...
        raw_token = (
            request.query_params.get("token")
            or request.query_params.get("access_token")
            or request.COOKIES.get("access_token")
            or request.COOKIES.get("jwt")
        )
        
        if not raw_token:
            return None
        
        try:
            validated_token = self.get_validated_token(raw_token)
        except (InvalidToken, TokenError) as exc:
            raise exceptions.AuthenticationFailed(str(exc))
        
        return self.get_user(validated_token), validated_token
    
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
        self.user: User | None = None
    
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
        
        self.user = user
        refresh = self.get_token(user)
        
        return {
            "refresh": str(refresh),
            "access": str(refresh.access_token),
        }
    
    def get_token(self, user):
        """Tạo JWT token cho user."""
        from rest_framework_simplejwt.settings import api_settings
        from rest_framework_simplejwt.tokens import RefreshToken
        
        token = RefreshToken.for_user(user)
        user_id_claim = api_settings.USER_ID_CLAIM
        token[user_id_claim] = str(user.id)
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
            "Login successfully.",
            {
                "tokens": data,
            },
        )

