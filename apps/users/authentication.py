from __future__ import annotations

from typing import Optional, Tuple

from rest_framework import authentication, exceptions
from rest_framework.request import Request
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

from apps.utils import api_success

from .mongo_models import User

class MongoEngineJWTAuthentication(JWTAuthentication):

    def authenticate(self, request: Request) -> Optional[Tuple[User, dict]]:

        auth_result = super().authenticate(request)
        if auth_result is not None:
            return auth_result

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
        try:
            user_id = validated_token["user_id"]
        except KeyError:
            raise InvalidToken("Token does not contain user_id")

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed("User does not exist")

        if not user.is_active:
            raise exceptions.AuthenticationFailed("User has been disabled")

        return user

class MongoEngineTokenObtainPairSerializer:

    username_field = "email"

    def __init__(self, *args, **kwargs):
        self.user: User | None = None

    def validate(self, attrs):
        email = attrs.get("email") or attrs.get("username")
        password = attrs.get("password")

        if not email or not password:
            raise exceptions.ValidationError("Email and password are required.")

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed("Email or password is incorrect.")

        if not user.check_password(password):
            raise exceptions.AuthenticationFailed("Email or password is incorrect.")

        if not user.is_active:
            raise exceptions.AuthenticationFailed("Account has been disabled.")

        self.user = user
        refresh = self.get_token(user)

        return {
            "refresh": str(refresh),
            "access": str(refresh.access_token),
        }

    def get_token(self, user):
        from rest_framework_simplejwt.settings import api_settings
        from rest_framework_simplejwt.tokens import RefreshToken

        token = RefreshToken.for_user(user)
        user_id_claim = api_settings.USER_ID_CLAIM
        token[user_id_claim] = str(user.id)
        token["email"] = user.email
        token["is_admin"] = user.is_admin
        return token

class MongoEngineTokenObtainPairView:

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

