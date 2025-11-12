"""Serializer cho người dùng."""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.utils.translation import gettext_lazy as _
from rest_framework import serializers

from apps.orders.models import Order
from apps.products.serializers import ProductSerializer

from .models import OutfitHistory, UserInteraction

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "email",
            "username",
            "first_name",
            "last_name",
            "is_staff",
            "height",
            "weight",
            "gender",
            "age",
            "preferences",
            "amazon_user_id",
        ]
        read_only_fields = ["is_staff"]


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = [
            "email",
            "password",
            "username",
            "first_name",
            "last_name",
            "height",
            "weight",
            "gender",
            "age",
        ]

    def validate_password(self, value):
        validate_password(value)
        return value


class UserDetailSerializer(UserSerializer):
    favorites = ProductSerializer(many=True, read_only=True)

    class Meta(UserSerializer.Meta):
        fields = UserSerializer.Meta.fields + [
            "favorites",
            "user_embedding",
            "content_profile",
        ]


class PasswordChangeSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True)
    new_password = serializers.CharField(write_only=True)

    def validate_new_password(self, value):
        validate_password(value)
        return value


class PasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()


class PasswordResetConfirmSerializer(serializers.Serializer):
    token = serializers.CharField()
    new_password = serializers.CharField()

    def validate_new_password(self, value):
        validate_password(value)
        return value


class UserInteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserInteraction
        fields = ["id", "product", "interaction_type", "rating", "timestamp"]


class OutfitHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = OutfitHistory
        fields = ["id", "outfit_id", "products", "interaction_type", "timestamp"]


class PurchaseHistorySummarySerializer(serializers.Serializer):
    has_purchase_history = serializers.BooleanField()
    order_count = serializers.IntegerField()


class GenderSummarySerializer(serializers.Serializer):
    has_gender = serializers.BooleanField()
    gender = serializers.CharField(allow_null=True)


class StylePreferenceSummarySerializer(serializers.Serializer):
    has_style_preference = serializers.BooleanField()
    style = serializers.CharField(allow_null=True)


class UserForTestingSerializer(serializers.ModelSerializer):
    order_count = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "email", "username", "age", "gender", "preferences", "order_count"]

    def get_order_count(self, obj):
        return Order.objects.filter(user=obj, is_paid=True).count()

