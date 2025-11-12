"""Serializers cho MongoEngine User models."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import serializers

from apps.products.mongo_serializers import ProductSerializer
from .mongo_models import OutfitHistory, PasswordResetAudit, User, UserInteraction


class UserSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    email = serializers.EmailField()
    username = serializers.CharField(required=False, allow_blank=True)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    is_staff = serializers.SerializerMethodField()
    height = serializers.FloatField(required=False, allow_null=True)
    weight = serializers.FloatField(required=False, allow_null=True)
    gender = serializers.CharField(required=False, allow_null=True)
    age = serializers.IntegerField(required=False, allow_null=True)
    preferences = serializers.DictField(default=dict)
    amazon_user_id = serializers.CharField(required=False, allow_null=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def get_is_staff(self, obj):
        return obj.is_admin
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        return {
            "id": str(instance.id),
            "email": instance.email,
            "username": instance.username,
            "first_name": instance.first_name or "",
            "last_name": instance.last_name or "",
            "is_staff": instance.is_admin,
            "height": instance.height,
            "weight": instance.weight,
            "gender": instance.gender,
            "age": instance.age,
            "preferences": instance.preferences,
            "amazon_user_id": instance.amazon_user_id,
        }
    
    def update(self, instance, validated_data):
        """Update user."""
        for key, value in validated_data.items():
            if key != "is_staff":  # Don't update is_staff directly
                setattr(instance, key, value)
        instance.save()
        return instance


class RegisterSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    username = serializers.CharField(required=False, allow_blank=True)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    height = serializers.FloatField(required=False, allow_null=True)
    weight = serializers.FloatField(required=False, allow_null=True)
    gender = serializers.CharField(required=False, allow_null=True)
    age = serializers.IntegerField(required=False, allow_null=True)
    
    def validate_password(self, value):
        """Validate password."""
        if len(value) < 8:
            raise serializers.ValidationError("Mật khẩu phải có ít nhất 8 ký tự.")
        return value


class UserDetailSerializer(UserSerializer):
    favorites = ProductSerializer(many=True, read_only=True)
    user_embedding = serializers.ListField(child=serializers.FloatField(), read_only=True)
    content_profile = serializers.DictField(read_only=True)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict with favorites."""
        data = super().to_representation(instance)
        
        # Load favorites
        from apps.products.mongo_models import Product
        favorite_products = Product.objects(id__in=instance.favorites)
        data["favorites"] = [ProductSerializer(product).to_representation(product) for product in favorite_products]
        
        data["user_embedding"] = instance.user_embedding
        data["content_profile"] = instance.content_profile
        
        return data


class PasswordChangeSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True)
    new_password = serializers.CharField(write_only=True)
    
    def validate_new_password(self, value):
        """Validate password."""
        if len(value) < 8:
            raise serializers.ValidationError("Mật khẩu phải có ít nhất 8 ký tự.")
        return value


class PasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()


class PasswordResetConfirmSerializer(serializers.Serializer):
    token = serializers.CharField()
    new_password = serializers.CharField()
    
    def validate_new_password(self, value):
        """Validate password."""
        if len(value) < 8:
            raise serializers.ValidationError("Mật khẩu phải có ít nhất 8 ký tự.")
        return value


class UserInteractionSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    product_id = serializers.CharField()
    interaction_type = serializers.CharField()
    rating = serializers.IntegerField(required=False, allow_null=True)
    timestamp = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        return {
            "id": str(instance.id),
            "product_id": str(instance.product_id),
            "interaction_type": instance.interaction_type,
            "rating": instance.rating,
            "timestamp": instance.timestamp,
        }
    
    def create(self, validated_data):
        """Create new interaction."""
        validated_data["product_id"] = ObjectId(validated_data["product_id"])
        validated_data["user_id"] = ObjectId(validated_data["user_id"])
        return UserInteraction(**validated_data).save()


class OutfitHistorySerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    outfit_id = serializers.CharField()
    product_ids = serializers.ListField(child=serializers.CharField())
    interaction_type = serializers.CharField()
    timestamp = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        return {
            "id": str(instance.id),
            "outfit_id": instance.outfit_id,
            "product_ids": [str(pid) for pid in instance.product_ids],
            "interaction_type": instance.interaction_type,
            "timestamp": instance.timestamp,
        }
    
    def create(self, validated_data):
        """Create new outfit history."""
        validated_data["product_ids"] = [ObjectId(pid) for pid in validated_data["product_ids"]]
        validated_data["user_id"] = ObjectId(validated_data["user_id"])
        return OutfitHistory(**validated_data).save()


class PurchaseHistorySummarySerializer(serializers.Serializer):
    has_purchase_history = serializers.BooleanField()
    order_count = serializers.IntegerField()


class GenderSummarySerializer(serializers.Serializer):
    has_gender = serializers.BooleanField()
    gender = serializers.CharField(allow_null=True)


class StylePreferenceSummarySerializer(serializers.Serializer):
    has_style_preference = serializers.BooleanField()
    style = serializers.CharField(allow_null=True)


class UserForTestingSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    email = serializers.EmailField()
    username = serializers.CharField()
    age = serializers.IntegerField(allow_null=True)
    gender = serializers.CharField(allow_null=True)
    preferences = serializers.DictField()
    order_count = serializers.IntegerField()
    
    def get_id(self, obj):
        return str(obj.id)

