"""Serializers for MongoEngine User models."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import serializers

from apps.products.mongo_serializers import ProductSerializer
from .mongo_models import OutfitHistory, PasswordResetAudit, User, UserInteraction


def convert_objectid_to_str(obj):
    """Recursively convert ObjectId instances to strings in nested data structures."""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_objectid_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    else:
        return obj


class UserSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    email = serializers.EmailField()
    name = serializers.CharField(required=False, allow_blank=True)
    username = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    is_staff = serializers.SerializerMethodField()
    is_active = serializers.BooleanField(required=False)
    height = serializers.FloatField(required=False, allow_null=True)
    weight = serializers.FloatField(required=False, allow_null=True)
    gender = serializers.CharField(required=False, allow_null=True)
    age = serializers.IntegerField(required=False, allow_null=True)
    preferences = serializers.DictField(default=dict)
    amazon_user_id = serializers.CharField(required=False, allow_null=True)
    favorites = serializers.ListField(child=serializers.CharField(), required=False)
    user_embedding = serializers.ListField(child=serializers.FloatField(), required=False)
    content_profile = serializers.DictField(required=False)
    interaction_history = serializers.ListField(required=False)
    outfit_history = serializers.ListField(required=False)
    created_at = serializers.DateTimeField(required=False, allow_null=True)
    updated_at = serializers.DateTimeField(required=False, allow_null=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def get_is_staff(self, obj):
        return obj.is_admin
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        # Get field values - mongoengine automatically maps db_field
        name = getattr(instance, 'name', None) or ""
        username = getattr(instance, 'username', None)
        first_name = getattr(instance, 'first_name', None) or ""
        last_name = getattr(instance, 'last_name', None) or ""
        height = getattr(instance, 'height', None)
        weight = getattr(instance, 'weight', None)
        gender = getattr(instance, 'gender', None)
        age = getattr(instance, 'age', None)
        preferences_raw = getattr(instance, 'preferences', None) or {}
        amazon_user_id = getattr(instance, 'amazon_user_id', None)
        is_admin = getattr(instance, 'is_admin', False)
        is_active = getattr(instance, 'is_active', True)
        
        # Get additional fields from database
        favorites = getattr(instance, 'favorites', None) or []
        user_embedding = getattr(instance, 'user_embedding', None) or []
        content_profile = getattr(instance, 'content_profile', None) or {}
        interaction_history_raw = getattr(instance, 'interaction_history', None) or []
        outfit_history_raw = getattr(instance, 'outfit_history', None) or []
        created_at = getattr(instance, 'created_at', None)
        updated_at = getattr(instance, 'updated_at', None)
        
        # If name is empty but we have first_name or last_name, use them
        if not name and (first_name or last_name):
            name = f"{first_name} {last_name}".strip()
        
        # Ensure preferences has complete structure
        preferences = {
            "priceRange": preferences_raw.get("priceRange", {"min": 0, "max": 1000000}),
            "style": preferences_raw.get("style", "casual"),
            "colorPreferences": preferences_raw.get("colorPreferences", []) or [],
            "brandPreferences": preferences_raw.get("brandPreferences", []) or []
        }
        # Merge any additional preference fields
        for key, value in preferences_raw.items():
            if key not in preferences:
                preferences[key] = value
        
        # Ensure content_profile has complete structure
        content_profile_result = {
            "featureVector": content_profile.get("featureVector", []) or [],
            "categoryWeights": content_profile.get("categoryWeights", []) or []
        }
        # Merge any additional content_profile fields
        for key, value in content_profile.items():
            if key not in content_profile_result:
                content_profile_result[key] = value
        
        # Convert favorites ObjectIds to strings
        favorites_list = [str(fav_id) for fav_id in favorites] if favorites else []
        
        # Convert ObjectId values in nested structures to strings
        interaction_history = convert_objectid_to_str(interaction_history_raw)
        outfit_history = convert_objectid_to_str(outfit_history_raw)
        content_profile_result = convert_objectid_to_str(content_profile_result)
        preferences = convert_objectid_to_str(preferences)
        
        # Format timestamps
        created_at_str = created_at.isoformat() if created_at else None
        updated_at_str = updated_at.isoformat() if updated_at else None
        
        result = {
            "id": str(instance.id),
            "email": getattr(instance, 'email', ''),
            "name": name,
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "is_staff": is_admin,
            "is_active": is_active,
            "height": height,
            "weight": weight,
            "gender": gender,
            "age": age,
            "preferences": preferences,
            "amazon_user_id": amazon_user_id,
            "favorites": favorites_list,
            "user_embedding": user_embedding,
            "content_profile": content_profile_result,
            "interaction_history": interaction_history,
            "outfit_history": outfit_history,
            "created_at": created_at_str,
            "updated_at": updated_at_str,
        }
        
        return result
    
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
            raise serializers.ValidationError("Password must be at least 8 characters.")
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
            raise serializers.ValidationError("Password must be at least 8 characters.")
        return value


class PasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()


class PasswordResetConfirmSerializer(serializers.Serializer):
    token = serializers.CharField()
    new_password = serializers.CharField()
    
    def validate_new_password(self, value):
        """Validate password."""
        if len(value) < 8:
            raise serializers.ValidationError("Password must be at least 8 characters.")
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

