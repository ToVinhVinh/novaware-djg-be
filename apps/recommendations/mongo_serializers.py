"""Serializers cho MongoEngine Recommendation models."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import serializers

from .mongo_models import (
    Outfit,
    RecommendationLog,
    RecommendationRequest,
    RecommendationResult,
)


class OutfitSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    name = serializers.CharField()
    product_ids = serializers.ListField(child=serializers.CharField(), required=False)
    style = serializers.CharField(required=False, allow_blank=True)
    season = serializers.CharField(required=False, allow_blank=True)
    total_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False, allow_null=True)
    compatibility_score = serializers.FloatField(required=False, allow_null=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        return {
            "id": str(instance.id),
            "name": instance.name,
            "product_ids": [str(pid) for pid in instance.product_ids],
            "style": instance.style,
            "season": instance.season,
            "total_price": float(instance.total_price) if instance.total_price else None,
            "compatibility_score": instance.compatibility_score,
            "created_at": instance.created_at,
            "updated_at": instance.updated_at,
        }
    
    def create(self, validated_data):
        """Create new outfit."""
        product_ids = validated_data.pop("product_ids", [])
        validated_data["product_ids"] = [ObjectId(pid) for pid in product_ids]
        return Outfit(**validated_data).save()
    
    def update(self, instance, validated_data):
        """Update outfit."""
        product_ids = validated_data.pop("product_ids", None)
        if product_ids is not None:
            validated_data["product_ids"] = [ObjectId(pid) for pid in product_ids]
        for key, value in validated_data.items():
            setattr(instance, key, value)
        instance.save()
        return instance


class RecommendationLogSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    message = serializers.CharField()
    created_at = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def create(self, validated_data):
        """Create new log."""
        return RecommendationLog(**validated_data).save()


class RecommendationResultSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    request_id = serializers.CharField()
    product_ids = serializers.ListField(child=serializers.CharField())
    metadata = serializers.DictField(default=dict)
    created_at = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        return {
            "id": str(instance.id),
            "request_id": str(instance.request_id),
            "product_ids": [str(pid) for pid in instance.product_ids],
            "metadata": instance.metadata,
            "created_at": instance.created_at,
        }
    
    def create(self, validated_data):
        """Create new result."""
        product_ids = validated_data.pop("product_ids", [])
        validated_data["product_ids"] = [ObjectId(pid) for pid in product_ids]
        validated_data["request_id"] = ObjectId(validated_data["request_id"])
        return RecommendationResult(**validated_data).save()


class RecommendationRequestSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    user_id = serializers.CharField(write_only=True, required=False)
    algorithm = serializers.CharField()
    parameters = serializers.DictField(default=dict)
    created_at = serializers.DateTimeField(read_only=True)
    result = RecommendationResultSerializer(read_only=True)
    logs = RecommendationLogSerializer(many=True, read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        data = {
            "id": str(instance.id),
            "user_id": str(instance.user_id),
            "algorithm": instance.algorithm,
            "parameters": instance.parameters,
            "created_at": instance.created_at,
        }
        
        # Load result
        try:
            result = RecommendationResult.objects.get(request_id=instance.id)
            data["result"] = RecommendationResultSerializer(result).to_representation(result)
        except RecommendationResult.DoesNotExist:
            data["result"] = None
        
        # Load logs
        logs = RecommendationLog.objects(request_id=instance.id)
        data["logs"] = [RecommendationLogSerializer(log).to_representation(log) for log in logs]
        
        return data
    
    def create(self, validated_data):
        """Create new recommendation request."""
        if "user_id" in validated_data:
            validated_data["user_id"] = ObjectId(validated_data["user_id"])
        return RecommendationRequest(**validated_data).save()
    
    def update(self, instance, validated_data):
        """Update recommendation request."""
        for key, value in validated_data.items():
            setattr(instance, key, value)
        instance.save()
        return instance

