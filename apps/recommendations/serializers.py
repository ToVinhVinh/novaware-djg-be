"""Serializer for recommendation system."""

from __future__ import annotations

from rest_framework import serializers

from .models import Outfit, RecommendationLog, RecommendationRequest, RecommendationResult


class OutfitSerializer(serializers.ModelSerializer):
    class Meta:
        model = Outfit
        fields = [
            "id",
            "name",
            "products",
            "style",
            "season",
            "total_price",
            "compatibility_score",
            "created_at",
            "updated_at",
        ]


class RecommendationLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecommendationLog
        fields = ["id", "message", "created_at"]


class RecommendationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecommendationResult
        fields = ["id", "request", "products", "metadata", "created_at"]


class RecommendationRequestSerializer(serializers.ModelSerializer):
    result = RecommendationResultSerializer(read_only=True)
    logs = RecommendationLogSerializer(many=True, read_only=True)

    class Meta:
        model = RecommendationRequest
        fields = ["id", "user", "algorithm", "parameters", "created_at", "result", "logs"]
        read_only_fields = ["user", "created_at", "result", "logs"]

