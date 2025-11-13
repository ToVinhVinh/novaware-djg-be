"""Serializers for the hybrid recommendation API."""

from __future__ import annotations

from rest_framework import serializers

from apps.recommendations.common.api import RecommendationRequestSerializer, TrainRequestSerializer


class HybridTrainSerializer(TrainRequestSerializer):
    alpha = serializers.FloatField(required=False, min_value=0.0, max_value=1.0)
    sync = serializers.BooleanField(required=False, default=False, help_text="Run training synchronously (for testing)")


class HybridRecommendationSerializer(RecommendationRequestSerializer):
    alpha = serializers.FloatField(required=False, min_value=0.0, max_value=1.0)

