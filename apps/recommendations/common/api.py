"""Shared DRF serializers for recommendation endpoints."""

from __future__ import annotations

from rest_framework import serializers

class TrainRequestSerializer(serializers.Serializer):
    force_retrain = serializers.BooleanField(default=False)
    sync = serializers.BooleanField(required=False, default=False, help_text="Run training synchronously (for testing)")
    task_id = serializers.CharField(required=False, allow_blank=True, help_text="Task ID to check training status")

class RecommendationRequestSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    current_product_id = serializers.CharField()
    top_k_personal = serializers.IntegerField(default=5, min_value=1, max_value=50)
    top_k_outfit = serializers.IntegerField(default=4, min_value=1, max_value=10)

