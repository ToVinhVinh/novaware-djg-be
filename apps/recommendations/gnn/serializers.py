"""Serializers for the GNN recommendation API."""

from __future__ import annotations

from apps.recommendations.common.api import RecommendationRequestSerializer, TrainRequestSerializer


class GNNTrainSerializer(TrainRequestSerializer):
    """No additional fields for now."""


class GNNRecommendationSerializer(RecommendationRequestSerializer):
    """Request payload for GNN recommendations."""

