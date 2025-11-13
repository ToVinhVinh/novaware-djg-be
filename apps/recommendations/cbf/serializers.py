"""Serializers for the content-based filtering API."""

from __future__ import annotations

from apps.recommendations.common.api import RecommendationRequestSerializer, TrainRequestSerializer


class CBFTrainSerializer(TrainRequestSerializer):
    """Optional overrides for CBF training payload."""


class CBFRecommendationSerializer(RecommendationRequestSerializer):
    """Payload for requesting content-based recommendations."""

