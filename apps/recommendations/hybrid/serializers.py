from __future__ import annotations

from rest_framework import serializers
from apps.recommendations.common.api import RecommendationRequestSerializer, TrainRequestSerializer

class HybridTrainSerializer(TrainRequestSerializer):
    pass

class HybridRecommendationSerializer(RecommendationRequestSerializer):
    alpha = serializers.FloatField(default=0.5, min_value=0.0, max_value=1.0, help_text="Trọng số Hybrid α (GNN ↔ CBF)")
    top_k_personalized = serializers.IntegerField(default=6, min_value=3, max_value=50, help_text="Số lượng sản phẩm Personalized")
    top_k_outfit = serializers.IntegerField(default=3, min_value=1, max_value=5, help_text="Số lượng outfit muốn xem")

