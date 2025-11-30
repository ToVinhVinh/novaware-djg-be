from __future__ import annotations

from apps.recommendations.common.api import RecommendationRequestSerializer, TrainRequestSerializer

class GNNTrainSerializer(TrainRequestSerializer):
    pass

class GNNRecommendationSerializer(RecommendationRequestSerializer):
    pass

