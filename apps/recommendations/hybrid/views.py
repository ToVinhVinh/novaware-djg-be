"""API endpoints for the hybrid recommendation engine."""

from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.recommendations.common.exceptions import ModelNotTrainedError

from .models import recommend_hybrid, train_hybrid_model
from .serializers import HybridRecommendationSerializer, HybridTrainSerializer


class TrainHybridView(APIView):
    serializer_class = HybridTrainSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        force_retrain = serializer.validated_data.get("force_retrain", False)
        alpha = serializer.validated_data.get("alpha")
        async_result = train_hybrid_model.delay(force_retrain=force_retrain, alpha=alpha)
        return Response(
            {
                "status": "training_started",
                "model": "hybrid",
                "task_id": async_result.id,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class RecommendHybridView(APIView):
    serializer_class = HybridRecommendationSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        alpha = serializer.validated_data.get("alpha")
        try:
            payload = recommend_hybrid(
                user_id=serializer.validated_data["user_id"],
                current_product_id=serializer.validated_data["current_product_id"],
                top_k_personal=serializer.validated_data["top_k_personal"],
                top_k_outfit=serializer.validated_data["top_k_outfit"],
                alpha=alpha,
                request_params=serializer.validated_data,
            )
        except ModelNotTrainedError as exc:
            return Response(
                {"detail": str(exc), "model": "hybrid"},
                status=status.HTTP_409_CONFLICT,
            )
        return Response(payload, status=status.HTTP_200_OK)

