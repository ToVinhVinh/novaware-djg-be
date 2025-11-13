"""API endpoints for the content-based filtering recommender."""

from __future__ import annotations

from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.recommendations.common.exceptions import ModelNotTrainedError

from .models import recommend_cbf, train_cbf_model
from .serializers import CBFRecommendationSerializer, CBFTrainSerializer


class TrainCBFView(APIView):
    serializer_class = CBFTrainSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        force_retrain = serializer.validated_data.get("force_retrain", False)
        async_result = train_cbf_model.delay(force_retrain=force_retrain)
        return Response(
            {
                "status": "training_started",
                "model": "cbf",
                "task_id": async_result.id,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class RecommendCBFView(APIView):
    serializer_class = CBFRecommendationSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            payload = recommend_cbf(
                user_id=serializer.validated_data["user_id"],
                current_product_id=serializer.validated_data["current_product_id"],
                top_k_personal=serializer.validated_data["top_k_personal"],
                top_k_outfit=serializer.validated_data["top_k_outfit"],
                request_params=serializer.validated_data,
            )
        except ModelNotTrainedError as exc:
            return Response(
                {"detail": str(exc), "model": "cbf"},
                status=status.HTTP_409_CONFLICT,
            )
        return Response(payload, status=status.HTTP_200_OK)

