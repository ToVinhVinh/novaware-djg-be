"""API endpoints for the GNN-based recommendation engine."""

from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.recommendations.common.exceptions import ModelNotTrainedError

from .models import recommend_gnn, train_gnn_model
from .serializers import GNNRecommendationSerializer, GNNTrainSerializer


class TrainGNNView(APIView):
    serializer_class = GNNTrainSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        force_retrain = serializer.validated_data.get("force_retrain", False)
        async_result = train_gnn_model.delay(force_retrain=force_retrain)
        return Response(
            {
                "status": "training_started",
                "model": "gnn",
                "task_id": async_result.id,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class RecommendGNNView(APIView):
    serializer_class = GNNRecommendationSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            payload = recommend_gnn(
                user_id=serializer.validated_data["user_id"],
                current_product_id=serializer.validated_data["current_product_id"],
                top_k_personal=serializer.validated_data["top_k_personal"],
                top_k_outfit=serializer.validated_data["top_k_outfit"],
                request_params=serializer.validated_data,
            )
        except ModelNotTrainedError as exc:
            return Response(
                {"detail": str(exc), "model": "gnn"},
                status=status.HTTP_409_CONFLICT,
            )
        return Response(payload, status=status.HTTP_200_OK)

