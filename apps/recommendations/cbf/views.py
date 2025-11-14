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
        sync_mode = serializer.validated_data.get("sync", False)
        
        if sync_mode:
            # Run synchronously
            try:
                result = train_cbf_model(force_retrain=force_retrain)
                # Get matrix data from stored artifacts
                from apps.recommendations.common.storage import ArtifactStorage
                storage = ArtifactStorage("cbf")
                stored = storage.load()
                matrix_data = stored.get("artifacts", {}).get("matrix_data")
                return Response(
                    {
                        "status": "training_completed",
                        "model": "cbf",
                        "result": result,
                        "matrix_data": matrix_data,
                    },
                    status=status.HTTP_200_OK,
                )
            except Exception as e:
                return Response(
                    {
                        "status": "training_failed",
                        "model": "cbf",
                        "error": str(e),
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
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
        import time
        from apps.recommendations.common.evaluation import calculate_evaluation_metrics
        
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Measure execution time
        start_time = time.time()
        try:
            payload = recommend_cbf(
                user_id=serializer.validated_data["user_id"],
                current_product_id=serializer.validated_data["current_product_id"],
                top_k_personal=serializer.validated_data["top_k_personal"],
                top_k_outfit=serializer.validated_data["top_k_outfit"],
                request_params=serializer.validated_data,
            )
            execution_time = time.time() - start_time
            
            # Calculate evaluation metrics
            personalized_recommendations = payload.get("personalized", [])
            metrics = calculate_evaluation_metrics(
                recommendations=personalized_recommendations,
                ground_truth=None,  # Can be added if ground truth data is available
                execution_time=execution_time,
            )
            metrics["model"] = "cbf"
            
            # Add metrics to response
            payload["evaluation_metrics"] = metrics
            
        except ModelNotTrainedError as exc:
            return Response(
                {"detail": str(exc), "model": "cbf"},
                status=status.HTTP_409_CONFLICT,
            )
        return Response(payload, status=status.HTTP_200_OK)

