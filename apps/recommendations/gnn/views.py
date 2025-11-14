"""API endpoints for the GNN-based recommendation engine."""

from __future__ import annotations

from django.core.exceptions import ValidationError
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import recommend_gnn, train_gnn_model
from .mongo_engine import recommend_gnn_mongo
from .serializers import GNNRecommendationSerializer, GNNTrainSerializer


class TrainGNNView(APIView):
    serializer_class = GNNTrainSerializer
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
                result = train_gnn_model(force_retrain=force_retrain)
                # Get matrix data from stored artifacts
                from apps.recommendations.common.storage import ArtifactStorage
                storage = ArtifactStorage("gnn")
                stored = storage.load()
                matrix_data = stored.get("artifacts", {}).get("matrix_data")
                return Response(
                    {
                        "status": "training_completed",
                        "model": "gnn",
                        "result": result,
                        "matrix_data": matrix_data,
                    },
                    status=status.HTTP_200_OK,
                )
            except Exception as e:
                return Response(
                    {
                        "status": "training_failed",
                        "model": "gnn",
                        "error": str(e),
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
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
            payload = recommend_gnn_mongo(
                user_id=serializer.validated_data["user_id"],
                current_product_id=serializer.validated_data["current_product_id"],
                top_k_personal=serializer.validated_data["top_k_personal"],
                top_k_outfit=serializer.validated_data["top_k_outfit"],
            )
            execution_time = time.time() - start_time
            
            # Calculate evaluation metrics
            personalized_recommendations = payload.get("personalized", [])
            metrics = calculate_evaluation_metrics(
                recommendations=personalized_recommendations,
                ground_truth=None,  # Can be added if ground truth data is available
                execution_time=execution_time,
            )
            metrics["model"] = "gnn"
            
            # Add metrics to response
            payload["evaluation_metrics"] = metrics
            
        except (ValueError, ValidationError) as exc:
            error_data = {}
            if hasattr(exc, "message_dict"):
                error_data = exc.message_dict
            elif hasattr(exc, "error_dict"):
                error_data = {k: v if isinstance(v, list) else [v] for k, v in exc.error_dict.items()}
            elif hasattr(exc, "error_list"):
                error_data = {"detail": exc.error_list}
            else:
                error_data = {"detail": [str(exc)]}
            return Response(
                error_data,
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            return Response(
                {"detail": [str(exc)]},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return Response(payload, status=status.HTTP_200_OK)

