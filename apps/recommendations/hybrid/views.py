"""API endpoints for the hybrid recommendation engine."""

from __future__ import annotations

import logging
import socket

from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.recommendations.common.exceptions import ModelNotTrainedError

from .models import recommend_hybrid, train_hybrid_model
from .serializers import HybridRecommendationSerializer, HybridTrainSerializer

logger = logging.getLogger(__name__)


def check_redis_connection(timeout=2):
    """Check if Redis is available by trying to connect."""
    try:
        from django.conf import settings
        broker_url = getattr(settings, "CELERY_BROKER_URL", "redis://localhost:6379/0")
        # Parse Redis URL
        if broker_url.startswith("redis://"):
            parts = broker_url.replace("redis://", "").split("/")
            host_port = parts[0].split(":")
            host = host_port[0] if host_port else "localhost"
            port = int(host_port[1]) if len(host_port) > 1 else 6379
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
    except Exception:
        pass
    return False


class TrainHybridView(APIView):
    serializer_class = HybridTrainSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        force_retrain = serializer.validated_data.get("force_retrain", False)
        alpha = serializer.validated_data.get("alpha")
        sync_mode = serializer.validated_data.get("sync", False)
        
        logger.info(f"[hybrid] Train request received: force_retrain={force_retrain}, alpha={alpha}, sync={sync_mode}")
        
        if sync_mode:
            # Run synchronously for testing/debugging
            logger.info("[hybrid] Running training in sync mode")
            try:
                result = train_hybrid_model(force_retrain=force_retrain, alpha=alpha)
                # Get matrix data from stored artifacts
                from apps.recommendations.common.storage import ArtifactStorage
                storage = ArtifactStorage("hybrid")
                stored = storage.load()
                matrix_data = stored.get("artifacts", {}).get("matrix_data")
                return Response(
                    {
                        "status": "training_completed",
                        "model": "hybrid",
                        "result": result,
                        "matrix_data": matrix_data,
                    },
                    status=status.HTTP_200_OK,
                )
            except Exception as e:
                logger.error(f"[hybrid] Training failed: {e}", exc_info=True)
                return Response(
                    {
                        "status": "training_failed",
                        "model": "hybrid",
                        "error": str(e),
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
            # Run asynchronously with Celery
            # First check if Redis is available
            if not check_redis_connection():
                logger.warning(
                    "[hybrid] Redis/Celery broker not available, automatically falling back to sync mode"
                )
                try:
                    result = train_hybrid_model(force_retrain=force_retrain, alpha=alpha)
                    # Get matrix data from stored artifacts
                    from apps.recommendations.common.storage import ArtifactStorage
                    storage = ArtifactStorage("hybrid")
                    stored = storage.load()
                    matrix_data = stored.get("artifacts", {}).get("matrix_data")
                    return Response(
                        {
                            "status": "training_completed",
                            "model": "hybrid",
                            "result": result,
                            "warning": "Celery/Redis unavailable, automatically ran in sync mode",
                            "matrix_data": matrix_data,
                        },
                        status=status.HTTP_200_OK,
                    )
                except Exception as sync_error:
                    logger.error(f"[hybrid] Sync training failed: {sync_error}", exc_info=True)
                    return Response(
                        {
                            "status": "training_failed",
                            "model": "hybrid",
                            "error": str(sync_error),
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )
            
            # Redis is available, try to use Celery
            try:
                async_result = train_hybrid_model.delay(force_retrain=force_retrain, alpha=alpha)
                logger.info(f"[hybrid] Training task started: task_id={async_result.id}")
                return Response(
                    {
                        "status": "training_started",
                        "model": "hybrid",
                        "task_id": async_result.id,
                    },
                    status=status.HTTP_202_ACCEPTED,
                )
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a Redis/Celery connection error
                is_connection_error = any(
                    keyword in error_msg
                    for keyword in ["redis", "connection", "broker", "celery", "timeout"]
                )
                
                if is_connection_error:
                    # Auto-fallback to sync mode
                    logger.warning(
                        f"[hybrid] Celery task failed ({e}), falling back to sync mode"
                    )
                    try:
                        result = train_hybrid_model(force_retrain=force_retrain, alpha=alpha)
                        # Get matrix data from stored artifacts
                        from apps.recommendations.common.storage import ArtifactStorage
                        storage = ArtifactStorage("hybrid")
                        stored = storage.load()
                        matrix_data = stored.get("artifacts", {}).get("matrix_data")
                        return Response(
                            {
                                "status": "training_completed",
                                "model": "hybrid",
                                "result": result,
                                "warning": "Celery task failed, ran in sync mode",
                                "matrix_data": matrix_data,
                            },
                            status=status.HTTP_200_OK,
                        )
                    except Exception as sync_error:
                        logger.error(f"[hybrid] Sync training also failed: {sync_error}", exc_info=True)
                        return Response(
                            {
                                "status": "training_failed",
                                "model": "hybrid",
                                "error": f"Celery failed: {e}. Sync fallback also failed: {sync_error}",
                            },
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        )
                else:
                    # Other errors - return as is
                    logger.error(f"[hybrid] Failed to start training task: {e}", exc_info=True)
                    return Response(
                        {
                            "status": "training_failed",
                            "model": "hybrid",
                            "error": str(e),
                            "suggestion": "Try with sync=true if Celery/Redis is not available",
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )


class RecommendHybridView(APIView):
    serializer_class = HybridRecommendationSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def post(self, request, *args, **kwargs):
        import time
        from apps.recommendations.common.evaluation import calculate_evaluation_metrics
        
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        alpha = serializer.validated_data.get("alpha")
        
        # Measure execution time
        start_time = time.time()
        try:
            payload = recommend_hybrid(
                user_id=serializer.validated_data["user_id"],
                current_product_id=serializer.validated_data["current_product_id"],
                top_k_personal=serializer.validated_data["top_k_personal"],
                top_k_outfit=serializer.validated_data["top_k_outfit"],
                alpha=alpha,
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
            metrics["model"] = "hybrid"
            
            # Add metrics to response
            payload["evaluation_metrics"] = metrics
            
        except ModelNotTrainedError as exc:
            return Response(
                {"detail": str(exc), "model": "hybrid"},
                status=status.HTTP_409_CONFLICT,
            )
        return Response(payload, status=status.HTTP_200_OK)

