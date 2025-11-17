"""API endpoints for the hybrid recommendation engine."""

from __future__ import annotations

import logging

from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.recommendations.common.background_tasks import get_task_status, submit_task
from apps.recommendations.common.exceptions import ModelNotTrainedError

from .models import recommend_hybrid, train_hybrid_model
from .serializers import HybridRecommendationSerializer, HybridTrainSerializer

logger = logging.getLogger(__name__)


class TrainHybridView(APIView):
    serializer_class = HybridTrainSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def _get_training_data(self, result: dict, include_artifacts: bool = True) -> dict:
        """Extract and format training data for frontend rendering."""
        from apps.recommendations.common.storage import ArtifactStorage
        
        training_data = {
            "status": result.get("status", "unknown"),
            "model": "hybrid",
            "result": result,
        }
        
        if include_artifacts:
            try:
                storage = ArtifactStorage("hybrid")
                stored = storage.load()
                artifacts = stored.get("artifacts", {})
                
                training_data["training_info"] = {
                    "trained_at": stored.get("trained_at"),
                    "model_name": stored.get("model", "hybrid"),
                }
                
                # Include matrix data if available
                if "matrix_data" in artifacts:
                    matrix_data = artifacts["matrix_data"]
                    training_data["matrix_data"] = {
                        "shape": matrix_data.get("shape") if isinstance(matrix_data, dict) else None,
                        "sparsity": matrix_data.get("sparsity") if isinstance(matrix_data, dict) else None,
                    }
                
                # Include training metrics if available
                if "metrics" in artifacts:
                    training_data["metrics"] = artifacts["metrics"]
                elif "training_metrics" in artifacts:
                    training_data["metrics"] = artifacts["training_metrics"]
                
                # Include model info if available
                if "model_info" in artifacts:
                    training_data["model_info"] = artifacts["model_info"]
                
                # Include alpha if available
                if "alpha" in artifacts:
                    training_data["alpha"] = artifacts["alpha"]
            except Exception as e:
                logger.warning(f"Could not load artifacts: {e}")
        
        return training_data

    def _get_task_status(self, task_id: str) -> dict:
        """Get the status of a training task."""
        try:
            task_status = get_task_status(task_id)
            
            if task_status is None:
                return {
                    "task_id": task_id,
                    "model": "hybrid",
                    "status": "not_found",
                    "message": "Task not found",
                    "error": "Task ID does not exist or has been cleaned up",
                }
            
            response_data = {
                "task_id": task_id,
                "model": "hybrid",
                "status": task_status.status,
            }
            
            if task_status.status == "pending":
                response_data.update({
                    "message": "Task is waiting to be processed",
                    "progress": 0,
                })
            elif task_status.status == "running":
                response_data.update({
                    "message": "Training in progress",
                    "progress": task_status.progress,
                    "current_step": task_status.current_step,
                    "total_steps": task_status.total_steps,
                })
            elif task_status.status == "success":
                result = task_status.result
                if result:
                    from apps.recommendations.common.storage import ArtifactStorage
                    try:
                        storage = ArtifactStorage("hybrid")
                        stored = storage.load()
                        artifacts = stored.get("artifacts", {})
                        
                        response_data.update({
                            "message": "Training completed successfully",
                            "progress": 100,
                            "result": result,
                            "training_info": {
                                "trained_at": stored.get("trained_at"),
                                "model_name": stored.get("model", "hybrid"),
                            },
                        })
                        
                        if "metrics" in artifacts:
                            response_data["metrics"] = artifacts["metrics"]
                        elif "training_metrics" in artifacts:
                            response_data["metrics"] = artifacts["training_metrics"]
                            
                        if "matrix_data" in artifacts:
                            matrix_data = artifacts["matrix_data"]
                            response_data["matrix_data"] = {
                                "shape": matrix_data.get("shape") if isinstance(matrix_data, dict) else None,
                                "sparsity": matrix_data.get("sparsity") if isinstance(matrix_data, dict) else None,
                            }
                        
                        if "alpha" in artifacts:
                            response_data["alpha"] = artifacts["alpha"]
                    except Exception as e:
                        response_data["result"] = result
                        response_data["warning"] = f"Could not load full artifacts: {e}"
                else:
                    response_data.update({
                        "message": "Training completed",
                        "progress": 100,
                    })
            elif task_status.status == "failure":
                response_data.update({
                    "message": "Training failed",
                    "error": task_status.error or "Unknown error",
                    "progress": task_status.progress,
                })
            else:
                response_data.update({
                    "message": f"Task status: {task_status.status}",
                    "progress": task_status.progress,
                })
            
            return response_data
        except Exception as e:
            return {
                "task_id": task_id,
                "model": "hybrid",
                "status": "error",
                "error": str(e),
            }

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Check if task_id is provided - if so, return status
        task_id = serializer.validated_data.get("task_id")
        if task_id and task_id.strip():
            status_data = self._get_task_status(task_id)
            status_code = status.HTTP_200_OK
            if status_data.get("status") == "error":
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return Response(status_data, status=status_code)
        
        # Otherwise, start training
        force_retrain = serializer.validated_data.get("force_retrain", False)
        alpha = serializer.validated_data.get("alpha")
        sync_mode = serializer.validated_data.get("sync", False)
        
        logger.info(f"[hybrid] Train request received: force_retrain={force_retrain}, alpha={alpha}, sync={sync_mode}")
        
        if sync_mode:
            # Run synchronously for testing/debugging
            logger.info("[hybrid] Running training in sync mode")
            try:
                result = train_hybrid_model(force_retrain=force_retrain, alpha=alpha)
                training_data = self._get_training_data(result, include_artifacts=True)
                return Response(training_data, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"[hybrid] Training failed: {e}", exc_info=True)
                return Response(
                    {
                        "status": "training_failed",
                        "model": "hybrid",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
            # Run asynchronously using background task manager
            try:
                from apps.recommendations.hybrid.models import engine
                
                # Create a wrapper function to handle alpha parameter
                def train_with_alpha():
                    if alpha is not None:
                        original_alpha = engine.alpha
                        engine.alpha = alpha
                        try:
                            return engine.train(force_retrain=force_retrain)
                        finally:
                            engine.alpha = original_alpha
                    else:
                        return engine.train(force_retrain=force_retrain)
                
                task_id = submit_task(train_with_alpha)
                logger.info(f"[hybrid] Training task started: task_id={task_id}")
                # Get initial status immediately
                status_data = self._get_task_status(task_id)
                return Response(status_data, status=status.HTTP_202_ACCEPTED)
            except Exception as e:
                logger.error(f"[hybrid] Failed to start background task: {e}", exc_info=True)
                # Fall back to synchronous execution
                try:
                    result = train_hybrid_model(force_retrain=force_retrain, alpha=alpha)
                    training_data = self._get_training_data(result, include_artifacts=True)
                    training_data["note"] = "Ran synchronously (background task failed)"
                    return Response(training_data, status=status.HTTP_200_OK)
                except Exception as sync_error:
                    logger.error(f"[hybrid] Sync training also failed: {sync_error}", exc_info=True)
                    return Response(
                        {
                            "status": "training_failed",
                            "model": "hybrid",
                            "error": str(sync_error),
                            "error_type": type(sync_error).__name__,
                            "note": "Both async and sync execution failed",
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

