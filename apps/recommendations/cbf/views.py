"""API endpoints for the content-based filtering recommender."""

from __future__ import annotations

import logging

from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.recommendations.common.background_tasks import get_task_status, submit_task
from apps.recommendations.common.exceptions import ModelNotTrainedError

from .models import recommend_cbf, train_cbf_model
from .serializers import CBFRecommendationSerializer, CBFTrainSerializer


class TrainCBFView(APIView):
    serializer_class = CBFTrainSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def _get_training_data(self, result: dict, include_artifacts: bool = True) -> dict:
        """Extract and format training data for frontend rendering."""
        from apps.recommendations.common.storage import ArtifactStorage
        
        training_data = {
            "status": result.get("status", "unknown"),
            "model": "cbf",
            "result": result,
        }
        
        # Extract metrics directly from result for easy access
        if isinstance(result, dict):
            # Training parameters
            training_data.update({
                "num_products": result.get("num_products"),
                "num_users": result.get("num_users"),
                "embedding_dim": result.get("embedding_dim", 384),  # Default SBERT dimension
                "test_size": 0.2,  # Default test split
            })
        
        if include_artifacts:
            try:
                storage = ArtifactStorage("cbf")
                stored = storage.load()
                artifacts = stored.get("artifacts", {})
                
                training_data["training_info"] = {
                    "trained_at": stored.get("trained_at") or result.get("trained_at"),
                    "model_name": stored.get("model", "cbf"),
                }
                
                # Include matrix data if available
                if "matrix_data" in artifacts:
                    matrix_data = artifacts["matrix_data"]
                    training_data["matrix_data"] = {
                        "shape": matrix_data.get("shape") if isinstance(matrix_data, dict) else None,
                        "sparsity": matrix_data.get("sparsity") if isinstance(matrix_data, dict) else None,
                    }
                
                # Include training metrics if available (prioritize artifacts over result)
                if "metrics" in artifacts:
                    training_data["metrics"] = artifacts["metrics"]
                    # Extract evaluation metrics if available
                    metrics = artifacts["metrics"]
                    if isinstance(metrics, dict):
                        training_data.update({
                            "mape": metrics.get("mape"),
                            "rmse": metrics.get("rmse"),
                            "precision": metrics.get("precision"),
                            "recall": metrics.get("recall"),
                            "f1": metrics.get("f1") or metrics.get("f1_score"),
                        })
                elif "training_metrics" in artifacts:
                    training_data["metrics"] = artifacts["training_metrics"]
                
                # Include model info if available
                if "model_info" in artifacts:
                    model_info = artifacts["model_info"]
                    training_data["model_info"] = model_info
                    # Extract additional info from model_info if available
                    if isinstance(model_info, dict):
                        training_data.update({
                            "embedding_dim": model_info.get("embedding_dim", training_data.get("embedding_dim", 384)),
                        })
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not load artifacts: {e}")
        
        return training_data

    def _get_task_status(self, task_id: str) -> dict:
        """Get the status of a training task."""
        try:
            task_status = get_task_status(task_id)
            
            if task_status is None:
                return {
                    "task_id": task_id,
                    "model": "cbf",
                    "status": "not_found",
                    "message": "Task not found",
                    "error": "Task ID does not exist or has been cleaned up",
                }
            
            response_data = {
                "task_id": task_id,
                "model": "cbf",
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
                        storage = ArtifactStorage("cbf")
                        stored = storage.load()
                        artifacts = stored.get("artifacts", {})
                        
                        response_data.update({
                            "message": "Training completed successfully",
                            "progress": 100,
                            "result": result,
                            "training_info": {
                                "trained_at": stored.get("trained_at"),
                                "model_name": stored.get("model", "cbf"),
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
                "model": "cbf",
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
        sync_mode = serializer.validated_data.get("sync", False)
        
        if sync_mode:
            # Run synchronously
            try:
                result = train_cbf_model(force_retrain=force_retrain)
                training_data = self._get_training_data(result, include_artifacts=True)
                return Response(training_data, status=status.HTTP_200_OK)
            except Exception as e:
                return Response(
                    {
                        "status": "training_failed",
                        "model": "cbf",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
            # Run asynchronously using background task manager
            try:
                from apps.recommendations.cbf.models import engine
                task_id = submit_task(
                    engine.train,
                    force_retrain=force_retrain
                )
                # Get initial status immediately
                status_data = self._get_task_status(task_id)
                return Response(status_data, status=status.HTTP_202_ACCEPTED)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to start background task: {e}", exc_info=True)
                # Fall back to synchronous execution
                try:
                    result = train_cbf_model(force_retrain=force_retrain)
                    training_data = self._get_training_data(result, include_artifacts=True)
                    training_data["note"] = "Ran synchronously (background task failed)"
                    return Response(training_data, status=status.HTTP_200_OK)
                except Exception as sync_error:
                    return Response(
                        {
                            "status": "training_failed",
                            "model": "cbf",
                            "error": str(sync_error),
                            "error_type": type(sync_error).__name__,
                            "note": "Both async and sync execution failed",
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )


class RecommendCBFView(APIView):
    serializer_class = CBFRecommendationSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    
    def get(self, request, *args, **kwargs):
        """Handle GET requests with query parameters."""
        import time
        from apps.recommendations.common.evaluation import calculate_evaluation_metrics
        
        # Extract query parameters
        user_id = request.query_params.get('user_id')
        product_id = request.query_params.get('product_id')
        top_k_personal = int(request.query_params.get('top_k_personal', 5))
        top_k_outfit = int(request.query_params.get('top_k_outfit', 4))
        
        if not user_id or not product_id:
            return Response(
                {"detail": "user_id and product_id are required query parameters"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        # Measure execution time
        start_time = time.time()
        try:
            payload = recommend_cbf(
                user_id=user_id,
                current_product_id=product_id,
                top_k_personal=top_k_personal,
                top_k_outfit=top_k_outfit,
                request_params=dict(request.query_params),
            )
            execution_time = time.time() - start_time
            
            # Calculate evaluation metrics
            personalized_recommendations = payload.get("personalized", [])
            metrics = calculate_evaluation_metrics(
                recommendations=personalized_recommendations,
                ground_truth=None,
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
        except Exception as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return Response(payload, status=status.HTTP_200_OK)

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

