"""API endpoints for the GNN-based recommendation engine."""

from __future__ import annotations

import logging

from django.core.exceptions import ValidationError
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.recommendations.common.background_tasks import get_task_status, submit_task
from apps.recommendations.common.exceptions import ModelNotTrainedError

from .models import recommend_gnn, train_gnn_model
from .serializers import GNNRecommendationSerializer, GNNTrainSerializer


class TrainGNNView(APIView):
    serializer_class = GNNTrainSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def _get_training_data(self, result: dict, include_artifacts: bool = True) -> dict:
        """Extract and format training data for frontend rendering."""
        from apps.recommendations.common.storage import ArtifactStorage
        
        training_data = {
            "status": result.get("status", "unknown"),
            "model": "gnn",
            "result": result,
        }
        
        # Extract metrics directly from result for easy access
        if isinstance(result, dict):
            # Training parameters - extract from result first, use defaults only if not present
            training_data.update({
                "num_users": result.get("num_users"),
                "num_products": result.get("num_products"),
                "num_interactions": result.get("num_interactions"),
                "num_training_samples": result.get("num_training_samples"),
                "embedding_dim": result.get("embedding_dim"),
                "epochs": result.get("epochs", 50),  # Extract from result, default 50
                "batch_size": result.get("batch_size", 2048),  # Extract from result, default 2048
                "learning_rate": result.get("learning_rate", 0.001),  # Extract from result, default 0.001
                "test_size": result.get("test_size", 0.2),  # Extract from result, default 0.2
                "training_time": result.get("training_time"),  # Extract training time if available
            })
            
            # Try to extract from matrix_data if available in result
            if "matrix_data" in result:
                matrix_data = result["matrix_data"]
                training_data["matrix_data"] = {
                    "shape": matrix_data.get("shape") if isinstance(matrix_data, dict) else None,
                    "sparsity": matrix_data.get("sparsity") if isinstance(matrix_data, dict) else None,
                }
                if isinstance(matrix_data, dict) and "shape" in matrix_data:
                    shape = matrix_data["shape"]
                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                        if training_data.get("num_users") is None:
                            training_data["num_users"] = shape[0]
                        if training_data.get("num_products") is None:
                            training_data["num_products"] = shape[1]
                        
                        # Calculate num_interactions from sparsity if available
                        if training_data.get("num_interactions") is None and "sparsity" in matrix_data:
                            total_possible = shape[0] * shape[1]
                            sparsity = matrix_data.get("sparsity", 0)
                            if sparsity < 1.0 and total_possible > 0:
                                training_data["num_interactions"] = int(total_possible * (1 - sparsity))
                        
                        # Also set num_training_samples from num_interactions if available
                        if training_data.get("num_training_samples") is None and training_data.get("num_interactions") is not None:
                            training_data["num_training_samples"] = training_data["num_interactions"]
            
            # Training info from result
            if "training_info" in result:
                training_data["training_info"] = result["training_info"]
            
            # Training metrics from result
            if "training_metrics" in result:
                training_data["training_metrics"] = result["training_metrics"]
        
        if include_artifacts:
            try:
                storage = ArtifactStorage("gnn")
                stored = storage.load()
                artifacts = stored.get("artifacts", {})
                
                # Set training_info if not already set from result
                if "training_info" not in training_data:
                    training_data["training_info"] = {
                        "trained_at": stored.get("trained_at") or result.get("trained_at"),
                        "model_name": stored.get("model", "gnn"),
                    }
                else:
                    # Merge with stored data if available
                    if stored.get("trained_at") and not training_data["training_info"].get("trained_at"):
                        training_data["training_info"]["trained_at"] = stored.get("trained_at")
                    if not training_data["training_info"].get("model_name"):
                        training_data["training_info"]["model_name"] = stored.get("model", "gnn")
                
                # First, try to extract training parameters directly from artifacts (only if not already set from result)
                # Priority: num_training_samples and embedding_dim are critical for BPR training display
                if isinstance(artifacts, dict):
                    # Extract all training parameters from artifacts
                    for key in ["num_users", "num_products", "num_interactions", "num_training_samples", "embedding_dim",
                               "epochs", "batch_size", "learning_rate", "test_size", "training_time"]:
                        if key in artifacts and training_data.get(key) is None:
                            training_data[key] = artifacts[key]
                    
                    # Special handling for num_training_samples (BPR training samples)
                    # If still None, try to calculate from num_interactions
                    if training_data.get("num_training_samples") is None:
                        if training_data.get("num_interactions") is not None:
                            training_data["num_training_samples"] = training_data["num_interactions"]
                        elif "num_interactions" in artifacts:
                            training_data["num_training_samples"] = artifacts["num_interactions"]
                    
                    # Special handling for embedding_dim - ensure it has a default if still None
                    if training_data.get("embedding_dim") is None:
                        # Try to infer from embeddings if available
                        if "user_embeddings" in artifacts and isinstance(artifacts["user_embeddings"], list):
                            if len(artifacts["user_embeddings"]) > 0:
                                if isinstance(artifacts["user_embeddings"][0], list):
                                    training_data["embedding_dim"] = len(artifacts["user_embeddings"][0])
                        # Default to 64 (LightGCN default) if still None
                        if training_data.get("embedding_dim") is None:
                            training_data["embedding_dim"] = 64
                
                # Include matrix data if available (only if not already set from result)
                if "matrix_data" in artifacts and "matrix_data" not in training_data:
                    matrix_data = artifacts["matrix_data"]
                    training_data["matrix_data"] = {
                        "shape": matrix_data.get("shape") if isinstance(matrix_data, dict) else None,
                        "sparsity": matrix_data.get("sparsity") if isinstance(matrix_data, dict) else None,
                    }
                    # Extract num_users and num_products from matrix shape if available
                    if isinstance(matrix_data, dict) and "shape" in matrix_data:
                        shape = matrix_data["shape"]
                        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                            if training_data.get("num_users") is None:
                                training_data["num_users"] = shape[0]
                            if training_data.get("num_products") is None:
                                training_data["num_products"] = shape[1]
                            
                            # Calculate num_interactions from sparsity if available
                            if training_data.get("num_interactions") is None and "sparsity" in matrix_data:
                                total_possible = shape[0] * shape[1]
                                sparsity = matrix_data.get("sparsity", 0)
                                if sparsity < 1.0 and total_possible > 0:
                                    training_data["num_interactions"] = int(total_possible * (1 - sparsity))
                
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
                    # Extract additional info from model_info if available (only if not already set)
                    if isinstance(model_info, dict):
                        for key in ["num_users", "num_products", "num_interactions", "num_training_samples",
                                   "embedding_dim", "epochs", "batch_size", "learning_rate", "test_size", "training_time"]:
                            if key in model_info and training_data.get(key) is None:
                                training_data[key] = model_info[key]
                        # Set defaults for epochs, batch_size, learning_rate if still None
                        if training_data.get("epochs") is None:
                            training_data["epochs"] = model_info.get("epochs", 50)
                        if training_data.get("batch_size") is None:
                            training_data["batch_size"] = model_info.get("batch_size", 2048)
                        if training_data.get("learning_rate") is None:
                            training_data["learning_rate"] = model_info.get("learning_rate", 0.001)
                        if training_data.get("test_size") is None:
                            training_data["test_size"] = model_info.get("test_size", 0.2)
                
                # Also try to get from stored training data in artifacts
                if "training_data" in artifacts:
                    stored_training = artifacts["training_data"]
                    if isinstance(stored_training, dict):
                        for key in ["num_users", "num_products", "num_interactions", "num_training_samples", 
                                   "embedding_dim", "epochs", "batch_size", "learning_rate", "test_size", "training_time"]:
                            if key in stored_training and training_data.get(key) is None:
                                training_data[key] = stored_training[key]
                
                # After loading artifacts, check if matrix_data was set and extract from it
                if "matrix_data" in training_data and training_data["matrix_data"]:
                    matrix_data = training_data["matrix_data"]
                    if isinstance(matrix_data, dict) and "shape" in matrix_data:
                        shape = matrix_data["shape"]
                        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                            if training_data.get("num_users") is None:
                                training_data["num_users"] = shape[0]
                            if training_data.get("num_products") is None:
                                training_data["num_products"] = shape[1]
                            
                            # Calculate num_interactions from sparsity if available
                            if training_data.get("num_interactions") is None and "sparsity" in matrix_data:
                                total_possible = shape[0] * shape[1]
                                sparsity = matrix_data.get("sparsity", 0)
                                if sparsity < 1.0 and total_possible > 0:
                                    training_data["num_interactions"] = int(total_possible * (1 - sparsity))
                            
                            # Also set num_training_samples from num_interactions if available
                            if training_data.get("num_training_samples") is None and training_data.get("num_interactions") is not None:
                                training_data["num_training_samples"] = training_data["num_interactions"]
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not load artifacts: {e}")
        
        # If still missing values and model was loaded, try to calculate from artifacts or model file
        if isinstance(result, dict) and result.get("status") in ["loaded", "already_trained"]:
            # Check if we still have any null values that need to be filled
            has_null_values = (
                training_data.get("num_users") is None or 
                training_data.get("num_products") is None or
                training_data.get("embedding_dim") is None or
                training_data.get("num_interactions") is None or
                training_data.get("num_training_samples") is None
            )
            
            if has_null_values:
                # First, try to calculate from artifacts if available
                if include_artifacts:
                    try:
                        storage = ArtifactStorage("gnn")
                        stored = storage.load()
                        artifacts = stored.get("artifacts", {})
                        
                        # Try to calculate num_interactions from user_ids/product_ids or edge_index
                        if training_data.get("num_interactions") is None:
                            # If we have user_ids and product_ids, we can estimate from matrix_data
                            if "matrix_data" in artifacts and isinstance(artifacts["matrix_data"], dict):
                                matrix_data = artifacts["matrix_data"]
                                if "sparsity" in matrix_data and "shape" in matrix_data:
                                    shape = matrix_data["shape"]
                                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                                        total_possible = shape[0] * shape[1]
                                        sparsity = matrix_data.get("sparsity", 0)
                                        if sparsity < 1.0:
                                            training_data["num_interactions"] = int(total_possible * (1 - sparsity))
                            
                            # Or try from user_embeddings/item_embeddings if available
                            if training_data.get("num_interactions") is None:
                                if "user_embeddings" in artifacts and "item_embeddings" in artifacts:
                                    # Estimate from embeddings (rough estimate)
                                    user_emb = artifacts["user_embeddings"]
                                    item_emb = artifacts["item_embeddings"]
                                    if isinstance(user_emb, list) and isinstance(item_emb, list):
                                        # Rough estimate: assume some interactions exist
                                        training_data["num_interactions"] = max(len(user_emb), len(item_emb))
                        
                        # Set num_training_samples to num_interactions if not set
                        if training_data.get("num_training_samples") is None and training_data.get("num_interactions") is not None:
                            training_data["num_training_samples"] = training_data["num_interactions"]
                        
                        # Set embedding_dim to default if not set and we have embeddings
                        if training_data.get("embedding_dim") is None:
                            if "user_embeddings" in artifacts and isinstance(artifacts["user_embeddings"], list):
                                if len(artifacts["user_embeddings"]) > 0:
                                    if isinstance(artifacts["user_embeddings"][0], list):
                                        training_data["embedding_dim"] = len(artifacts["user_embeddings"][0])
                                    elif isinstance(artifacts["user_embeddings"][0], (int, float)):
                                        # If it's a flat list, we can't determine dim
                                        pass
                            
                            # Default to 64 if still None (LightGCN default)
                            if training_data.get("embedding_dim") is None:
                                training_data["embedding_dim"] = 64
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Could not calculate from artifacts: {e}")
                
                # If still missing, try to load from model file
                if (training_data.get("num_users") is None or 
                    training_data.get("num_products") is None or
                    training_data.get("embedding_dim") is None or
                    training_data.get("num_interactions") is None):
                    try:
                        import pickle
                        from pathlib import Path
                        from django.conf import settings
                        
                        # Try to load from model file
                        model_dir = Path(settings.BASE_DIR) / "models"
                        model_path = model_dir / "gnn_lightgcn.pkl"
                        
                        if model_path.exists():
                            with open(model_path, 'rb') as f:
                                model_data = pickle.load(f)
                            
                            # Extract info from model file (only if still null)
                            if training_data.get("num_users") is None:
                                if "num_users" in model_data:
                                    training_data["num_users"] = model_data["num_users"]
                                elif "user_id_map" in model_data:
                                    training_data["num_users"] = len(model_data["user_id_map"])
                            
                            if training_data.get("num_products") is None:
                                if "num_products" in model_data:
                                    training_data["num_products"] = model_data["num_products"]
                                elif "product_id_map" in model_data:
                                    training_data["num_products"] = len(model_data["product_id_map"])
                            
                            if training_data.get("embedding_dim") is None:
                                if "embedding_dim" in model_data:
                                    training_data["embedding_dim"] = model_data["embedding_dim"]
                                elif "model" in model_data and hasattr(model_data["model"], "embedding_dim"):
                                    training_data["embedding_dim"] = model_data["model"].embedding_dim
                                else:
                                    training_data["embedding_dim"] = 64  # Default
                            
                            # Try to calculate num_interactions from edge_index
                            if training_data.get("num_interactions") is None:
                                if "edge_index" in model_data:
                                    edge_index = model_data["edge_index"]
                                    try:
                                        if hasattr(edge_index, 'shape'):
                                            # PyTorch tensor or numpy array
                                            if len(edge_index.shape) >= 2:
                                                training_data["num_interactions"] = int(edge_index.shape[1] // 2)  # Divide by 2 for undirected
                                            elif len(edge_index.shape) == 1:
                                                training_data["num_interactions"] = int(edge_index.shape[0] // 2)
                                        elif isinstance(edge_index, (list, tuple)) and len(edge_index) == 2:
                                            # Tuple of tensors/arrays
                                            if hasattr(edge_index[0], '__len__'):
                                                training_data["num_interactions"] = len(edge_index[0]) // 2
                                    except Exception:
                                        pass
                            
                            # Set num_training_samples to num_interactions if not set
                            if training_data.get("num_training_samples") is None and training_data.get("num_interactions") is not None:
                                training_data["num_training_samples"] = training_data["num_interactions"]
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Could not load model file info: {e}")
        
        # FINAL FALLBACK: Ensure critical values are NEVER None before returning
        # This is the last chance to set these values - they MUST have a value
        
        # Calculate num_interactions from matrix_data if still missing
        if training_data.get("num_interactions") is None:
            if "matrix_data" in training_data and isinstance(training_data["matrix_data"], dict):
                matrix_data = training_data["matrix_data"]
                if "shape" in matrix_data and "sparsity" in matrix_data:
                    shape = matrix_data["shape"]
                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                        total_possible = shape[0] * shape[1]
                        sparsity = matrix_data.get("sparsity", 0)
                        if sparsity < 1.0 and total_possible > 0:
                            training_data["num_interactions"] = int(total_possible * (1 - sparsity))
        
        # Ensure embedding_dim is NEVER None
        if training_data.get("embedding_dim") is None:
            # Default embedding dimension for LightGCN
            training_data["embedding_dim"] = 64
        
        # Ensure num_training_samples is NEVER None
        if training_data.get("num_training_samples") is None:
            # Use num_interactions if available (BPR samples = interactions for GNN)
            if training_data.get("num_interactions") is not None:
                training_data["num_training_samples"] = training_data["num_interactions"]
            # Otherwise try to calculate from matrix_data
            elif "matrix_data" in training_data and isinstance(training_data["matrix_data"], dict):
                matrix_data = training_data["matrix_data"]
                if "shape" in matrix_data and "sparsity" in matrix_data:
                    shape = matrix_data["shape"]
                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                        total_possible = shape[0] * shape[1]
                        sparsity = matrix_data.get("sparsity", 0)
                        if sparsity < 1.0 and total_possible > 0:
                            training_data["num_training_samples"] = int(total_possible * (1 - sparsity))
            # Last resort: use num_users * num_products as estimate (very rough)
            if training_data.get("num_training_samples") is None:
                num_users = training_data.get("num_users")
                num_products = training_data.get("num_products")
                if num_users is not None and num_products is not None and num_users > 0 and num_products > 0:
                    # Rough estimate: assume 10% of possible interactions exist
                    training_data["num_training_samples"] = max(1, int(num_users * num_products * 0.1))
                else:
                    # Absolute last resort: set to 0 (better than None)
                    training_data["num_training_samples"] = 0
        
        return training_data

    def _get_task_status(self, task_id: str) -> dict:
        """Get the status of a training task."""
        try:
            task_status = get_task_status(task_id)
            
            if task_status is None:
                return {
                    "task_id": task_id,
                    "model": "gnn",
                    "status": "not_found",
                    "message": "Task not found",
                    "error": "Task ID does not exist or has been cleaned up",
                }
            
            response_data = {
                "task_id": task_id,
                "model": "gnn",
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
                    # Get training data similar to sync response
                    from apps.recommendations.common.storage import ArtifactStorage
                    try:
                        storage = ArtifactStorage("gnn")
                        stored = storage.load()
                        artifacts = stored.get("artifacts", {})
                        
                        response_data.update({
                            "message": "Training completed successfully",
                            "progress": 100,
                            "result": result,
                            "training_info": {
                                "trained_at": stored.get("trained_at"),
                                "model_name": stored.get("model", "gnn"),
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
                            # Extract num_users and num_products from matrix shape if available
                            if isinstance(matrix_data, dict) and "shape" in matrix_data:
                                shape = matrix_data["shape"]
                                if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                                    if "num_users" not in response_data or response_data["num_users"] is None:
                                        response_data["num_users"] = shape[0]
                                    if "num_products" not in response_data or response_data["num_products"] is None:
                                        response_data["num_products"] = shape[1]
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
                "model": "gnn",
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
                result = train_gnn_model(force_retrain=force_retrain)
                training_data = self._get_training_data(result, include_artifacts=True)
                return Response(training_data, status=status.HTTP_200_OK)
            except Exception as e:
                return Response(
                    {
                        "status": "training_failed",
                        "model": "gnn",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
            # Run asynchronously using background task manager
            try:
                # Import the actual function (not the Celery task)
                from apps.recommendations.gnn.models import engine
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
                    result = train_gnn_model(force_retrain=force_retrain)
                    training_data = self._get_training_data(result, include_artifacts=True)
                    training_data["note"] = "Ran synchronously (background task failed)"
                    return Response(training_data, status=status.HTTP_200_OK)
                except Exception as sync_error:
                    return Response(
                        {
                            "status": "training_failed",
                            "model": "gnn",
                            "error": str(sync_error),
                            "error_type": type(sync_error).__name__,
                            "note": "Both async and sync execution failed",
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )


class RecommendGNNView(APIView):
    serializer_class = GNNRecommendationSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    
    def get(self, request, *args, **kwargs):
        """Handle GET requests with query parameters."""
        import time
        from apps.recommendations.common.evaluation import calculate_evaluation_metrics
        from apps.users.models import UserInteraction
        
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
        
        # Get ground truth: products that user has interacted with (excluding current product)
        ground_truth = []
        try:
            # Try Django ORM first
            try:
                user_interactions = UserInteraction.objects.filter(
                    user_id=user_id
                ).exclude(
                    product_id=product_id
                ).select_related('product').order_by('-timestamp')[:50]
                
                if not user_interactions.exists():
                    # Try with integer conversion
                    try:
                        user_id_int = int(user_id)
                        user_interactions = UserInteraction.objects.filter(
                            user_id=user_id_int
                        ).exclude(
                            product_id=product_id
                        ).select_related('product').order_by('-timestamp')[:50]
                    except (ValueError, TypeError):
                        pass
            except Exception:
                user_interactions = UserInteraction.objects.none()
            
            # If Django ORM didn't find anything, try MongoDB
            if not user_interactions.exists():
                try:
                    from apps.users.mongo_models import UserInteraction as MongoInteraction
                    from bson import ObjectId
                    
                    # Try to convert user_id to ObjectId if it's a MongoDB ID
                    try:
                        user_obj_id = ObjectId(user_id) if len(user_id) == 24 else user_id
                        product_obj_id = ObjectId(product_id) if len(str(product_id)) == 24 else product_id
                    except:
                        user_obj_id = user_id
                        product_obj_id = product_id
                    
                    mongo_interactions = MongoInteraction.objects(
                        user_id=user_obj_id
                    ).exclude(
                        product_id=product_obj_id
                    ).order_by('-timestamp').limit(50)
                    
                    for interaction in mongo_interactions:
                        item = {"id": str(interaction.product_id)}
                        if hasattr(interaction, 'rating') and interaction.rating is not None:
                            item["rating"] = float(interaction.rating)
                        ground_truth.append(item)
                except Exception:
                    pass
            
            # Get product IDs and ratings from Django ORM results
            for interaction in user_interactions:
                item = {"id": str(interaction.product_id)}
                if interaction.rating is not None:
                    item["rating"] = float(interaction.rating)
                ground_truth.append(item)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not fetch ground truth for user {user_id}: {e}")
            ground_truth = []
        
        # If still no ground truth, try to get from user's interaction_history (fallback)
        if not ground_truth:
            try:
                from apps.users.mongo_models import User as MongoUser
                from bson import ObjectId
                
                # Try to get user by ID
                try:
                    user_obj_id = ObjectId(user_id) if len(user_id) == 24 else user_id
                except:
                    user_obj_id = user_id
                
                mongo_user = MongoUser.objects(id=user_obj_id).first()
                if mongo_user and hasattr(mongo_user, 'interaction_history') and mongo_user.interaction_history:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Using interaction_history from user profile for user {user_id}")
                    
                    # Convert interaction_history to ground_truth format
                    for interaction in mongo_user.interaction_history:
                        # Skip current product
                        if str(interaction.get('product_id')) == str(product_id):
                            continue
                        
                        item = {"id": str(interaction.get('product_id'))}
                        
                        # Try to get rating if available (might be in rating field or inferred from interaction_type)
                        rating = interaction.get('rating')
                        if rating is not None:
                            item["rating"] = float(rating)
                        else:
                            # Infer rating from interaction_type if no explicit rating
                            interaction_type = interaction.get('interaction_type', '').lower()
                            if interaction_type == 'purchase':
                                item["rating"] = 5.0
                            elif interaction_type == 'like':
                                item["rating"] = 4.0
                            elif interaction_type == 'cart':
                                item["rating"] = 3.0
                            elif interaction_type == 'view':
                                item["rating"] = 2.0
                        
                        ground_truth.append(item)
                        
                        # Limit to 50 most recent
                        if len(ground_truth) >= 50:
                            break
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not fetch interaction_history from user profile: {e}")
        
        # Convert to None if empty for backward compatibility
        if not ground_truth:
            ground_truth = None
        else:
            # Log successful ground truth fetch
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Found {len(ground_truth)} ground truth items for user {user_id}, "
                f"product {product_id}"
            )
        
        # Measure execution time
        start_time = time.time()
        try:
            payload = recommend_gnn(
                user_id=user_id,
                current_product_id=product_id,
                top_k_personal=top_k_personal,
                top_k_outfit=top_k_outfit,
                request_params=dict(request.query_params),
            )
            execution_time = time.time() - start_time
            
            # Calculate evaluation metrics
            personalized_recommendations = payload.get("personalized", [])
            
            # Debug logging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Calculating metrics: "
                f"recommendations={len(personalized_recommendations)}, "
                f"ground_truth={len(ground_truth) if ground_truth else 0}, "
                f"user_id={user_id}"
            )
            
            metrics = calculate_evaluation_metrics(
                recommendations=personalized_recommendations,
                ground_truth=ground_truth,
                execution_time=execution_time,
            )
            metrics["model"] = "gnn"
            
            # Add debug info to help diagnose issues
            if metrics.get("_debug"):
                logger.info(f"Evaluation metrics debug: {metrics['_debug']}")
            
            # Add metrics to response
            payload["evaluation_metrics"] = metrics
            
        except ModelNotTrainedError as exc:
            return Response(
                {"detail": str(exc), "model": "gnn"},
                status=status.HTTP_409_CONFLICT,
            )
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

    def post(self, request, *args, **kwargs):
        import time
        from apps.recommendations.common.evaluation import calculate_evaluation_metrics
        from apps.users.models import UserInteraction
        
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user_id = serializer.validated_data["user_id"]
        current_product_id = serializer.validated_data["current_product_id"]
        
        # Get ground truth: products that user has interacted with (excluding current product)
        ground_truth = []
        try:
            # Try Django ORM first
            try:
                user_interactions = UserInteraction.objects.filter(
                    user_id=user_id
                ).exclude(
                    product_id=current_product_id
                ).select_related('product').order_by('-timestamp')[:50]
                
                if not user_interactions.exists():
                    # Try with integer conversion
                    try:
                        user_id_int = int(user_id)
                        user_interactions = UserInteraction.objects.filter(
                            user_id=user_id_int
                        ).exclude(
                            product_id=current_product_id
                        ).select_related('product').order_by('-timestamp')[:50]
                    except (ValueError, TypeError):
                        pass
            except Exception:
                user_interactions = UserInteraction.objects.none()
            
            # If Django ORM didn't find anything, try MongoDB
            if not user_interactions.exists():
                try:
                    from apps.users.mongo_models import UserInteraction as MongoInteraction
                    from bson import ObjectId
                    
                    # Try to convert user_id to ObjectId if it's a MongoDB ID
                    try:
                        user_obj_id = ObjectId(user_id) if len(user_id) == 24 else user_id
                        product_obj_id = ObjectId(current_product_id) if len(str(current_product_id)) == 24 else current_product_id
                    except:
                        user_obj_id = user_id
                        product_obj_id = current_product_id
                    
                    mongo_interactions = MongoInteraction.objects(
                        user_id=user_obj_id
                    ).exclude(
                        product_id=product_obj_id
                    ).order_by('-timestamp').limit(50)
                    
                    for interaction in mongo_interactions:
                        item = {"id": str(interaction.product_id)}
                        if hasattr(interaction, 'rating') and interaction.rating is not None:
                            item["rating"] = float(interaction.rating)
                        ground_truth.append(item)
                except Exception:
                    pass
            
            # Get product IDs and ratings from Django ORM results
            for interaction in user_interactions:
                item = {"id": str(interaction.product_id)}
                if interaction.rating is not None:
                    item["rating"] = float(interaction.rating)
                ground_truth.append(item)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not fetch ground truth for user {user_id}: {e}")
            ground_truth = []
        
        # If still no ground truth, try to get from user's interaction_history (fallback)
        if not ground_truth:
            try:
                from apps.users.mongo_models import User as MongoUser
                from bson import ObjectId
                
                # Try to get user by ID
                try:
                    user_obj_id = ObjectId(user_id) if len(user_id) == 24 else user_id
                except:
                    user_obj_id = user_id
                
                mongo_user = MongoUser.objects(id=user_obj_id).first()
                if mongo_user and hasattr(mongo_user, 'interaction_history') and mongo_user.interaction_history:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Using interaction_history from user profile for user {user_id}")
                    
                    # Convert interaction_history to ground_truth format
                    for interaction in mongo_user.interaction_history:
                        # Skip current product
                        if str(interaction.get('product_id')) == str(current_product_id):
                            continue
                        
                        item = {"id": str(interaction.get('product_id'))}
                        
                        # Try to get rating if available (might be in rating field or inferred from interaction_type)
                        rating = interaction.get('rating')
                        if rating is not None:
                            item["rating"] = float(rating)
                        else:
                            # Infer rating from interaction_type if no explicit rating
                            interaction_type = interaction.get('interaction_type', '').lower()
                            if interaction_type == 'purchase':
                                item["rating"] = 5.0
                            elif interaction_type == 'like':
                                item["rating"] = 4.0
                            elif interaction_type == 'cart':
                                item["rating"] = 3.0
                            elif interaction_type == 'view':
                                item["rating"] = 2.0
                        
                        ground_truth.append(item)
                        
                        # Limit to 50 most recent
                        if len(ground_truth) >= 50:
                            break
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not fetch interaction_history from user profile: {e}")
        
        # Convert to None if empty for backward compatibility
        if not ground_truth:
            ground_truth = None
        else:
            # Log successful ground truth fetch
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Found {len(ground_truth)} ground truth items for user {user_id}, "
                f"product {current_product_id}"
            )
        
        # Measure execution time
        start_time = time.time()
        try:
            payload = recommend_gnn(
                user_id=user_id,
                current_product_id=current_product_id,
                top_k_personal=serializer.validated_data["top_k_personal"],
                top_k_outfit=serializer.validated_data["top_k_outfit"],
                request_params=serializer.validated_data,
            )
            execution_time = time.time() - start_time
            
            # Calculate evaluation metrics
            personalized_recommendations = payload.get("personalized", [])
            
            # Debug logging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Calculating metrics: "
                f"recommendations={len(personalized_recommendations)}, "
                f"ground_truth={len(ground_truth) if ground_truth else 0}, "
                f"user_id={user_id}"
            )
            
            metrics = calculate_evaluation_metrics(
                recommendations=personalized_recommendations,
                ground_truth=ground_truth,
                execution_time=execution_time,
            )
            metrics["model"] = "gnn"
            
            # Add debug info to help diagnose issues
            if metrics.get("_debug"):
                logger.info(f"Evaluation metrics debug: {metrics['_debug']}")
            
            # Add metrics to response
            payload["evaluation_metrics"] = metrics
            
        except ModelNotTrainedError as exc:
            return Response(
                {"detail": str(exc), "model": "gnn"},
                status=status.HTTP_409_CONFLICT,
            )
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

