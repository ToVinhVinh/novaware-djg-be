"""Evaluation metrics for recommendation models."""

from __future__ import annotations

import time
from typing import Any


def calculate_evaluation_metrics(
    recommendations: list[Any],
    ground_truth: list[Any] | None = None,
    execution_time: float | None = None,
) -> dict[str, Any]:
    """
    Calculate evaluation metrics for recommendations.
    
    Args:
        recommendations: List of recommended items with scores (from payload.as_dict())
        ground_truth: Optional ground truth items for comparison
        execution_time: Optional execution time in seconds
        
    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = {
        "model": None,  # Will be set by caller
        "mape": None,
        "rmse": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "time": round(execution_time, 4) if execution_time is not None else None,
    }
    
    recommended_ids = []
    for rec in recommendations:
        if isinstance(rec, dict):
            product = rec.get("product", {})
            if isinstance(product, dict):
                rec_id = product.get("id")
                if rec_id is not None:
                    recommended_ids.append(str(rec_id))
        elif hasattr(rec, "product"):
            product = rec.product
            if hasattr(product, "id"):
                rec_id = product.id
                if rec_id is not None:
                    recommended_ids.append(str(rec_id))
    
    # If we have ground truth, calculate metrics
    if ground_truth is not None and len(ground_truth) > 0:
        # Extract ground truth IDs
        if isinstance(ground_truth[0], dict):
            ground_truth_ids = [str(item.get("id")) for item in ground_truth if item.get("id")]
        else:
            ground_truth_ids = [str(getattr(item, "id", None)) for item in ground_truth if hasattr(item, "id") and getattr(item, "id") is not None]
        
        # Calculate precision, recall, F1
        if recommended_ids and ground_truth_ids:
            true_positives = len(set(recommended_ids) & set(ground_truth_ids))
            precision = true_positives / len(recommended_ids) if recommended_ids else 0.0
            recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics["precision"] = round(precision, 4)
            metrics["recall"] = round(recall, 4)
            metrics["f1"] = round(f1, 4)
    
    # For MAPE and RMSE, we would need actual ratings/predictions
    # These require ground truth ratings and predicted ratings
    # For now, we'll set them to None as they need rating data
    
    return metrics


def format_time(seconds: float | None) -> str | None:
    """Format time in seconds to a readable string."""
    if seconds is None:
        return None
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"

