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
    import logging
    logger = logging.getLogger(__name__)
    
    metrics = {
        "model": None,  # Will be set by caller
        "mape": None,
        "rmse": None,
        "precision": 0.0,  # Default to 0.0 instead of None
        "recall": 0.0,
        "f1": 0.0,
        "time": round(execution_time, 4) if execution_time is not None else None,
        "_debug": {  # Debug info to help diagnose issues
            "num_recommendations": len(recommendations) if recommendations else 0,
            "num_ground_truth": len(ground_truth) if ground_truth else 0,
            "has_scores": False,
            "has_ratings": False,
        }
    }
    
    # Extract recommended product IDs and scores
    recommended_ids = []
    recommended_scores = {}  # product_id -> score mapping
    
    for rec in recommendations:
        rec_id = None
        score = None
        
        if isinstance(rec, dict):
            product = rec.get("product", {})
            if isinstance(product, dict):
                rec_id = product.get("id")
            score = rec.get("score")
        elif hasattr(rec, "product"):
            product = rec.product
            if hasattr(product, "id"):
                rec_id = product.id
            if hasattr(rec, "score"):
                score = rec.score
        
        if rec_id is not None:
            recommended_ids.append(str(rec_id))
            if score is not None:
                recommended_scores[str(rec_id)] = float(score)
                metrics["_debug"]["has_scores"] = True
    
    # Calculate basic metrics even without ground truth (based on scores)
    if recommended_scores:
        # Calculate score statistics as fallback metrics
        scores_list = list(recommended_scores.values())
        if scores_list:
            # Normalize scores to 0-5 range for display
            max_score = max(scores_list) if scores_list else 1.0
            min_score = min(scores_list) if scores_list else 0.0
            
            # If we have scores but no ground truth, we can still provide some metrics
            # For example, score distribution or confidence metrics
            pass  # Placeholder for future score-based metrics
    
    # Update debug info
    metrics["_debug"]["num_recommended_ids"] = len(recommended_ids)
    metrics["_debug"]["num_recommended_scores"] = len(recommended_scores)
    
    # Log what we have
    logger.info(
        f"Evaluation input: recommendations={len(recommendations)}, "
        f"recommended_ids={len(recommended_ids)}, "
        f"ground_truth={len(ground_truth) if ground_truth else 0}"
    )
    
    # If we have ground truth, calculate metrics
    if ground_truth is not None and len(ground_truth) > 0:
        # Extract ground truth IDs and ratings
        ground_truth_ids = []
        ground_truth_ratings = {}  # product_id -> rating mapping
        
        if isinstance(ground_truth[0], dict):
            for item in ground_truth:
                item_id = item.get("id")
                if item_id:
                    ground_truth_ids.append(str(item_id))
                    # Try to get rating if available
                    rating = item.get("rating") or item.get("score")
                    if rating is not None:
                        ground_truth_ratings[str(item_id)] = float(rating)
                        metrics["_debug"]["has_ratings"] = True
        else:
            for item in ground_truth:
                if hasattr(item, "id") and getattr(item, "id") is not None:
                    item_id = str(getattr(item, "id"))
                    ground_truth_ids.append(item_id)
                    # Try to get rating if available
                    if hasattr(item, "rating"):
                        rating = getattr(item, "rating")
                        if rating is not None:
                            ground_truth_ratings[item_id] = float(rating)
                            metrics["_debug"]["has_ratings"] = True
        
        # Update debug info
        metrics["_debug"]["num_ground_truth_ids"] = len(ground_truth_ids)
        metrics["_debug"]["num_ground_truth_ratings"] = len(ground_truth_ratings)
        
        logger.info(
            f"Ground truth extracted: ids={len(ground_truth_ids)}, "
            f"ratings={len(ground_truth_ratings)}, "
            f"sample_ids={ground_truth_ids[:5] if ground_truth_ids else []}"
        )
        
        # Calculate precision, recall, F1
        if recommended_ids and ground_truth_ids:
            # Try exact match first
            true_positives_set = set(recommended_ids) & set(ground_truth_ids)
            true_positives = len(true_positives_set)
            
            # If no exact match, try converting to int for comparison (handle string vs int mismatch)
            if true_positives == 0:
                try:
                    # Convert all to int for comparison
                    rec_ids_int = {}
                    for id in recommended_ids:
                        if id and str(id).strip():
                            try:
                                int_id = int(str(id).strip())
                                rec_ids_int[int_id] = str(id)
                            except (ValueError, TypeError):
                                pass
                    
                    gt_ids_int = {}
                    for id in ground_truth_ids:
                        if id and str(id).strip():
                            try:
                                int_id = int(str(id).strip())
                                gt_ids_int[int_id] = str(id)
                            except (ValueError, TypeError):
                                pass
                    
                    # Find overlap in int space
                    common_int_ids = set(rec_ids_int.keys()) & set(gt_ids_int.keys())
                    true_positives = len(common_int_ids)
                    
                    if true_positives > 0:
                        logger.info(f"Found overlap after converting IDs to int: {true_positives} matches")
                        # Update true_positives_set with string versions for later use
                        true_positives_set = {rec_ids_int[int_id] for int_id in common_int_ids if int_id in rec_ids_int}
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert IDs to int for comparison: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error in ID conversion: {e}")
            
            precision = true_positives / len(recommended_ids) if recommended_ids else 0.0
            recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics["precision"] = round(precision, 4)
            metrics["recall"] = round(recall, 4)
            metrics["f1"] = round(f1, 4)
            
            # Debug: log why metrics might be 0
            if true_positives == 0:
                logger.warning(
                    f"No overlap between recommendations ({len(recommended_ids)} items) "
                    f"and ground truth ({len(ground_truth_ids)} items). "
                    f"Recommended IDs sample: {recommended_ids[:5] if recommended_ids else []}, "
                    f"Ground truth IDs sample: {ground_truth_ids[:5] if ground_truth_ids else []}"
                )
                # Add to debug info
                metrics["_debug"]["overlap_attempted"] = True
                metrics["_debug"]["overlap_found"] = False
                metrics["_debug"]["sample_recommended_ids"] = recommended_ids[:5] if recommended_ids else []
                metrics["_debug"]["sample_ground_truth_ids"] = ground_truth_ids[:5] if ground_truth_ids else []
                
                # Additional debug: check ID types
                if recommended_ids and ground_truth_ids:
                    rec_types = [type(id).__name__ for id in recommended_ids[:3]]
                    gt_types = [type(id).__name__ for id in ground_truth_ids[:3]]
                    metrics["_debug"]["recommended_id_types"] = rec_types
                    metrics["_debug"]["ground_truth_id_types"] = gt_types
                    
                    # Try to see if any IDs match when converted
                    rec_set = set(str(id).strip() for id in recommended_ids)
                    gt_set = set(str(id).strip() for id in ground_truth_ids)
                    string_overlap = rec_set & gt_set
                    metrics["_debug"]["string_overlap_count"] = len(string_overlap)
                    if string_overlap:
                        metrics["_debug"]["string_overlap_ids"] = list(string_overlap)[:5]
            else:
                metrics["_debug"]["overlap_found"] = True
                metrics["_debug"]["true_positives"] = true_positives
        else:
            logger.warning(
                f"Cannot calculate precision/recall: "
                f"recommended_ids={len(recommended_ids)}, ground_truth_ids={len(ground_truth_ids)}"
            )
        
        # Calculate MAPE and RMSE if we have both predicted scores and actual ratings
        if recommended_scores and ground_truth_ratings:
            # Find common products - try exact match first
            common_ids = set(recommended_scores.keys()) & set(ground_truth_ratings.keys())
            
            # Create mappings for int-based comparison
            rec_scores_by_int = {}
            gt_ratings_by_int = {}
            rec_int_to_str = {}  # Map int -> str for recommended
            gt_int_to_str = {}   # Map int -> str for ground truth
            
            # If no exact match, try converting to int
            if not common_ids:
                try:
                    # Create mapping: int_id -> (score/rating, str_id) for both
                    for k, v in recommended_scores.items():
                        if str(k).isdigit():
                            int_id = int(k)
                            rec_scores_by_int[int_id] = v
                            rec_int_to_str[int_id] = k
                    
                    for k, v in ground_truth_ratings.items():
                        if str(k).isdigit():
                            int_id = int(k)
                            gt_ratings_by_int[int_id] = v
                            gt_int_to_str[int_id] = k
                    
                    common_int_ids = set(rec_scores_by_int.keys()) & set(gt_ratings_by_int.keys())
                    
                    if common_int_ids:
                        logger.info(f"Found {len(common_int_ids)} common products after converting IDs to int for MAPE/RMSE")
                        # Use int IDs for processing - need to get scores/ratings using int keys
                        # Build mappings from int IDs back to original string keys
                        for int_id in common_int_ids:
                            if int_id in rec_scores_by_int and int_id in gt_ratings_by_int:
                                # We have both score and rating for this int_id
                                # Map back to string keys for common_ids
                                if int_id in rec_int_to_str:
                                    common_ids.add(rec_int_to_str[int_id])
                        common_ids_int = common_int_ids
                    else:
                        common_ids_int = set()
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert IDs to int for MAPE/RMSE: {e}")
                    common_ids_int = set()
            else:
                # We have exact matches, convert to int for processing
                try:
                    common_ids_int = {int(id) for id in common_ids if str(id).isdigit()}
                    # Build mappings
                    for k, v in recommended_scores.items():
                        if k in common_ids and str(k).isdigit():
                            int_id = int(k)
                            rec_scores_by_int[int_id] = v
                            rec_int_to_str[int_id] = k
                    for k, v in ground_truth_ratings.items():
                        if k in common_ids and str(k).isdigit():
                            int_id = int(k)
                            gt_ratings_by_int[int_id] = v
                            gt_int_to_str[int_id] = k
                except (ValueError, TypeError):
                    common_ids_int = set()
            
            if common_ids_int:
                predicted_values = []
                actual_values = []
                
                for product_id_int in common_ids_int:
                    pred_score = rec_scores_by_int[product_id_int]
                    actual_rating = gt_ratings_by_int[product_id_int]
                    
                    # Normalize scores to 0-5 range if needed (assuming max score is around 1.0)
                    # You may need to adjust this based on your actual score range
                    normalized_pred = min(max(pred_score * 5.0, 0.0), 5.0)  # Scale to 0-5
                    
                    predicted_values.append(normalized_pred)
                    actual_values.append(actual_rating)
                
                if predicted_values and actual_values:
                    # Calculate RMSE
                    squared_errors = [(p - a) ** 2 for p, a in zip(predicted_values, actual_values)]
                    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
                    metrics["rmse"] = round(rmse, 4)
                    
                    # Calculate MAPE (avoid division by zero)
                    absolute_percentage_errors = []
                    for p, a in zip(predicted_values, actual_values):
                        if a != 0:
                            ape = abs((p - a) / a) * 100
                            absolute_percentage_errors.append(ape)
                    
                    if absolute_percentage_errors:
                        mape = sum(absolute_percentage_errors) / len(absolute_percentage_errors)
                        metrics["mape"] = round(mape, 4)
            else:
                logger.warning(
                    f"Cannot calculate MAPE/RMSE: "
                    f"common_ids={len(common_ids)}, "
                    f"recommended_scores={len(recommended_scores)}, "
                    f"ground_truth_ratings={len(ground_truth_ratings)}"
                )
        else:
            logger.warning(
                f"Cannot calculate MAPE/RMSE: "
                f"recommended_scores={len(recommended_scores)}, "
                f"ground_truth_ratings={len(ground_truth_ratings)}"
            )
    else:
        logger.info(
            f"No ground truth provided. Metrics will be 0.0. "
            f"Recommendations: {len(recommendations)}, "
            f"Recommended IDs: {len(recommended_ids)}, "
            f"Scores: {len(recommended_scores)}"
        )
    
    # Fallback: If no ground truth but we have scores, calculate basic score-based metrics
    if (ground_truth is None or len(ground_truth) == 0) and recommended_scores:
        scores_list = list(recommended_scores.values())
        if scores_list:
            # Calculate score statistics
            avg_score = sum(scores_list) / len(scores_list)
            max_score = max(scores_list)
            min_score = min(scores_list)
            
            # Use score distribution as a proxy for model confidence
            # Higher variance = more diverse recommendations
            if len(scores_list) > 1:
                variance = sum((s - avg_score) ** 2 for s in scores_list) / len(scores_list)
                std_dev = variance ** 0.5
                
                # Normalize to 0-1 range for display
                # These are not true precision/recall but score-based metrics
                # We keep precision/recall/f1 as 0.0 when no ground truth
    
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

