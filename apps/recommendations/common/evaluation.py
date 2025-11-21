"""Evaluation metrics for recommendation models."""

from __future__ import annotations

import math
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
        execution_time: Optional execution time in seconds (inference time per user)
        
    Returns:
        Dictionary containing evaluation metrics:
        - recall_at_10: Recall@10 metric
        - recall_at_20: Recall@20 metric
        - ndcg_at_10: NDCG@10 metric
        - ndcg_at_20: NDCG@20 metric
        - inference_time: Inference time per user in milliseconds
    """
    import logging
    logger = logging.getLogger(__name__)
    
    metrics = {
        "model": None,  # Will be set by caller
        "recall_at_10": 0.0,
        "recall_at_20": 0.0,
        "ndcg_at_10": 0.0,
        "ndcg_at_20": 0.0,
        "inference_time": round(execution_time * 1000, 2) if execution_time is not None else None,  # Convert to ms
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
        
        # Calculate Recall@K and NDCG@K metrics
        if recommended_ids and ground_truth_ids:
            # Normalize IDs to sets for comparison (handle string vs int mismatch)
            rec_ids_set = set()
            gt_ids_set = set()
            
            # Convert recommended IDs to normalized set
            for id in recommended_ids:
                if id and str(id).strip():
                    try:
                        # Try int conversion first
                        rec_ids_set.add(int(str(id).strip()))
                    except (ValueError, TypeError):
                        # Keep as string if not numeric
                        rec_ids_set.add(str(id).strip())
            
            # Convert ground truth IDs to normalized set
            for id in ground_truth_ids:
                if id and str(id).strip():
                    try:
                        # Try int conversion first
                        gt_ids_set.add(int(str(id).strip()))
                    except (ValueError, TypeError):
                        # Keep as string if not numeric
                        gt_ids_set.add(str(id).strip())
            
            # Find overlap
            overlap = rec_ids_set & gt_ids_set
            
            if len(overlap) > 0:
                logger.info(f"Found {len(overlap)} overlapping items between recommendations and ground truth")
                metrics["_debug"]["overlap_found"] = True
                metrics["_debug"]["overlap_count"] = len(overlap)
            else:
                logger.warning(
                    f"No overlap between recommendations ({len(recommended_ids)} items) "
                    f"and ground truth ({len(ground_truth_ids)} items). "
                    f"Recommended IDs sample: {recommended_ids[:5] if recommended_ids else []}, "
                    f"Ground truth IDs sample: {ground_truth_ids[:5] if ground_truth_ids else []}"
                )
                metrics["_debug"]["overlap_found"] = False
                metrics["_debug"]["sample_recommended_ids"] = recommended_ids[:5] if recommended_ids else []
                metrics["_debug"]["sample_ground_truth_ids"] = ground_truth_ids[:5] if ground_truth_ids else []
            
            # Calculate Recall@K
            # Recall@K = (number of relevant items in top K) / (total number of relevant items)
            if len(gt_ids_set) > 0:
                # Get top K recommendations (first K items)
                top_10_ids = set()
                top_20_ids = set()
                
                # Normalize recommended_ids to same format as sets
                for i, id in enumerate(recommended_ids):
                    normalized_id = None
                    if id and str(id).strip():
                        try:
                            normalized_id = int(str(id).strip())
                        except (ValueError, TypeError):
                            normalized_id = str(id).strip()
                    
                    if normalized_id is not None:
                        if i < 10:
                            top_10_ids.add(normalized_id)
                        if i < 20:
                            top_20_ids.add(normalized_id)
                
                # Calculate Recall@10
                relevant_at_10 = len(top_10_ids & gt_ids_set)
                recall_at_10 = relevant_at_10 / len(gt_ids_set) if len(gt_ids_set) > 0 else 0.0
                metrics["recall_at_10"] = round(recall_at_10, 4)
                
                # Calculate Recall@20
                relevant_at_20 = len(top_20_ids & gt_ids_set)
                recall_at_20 = relevant_at_20 / len(gt_ids_set) if len(gt_ids_set) > 0 else 0.0
                metrics["recall_at_20"] = round(recall_at_20, 4)
                
                # Calculate NDCG@K
                # NDCG@K = DCG@K / IDCG@K
                # DCG@K = sum(rel_i / log2(i+1)) for i in [1, K]
                # IDCG@K = DCG of ideal ranking (all relevant items ranked first)
                
                def calculate_dcg(relevance_list: list[float], k: int) -> float:
                    """Calculate DCG@K given a list of relevance scores."""
                    dcg = 0.0
                    for i in range(min(len(relevance_list), k)):
                        rel = relevance_list[i]
                        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0, we want log2(2) = 1
                    return dcg
                
                # Build mapping from normalized IDs to original string keys for ratings lookup
                # ground_truth_ratings uses string keys from ground_truth_ids
                normalized_to_string = {}
                for orig_id in ground_truth_ids:
                    if orig_id and str(orig_id).strip():
                        try:
                            normalized_id = int(str(orig_id).strip())
                            normalized_to_string[normalized_id] = str(orig_id).strip()
                        except (ValueError, TypeError):
                            normalized_id = str(orig_id).strip()
                            normalized_to_string[normalized_id] = str(orig_id).strip()
                
                # Build relevance lists for top 10 and top 20
                # Relevance = 1 if item is in ground truth, 0 otherwise
                # Use ratings if available, otherwise binary relevance
                relevance_list_10 = []
                relevance_list_20 = []
                
                for i, id in enumerate(recommended_ids):
                    normalized_id = None
                    if id and str(id).strip():
                        try:
                            normalized_id = int(str(id).strip())
                        except (ValueError, TypeError):
                            normalized_id = str(id).strip()
                    
                    if normalized_id is not None:
                        # Check if in ground truth
                        if normalized_id in gt_ids_set:
                            # Use rating if available, otherwise 1.0
                            # Try to get rating using original string key
                            string_key = normalized_to_string.get(normalized_id, str(normalized_id))
                            if string_key in ground_truth_ratings:
                                rel = float(ground_truth_ratings[string_key])
                                # Normalize rating to 0-1 scale (assuming max rating is 5.0)
                                rel = min(rel / 5.0, 1.0)
                            else:
                                rel = 1.0  # Binary relevance
                        else:
                            rel = 0.0
                        
                        if i < 10:
                            relevance_list_10.append(rel)
                        if i < 20:
                            relevance_list_20.append(rel)
                
                # Calculate DCG@10 and DCG@20
                dcg_at_10 = calculate_dcg(relevance_list_10, 10)
                dcg_at_20 = calculate_dcg(relevance_list_20, 20)
                
                # Calculate IDCG@K (ideal DCG - all relevant items ranked first)
                # Get all relevance scores from ground truth, sort descending
                ideal_relevance = []
                for normalized_id in gt_ids_set:
                    string_key = normalized_to_string.get(normalized_id, str(normalized_id))
                    if string_key in ground_truth_ratings:
                        rel = float(ground_truth_ratings[string_key])
                        rel = min(rel / 5.0, 1.0)
                    else:
                        rel = 1.0  # Binary relevance
                    ideal_relevance.append(rel)
                
                ideal_relevance.sort(reverse=True)  # Sort descending
                
                idcg_at_10 = calculate_dcg(ideal_relevance, 10)
                idcg_at_20 = calculate_dcg(ideal_relevance, 20)
                
                # Calculate NDCG@10 and NDCG@20
                ndcg_at_10 = dcg_at_10 / idcg_at_10 if idcg_at_10 > 0 else 0.0
                ndcg_at_20 = dcg_at_20 / idcg_at_20 if idcg_at_20 > 0 else 0.0
                
                metrics["ndcg_at_10"] = round(ndcg_at_10, 4)
                metrics["ndcg_at_20"] = round(ndcg_at_20, 4)
            else:
                logger.warning(
                    f"Cannot calculate Recall/NDCG: "
                    f"recommended_ids={len(recommended_ids)}, ground_truth_ids={len(ground_truth_ids)}"
                )
    else:
        logger.info(
            f"No ground truth provided. Metrics will be 0.0. "
            f"Recommendations: {len(recommendations)}, "
            f"Recommended IDs: {len(recommended_ids)}, "
            f"Scores: {len(recommended_scores)}"
        )
    
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

