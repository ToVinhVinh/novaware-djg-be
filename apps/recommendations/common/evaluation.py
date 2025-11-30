from __future__ import annotations

import math
import time
from typing import Any

def _normalize_id(id_value: Any) -> int | str:
    if id_value is None:
        return None
    id_str = str(id_value).strip()
    if not id_str:
        return None
    try:
        return int(id_str)
    except (ValueError, TypeError):
        return id_str

def calculate_evaluation_metrics(
    recommendations: list[Any],
    ground_truth: list[Any] | None = None,
    execution_time: float | None = None,
) -> dict[str, Any]:

    import logging
    logger = logging.getLogger(__name__)

    metrics = {
        "model": None,
        "recall_at_10": 0.0,
        "recall_at_20": 0.0,
        "ndcg_at_10": 0.0,
        "ndcg_at_20": 0.0,
        "inference_time": round(execution_time * 1000, 2) if execution_time is not None else None,
        "_debug": {
            "num_recommendations": len(recommendations) if recommendations else 0,
            "num_ground_truth": len(ground_truth) if ground_truth else 0,
            "has_scores": False,
            "has_ratings": False,
        }
    }

    recommended_ids = []
    recommended_scores = {}

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

    if recommended_scores:
        scores_list = list(recommended_scores.values())
        if scores_list:
            max_score = max(scores_list) if scores_list else 1.0
            min_score = min(scores_list) if scores_list else 0.0

            pass

    metrics["_debug"]["num_recommended_ids"] = len(recommended_ids)
    metrics["_debug"]["num_recommended_scores"] = len(recommended_scores)

    logger.info(
        f"Evaluation input: recommendations={len(recommendations)}, "
        f"recommended_ids={len(recommended_ids)}, "
        f"ground_truth={len(ground_truth) if ground_truth else 0}"
    )

    if ground_truth is not None and len(ground_truth) > 0:
        ground_truth_ids = []
        ground_truth_ratings = {}

        if isinstance(ground_truth[0], dict):
            for item in ground_truth:
                item_id = item.get("id")
                if item_id:
                    ground_truth_ids.append(str(item_id))
                    rating = item.get("rating") or item.get("score")
                    if rating is not None:
                        ground_truth_ratings[str(item_id)] = float(rating)
                        metrics["_debug"]["has_ratings"] = True
        else:
            for item in ground_truth:
                if hasattr(item, "id") and getattr(item, "id") is not None:
                    item_id = str(getattr(item, "id"))
                    ground_truth_ids.append(item_id)
                    if hasattr(item, "rating"):
                        rating = getattr(item, "rating")
                        if rating is not None:
                            ground_truth_ratings[item_id] = float(rating)
                            metrics["_debug"]["has_ratings"] = True

        metrics["_debug"]["num_ground_truth_ids"] = len(ground_truth_ids)
        metrics["_debug"]["num_ground_truth_ratings"] = len(ground_truth_ratings)

        logger.info(
            f"Ground truth extracted: ids={len(ground_truth_ids)}, "
            f"ratings={len(ground_truth_ratings)}, "
            f"sample_ids={ground_truth_ids[:5] if ground_truth_ids else []}"
        )

        if recommended_ids and ground_truth_ids:
            rec_ids_set = set()
            gt_ids_set = set()

            for id in recommended_ids:
                normalized = _normalize_id(id)
                if normalized is not None:
                    rec_ids_set.add(normalized)

            for id in ground_truth_ids:
                normalized = _normalize_id(id)
                if normalized is not None:
                    gt_ids_set.add(normalized)

            overlap = rec_ids_set & gt_ids_set

            logger.info(
                f"Normalized recommended IDs (first 5): {list(rec_ids_set)[:5] if rec_ids_set else []}, "
                f"Normalized ground truth IDs (first 5): {list(gt_ids_set)[:5] if gt_ids_set else []}"
            )

            if len(overlap) > 0:
                logger.info(f"Found {len(overlap)} overlapping items between recommendations and ground truth: {list(overlap)[:10]}")
                metrics["_debug"]["overlap_found"] = True
                metrics["_debug"]["overlap_count"] = len(overlap)
                metrics["_debug"]["overlap_ids"] = [str(x) for x in list(overlap)[:10]]
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
                metrics["_debug"]["normalized_rec_ids"] = [str(x) for x in list(rec_ids_set)[:5]] if rec_ids_set else []
                metrics["_debug"]["normalized_gt_ids"] = [str(x) for x in list(gt_ids_set)[:5]] if gt_ids_set else []

            if len(gt_ids_set) > 0:
                top_10_ids = set()
                top_20_ids = set()

                for i, id in enumerate(recommended_ids):
                    normalized_id = _normalize_id(id)
                    if normalized_id is not None:
                        if i < 10:
                            top_10_ids.add(normalized_id)
                        if i < 20:
                            top_20_ids.add(normalized_id)

                relevant_at_10 = len(top_10_ids & gt_ids_set)
                recall_at_10 = relevant_at_10 / len(gt_ids_set) if len(gt_ids_set) > 0 else 0.0
                metrics["recall_at_10"] = round(recall_at_10, 4)

                relevant_at_20 = len(top_20_ids & gt_ids_set)
                recall_at_20 = relevant_at_20 / len(gt_ids_set) if len(gt_ids_set) > 0 else 0.0
                metrics["recall_at_20"] = round(recall_at_20, 4)

                def calculate_dcg(relevance_list: list[float], k: int) -> float:
                    dcg = 0.0
                    for i in range(min(len(relevance_list), k)):
                        rel = relevance_list[i]
                        dcg += rel / math.log2(i + 2)
                    return dcg

                normalized_to_string = {}
                for orig_id in ground_truth_ids:
                    normalized_id = _normalize_id(orig_id)
                    if normalized_id is not None:
                        normalized_to_string[normalized_id] = str(orig_id).strip()

                relevance_list_10 = []
                relevance_list_20 = []

                for i, id in enumerate(recommended_ids):
                    normalized_id = _normalize_id(id)
                    if normalized_id is not None:
                        if normalized_id in gt_ids_set:
                            string_key = normalized_to_string.get(normalized_id, str(normalized_id))
                            if string_key in ground_truth_ratings:
                                rel = float(ground_truth_ratings[string_key])
                                rel = min(rel / 5.0, 1.0)
                            else:
                                rel = 1.0
                        else:
                            rel = 0.0

                        if i < 10:
                            relevance_list_10.append(rel)
                        if i < 20:
                            relevance_list_20.append(rel)

                dcg_at_10 = calculate_dcg(relevance_list_10, 10)
                dcg_at_20 = calculate_dcg(relevance_list_20, 20)

                ideal_relevance = []
                for normalized_id in gt_ids_set:
                    string_key = normalized_to_string.get(normalized_id)
                    if string_key and string_key in ground_truth_ratings:
                        rel = float(ground_truth_ratings[string_key])
                        rel = min(rel / 5.0, 1.0)
                    else:
                        rel = 1.0
                    ideal_relevance.append(rel)

                ideal_relevance.sort(reverse=True)

                idcg_at_10 = calculate_dcg(ideal_relevance, 10)
                idcg_at_20 = calculate_dcg(ideal_relevance, 20)

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

