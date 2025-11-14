"""GNN-inspired recommendation engine leveraging interaction graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from celery import shared_task

from apps.recommendations.common import BaseRecommendationEngine, CandidateFilter
from apps.recommendations.common.constants import INTERACTION_WEIGHTS
from apps.recommendations.common.context import RecommendationContext
from apps.users.models import UserInteraction


class GNNRecommendationEngine(BaseRecommendationEngine):
    model_name = "gnn"

    def _train_impl(self) -> dict[str, Any]:
        co_occurrence: dict[int, dict[int, float]] = defaultdict(dict)
        product_frequency: dict[int, float] = defaultdict(float)

        interactions = (
            UserInteraction.objects.select_related("product")
            .order_by("user_id", "timestamp")
        )

        user_histories: dict[int, list[tuple[int, float]]] = defaultdict(list)

        for interaction in interactions:
            if not interaction.product_id:
                continue
            weight = INTERACTION_WEIGHTS.get(interaction.interaction_type, 1.0)
            user_history = user_histories[interaction.user_id]
            for other_product_id, other_weight in user_history:
                total = weight + other_weight
                co_occurrence[interaction.product_id][other_product_id] = (
                    co_occurrence[interaction.product_id].get(other_product_id, 0.0) + total
                )
                co_occurrence[other_product_id][interaction.product_id] = (
                    co_occurrence[other_product_id].get(interaction.product_id, 0.0) + total
                )
            user_history.append((interaction.product_id, weight))
            product_frequency[interaction.product_id] += weight

        serialized_graph = {
            product_id: dict(neighbours) for product_id, neighbours in co_occurrence.items()
        }
        
        # Create matrix data for display (co-occurrence matrix)
        all_product_ids = sorted(set(list(serialized_graph.keys()) + list(product_frequency.keys())))
        # Show approximately 5 rows for display
        max_display = min(5, len(all_product_ids))
        display_product_ids = all_product_ids[:max_display].copy()
        
        # Build matrix data
        matrix_data_list = []
        for i, product_id_i in enumerate(display_product_ids):
            row = []
            for j, product_id_j in enumerate(display_product_ids):
                if product_id_i == product_id_j:
                    row.append(0.0)  # Self-co-occurrence is 0
                else:
                    # Get co-occurrence value (symmetric)
                    value = serialized_graph.get(product_id_i, {}).get(product_id_j, 0.0)
                    row.append(float(value))
            matrix_data_list.append(row)
        
        # If we have fewer products than desired, pad with empty rows for better visualization
        display_rows = 5
        if len(all_product_ids) < display_rows:
            # Pad matrix with zero rows and columns
            while len(matrix_data_list) < display_rows:
                # Add a new row with zeros
                matrix_data_list.append([0.0] * len(matrix_data_list[0]) if matrix_data_list else [0.0] * display_rows)
                # Add a new column to all existing rows
                for row in matrix_data_list[:-1]:
                    row.append(0.0)
                # Use placeholder product IDs (negative numbers to indicate they're not real)
                display_product_ids.append(-(len(matrix_data_list)))
        
        matrix_data = {
            "shape": [len(all_product_ids), len(all_product_ids)],
            "display_shape": [len(matrix_data_list), len(matrix_data_list[0]) if matrix_data_list else 0],
            "data": matrix_data_list[:display_rows],
            "product_ids": display_product_ids[:display_rows],
            "description": "Product Co-occurrence Matrix",
            "row_label": "Product ID",
            "col_label": "Product ID",
            "value_description": "Co-occurrence weight (0 = no co-occurrence, >0 = co-occurrence strength)",
        }
        
        return {
            "co_occurrence": serialized_graph,
            "product_frequency": dict(product_frequency),
            "matrix_data": matrix_data,
        }

    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        graph: dict[int, dict[int, float]] = artifacts.get("co_occurrence", {})
        frequency: dict[int, float] = artifacts.get("product_frequency", {})
        current_neighbors = graph.get(context.current_product.id, {}) if context.current_product.id else {}

        history_ids = list(context.iter_history_ids())
        candidate_scores: dict[int, float] = {}

        for candidate in context.candidate_products:
            candidate_id = candidate.id
            if candidate_id is None:
                continue
            score = 0.0
            candidate_neighbors = graph.get(candidate_id, {})
            for history_product_id in history_ids:
                score += candidate_neighbors.get(history_product_id, 0.0)
            score += current_neighbors.get(candidate_id, 0.0) * 1.2
            score += sum(context.style_weight(token) for token in _style_tokens(candidate))
            score += 0.1 * frequency.get(candidate_id, 1.0)
            score += 0.2 * context.brand_weight(candidate.brand_id)
            candidate_scores[candidate_id] = score

        if not candidate_scores:
            # fallback to style-based scoring
            for candidate in context.candidate_products:
                if candidate.id is None:
                    continue
                candidate_scores[candidate.id] = sum(context.style_weight(token) for token in _style_tokens(candidate))
        return candidate_scores


def _style_tokens(product) -> Iterable[str]:
    tokens = []
    if isinstance(getattr(product, "style_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.style_tags if tag)
    if isinstance(getattr(product, "outfit_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.outfit_tags if tag)
    if getattr(product, "category_type", None):
        tokens.append(product.category_type.lower())
    return tokens


engine = GNNRecommendationEngine()


@shared_task
def train_gnn_model(force_retrain: bool = False) -> dict[str, Any]:
    return engine.train(force_retrain=force_retrain)


def recommend_gnn(
    *,
    user_id: str | int,
    current_product_id: str | int,
    top_k_personal: int,
    top_k_outfit: int,
    request_params: dict | None = None,
) -> dict[str, Any]:
    context = CandidateFilter.build_context(
        user_id=user_id,
        current_product_id=current_product_id,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
        request_params=request_params,
    )
    payload = engine.recommend(context)
    return payload.as_dict()

