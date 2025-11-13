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
        return {
            "co_occurrence": serialized_graph,
            "product_frequency": dict(product_frequency),
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

