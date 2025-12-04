from __future__ import annotations

import logging
from typing import Any

from celery import shared_task

from apps.recommendations.common import BaseRecommendationEngine
from apps.recommendations.common.context import RecommendationContext

logger = logging.getLogger(__name__)


class HybridRecommendationEngine(BaseRecommendationEngine):
    model_name = "hybrid"

    def _train_impl(self) -> dict[str, Any]:
        """
        Train hybrid model by ensuring both CBF and GNN models are trained.
        The hybrid model combines predictions from both models.
        """
        logger.info(f"[{self.model_name}] Hybrid training entrypoint disabled (CBF/GNN engines removed).")
        return {
            "note": "Hybrid model training is disabled because standalone CBF/GNN engines have been removed."
        }

    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        """
        Deprecated: legacy hybrid engine scoring depended on CBF/GNN engines.
        The HTTP API `/api/v1/hybrid/recommend/` now contains the authoritative logic.
        """
        logger.warning("[%s] _score_candidates is deprecated; use HTTP hybrid API instead.", self.model_name)
        return {}


engine = HybridRecommendationEngine()


@shared_task
def train_hybrid_model(force_retrain: bool = False) -> dict[str, Any]:
    return engine.train(force_retrain=force_retrain)


def recommend_hybrid(
    *,
    user_id: str | int,
    current_product_id: str | int,
    top_k_personal: int,
    top_k_outfit: int,
    request_params: dict | None = None,
) -> dict[str, Any]:
    """
    Recommend using hybrid model.
    Note: The HTTP API `/api/v1/hybrid/recommend/` now contains the
    authoritative recommendation logic; this function delegates to the
    generic engine-based flow for CLI/legacy use.
    """
    from apps.recommendations.common.filters import CandidateFilter

    context = CandidateFilter.build_context(
        user_id=user_id,
        current_product_id=current_product_id,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
        request_params=request_params,
    )
    payload = engine.recommend(context)
    return payload.as_dict()

