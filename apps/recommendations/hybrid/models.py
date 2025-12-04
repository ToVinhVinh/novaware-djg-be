from __future__ import annotations

import logging
from typing import Any

from celery import shared_task

from apps.recommendations.common import BaseRecommendationEngine
from apps.recommendations.common.context import RecommendationContext
from apps.recommendations.cbf.models import engine as cbf_engine
from apps.recommendations.gnn.models import engine as gnn_engine

logger = logging.getLogger(__name__)


class HybridRecommendationEngine(BaseRecommendationEngine):
    model_name = "hybrid"

    def _train_impl(self) -> dict[str, Any]:
        """
        Train hybrid model by ensuring both CBF and GNN models are trained.
        The hybrid model combines predictions from both models.
        """
        logger.info(f"[{self.model_name}] Ensuring CBF and GNN models are trained...")
        
        # Ensure CBF is trained
        if not cbf_engine.storage.exists():
            logger.info(f"[{self.model_name}] Training CBF model first...")
            cbf_engine.train(force_retrain=False)
        
        # Ensure GNN is trained
        if not gnn_engine.storage.exists():
            logger.info(f"[{self.model_name}] Training GNN model first...")
            gnn_engine.train(force_retrain=False)
        
        logger.info(f"[{self.model_name}] Hybrid model ready (combines CBF and GNN)")
        
        return {
            "cbf_trained": cbf_engine.storage.exists(),
            "gnn_trained": gnn_engine.storage.exists(),
            "note": "Hybrid model combines CBF and GNN predictions"
        }

    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        """
        Score candidates using hybrid approach (combines CBF and GNN scores).
        This is a simplified version - the full implementation is in the API endpoint.
        """
        # Get CBF scores
        cbf_scores = {}
        if cbf_engine.storage.exists():
            try:
                cbf_context = RecommendationContext(
                    user_id=context.user_id,
                    current_product_id=context.current_product_id,
                    top_k_personal=context.top_k_personal,
                    top_k_outfit=context.top_k_outfit,
                    request_params=context.request_params,
                )
                cbf_artifacts = cbf_engine.storage.load().get("artifacts", {})
                cbf_scores = cbf_engine._score_candidates(cbf_context, cbf_artifacts)
            except Exception as e:
                logger.warning(f"[{self.model_name}] Could not get CBF scores: {e}")
        
        # Get GNN scores
        gnn_scores = {}
        if gnn_engine.storage.exists():
            try:
                gnn_context = RecommendationContext(
                    user_id=context.user_id,
                    current_product_id=context.current_product_id,
                    top_k_personal=context.top_k_personal,
                    top_k_outfit=context.top_k_outfit,
                    request_params=context.request_params,
                )
                gnn_artifacts = gnn_engine.storage.load().get("artifacts", {})
                gnn_scores = gnn_engine._score_candidates(gnn_context, gnn_artifacts)
            except Exception as e:
                logger.warning(f"[{self.model_name}] Could not get GNN scores: {e}")
        
        # Combine scores (default alpha = 0.5)
        alpha = 0.5
        if context.request_params:
            alpha = context.request_params.get("alpha", 0.5)
        
        # Normalize and combine
        hybrid_scores = {}
        all_product_ids = set(cbf_scores.keys()) | set(gnn_scores.keys())
        
        if all_product_ids:
            # Normalize CBF scores
            cbf_values = list(cbf_scores.values())
            cbf_min = min(cbf_values) if cbf_values else 0.0
            cbf_max = max(cbf_values) if cbf_values else 1.0
            cbf_range = cbf_max - cbf_min if cbf_max != cbf_min else 1.0
            
            # Normalize GNN scores
            gnn_values = list(gnn_scores.values())
            gnn_min = min(gnn_values) if gnn_values else 0.0
            gnn_max = max(gnn_values) if gnn_values else 1.0
            gnn_range = gnn_max - gnn_min if gnn_max != gnn_min else 1.0
            
            # Combine normalized scores
            for product_id in all_product_ids:
                cbf_score = cbf_scores.get(product_id, 0.0)
                gnn_score = gnn_scores.get(product_id, 0.0)
                
                cbf_norm = (cbf_score - cbf_min) / cbf_range if cbf_range > 0 else 0.0
                gnn_norm = (gnn_score - gnn_min) / gnn_range if gnn_range > 0 else 0.0
                
                hybrid_score = alpha * gnn_norm + (1 - alpha) * cbf_norm
                hybrid_scores[product_id] = hybrid_score
        
        return hybrid_scores


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
    Recommend using hybrid model (combines CBF and GNN).
    Note: The API endpoint /api/v1/hybrid/recommend uses a different implementation
    that matches the Streamlit tab exactly.
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

