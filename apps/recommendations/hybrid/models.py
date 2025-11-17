"""Hybrid recommendation engine combining LightGCN (CF) + SBERT (Content) with late fusion."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from celery import shared_task

from apps.recommendations.cbf.models import ContentBasedRecommendationEngine, _style_tokens
from apps.recommendations.common import CandidateFilter
from apps.recommendations.common.context import RecommendationContext
from apps.recommendations.common.gender_utils import normalize_gender
from apps.recommendations.gnn.models import GNNRecommendationEngine

logger = logging.getLogger(__name__)


class HybridRecommendationEngine(ContentBasedRecommendationEngine):
    model_name = "hybrid"
    alpha = 0.6

    def _train_impl(self) -> dict[str, Any]:
        logger.info(f"[{self.model_name}] Starting hybrid training...")
        
        logger.info(f"[{self.model_name}] Training CBF component (SBERT + FAISS)...")
        cbf_artifacts = super()._train_impl()
        
        # Train GNN component (LightGCN)
        logger.info(f"[{self.model_name}] Training GNN component (LightGCN)...")
        gnn_engine = GNNRecommendationEngine()
        gnn_artifacts = gnn_engine._train_impl()
        
        # Combine artifacts
        logger.info(f"[{self.model_name}] Combining artifacts from CBF and GNN...")
        
        # Create combined matrix data
        cbf_matrix = cbf_artifacts.get("matrix_data", {})
        gnn_matrix = gnn_artifacts.get("matrix_data", {})
        
        # Use CBF matrix as base, or create a combined visualization
        combined_matrix_data = cbf_matrix.copy()
        combined_matrix_data["description"] = "Hybrid Similarity Matrix (LightGCN + SBERT)"
        combined_matrix_data["value_description"] = f"Hybrid score (alpha={self.alpha} CF + {1-self.alpha} Content)"
        
        return {
            **cbf_artifacts,
            "gnn_artifacts": gnn_artifacts,
            "alpha": self.alpha,
            "matrix_data": combined_matrix_data,
        }

    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        """Score candidates using late fusion of LightGCN (CF) and SBERT (Content)."""
        alpha = artifacts.get("alpha", self.alpha)
        
        # Get CBF (Content-based) scores using SBERT
        logger.debug(f"[{self.model_name}] Computing content-based scores (SBERT)...")
        cbf_scores = super()._score_candidates(context, artifacts)
        
        # Get GNN (Collaborative filtering) scores using LightGCN
        logger.debug(f"[{self.model_name}] Computing collaborative filtering scores (LightGCN)...")
        gnn_artifacts = artifacts.get("gnn_artifacts", {})
        gnn_engine = GNNRecommendationEngine()
        gnn_scores = gnn_engine._score_candidates(context, gnn_artifacts)
        
        # Normalize scores to [0, 1] range for fair fusion
        def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
            if not scores:
                return {}
            values = list(scores.values())
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return {k: 0.5 for k in scores.keys()}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
        
        cbf_scores_norm = normalize_scores(cbf_scores)
        gnn_scores_norm = normalize_scores(gnn_scores)
        
        # Late fusion: weighted sum
        logger.debug(f"[{self.model_name}] Fusing scores: alpha={alpha} (CF) + {1-alpha} (Content)")
        candidate_scores: dict[int, float] = {}
        
        # Get all candidate IDs
        all_candidate_ids = set(cbf_scores_norm.keys()) | set(gnn_scores_norm.keys())
        
        for candidate_id in all_candidate_ids:
            cbf_score = cbf_scores_norm.get(candidate_id, 0.0)
            gnn_score = gnn_scores_norm.get(candidate_id, 0.0)
            
            # Late fusion: weighted sum
            fused_score = alpha * gnn_score + (1 - alpha) * cbf_score
            
            # Add style and brand bonuses
            candidate = next((c for c in context.candidate_products if c.id == candidate_id), None)
            if candidate:
                style_bonus = 0.05 * sum(context.style_weight(token) for token in _style_tokens(candidate))
                brand_bonus = 0.0  # Brand field removed from Product model
                fused_score += style_bonus + brand_bonus
            
            candidate_scores[candidate_id] = fused_score
        
        return candidate_scores
    
    def _build_reason(self, product, context: RecommendationContext) -> str:
        """Build detailed reason based on user age, gender, interaction history, style, and color."""
        from apps.recommendations.utils.english_reasons import build_english_reason_from_context
        return build_english_reason_from_context(product, context, "hybrid")


engine = HybridRecommendationEngine()


@shared_task
def train_hybrid_model(force_retrain: bool = False, alpha: float | None = None) -> dict[str, Any]:
    logger.info(f"[hybrid] Celery task started: force_retrain={force_retrain}, alpha={alpha}")
    if alpha is not None:
        engine.alpha = alpha
        logger.info(f"[hybrid] Alpha set to {alpha}")
    result = engine.train(force_retrain=force_retrain)
    logger.info(f"[hybrid] Celery task completed")
    return result


def recommend_hybrid(
    *,
    user_id: str | int,
    current_product_id: str | int,
    top_k_personal: int,
    top_k_outfit: int,
    alpha: float | None = None,
    request_params: dict | None = None,
) -> dict[str, Any]:
    if alpha is not None:
        previous_alpha = engine.alpha
        engine.alpha = alpha
    else:
        previous_alpha = None
    
    if request_params is None:
        request_params = {}
    if alpha is not None:
        request_params["alpha"] = alpha
    
    context = CandidateFilter.build_context(
        user_id=user_id,
        current_product_id=current_product_id,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
        request_params=request_params,
    )
    try:
        payload = engine.recommend(context)
    finally:
        if previous_alpha is not None:
            engine.alpha = previous_alpha
    return payload.as_dict()
