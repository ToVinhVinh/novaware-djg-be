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
        parts = []
        
        # Gender alignment
        user_gender = normalize_gender(getattr(context.user, "gender", "")) if hasattr(context.user, "gender") else ""
        product_gender = normalize_gender(getattr(product, "gender", "")) if hasattr(product, "gender") else ""
        
        if user_gender and product_gender:
            if product_gender == user_gender and product_gender in ("male", "female"):
                parts.append(f"phù hợp với giới tính {user_gender} của bạn")
            elif product_gender == "unisex":
                parts.append("phù hợp cho mọi giới tính")
        
        # Age alignment
        user_age = getattr(context.user, "age", None) if hasattr(context.user, "age") else None
        if user_age:
            if 18 <= user_age <= 35:
                parts.append("phù hợp với độ tuổi trẻ của bạn")
            elif 36 <= user_age <= 50:
                parts.append("phù hợp với độ tuổi trung niên của bạn")
            elif user_age > 50:
                parts.append("phù hợp với độ tuổi của bạn")
        
        # User preferences from profile
        user_preferences = getattr(context.user, "preferences", {}) or {}
        user_style = user_preferences.get("style", "").lower()
        product_usage = getattr(product, "usage", "").lower() if hasattr(product, "usage") else ""
        
        if user_style and product_usage and user_style == product_usage:
            parts.append(f"phù hợp với phong cách {user_style} của bạn")
        
        # Color preferences from user preferences
        color_preferences = user_preferences.get("colorPreferences", []) or []
        product_color = getattr(product, "baseColour", "") if hasattr(product, "baseColour") else ""
        
        if product_color and color_preferences:
            for pref_color in color_preferences:
                if pref_color.lower() in product_color.lower() or product_color.lower() in pref_color.lower():
                    parts.append(f"màu sắc {product_color} phù hợp với sở thích của bạn")
                    break
        
        # Interaction history - style and color preferences
        style_tokens = list(_style_tokens(product))
        matched_styles = [token for token in style_tokens if context.style_weight(token) > 0]
        if matched_styles:
            parts.append(f"tương tự với sản phẩm bạn đã xem: {', '.join(matched_styles[:2])}")
        
        # Product category matching with user history
        product_article_type = getattr(product, "articleType", "") if hasattr(product, "articleType") else ""
        if product_article_type and context.style_weight(product_article_type.lower()) > 0:
            parts.append(f"bạn đã quan tâm đến {product_article_type.lower()}")
        
        # Hybrid approach
        alpha = getattr(context, 'request_params', {}).get('alpha', self.alpha) if hasattr(context, 'request_params') else self.alpha
        parts.append(f"kết hợp LightGCN ({alpha:.0%}) và SBERT ({1-alpha:.0%})")
        
        if not parts:
            return "gợi ý dựa trên mô hình Hybrid và lịch sử tương tác của bạn"
        
        return "; ".join(parts)


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
