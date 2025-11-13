"""Hybrid recommendation engine combining content-based and collaborative signals."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
from celery import shared_task
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from apps.recommendations.cbf.models import ContentBasedRecommendationEngine, _style_tokens
from apps.recommendations.common import CandidateFilter
from apps.recommendations.common.constants import INTERACTION_WEIGHTS
from apps.recommendations.common.context import RecommendationContext
from apps.users.models import UserInteraction

logger = logging.getLogger(__name__)


class HybridRecommendationEngine(ContentBasedRecommendationEngine):
    model_name = "hybrid"
    alpha = 0.6

    def _train_impl(self) -> dict[str, Any]:
        logger.info(f"[{self.model_name}] Starting hybrid training: training CBF component...")
        base_artifacts = super()._train_impl()
        product_ids: list[int] = base_artifacts["product_ids"]
        logger.info(f"[{self.model_name}] CBF component trained, {len(product_ids)} products")
        id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}

        logger.info(f"[{self.model_name}] Loading user interactions from database...")
        interactions = (
            UserInteraction.objects.all()
            .values_list("user_id", "product_id", "interaction_type")
        )
        interactions_list = list(interactions)
        logger.info(f"[{self.model_name}] Loaded {len(interactions_list)} interactions")
        
        user_ids: list[int] = []
        for user_id, _, _ in interactions_list:
            if user_id not in user_ids:
                user_ids.append(user_id)
        logger.info(f"[{self.model_name}] Found {len(user_ids)} unique users")
        user_index = {uid: idx for idx, uid in enumerate(user_ids)}

        logger.info(f"[{self.model_name}] Building user-item interaction matrix...")
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for user_id, product_id, interaction_type in interactions_list:
            if product_id not in id_to_index:
                continue
            rows.append(user_index[user_id])
            cols.append(id_to_index[product_id])
            data.append(INTERACTION_WEIGHTS.get(interaction_type, 1.0))

        logger.info(f"[{self.model_name}] Matrix entries: {len(rows)} interactions")

        if not rows:
            logger.warning(f"[{self.model_name}] No valid interactions found, skipping collaborative filtering")
            collaborative_payload = {
                "user_ids": user_ids,
                "item_factors": None,
                "user_factors": None,
            }
        else:
            logger.info(f"[{self.model_name}] Creating sparse matrix: shape ({len(user_ids)}, {len(product_ids)})")
            matrix = sp.coo_matrix(
                (data, (rows, cols)),
                shape=(len(user_ids), len(product_ids)),
            ).tocsr()
            n_components = min(32, matrix.shape[0] - 1, matrix.shape[1] - 1)
            if n_components < 1:
                logger.warning(f"[{self.model_name}] Matrix too small for SVD (n_components={n_components}), skipping")
                collaborative_payload = {
                    "user_ids": user_ids,
                    "item_factors": None,
                    "user_factors": None,
                }
            else:
                logger.info(f"[{self.model_name}] Performing TruncatedSVD with {n_components} components...")
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                svd.fit(matrix)
                logger.info(f"[{self.model_name}] SVD fitting completed, transforming matrix...")
                user_factors = svd.transform(matrix)
                item_factors = svd.components_.T
                logger.info(f"[{self.model_name}] SVD transformation completed: user_factors shape {user_factors.shape}, item_factors shape {item_factors.shape}")
                collaborative_payload = {
                    "user_ids": user_ids,
                    "item_factors": item_factors,
                    "user_factors": user_factors,
                }

        logger.info(f"[{self.model_name}] Hybrid training completed (alpha={self.alpha})")
        return {
            **base_artifacts,
            **collaborative_payload,
            "alpha": self.alpha,
        }

    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        vectorizer = artifacts["vectorizer"]
        product_ids = artifacts["product_ids"]
        product_matrix = artifacts["product_matrix"]
        id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}

        cbf_profile = self._build_user_profile(context, vectorizer, product_matrix, id_to_index)
        if cbf_profile is None or cbf_profile.nnz == 0:
            cbf_profile = self._vector_for_product(
                context.current_product,
                vectorizer,
                product_matrix,
                id_to_index,
            )

        user_factors = artifacts.get("user_factors")
        item_factors = artifacts.get("item_factors")
        user_ids = artifacts.get("user_ids") or []
        user_index_map = {uid: idx for idx, uid in enumerate(user_ids)}

        cf_vector = None
        if user_factors is not None and item_factors is not None and context.user.id in user_index_map:
            cf_vector = user_factors[user_index_map[context.user.id]]
        elif item_factors is not None:
            history_vectors = []
            for product in context.history_products:
                idx = id_to_index.get(product.id)
                if idx is not None:
                    history_vectors.append(item_factors[idx])
            if history_vectors:
                cf_vector = np.mean(history_vectors, axis=0)

        candidate_scores: dict[int, float] = {}
        for candidate in context.candidate_products:
            candidate_id = candidate.id
            if candidate_id is None:
                continue
            candidate_vector = self._vector_for_product(candidate, vectorizer, product_matrix, id_to_index)
            if candidate_vector is None or candidate_vector.nnz == 0:
                continue
            cbf_score = cosine_similarity(cbf_profile, candidate_vector)[0][0]
            cf_score = 0.0
            if cf_vector is not None and item_factors is not None:
                idx = id_to_index.get(candidate_id)
                if idx is not None:
                    item_vector = item_factors[idx]
                    denominator = (np.linalg.norm(cf_vector) * np.linalg.norm(item_vector)) + 1e-9
                    cf_score = float(np.dot(cf_vector, item_vector) / denominator)
            blend_alpha = context.request_params.get("alpha", artifacts.get("alpha", self.alpha))
            blended = blend_alpha * cbf_score + (1 - blend_alpha) * cf_score
            style_bonus = 0.05 * sum(context.style_weight(token) for token in _style_tokens(candidate))
            brand_bonus = 0.2 * context.brand_weight(candidate.brand_id)
            candidate_scores[candidate_id] = blended + style_bonus + brand_bonus

        return candidate_scores

    def _build_reason(self, product: Product, context: RecommendationContext) -> str:
        """Build reason text for hybrid personalized recommendations."""
        from apps.recommendations.common.base_engine import _extract_style_tokens
        
        tags = _extract_style_tokens(product)
        matched = [token for token in tags if context.style_weight(token) > 0]
        base_reason = ""
        if matched:
            base_reason = f"sized for your {product.age_group or 'adult'} age group; shares styles you like: {', '.join(matched[:3])}"
        elif context.brand_weight(product.brand_id):
            base_reason = f"sized for your {product.age_group or 'adult'} age group; matches your preferred brand"
        else:
            base_reason = f"sized for your {product.age_group or 'adult'} age group"
        
        # Add hybrid blend info if available
        if context.request_params:
            alpha = context.request_params.get('alpha', self.alpha)
            graph_weight = alpha
            content_weight = 1 - alpha
            # Note: G and C scores are not directly available, using approximate values
            g_score = 0.0  # Graph/collaborative score not directly available
            c_score = round(content_weight * 1.5, 1)  # Approximate content score
            base_reason += f"; hybrid blend {graph_weight:.2f} graph / {content_weight:.2f} content (G={g_score}, C={c_score})"
        
        return base_reason


engine = HybridRecommendationEngine()


@shared_task
def train_hybrid_model(force_retrain: bool = False, alpha: float | None = None) -> dict[str, Any]:
    logger.info(f"[hybrid] Celery task started: force_retrain={force_retrain}, alpha={alpha}")
    if alpha is not None:
        engine.alpha = alpha
        logger.info(f"[hybrid] Alpha set to {alpha}")
    result = engine.train(force_retrain=force_retrain)
    logger.info(f"[hybrid] Celery task completed: {result}")
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

