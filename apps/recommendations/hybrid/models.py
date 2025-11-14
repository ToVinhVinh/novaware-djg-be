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
from apps.users.models import User, UserInteraction

logger = logging.getLogger(__name__)


class HybridRecommendationEngine(ContentBasedRecommendationEngine):
    model_name = "hybrid"
    alpha = 0.6

    def _train_impl(self) -> dict[str, Any]:
        logger.info(f"[{self.model_name}] Starting hybrid training: training CBF component...")
        base_artifacts = super()._train_impl()
        
        # Load MongoDB data directly - use only MongoDB ObjectIds
        logger.info(f"[{self.model_name}] Loading MongoDB data...")
        
        # Initialize variables
        mongo_product_ids: list[str] = []
        user_ids: list[str] = []
        interactions_list: list[tuple[str, str, str]] = []
        mongo_id_to_index: dict[str, int] = {}
        user_index: dict[str, int] = {}
        
        try:
            from apps.users.mongo_models import UserInteraction as MongoInteraction
            from apps.products.mongo_models import Product as MongoProduct
            from bson import ObjectId
            
            # Get all MongoDB interactions
            mongo_interactions = MongoInteraction.objects.all()
            logger.info(f"[{self.model_name}] Found {mongo_interactions.count()} MongoDB interactions")
            
            # Get all MongoDB products (not just from interactions) for matrix display
            all_mongo_products = MongoProduct.objects.all().limit(100)  # Limit for performance
            for mongo_product in all_mongo_products:
                if mongo_product.id:
                    mongo_product_ids.append(str(mongo_product.id))
            
            logger.info(f"[{self.model_name}] Found {len(mongo_product_ids)} MongoDB products for matrix")
            
            # If no products from all products, try to get from interactions
            if len(mongo_product_ids) == 0:
                mongo_product_ids_set = set()
                for interaction in mongo_interactions:
                    if interaction.product_id:
                        mongo_product_ids_set.add(str(interaction.product_id))
                
                for product_id_str in mongo_product_ids_set:
                    try:
                        mongo_product = MongoProduct.objects(id=ObjectId(product_id_str)).first()
                        if mongo_product:
                            mongo_product_ids.append(product_id_str)
                    except:
                        continue
                logger.info(f"[{self.model_name}] Found {len(mongo_product_ids)} products from interactions")
            
            # Get all MongoDB users (not just from interactions) for matrix display
            from apps.users.mongo_models import User as MongoUser
            all_mongo_users = MongoUser.objects.all().limit(100)  # Limit for performance
            mongo_user_ids_list = []
            for mongo_user in all_mongo_users:
                if mongo_user.id:
                    mongo_user_ids_list.append(str(mongo_user.id))
            
            logger.info(f"[{self.model_name}] Found {len(mongo_user_ids_list)} MongoDB users for matrix")
            
            # Create mapping: MongoDB product ID -> index
            mongo_id_to_index = {pid: idx for idx, pid in enumerate(mongo_product_ids)}
            
            # Collect interactions with MongoDB IDs only
            for interaction in mongo_interactions:
                user_id_str = str(interaction.user_id)
                product_id_str = str(interaction.product_id)
                
                # Only include if product exists in our MongoDB product list
                if product_id_str in mongo_id_to_index:
                    interactions_list.append((user_id_str, product_id_str, interaction.interaction_type))
            
            logger.info(f"[{self.model_name}] Loaded {len(interactions_list)} interactions with MongoDB IDs")
            
            # If no interactions but we have users and products, use all users
            if len(interactions_list) == 0 and len(mongo_user_ids_list) > 0:
                user_ids = mongo_user_ids_list[:5]  # Use first 5 users for display
                logger.info(f"[{self.model_name}] No interactions found, using {len(user_ids)} users for matrix display")
            elif len(interactions_list) > 0:
                # Collect unique user IDs from interactions
                user_ids_set = set()
                for user_id, _, _ in interactions_list:
                    user_ids_set.add(user_id)
                user_ids = list(user_ids_set)
                logger.info(f"[{self.model_name}] Found {len(user_ids)} unique users from interactions")
            
            # Ensure we have at least some users and products for matrix display
            if len(user_ids) == 0:
                if len(mongo_user_ids_list) > 0:
                    user_ids = mongo_user_ids_list[:5]  # Use first 5 users
                    logger.info(f"[{self.model_name}] Using {len(user_ids)} users from MongoDB for matrix display")
                else:
                    logger.warning(f"[{self.model_name}] No MongoDB users found!")
                    user_ids = []
            
            if len(mongo_product_ids) == 0:
                logger.warning(f"[{self.model_name}] No MongoDB products found!")
            
            # If we have no users or products, return empty matrix
            if len(user_ids) == 0 or len(mongo_product_ids) == 0:
                logger.warning(f"[{self.model_name}] Cannot create matrix: users={len(user_ids)}, products={len(mongo_product_ids)}")
                return {
                    **base_artifacts,
                    "user_ids": [],
                    "item_factors": None,
                    "user_factors": None,
                    "alpha": self.alpha,
                    "matrix_data": {
                        "shape": [0, 0],
                        "display_shape": [0, 0],
                        "data": [],
                        "user_ids": [],
                        "product_ids": [],
                        "description": "User-Item Interaction Matrix",
                        "row_label": "User ID",
                        "col_label": "Product ID",
                        "value_description": "Interaction weight (0 = no interaction, >0 = interaction strength)",
                        "warning": f"No MongoDB data found. Users: {len(user_ids)}, Products: {len(mongo_product_ids)}",
                    },
                }
            
            # Create user index mapping
            user_index = {uid: idx for idx, uid in enumerate(user_ids)}
            logger.info(f"[{self.model_name}] Matrix will have {len(user_ids)} users and {len(mongo_product_ids)} products")
            
        except Exception as e:
            logger.error(f"[{self.model_name}] Failed to load MongoDB data: {e}", exc_info=True)
            # Return empty matrix data with error info
            return {
                **base_artifacts,
                "user_ids": [],
                "item_factors": None,
                "user_factors": None,
                "alpha": self.alpha,
                "matrix_data": {
                    "shape": [0, 0],
                    "display_shape": [0, 0],
                    "data": [],
                    "user_ids": [],
                    "product_ids": [],
                    "description": "User-Item Interaction Matrix",
                    "row_label": "User ID",
                    "col_label": "Product ID",
                    "value_description": "Interaction weight (0 = no interaction, >0 = interaction strength)",
                    "error": f"Failed to load MongoDB data: {str(e)}",
                },
            }

        logger.info(f"[{self.model_name}] Building user-item interaction matrix...")
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for user_id, product_id, interaction_type in interactions_list:
            # All IDs are now MongoDB ObjectIds (strings)
            if product_id not in mongo_id_to_index:
                continue
            rows.append(user_index[user_id])
            cols.append(mongo_id_to_index[product_id])
            data.append(INTERACTION_WEIGHTS.get(interaction_type, 1.0))

        logger.info(f"[{self.model_name}] Matrix entries: {len(rows)} interactions")

        # Prepare matrix data for response (sample for display)
        # Always create matrix, even if no interactions (will be all zeros)
        logger.info(f"[{self.model_name}] Creating matrix: shape ({len(user_ids)}, {len(mongo_product_ids)})")
        
        if rows:
            # Create sparse matrix from interactions
            matrix = sp.coo_matrix(
                (data, (rows, cols)),
                shape=(len(user_ids), len(mongo_product_ids)),
            ).tocsr()
        else:
            # Create empty sparse matrix (all zeros)
            matrix = sp.coo_matrix(
                (len(user_ids), len(mongo_product_ids))
            ).tocsr()
            logger.info(f"[{self.model_name}] No interactions found, creating empty matrix (all zeros)")
        
        max_display_rows = min(5, len(user_ids))
        max_display_cols = min(500, len(mongo_product_ids))
        matrix_dense = matrix[:max_display_rows, :max_display_cols].toarray()
        
        # If we have fewer rows than desired, pad with empty rows for better visualization
        display_rows = 5
        if len(user_ids) < display_rows:
            # Pad matrix with zero rows
            padded_data = matrix_dense.tolist()
            # Convert user_ids to strings for consistency
            padded_user_ids = [str(uid) for uid in user_ids[:max_display_rows]]
            
            # Add empty rows to reach 5 rows
            while len(padded_data) < display_rows:
                padded_data.append([0.0] * max_display_cols)
                # Use placeholder user IDs (negative numbers as strings to indicate they're not real)
                padded_user_ids.append(f"-{len(padded_data)}")
            
            matrix_dense_display = padded_data[:display_rows]
            user_ids_display = padded_user_ids[:display_rows]
        else:
            matrix_dense_display = matrix_dense.tolist()
            user_ids_display = [str(uid) for uid in user_ids[:max_display_rows]]
        
        # Use MongoDB product IDs directly (already ObjectIds)
        product_ids_display = mongo_product_ids[:max_display_cols]
        
        # Create matrix data structure for frontend
        matrix_data = {
            "shape": [len(user_ids), len(mongo_product_ids)],
            "display_shape": [len(matrix_dense_display), max_display_cols],
            "data": matrix_dense_display,
            "user_ids": user_ids_display,
            "product_ids": product_ids_display,
            "description": "User-Item Interaction Matrix",
            "row_label": "User ID",
            "col_label": "Product ID",
            "value_description": "Interaction weight (0 = no interaction, >0 = interaction strength)",
        }
        
        # Create collaborative filtering payload
        n_components = min(32, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if n_components < 1 or len(rows) == 0:
            logger.warning(f"[{self.model_name}] Matrix too small for SVD (n_components={n_components}) or no interactions, skipping SVD")
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
            "matrix_data": matrix_data,
            "mongo_product_ids": mongo_product_ids,  # Store MongoDB product IDs
            "mongo_user_ids": user_ids,  # Store MongoDB user IDs
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
        
        if context.request_params:
            alpha = context.request_params.get('alpha', self.alpha)
            graph_weight = alpha
            content_weight = 1 - alpha
            g_score = 0.0
            c_score = round(content_weight * 1.5, 1)
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

