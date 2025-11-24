"""Hybrid recommendation engine combining GNN (LightGCN) + Content-based Filtering (Sentence-BERT + FAISS) with late fusion."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from bson import ObjectId
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
        logger.info(f"[{self.model_name}] Starting hybrid training (GNN + CBF)...")
        
        logger.info(f"[{self.model_name}] Training CBF component (Sentence-BERT + FAISS)...")
        cbf_artifacts = super()._train_impl()
        
        # Train GNN component (LightGCN)
        logger.info(f"[{self.model_name}] Training GNN component (LightGCN)...")
        gnn_engine = GNNRecommendationEngine()
        gnn_artifacts = gnn_engine._train_impl()
        
        # Combine artifacts
        logger.info(f"[{self.model_name}] Combining artifacts from GNN (LightGCN) and CBF (SBERT + FAISS)...")
        
        # Create combined matrix data
        cbf_matrix = cbf_artifacts.get("matrix_data", {})
        gnn_matrix = gnn_artifacts.get("matrix_data", {})
        
        # Use CBF matrix as base, or create a combined visualization
        combined_matrix_data = cbf_matrix.copy()
        combined_matrix_data["description"] = "Hybrid Similarity Matrix (GNN LightGCN + CBF SBERT)"
        combined_matrix_data["value_description"] = f"Hybrid score (alpha={self.alpha} GNN + {1-self.alpha} CBF)"
        
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
        """Score candidates using late fusion of GNN (LightGCN) and Content-based Filtering (Sentence-BERT + FAISS)."""
        alpha = artifacts.get("alpha", self.alpha)
        
        # Get CBF (Content-based Filtering) scores using Sentence-BERT + FAISS
        logger.debug(f"[{self.model_name}] Computing content-based scores (Sentence-BERT + FAISS)...")
        cbf_scores = super()._score_candidates(context, artifacts)
        
        # Get GNN (LightGCN) scores
        logger.debug(f"[{self.model_name}] Computing GNN scores (LightGCN)...")
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
        logger.debug(f"[{self.model_name}] Fusing scores: alpha={alpha} (GNN LightGCN) + {1-alpha} (CBF SBERT)")
        candidate_scores: dict[int, float] = {}
        
        # Get all candidate IDs
        all_candidate_ids = set(cbf_scores_norm.keys()) | set(gnn_scores_norm.keys())
        
        for candidate_id in all_candidate_ids:
            cbf_score = cbf_scores_norm.get(candidate_id, 0.0)
            gnn_score = gnn_scores_norm.get(candidate_id, 0.0)
            
            # Late fusion: weighted sum of GNN (LightGCN) and CBF (Sentence-BERT + FAISS)
            fused_score = alpha * gnn_score + (1 - alpha) * cbf_score
            
            # Add style and brand bonuses
            candidate = next((c for c in context.candidate_products if c.id == candidate_id), None)
            if candidate:
                style_bonus = 0.05 * sum(context.style_weight(token) for token in _style_tokens(candidate))
                brand_bonus = 0.0 
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


def _load_hybrid_model_from_recommendation_system():
    """Load HybridRecommender model from recommendation_system module"""
    import pickle
    from pathlib import Path
    from django.conf import settings
    import os
    import sys
    
    # Add recommendation_system to path
    base_dir = Path(settings.BASE_DIR)
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    
    # Load preprocessor
    preprocessor_path = base_dir / "recommendation_system" / "data" / "preprocessor.pkl"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {preprocessor_path}. "
            "Please run train_recommendation.py first to create the preprocessor."
        )
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load hybrid model
    hybrid_model_path = base_dir / "recommendation_system" / "models" / "hybrid_model.pkl"
    if not hybrid_model_path.exists():
        raise FileNotFoundError(
            f"Hybrid model not found at {hybrid_model_path}. "
            "Please run train_recommendation.py to train the hybrid model first."
        )
    
    with open(hybrid_model_path, 'rb') as f:
        hybrid_model = pickle.load(f)
    
    return preprocessor, hybrid_model


def _build_personalized_reason(payload_product_info: dict, recommended_product_info: dict, score: float) -> str:
    """
    Build a natural language reason by comparing payload product and recommended product.
    Identifies common attributes and creates a conversational explanation.
    """
    reasons = []

    
    # Compare category/type
    payload_category = payload_product_info.get('masterCategory', '')
    recommended_category = recommended_product_info.get('masterCategory', '')
    
    payload_subcategory = payload_product_info.get('subCategory', '')
    recommended_subcategory = recommended_product_info.get('subCategory', '')
    
    payload_article = payload_product_info.get('articleType', '')
    recommended_article = recommended_product_info.get('articleType', '')
    
    # Compare color
    payload_color = payload_product_info.get('baseColour', '')
    recommended_color = recommended_product_info.get('baseColour', '')
    
    # Compare gender
    payload_gender = payload_product_info.get('gender', '')
    recommended_gender = recommended_product_info.get('gender', '')
    
    # Compare season/usage
    payload_season = payload_product_info.get('season', '') if payload_product_info.get('season') else ''
    recommended_season = recommended_product_info.get('season', '') if recommended_product_info.get('season') else ''
    
    payload_usage = payload_product_info.get('usage', '') if payload_product_info.get('usage') else ''
    recommended_usage = recommended_product_info.get('usage', '') if recommended_product_info.get('usage') else ''
    
    # Build reason based on similarities
    if payload_article and recommended_article and payload_article == recommended_article:
        reasons.append(f"Both are {recommended_article}s")
    
    if payload_color and recommended_color and payload_color == recommended_color:
        reasons.append(f"Same {recommended_color} color")
    elif payload_color and recommended_color:
        reasons.append(f"Complements your {payload_color} style")
    
    if payload_gender and recommended_gender and payload_gender == recommended_gender:
        reasons.append(f"Designed for {recommended_gender}")
    
    if payload_season and recommended_season and payload_season == recommended_season:
        reasons.append(f"Perfect for {recommended_season}")
    
    if payload_usage and recommended_usage and payload_usage == recommended_usage:
        reasons.append(f"Great for {recommended_usage}")
    
    if payload_subcategory and recommended_subcategory and payload_subcategory == recommended_subcategory:
        reasons.append(f"Similar {recommended_subcategory} style")
    
    # If no specific matches, provide a generic reason
    if not reasons:
        if payload_category and recommended_category:
            reasons.append(f"Matches your {payload_category} preference")
        else:
            reasons.append("Highly recommended based on your style")
    
    # Combine reasons into natural language
    if len(reasons) == 1:
        reason_text = reasons[0]
    elif len(reasons) == 2:
        reason_text = f"{reasons[0]} and {reasons[1]}"
    else:
        reason_text = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
    
    # Add confidence indicator based on score
    if score >= 0.8:
        confidence = "highly recommended"
    elif score >= 0.6:
        confidence = "recommended"
    else:
        confidence = "worth considering"
    
    return f"{reason_text}. This item is {confidence} for you."


def _get_product_data_with_images(product_info: dict, product_id: str | int) -> dict:
    """Get product data including images and variants from MongoDB"""
    try:
        from apps.products.mongo_models import Product as MongoProduct, ProductVariant
        
        # MongoDB Product uses IntField for id, so convert to int
        product_id_int = None
        try:
            product_id_int = int(product_id)
        except (ValueError, TypeError):
            # If conversion fails, try to get from product_info
            product_id_int = product_info.get('id')
            if product_id_int:
                try:
                    product_id_int = int(product_id_int)
                except (ValueError, TypeError):
                    product_id_int = None
        
        mongo_product = None
        if product_id_int is not None:
            # MongoDB Product.id is IntField, so query with integer
            mongo_product = MongoProduct.objects(id=product_id_int).first()
        
        # Build product data
        product_data = {
            "id": str(product_info.get('id', product_id)),
            "productDisplayName": product_info.get('productDisplayName', 'N/A'),
            "masterCategory": product_info.get('masterCategory', 'N/A'),
            "subCategory": product_info.get('subCategory', 'N/A'),
            "articleType": product_info.get('articleType', 'N/A'),
            "baseColour": product_info.get('baseColour', 'N/A'),
            "gender": product_info.get('gender', 'N/A'),
            "season": product_info.get('season', None),
            "year": product_info.get('year', None),
            "usage": product_info.get('usage', None),
            "images": [],
            "variants": [],
        }
        
        # Get images and variants from MongoDB if product found
        if mongo_product:
            # Get images
            images = getattr(mongo_product, 'images', None)
            if images:
                product_data["images"] = list(images) if isinstance(images, (list, tuple)) else []
            
            # Get variants
            try:
                variants = ProductVariant.objects(product_id=product_id_int)
                variants_list = []
                for variant in variants:
                    variants_list.append({
                        "_id": str(variant.id),
                        "color": getattr(variant, 'color', ''),
                        "size": getattr(variant, 'size', ''),
                        "price": float(getattr(variant, 'price', 0.0)),
                        "stock": int(getattr(variant, 'stock', 0)),
                    })
                product_data["variants"] = variants_list
            except Exception as variant_error:
                logger.warning(f"Could not fetch variants for product {product_id}: {variant_error}")
                product_data["variants"] = []
        
        return product_data
        
    except Exception as e:
        logger.warning(f"Could not fetch product data for product {product_id}: {e}")
        # Return product data without images/variants if MongoDB fetch fails
        return {
            "id": str(product_info.get('id', product_id)),
            "productDisplayName": product_info.get('productDisplayName', 'N/A'),
            "masterCategory": product_info.get('masterCategory', 'N/A'),
            "subCategory": product_info.get('subCategory', 'N/A'),
            "articleType": product_info.get('articleType', 'N/A'),
            "baseColour": product_info.get('baseColour', 'N/A'),
            "gender": product_info.get('gender', 'N/A'),
            "season": product_info.get('season', None),
            "year": product_info.get('year', None),
            "usage": product_info.get('usage', None),
            "images": [],
            "variants": [],
        }


def recommend_hybrid(
    *,
    user_id: str | int,
    current_product_id: str | int,
    top_k_personal: int,
    top_k_outfit: int,
    alpha: float | None = None,
    request_params: dict | None = None,
) -> dict[str, Any]:
    """
    Hybrid recommendation using HybridRecommender from recommendation_system.
    
    This function integrates the HybridRecommender algorithm from 
    recommendation_system/models/hybrid_model.py into the API endpoint.
    """
    try:
        preprocessor, hybrid_model = _load_hybrid_model_from_recommendation_system()
        
        user_id_str = str(user_id)
        product_id_str = str(current_product_id)
        
        user_row = preprocessor.users_df[preprocessor.users_df['id'] == user_id_str]
        if user_row.empty:
            try:
                user_id_int = int(user_id_str)
                user_row = preprocessor.users_df[preprocessor.users_df['id'] == user_id_int]
            except (ValueError, TypeError):
                pass
        
        if user_row.empty:
            raise ValueError(f"User {user_id} not found in preprocessor")
        
        user_idx = int(user_row.iloc[0]['user_idx'])
        user_info = preprocessor.get_user_info(user_idx)
        user_history = preprocessor.get_user_interaction_history(user_idx)
        
        product_row = preprocessor.products_df[preprocessor.products_df['id'] == product_id_str]
        if product_row.empty:
            try:
                product_id_int = int(product_id_str)
                product_row = preprocessor.products_df[preprocessor.products_df['id'] == product_id_int]
            except (ValueError, TypeError):
                pass
        
        if product_row.empty:
            raise ValueError(f"Product {current_product_id} not found in preprocessor")
        
        product_idx = int(product_row.iloc[0]['product_idx'])
        
        # Get payload product info to filter by articleType
        payload_product_info = preprocessor.get_product_info(product_idx)
        payload_article_type = payload_product_info.get('articleType')
        
        if not payload_article_type:
            logger.warning(f"Payload product {current_product_id} has no articleType, cannot filter recommendations")
        
        # Update alpha if provided
        if alpha is not None:
            hybrid_model.alpha = alpha
        
        # Get recommendations using HybridRecommender
        recommendations, inference_time = hybrid_model.recommend_personalized(
            user_info=user_info,
            user_idx=user_idx,
            user_history=user_history,
            payload_product_idx=product_idx,
            top_k=top_k_personal
        )
        
        # Convert recommendations to API format and filter by articleType
        personalized = []
        for prod_idx, score in recommendations:
            product_info = preprocessor.get_product_info(prod_idx)
            
            # Filter: only include products with same articleType as payload
            if payload_article_type:
                candidate_article_type = product_info.get('articleType')
                if candidate_article_type != payload_article_type:
                    logger.debug(
                        f"Skipping product {product_info.get('id')} with articleType '{candidate_article_type}' "
                        f"(payload has '{payload_article_type}')"
                    )
                    continue
            
            product_id = product_info.get('id', str(prod_idx))
            
            # Get product data with images and variants
            product_data = _get_product_data_with_images(product_info, product_id)
            
            # Build natural language reason by comparing payload and recommended product
            reason = _build_personalized_reason(payload_product_info, product_info, score)
            
            personalized.append({
                "product": product_data,
                "score": float(score),
                "reason": reason
            })
        
        # Generate outfit recommendations
        outfit = {
            "topwear": [],
            "bottomwear": [],
            "footwear": [],
            "accessories": [],
            "dress": [],
            "innerwear": []
        }
        
        # Determine payload product category based on subCategory or masterCategory
        payload_subcategory = payload_product_info.get('subCategory', '').lower()
        payload_mastercategory = payload_product_info.get('masterCategory', '').lower()
        
        payload_category = None
        if 'topwear' in payload_subcategory or ('apparel' in payload_mastercategory and 'top' in payload_subcategory):
            payload_category = 'topwear'
        elif 'bottomwear' in payload_subcategory or ('apparel' in payload_mastercategory and ('bottom' in payload_subcategory or 'trouser' in payload_subcategory or 'short' in payload_subcategory or 'skirt' in payload_subcategory)):
            payload_category = 'bottomwear'
        elif 'footwear' in payload_mastercategory or 'shoe' in payload_subcategory or 'sandal' in payload_subcategory or 'flip' in payload_subcategory:
            payload_category = 'footwear'
        elif 'accessories' in payload_mastercategory or 'bag' in payload_subcategory or 'watch' in payload_subcategory or 'belt' in payload_subcategory or 'headwear' in payload_subcategory:
            payload_category = 'accessories'
        elif 'dress' in payload_subcategory:
            payload_category = 'dress'
        elif 'innerwear' in payload_subcategory:
            payload_category = 'innerwear'
        
        logger.debug(f"Payload product category determined: {payload_category} (subCategory: {payload_subcategory}, masterCategory: {payload_mastercategory})")
        
        # Get payload product data
        payload_product_data = _get_product_data_with_images(payload_product_info, current_product_id)
        
        # If payload belongs to a category, add it to that category first
        if payload_category and payload_category in outfit:
            outfit[payload_category].append({
                "product": payload_product_data,
                "score": 1.0
            })
            logger.debug(f"Added payload product to {payload_category} category")
        
        try:
            outfit_recs, outfit_inference_time = hybrid_model.recommend_outfit(
                user_info=user_info,
                payload_product_idx=product_idx,
                user_history=user_history
            )
            
            logger.debug(f"Outfit recommendations received: {list(outfit_recs.keys())}")
            
            for category, items in outfit_recs.items():
                if category == 'payload':
                    continue
                
                if category not in outfit:
                    outfit[category] = []
                
                # If this category is the payload category, skip adding recommendations
                # (only payload product should be in this category)
                if category == payload_category:
                    logger.debug(f"Skipping recommendations for {category} as it contains payload product")
                    continue
                
                # Check if items is a list and has elements
                if not isinstance(items, list) or len(items) == 0:
                    logger.debug(f"Category {category} has no items or invalid format")
                    continue
                
                # Process items (should be List[Tuple[int, float]])
                for item in items[:top_k_outfit]:
                    try:
                        # Handle tuple format (prod_idx, score)
                        if isinstance(item, tuple) and len(item) == 2:
                            prod_idx, score = item
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            prod_idx, score = item[0], item[1]
                        else:
                            logger.warning(f"Invalid item format in {category}: {item}")
                            continue
                        
                        # Skip if this is the payload product
                        if int(prod_idx) == product_idx:
                            logger.debug(f"Skipping payload product from {category} recommendations")
                            continue
                        
                        # Get product info
                        product_info = preprocessor.get_product_info(int(prod_idx))
                        product_id = product_info.get('id', str(prod_idx))
                        
                        # Get product data with images
                        product_data = _get_product_data_with_images(product_info, product_id)
                        
                        outfit[category].append({
                            "product": product_data,
                            "score": float(score)
                        })
                    except Exception as item_error:
                        logger.warning(f"Error processing item in {category}: {item_error}")
                        continue
            
            logger.info(f"Generated outfit with {sum(len(v) for v in outfit.values())} total items")
            
        except Exception as e:
            logger.error(f"Could not generate outfit recommendations: {e}", exc_info=True)
            # Keep empty outfit structure as initialized above
        
        return {
            "personalized": personalized,
            "outfit": outfit,
            "model": "hybrid",
            "alpha": hybrid_model.alpha,
            "inference_time": inference_time,
            "user_id": str(user_id),
            "current_product_id": str(current_product_id)
        }
        
    except (FileNotFoundError, ValueError) as e:
        # Fallback to original engine if recommendation_system models are not available
        logger.warning(
            f"Could not use HybridRecommender from recommendation_system: {e}. "
            "Falling back to HybridRecommendationEngine."
        )
        
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
            return payload.as_dict()
        finally:
            if previous_alpha is not None:
                engine.alpha = previous_alpha
