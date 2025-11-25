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


def _normalize_article_type(article_type: str | None) -> str | None:
    """Normalize articleType for safer comparisons."""
    if not article_type:
        return None
    normalized = str(article_type).strip().lower()
    return normalized or None


def _normalize_gender_value(gender: str | None) -> str | None:
    """Normalize gender labels to a consistent lowercase representation."""
    if not gender:
        return None
    normalized = str(gender).strip().lower()
    return normalized or None


def _allowed_outfit_genders(user_gender: str | None, payload_gender: str | None) -> set[str]:
    """
    Determine which product genders are allowed for outfit items based on user gender
    and the gender label of the currently viewed product.
    """
    user_gender_norm = _normalize_gender_value(user_gender)
    payload_gender_norm = _normalize_gender_value(payload_gender)
    
    male_rules = {
        "women": {"women", "unisex"},
        "unisex": {"women", "unisex"},
        "boys": {"boys", "unisex"},
        "girls": {"girls", "unisex"},
        "men": {"men", "unisex"},
    }
    
    female_rules = {
        "men": {"men", "unisex"},
        "unisex": {"men", "unisex"},
        "boys": {"boys", "unisex"},
        "girls": {"girls", "unisex"},
        "women": {"women", "unisex"},
    }
    
    if user_gender_norm == "male":
        allowed = male_rules.get(payload_gender_norm)
    elif user_gender_norm == "female":
        allowed = female_rules.get(payload_gender_norm)
    else:
        allowed = None
    
    if allowed:
        return allowed
    
    return set()


_OUTFIT_CATEGORY_SUBCATEGORY_MAP = {
    "topwear": ["Topwear"],
    "bottomwear": ["Bottomwear"],
    "footwear": ["Footwear"],
    "accessories": ["Accessories", "Bags", "Belts", "Headwear", "Watches"],
    "dress": ["Dress"],
    "innerwear": ["Innerwear"],
}


def _build_outfit_template(payload_gender: str | None) -> dict[str, list]:
    """
    Build the base outfit bucket structure depending on the payload product gender.
    Women/Girls must include dress recommendations, while Men/Boys omit that slot.
    """
    template = {
        "topwear": [],
        "bottomwear": [],
        "footwear": [],
        "accessories": [],
    }

    payload_gender_norm = _normalize_gender_value(payload_gender)
    if payload_gender_norm in {"women", "girls"}:
        template["dress"] = []

    template["innerwear"] = []

    return template


def _ensure_outfit_categories(
    *,
    outfit: dict[str, list],
    preprocessor,
    allowed_genders: set[str] | None,
    used_product_ids: set[str],
    payload_product_info: dict,
    top_k: int,
) -> dict[str, list]:
    """
    Ensure each required outfit category has at least one item by pulling
    fallback products from the catalog when recommendation slots are empty.
    """
    payload_gender_norm = _normalize_gender_value(payload_product_info.get('gender'))
    
    for category, items in outfit.items():
        if items:
            continue
        
        fallback_items = _fetch_category_fallback_products(
            preprocessor=preprocessor,
            category_key=category,
            allowed_genders=allowed_genders,
            used_product_ids=used_product_ids,
            payload_product_info=payload_product_info,
            top_k=top_k,
            payload_gender_norm=payload_gender_norm,
        )
        
        if not fallback_items:
            continue
        
        outfit[category].extend(fallback_items)
        for fallback in fallback_items:
            product_id = str(fallback["product"].get("id"))
            if product_id:
                used_product_ids.add(product_id)
    
    return outfit


def _fetch_category_fallback_products(
    *,
    preprocessor,
    category_key: str,
    allowed_genders: set[str] | None,
    used_product_ids: set[str],
    payload_product_info: dict,
    top_k: int,
    payload_gender_norm: str | None,
    force_article_types: set[str] | None = None,
) -> list[dict]:
    """Fetch fallback products for a given outfit category when the recommender has gaps."""
    products_df = getattr(preprocessor, "products_df", None)
    if products_df is None or products_df.empty:
        return []
    
    subcategories = _OUTFIT_CATEGORY_SUBCATEGORY_MAP.get(category_key, [])
    if not subcategories:
        return []
    
    normalized_subcats = {subcat.lower() for subcat in subcategories}
    df = products_df.copy()
    
    def _normalize_series(series):
        return (
            series.fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    
    if "subCategory" not in df.columns:
        return []
    
    subcategory_series = _normalize_series(df["subCategory"])
    mask = subcategory_series.isin(normalized_subcats)
    
    if allowed_genders and "gender" in df.columns:
        gender_series = _normalize_series(df["gender"])
        mask &= gender_series.isin(allowed_genders)
    elif allowed_genders:
        return []
    
    df = df.loc[mask]
    if df.empty:
        return []
    
    preferred_article_types = None
    if category_key == "accessories" and payload_gender_norm in {"women", "girls"}:
        preferred_article_types = {"handbags"}
    
    target_article_types = force_article_types or preferred_article_types
    if target_article_types and "articleType" in df.columns:
        article_series = _normalize_series(df["articleType"])
        preferred_mask = article_series.isin({art.lower() for art in target_article_types})
        preferred_df = df.loc[preferred_mask]
        if not preferred_df.empty:
            df = preferred_df
        elif force_article_types:
            return []
    
    # Exclude already used products
    df["id_str"] = df["id"].astype(str)
    df = df[~df["id_str"].isin(used_product_ids)]
    if df.empty:
        return []
    
    sort_fields = [col for col in ("rating", "count_in_stock", "year") if col in df.columns]
    if sort_fields:
        df = df.sort_values(by=sort_fields, ascending=False)
    
    fallback_items = []
    for _, row in df.head(top_k * 2).iterrows():
        product_info = row.to_dict()
        product_id = product_info.get("id")
        if product_id is None:
            continue
        
        product_data = _get_product_data_with_images(product_info, product_id)
        reason = _build_personalized_reason(payload_product_info, product_info, score=0.3)
        fallback_items.append(
            {
                "product": product_data,
                "score": 0.3,
                "reason": reason,
            }
        )
        
        if len(fallback_items) >= max(1, top_k):
            break
    
    return fallback_items


def _build_article_type_fallback(
    *,
    preprocessor,
    payload_product_info: dict,
    payload_article_type: str | None,
    top_k: int,
    exclude_product_ids: set[str],
) -> list[dict]:
    """Return relaxed recommendations scoped only by articleType when strict filter yields nothing."""
    target_article_type = _normalize_article_type(payload_article_type)
    if not target_article_type:
        return []

    try:
        products_df = getattr(preprocessor, "products_df", None)
        if products_df is None or "articleType" not in products_df.columns:
            return []

        article_series = (
            products_df["articleType"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        matches = products_df.loc[article_series == target_article_type]
        if matches.empty:
            return []

        sort_fields = [col for col in ("rating", "count_in_stock", "year") if col in matches.columns]
        if sort_fields:
            matches = matches.sort_values(by=sort_fields, ascending=False)

        relaxed = []
        for _, row in matches.iterrows():
            product_info = row.to_dict()
            product_id = str(product_info.get("id") or "")
            if not product_id or product_id in exclude_product_ids:
                continue

            product_data = _get_product_data_with_images(product_info, product_id)
            reason = _build_personalized_reason(payload_product_info, product_info, score=0.35)
            relaxed.append(
                {
                    "product": product_data,
                    "score": 0.35,
                    "reason": reason,
                }
            )
            if len(relaxed) >= top_k:
                break
        return relaxed
    except Exception as exc:
        logger.warning(f"Could not build articleType fallback recommendations: {exc}")
        return []


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
        
        payload_product_info = preprocessor.get_product_info(product_idx)
        payload_article_type = payload_product_info.get('articleType')
        payload_article_type_norm = _normalize_article_type(payload_article_type)
        payload_gender = payload_product_info.get('gender')
        payload_gender_norm = _normalize_gender_value(payload_gender)
        user_gender = (
            user_info.get('gender')
            if isinstance(user_info, dict)
            else getattr(user_info, 'gender', None)
        )
        if isinstance(user_info, dict) and not user_gender:
            user_gender = user_info.get('Gender')
        allowed_outfit_genders = _allowed_outfit_genders(user_gender, payload_gender)
        
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
        
        current_product_id_str = str(current_product_id)
        personalized = []
        for prod_idx, score in recommendations:
            product_info = preprocessor.get_product_info(prod_idx)
            
            if payload_article_type_norm:
                candidate_article_type = product_info.get('articleType')
                candidate_article_type_norm = _normalize_article_type(candidate_article_type)
                if candidate_article_type_norm != payload_article_type_norm:
                    logger.debug(
                        f"Skipping product {product_info.get('id')} with articleType '{candidate_article_type}' "
                        f"(payload has '{payload_article_type}')"
                    )
                    continue
            
            product_id = str(product_info.get('id', prod_idx))
            if product_id == current_product_id_str:
                logger.debug(f"Skipping payload product {product_id} from personalized list")
                continue
            
            product_data = _get_product_data_with_images(product_info, product_id)
            
            reason = _build_personalized_reason(payload_product_info, product_info, score)
            
            personalized.append({
                "product": product_data,
                "score": float(score),
                "reason": reason
            })

        if not personalized:
            logger.info(
                "[hybrid] No strict personalized recommendations, relaxing filter to articleType matches"
            )
            fallback_items = _build_article_type_fallback(
                preprocessor=preprocessor,
                payload_product_info=payload_product_info,
                payload_article_type=payload_article_type,
                top_k=top_k_personal,
                exclude_product_ids={current_product_id_str},
            )
            personalized.extend(fallback_items)
        
        outfit = _build_outfit_template(payload_product_info.get('gender'))
        allowed_outfit_categories = set(outfit.keys())
        used_outfit_product_ids: set[str] = set()
        
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
            used_outfit_product_ids.add(str(payload_product_data.get("id")))
        
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
                
                if category not in allowed_outfit_categories:
                    logger.debug(f"Skipping category {category} as it is not required for payload gender {payload_gender}")
                    continue
                
                if category == payload_category:
                    logger.debug(f"Skipping recommendations for {category} as it contains payload product")
                    continue
                
                # Check if items is a list and has elements
                if not isinstance(items, list) or len(items) == 0:
                    logger.debug(f"Category {category} has no items or invalid format")
                    continue
                
                prefer_handbags = (
                    category == "accessories"
                    and payload_gender_norm in {"women", "girls"}
                )
                deferred_accessory_items: list[dict] = []
                
                for item in items:
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
                        
                        candidate_gender_norm = _normalize_gender_value(product_info.get('gender'))
                        if allowed_outfit_genders:
                            if not candidate_gender_norm:
                                logger.debug(
                                    "Skipping product %s in %s due to missing gender metadata",
                                    product_id,
                                    category,
                                )
                                continue
                            if candidate_gender_norm not in allowed_outfit_genders:
                                logger.debug(
                                    "Skipping product %s in %s: gender '%s' not allowed for user '%s' and payload '%s'",
                                    product_id,
                                    category,
                                    candidate_gender_norm,
                                    user_gender,
                                    payload_gender,
                                )
                                continue
                        
                        # Get product data with images
                        product_data = _get_product_data_with_images(product_info, product_id)
                        
                        entry = {
                            "product": product_data,
                            "score": float(score)
                        }
                        
                        candidate_article_type_norm = _normalize_article_type(product_info.get('articleType'))
                        if (
                            prefer_handbags
                            and candidate_article_type_norm != "handbags"
                        ):
                            deferred_accessory_items.append(entry)
                            continue
                        
                        outfit[category].append(entry)
                        used_outfit_product_ids.add(str(product_data.get("id")))
                        
                        if len(outfit[category]) >= top_k_outfit:
                            break
                    except Exception as item_error:
                        logger.warning(f"Error processing item in {category}: {item_error}")
                        continue
                
                if prefer_handbags and not outfit[category]:
                    handbag_fallback = _fetch_category_fallback_products(
                        preprocessor=preprocessor,
                        category_key=category,
                        allowed_genders=allowed_outfit_genders,
                        used_product_ids=used_outfit_product_ids,
                        payload_product_info=payload_product_info,
                        top_k=max(1, top_k_outfit),
                        payload_gender_norm=payload_gender_norm,
                        force_article_types={"handbags"},
                    )
                    if handbag_fallback:
                        outfit[category].extend(handbag_fallback)
                        for entry in handbag_fallback:
                            used_outfit_product_ids.add(str(entry["product"].get("id")))
                
                if len(outfit[category]) < top_k_outfit and deferred_accessory_items:
                    needed = top_k_outfit - len(outfit[category])
                    for entry in deferred_accessory_items[:needed]:
                        outfit[category].append(entry)
                        used_outfit_product_ids.add(str(entry["product"].get("id")))
            
            outfit = _ensure_outfit_categories(
                outfit=outfit,
                preprocessor=preprocessor,
                allowed_genders=allowed_outfit_genders,
                used_product_ids=used_outfit_product_ids,
                payload_product_info=payload_product_info,
                top_k=max(1, top_k_outfit),
            )
            
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
