"""Helpers for outfit completion logic."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .constants import BRAND_MATCH_BONUS, OUTFIT_COMPLETION_RULES, OUTFIT_SCORE_FLOOR
from .context import RecommendationContext
from .gender_utils import gender_filter_values, normalize_gender
from .schema import OutfitRecommendation


class OutfitBuilder:
    """Create outfit suggestions based on the current product."""

    @classmethod
    def required_categories(cls, current_category: str) -> List[str]:
        return OUTFIT_COMPLETION_RULES.get(current_category or "", [])

    @classmethod
    def build(
        cls,
        context: RecommendationContext,
        scored_candidates: dict[int, float],
        top_k: int,
    ) -> tuple[dict[str, OutfitRecommendation], float]:
        # Use subCategory and articleType from Product model
        current_sub_category = (getattr(context.current_product, "subCategory", "") or "").lower()
        current_article_type = (getattr(context.current_product, "articleType", "") or "").lower()
        
        required_categories = []
        
        # Map based on actual category hierarchy from API
        topwear_articles = ["jackets", "shirts", "sweaters", "sweatshirts", "tops", "tshirts", "tunics"]
        bottomwear_articles = ["capris", "jeans", "shorts", "skirts", "track pants", "tracksuits", "trousers"]
        dress_articles = ["dresses"]
        footwear_articles = ["flip flops", "sandals", "sports sandals", "casual shoes", "flats", "formal shoes", "heels", "sports shoes"]
        accessories_articles = ["backpacks", "handbags", "belts", "caps", "watches"]
        
        if current_sub_category == "topwear" or current_article_type.lower() in topwear_articles:
            required_categories = ["bottomwear", "footwear", "accessories"]
        elif current_sub_category == "bottomwear" or current_article_type.lower() in bottomwear_articles:
            required_categories = ["topwear", "footwear", "accessories"]
        elif current_sub_category == "dress" or current_article_type.lower() in dress_articles:
            # Only suggest shoes and accessories for dresses
            required_categories = ["footwear", "accessories"]
        elif current_sub_category in ["shoes", "sandal", "flip flops"] or current_article_type.lower() in footwear_articles:
            required_categories = ["topwear", "bottomwear", "accessories"]
        elif current_sub_category in ["bags", "belts", "headwear", "watches"] or current_article_type.lower() in accessories_articles:
            required_categories = ["topwear", "bottomwear", "footwear"]
        else:
            # Default fallback
            required_categories = ["topwear", "bottomwear", "footwear", "accessories"]
        
        if not required_categories:
            return {}, 0.0

        candidate_map = context.candidate_map
        remaining_ids = set(candidate_map.keys())
        used_ids: set[int] = set()

        outfit_payload: dict[str, OutfitRecommendation] = {}
        category_scores: list[float] = []

        # Don't include the current product in outfit recommendations
        # Outfit should complement the current product, not include it

        # First pass: try to find products matching each category
        for category in required_categories:
            product_found = False
            
            # Step 1: Try to find in candidate pool matching category
            ranked = cls._rank_for_category(
                category=category,
                scored_candidates=scored_candidates,
                candidate_map=candidate_map,
                remaining_ids=remaining_ids,
                context=context,
            )
            if ranked:
                # Take only the top item for each category
                product_id, score = ranked[0]
                product = candidate_map.get(product_id)
                if product and product_id not in used_ids:
                    reason = cls._build_reason(product, context, category)
                    outfit_payload[category] = OutfitRecommendation(category, product, score, reason, context=context)
                    remaining_ids.discard(product_id)
                    used_ids.add(product_id)
                    category_scores.append(score)
                    product_found = True
            
            if not product_found:
                # Step 2: Try to find from database matching category
                fallback_product = cls._fallback_from_database(category, context, used_ids)
                if fallback_product:
                    fallback_id = fallback_product.id
                    if fallback_id and fallback_id not in used_ids:
                        fallback_score = scored_candidates.get(fallback_id, 0.0)
                        reason = cls._build_reason(fallback_product, context, category, is_fallback=True)
                        outfit_payload[category] = OutfitRecommendation(category, fallback_product, fallback_score, reason, context=context)
                        used_ids.add(fallback_id)
                        category_scores.append(fallback_score)
                        product_found = True
            
            if not product_found:
                # Step 3: Try to find matching category in candidate pool
                # Only use products that match the category (case-insensitive comparison)
                for product_id in list(remaining_ids):
                    if product_id in used_ids:
                        continue
                    product = candidate_map.get(product_id)
                    if product:
                        if cls._product_matches_category(product, category):
                            score = scored_candidates.get(product_id, 0.0)
                            reason = cls._build_reason(product, context, category, is_fallback=False)
                            outfit_payload[category] = OutfitRecommendation(category, product, score, reason, context=context)
                            remaining_ids.discard(product_id)
                            used_ids.add(product_id)
                            category_scores.append(score)
                            product_found = True
                            break
                
            if not product_found:
                relaxed_product = cls._fallback_from_database_relaxed(category, context, used_ids)
                if relaxed_product:
                    relaxed_id = relaxed_product.id
                    if relaxed_id and relaxed_id not in used_ids:
                        relaxed_score = scored_candidates.get(relaxed_id, 0.0)
                        reason = cls._build_reason(relaxed_product, context, category, is_fallback=True)
                        outfit_payload[category] = OutfitRecommendation(category, relaxed_product, relaxed_score, reason, context=context)
                        used_ids.add(relaxed_id)
                        category_scores.append(relaxed_score)
                        product_found = True
            
            # If still no product found, log warning but continue
            if not product_found:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not find product for outfit category: {category}")

        if not outfit_payload:
            return {}, 0.0

        completeness_ratio = len(outfit_payload) / len(required_categories)
        averaged_score = sum(category_scores) / len(category_scores) if category_scores else OUTFIT_SCORE_FLOOR
        outfit_score = max(OUTFIT_SCORE_FLOOR, min(1.0, (averaged_score + completeness_ratio) / 2))
        return dict(outfit_payload), outfit_score

    @staticmethod
    def _fallback_from_database(category: str, context: RecommendationContext, used_ids: set[int]):
        """Try to get a product from database for the category with strict gender/age filtering."""
        from apps.products.models import Product
        
        try:
            excluded = used_ids | context.excluded_product_ids
            allowed_genders = OutfitBuilder._allowed_genders(context.resolved_gender)
            
            # Strict filtering: must match user's gender and age group
            # Exclude children's items for adult users
            query = Product.objects.filter(
                gender__in=allowed_genders,
                age_group=context.resolved_age_group,
            ).exclude(id__in=excluded)
            
            for product in query:
                if cls._product_matches_category(product, category):
                    # Double-check gender and age match
                    product_gender = (getattr(product, "gender", "") or "").lower()
                    product_age = (getattr(product, "age_group", "") or "").lower()
                    user_gender = context.resolved_gender.lower()
                    user_age = context.resolved_age_group.lower()
                    
                    # Gender check: product must match user gender or be unisex
                    if product_gender not in allowed_genders:
                        continue
                    
                    # Age check: must match exactly (no children's items for adults)
                    if product_age != user_age:
                        continue
                    
                    return product
            return None
        except Exception:
            return None

    @staticmethod
    def _fallback_from_database_relaxed(category: str, context: RecommendationContext, used_ids: set[int]):
        """Try to get a product from database with relaxed constraints."""
        from apps.products.models import Product
        
        try:
            excluded = used_ids | context.excluded_product_ids
            category_lower = str(category).lower().strip()
            
            # First try: same category, any gender/age (case-insensitive)
            query = Product.objects.exclude(id__in=excluded)
            for product in query:
                if cls._product_matches_category(product, category_lower):
                    return product
            
            # Second try: same category, relaxed gender constraints
            allowed_genders = OutfitBuilder._allowed_genders(context.resolved_gender)
            query = Product.objects.filter(
                gender__in=allowed_genders,
            ).exclude(id__in=excluded)
            for product in query:
                if cls._product_matches_category(product, category_lower):
                    return product
            
            # Third try: same category, any constraints
            query = Product.objects.exclude(id__in=excluded)
            for product in query:
                if cls._product_matches_category(product, category_lower):
                    return product
            
            # Don't fall back to any product - return None to avoid duplicates
            # This ensures each category gets a unique product or none
            return None
        except Exception:
            return None

    @staticmethod
    def _allowed_genders(gender: str) -> list[str]:
        return gender_filter_values(gender)

    @staticmethod
    def _build_reason(product, context: RecommendationContext, category: str, is_fallback: bool = False) -> str:
        """Build detailed reason for outfit recommendation based on age, gender, style, and color."""
        from .base_engine import _extract_style_tokens
        
        parts = []
        
        # Age and gender alignment
        user_gender = normalize_gender(getattr(context.user, "gender", "")) if hasattr(context.user, "gender") else ""
        product_gender = normalize_gender(getattr(product, "gender", "")) if hasattr(product, "gender") else ""
        
        if user_gender and product_gender:
            if product_gender == user_gender and product_gender in ("male", "female"):
                parts.append(f"phù hợp với giới tính {user_gender} của bạn")
            elif product_gender == "unisex":
                parts.append("phù hợp cho mọi giới tính")
        
        # Age group
        user_age = getattr(context.user, "age", None) if hasattr(context.user, "age") else None
        product_age_group = getattr(product, "age_group", "").lower() if hasattr(product, "age_group") else ""
        
        if user_age:
            if user_age <= 12:
                user_age_group = "kid"
            elif user_age <= 19:
                user_age_group = "teen"
            else:
                user_age_group = "adult"
            
            if product_age_group == user_age_group:
                parts.append(f"phù hợp với độ tuổi {user_age_group} của bạn")
        
        # Style matching
        tags = _extract_style_tokens(product)
        matched = [token for token in tags if context.style_weight(token) > 0]
        if matched:
            parts.append(f"có phong cách tương tự: {', '.join(matched[:3])}")
        
        # Color preferences
        if hasattr(product, "baseColour") and product.baseColour:
            color = product.baseColour.lower()
            if context.style_weight(color) > 0:
                parts.append(f"màu sắc {color} phù hợp với sở thích của bạn")
        
        # Brand preference
        # Brand field removed from Product model
        
        # Outfit completion context - use current product's category info
        current_sub_category = (getattr(context.current_product, "subCategory", "") or "").lower()
        current_article_type = (getattr(context.current_product, "articleType", "") or "").lower()
        
        if category != current_sub_category and category != current_article_type:
            if current_sub_category:
                parts.append(f"bổ sung cho {current_sub_category} bạn đang xem")
            elif current_article_type:
                parts.append(f"bổ sung cho {current_article_type} bạn đang xem")
        
        if not parts:
            parts.append("phù hợp để hoàn thiện bộ trang phục")
        
        # Add fallback indicator if needed
        if is_fallback:
            parts.append("(gợi ý bổ sung)")
        
        return "; ".join(parts)

    @staticmethod
    def _rank_for_category(
        category: str,
        scored_candidates: dict[int, float],
        candidate_map: dict[int, object],
        remaining_ids: set[int],
        context: RecommendationContext,
    ) -> list[tuple[int, float]]:
        ranked: list[tuple[int, float]] = []
        category_lower = str(category).lower().strip()
        for product_id in list(remaining_ids):
            product = candidate_map.get(product_id)
            if not product:
                continue
            
            if not cls._product_matches_category(product, category_lower):
                continue
            base_score = scored_candidates.get(product_id, 0.0)
            style_bonus = sum(context.style_weight(tag) for tag in _extract_style_tokens(product))
            brand_bonus = 0.0  # Brand field removed from Product model
            total_score = base_score + style_bonus + brand_bonus
            ranked.append((product_id, total_score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


    @staticmethod
    def _product_matches_category(product, target_category: str) -> bool:
        """Check if a product matches a target outfit category based on the hierarchy."""
        product_sub_category = (getattr(product, "subCategory", "") or "").lower().strip()
        product_article_type = (getattr(product, "articleType", "") or "").lower().strip()
        target_category = target_category.lower().strip()

        topwear_articles = ["jackets", "shirts", "sweaters", "sweatshirts", "tops", "tshirts", "tunics"]
        bottomwear_articles = ["capris", "jeans", "shorts", "skirts", "track pants", "tracksuits", "trousers"]
        footwear_articles = ["flip flops", "sandals", "sports sandals", "casual shoes", "flats", "formal shoes", "heels", "sports shoes"]
        accessories_articles = ["backpacks", "handbags", "belts", "caps", "watches"]

        if target_category == "topwear":
            return product_sub_category == "topwear" or product_article_type in topwear_articles
        if target_category == "bottomwear":
            return product_sub_category == "bottomwear" or product_article_type in bottomwear_articles
        if target_category == "footwear":
            return product_sub_category in ["shoes", "sandal", "flip flops"] or product_article_type in footwear_articles
        if target_category == "accessories":
            return product_sub_category in ["bags", "belts", "headwear", "watches"] or product_article_type in accessories_articles
        
        return False

def _extract_style_tokens(product) -> Iterable[str]:
    style_tags = []
    if isinstance(getattr(product, "style_tags", None), list):
        style_tags.extend(product.style_tags)
    if isinstance(getattr(product, "outfit_tags", None), list):
        style_tags.extend(product.outfit_tags)
    if getattr(product, "articleType", None):
        style_tags.append(product.articleType)
    if getattr(product, "subCategory", None):
        style_tags.append(product.subCategory)
    return {str(token).lower() for token in style_tags if token}

