"""Helpers for outfit completion logic."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .constants import BRAND_MATCH_BONUS, OUTFIT_COMPLETION_RULES, OUTFIT_SCORE_FLOOR
from .context import RecommendationContext
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
        current_category = context.current_product.category_type
        
        # Always return all standard categories, not just based on current_category
        # This matches the mongo_engine behavior
        required_categories = ["accessories", "bottoms", "shoes", "tops"]
        # Add dresses if user is female
        if hasattr(context.user, "gender") and (context.user.gender or "").lower() == "female":
            required_categories.insert(2, "dresses")
        
        if not required_categories:
            return {}, 0.0

        candidate_map = context.candidate_map
        remaining_ids = set(candidate_map.keys())
        used_ids: set[int] = set()

        outfit_payload: dict[str, OutfitRecommendation] = {}
        category_scores: list[float] = []

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
                    reason = cls._build_reason(product, context, category, current_category)
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
                        reason = cls._build_reason(fallback_product, context, category, current_category, is_fallback=True)
                        outfit_payload[category] = OutfitRecommendation(category, fallback_product, fallback_score, reason, context=context)
                        used_ids.add(fallback_id)
                        category_scores.append(fallback_score)
                        product_found = True
            
            if not product_found:
                # Step 3: Try to find matching category in candidate pool
                # Only use products that match the category
                for product_id in list(remaining_ids):
                    if product_id in used_ids:
                        continue
                    product = candidate_map.get(product_id)
                    if product and product.category_type == category:
                        score = scored_candidates.get(product_id, 0.0)
                        reason = cls._build_reason(product, context, category, current_category, is_fallback=False)
                        outfit_payload[category] = OutfitRecommendation(category, product, score, reason, context=context)
                        remaining_ids.discard(product_id)
                        used_ids.add(product_id)
                        category_scores.append(score)
                        product_found = True
                        break
                
                # Don't use products from other categories here - let Step 4 handle it from database
            
            if not product_found:
                # Step 4: Final fallback - get any product from database (relax constraints)
                relaxed_product = cls._fallback_from_database_relaxed(category, context, used_ids)
                if relaxed_product:
                    relaxed_id = relaxed_product.id
                    if relaxed_id and relaxed_id not in used_ids:
                        relaxed_score = scored_candidates.get(relaxed_id, 0.0)
                        reason = cls._build_reason(relaxed_product, context, category, current_category, is_fallback=True)
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
        """Try to get a product from database for the category."""
        from apps.products.models import Product
        
        try:
            excluded = used_ids | context.excluded_product_ids
            allowed_genders = OutfitBuilder._allowed_genders(context.resolved_gender)
            query = Product.objects.filter(
                category_type=category,
                gender__in=allowed_genders,
                age_group=context.resolved_age_group,
            ).exclude(id__in=excluded).select_related("brand", "category")
            
            product = query.first()
            return product
        except Exception:
            return None

    @staticmethod
    def _fallback_from_database_relaxed(category: str, context: RecommendationContext, used_ids: set[int]):
        """Try to get a product from database with relaxed constraints."""
        from apps.products.models import Product
        
        try:
            excluded = used_ids | context.excluded_product_ids
            # First try: same category, any gender/age
            query = Product.objects.filter(
                category_type=category,
            ).exclude(id__in=excluded).select_related("brand", "category")
            product = query.first()
            if product:
                return product
            
            # Second try: any product not excluded
            query = Product.objects.exclude(id__in=excluded).select_related("brand", "category")
            product = query.first()
            return product
        except Exception:
            return None

    @staticmethod
    def _allowed_genders(gender: str) -> list[str]:
        allowed = ["unisex"]
        if gender:
            allowed.insert(0, gender)
        return list(dict.fromkeys(allowed))

    @staticmethod
    def _build_reason(product, context: RecommendationContext, category: str, current_category: str, is_fallback: bool = False) -> str:
        """Build reason text for outfit recommendation."""
        from .base_engine import _extract_style_tokens
        
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
        if hasattr(context, 'request_params') and context.request_params:
            alpha = context.request_params.get('alpha', 0.6)
            graph_weight = alpha
            content_weight = 1 - alpha
            # Note: G and C scores are not available in the current implementation
            # Using approximate values based on the blend weights
            g_score = 0.0  # Graph score not directly available
            c_score = round(content_weight * 1.5, 1)  # Approximate content score
            base_reason += f"; hybrid blend {graph_weight:.2f} graph / {content_weight:.2f} content (G={g_score}, C={c_score})"
        
        # Add fallback indicator if it's a fallback or category doesn't match
        if is_fallback or (category != current_category and product.category_type != category):
            # Only add if not already present
            if "; fallback to complete outfit" not in base_reason:
                base_reason += "; fallback to complete outfit"
        
        return base_reason

    @staticmethod
    def _rank_for_category(
        category: str,
        scored_candidates: dict[int, float],
        candidate_map: dict[int, object],
        remaining_ids: set[int],
        context: RecommendationContext,
    ) -> list[tuple[int, float]]:
        ranked: list[tuple[int, float]] = []
        for product_id in list(remaining_ids):
            product = candidate_map.get(product_id)
            if not product or product.category_type != category:
                continue
            base_score = scored_candidates.get(product_id, 0.0)
            style_bonus = sum(context.style_weight(tag) for tag in _extract_style_tokens(product))
            brand_bonus = BRAND_MATCH_BONUS if product.brand_id and context.brand_weight(product.brand_id) else 0.0
            total_score = base_score + style_bonus + brand_bonus
            ranked.append((product_id, total_score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


def _extract_style_tokens(product) -> Iterable[str]:
    style_tags = []
    if isinstance(getattr(product, "style_tags", None), list):
        style_tags.extend(product.style_tags)
    if isinstance(getattr(product, "outfit_tags", None), list):
        style_tags.extend(product.outfit_tags)
    if product.category_type:
        style_tags.append(product.category_type)
    return {str(token).lower() for token in style_tags if token}

