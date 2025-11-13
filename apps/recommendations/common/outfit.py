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
    ) -> tuple[dict[str, list[OutfitRecommendation]], float]:
        current_category = context.current_product.category_type
        required_categories = cls.required_categories(current_category)
        if not required_categories:
            return {}, 0.0

        candidate_map = context.candidate_map
        remaining_ids = set(candidate_map.keys())

        outfit_payload: dict[str, list[OutfitRecommendation]] = defaultdict(list)
        category_scores: list[float] = []

        for category in required_categories:
            ranked = cls._rank_for_category(
                category=category,
                scored_candidates=scored_candidates,
                candidate_map=candidate_map,
                remaining_ids=remaining_ids,
                context=context,
            )
            if not ranked:
                continue
            top_entries = ranked[:top_k]
            for product_id, score in top_entries:
                product = candidate_map.get(product_id)
                if not product:
                    continue
                outfit_payload[category].append(OutfitRecommendation(category, product, score))
                remaining_ids.discard(product_id)
                category_scores.append(score)

        if not outfit_payload:
            return {}, 0.0

        completeness_ratio = len(outfit_payload) / len(required_categories)
        averaged_score = sum(category_scores) / len(category_scores) if category_scores else OUTFIT_SCORE_FLOOR
        outfit_score = max(OUTFIT_SCORE_FLOOR, min(1.0, (averaged_score + completeness_ratio) / 2))
        return dict(outfit_payload), outfit_score

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

