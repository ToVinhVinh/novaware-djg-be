from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable

from django.utils import timezone

from apps.products.models import Product

from .context import RecommendationContext
from .outfit import OutfitBuilder
from .schema import PersonalizedRecommendation, RecommendationPayload
from .storage import ArtifactStorage

logger = logging.getLogger(__name__)

class BaseRecommendationEngine(ABC):

    model_name: str = "base"

    def __init__(self) -> None:
        self.storage = ArtifactStorage(self.model_name)

    def train(self, force_retrain: bool = False) -> dict[str, Any]:
        logger.info(f"[{self.model_name}] Starting training process (force_retrain={force_retrain})")
        if not force_retrain and self.storage.exists():
            logger.info(f"[{self.model_name}] Model already trained, skipping")
            return {"status": "already_trained", "model": self.model_name}
        logger.info(f"[{self.model_name}] Training implementation started...")
        artifacts = self._train_impl()
        logger.info(f"[{self.model_name}] Training implementation completed, saving artifacts...")
        metadata = {
            "model": self.model_name,
            "trained_at": timezone.now().isoformat(),
            "artifacts": artifacts,
        }
        self.storage.save(metadata)
        logger.info(f"[{self.model_name}] Training completed successfully")
        return {"status": "training_completed", "model": self.model_name}

    def recommend(self, context: RecommendationContext) -> RecommendationPayload:
        stored = self.storage.load()
        artifacts = stored.get("artifacts", {})
        scored_candidates = self._score_candidates(context, artifacts)
        personalized = self._select_personalized(context, scored_candidates)
        outfit_payload, outfit_score = OutfitBuilder.build(
            context=context,
            scored_candidates=scored_candidates,
            top_k=context.top_k_outfit,
        )
        return RecommendationPayload(
            personalized=personalized,
            outfit=outfit_payload,
            outfit_complete_score=outfit_score,
        )

    def _select_personalized(
        self,
        context: RecommendationContext,
        scored_candidates: dict[int, float],
    ) -> list[PersonalizedRecommendation]:
        candidate_map = context.candidate_map
        ranked = sorted(
            (
                (product_id, score)
                for product_id, score in scored_candidates.items()
                if product_id in candidate_map
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        results: list[PersonalizedRecommendation] = []
        for product_id, score in ranked[: context.top_k_personal]:
            product = candidate_map[product_id]
            reason = self._build_reason(product, context)
            results.append(PersonalizedRecommendation(product, score, reason, context=context))
        return results

    def _build_reason(self, product: Product, context: RecommendationContext) -> str:
        tags = _extract_style_tokens(product)
        matched = [token for token in tags if context.style_weight(token) > 0]
        if matched:
            return f"Similar to your favorite ({', '.join(matched[:3])})"
        return "Suggestions based on your interaction history"

    @abstractmethod
    def _train_impl(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        pass

def _extract_style_tokens(product: Product) -> list[str]:
    tokens: list[str] = []
    if isinstance(getattr(product, "style_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.style_tags if tag)
    if isinstance(getattr(product, "outfit_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.outfit_tags if tag)
    if getattr(product, "category_type", None):
        tokens.append(product.category_type.lower())
    return tokens

