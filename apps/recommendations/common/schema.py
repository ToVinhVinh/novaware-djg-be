"""Serializable schema objects for recommendation responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from apps.products.models import Product


def _serialize_product(product: Product) -> dict[str, Any]:
    images = []
    if isinstance(product.images, list):
        images = product.images
    image_url = images[0] if images else None
    return {
        "product_id": str(product.id),
        "name": product.name,
        "slug": product.slug,
        "price": float(product.price),
        "score": None,
        "image_url": image_url,
        "gender": product.gender,
        "age_group": product.age_group,
        "category_type": product.category_type,
    }


@dataclass(slots=True)
class PersonalizedRecommendation:
    product: Product
    score: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        payload = _serialize_product(self.product)
        payload["score"] = float(self.score)
        payload["reason"] = self.reason
        return payload


@dataclass(slots=True)
class OutfitRecommendation:
    category: str
    product: Product
    score: float

    def as_dict(self) -> dict[str, Any]:
        payload = _serialize_product(self.product)
        payload["score"] = float(self.score)
        return payload


@dataclass(slots=True)
class RecommendationPayload:
    personalized: List[PersonalizedRecommendation]
    outfit: Dict[str, List[OutfitRecommendation]]
    outfit_complete_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "personalized": [item.as_dict() for item in self.personalized],
            "outfit": {
                category: [entry.as_dict() for entry in entries] for category, entries in self.outfit.items()
            },
            "outfit_complete_score": round(self.outfit_complete_score, 4),
        }

