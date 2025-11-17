"""Context dataclass for recommendation pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from django.utils import timezone

from apps.products.mongo_models import Product as MongoProduct
from apps.users.mongo_models import User as MongoUser


@dataclass(slots=True)
class RecommendationContext:
    """Runtime context shared across recommendation engines."""

    user: MongoUser
    current_product: MongoProduct
    top_k_personal: int
    top_k_outfit: int
    interactions: list = field(default_factory=list)  # Using generic list for now
    history_products: list[MongoProduct] = field(default_factory=list)
    candidate_products: list[MongoProduct] = field(default_factory=list)
    excluded_product_ids: set[int] = field(default_factory=set)
    style_counter: dict[str, float] = field(default_factory=dict)
    brand_counter: dict[int, float] = field(default_factory=dict)
    interaction_weights: dict[int, float] = field(default_factory=dict)
    resolved_gender: str = "unisex"
    resolved_age_group: str = "adult"
    request_params: dict[str, Any] = field(default_factory=dict)
    prepared_at: datetime = field(default_factory=timezone.now)
    # Mapping from Django product ID to MongoDB ObjectId
    product_id_to_mongo_id: dict[int, str] = field(default_factory=dict)

    def iter_history_ids(self) -> Iterable[int]:
        for product in self.history_products:
            if product.id is not None:
                yield product.id

    @property
    def candidate_map(self) -> dict[Any, MongoProduct]:
        return {candidate.id: candidate for candidate in self.candidate_products if candidate.id is not None}

    def style_weight(self, token: str) -> float:
        return self.style_counter.get(token.lower(), 0.0)

    def brand_weight(self, brand_id: int | None) -> float:
        if brand_id is None:
            return 0.0
        return self.brand_counter.get(brand_id, 0.0)

    def interaction_weight(self, product_id: int | None) -> float:
        if product_id is None:
            return 0.0
        return self.interaction_weights.get(product_id, 0.0)

