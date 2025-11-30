from __future__ import annotations

from datetime import timedelta

INTERACTION_WEIGHTS: dict[str, float] = {
    "purchase": 5.0,
    "cart": 3.5,
    "like": 2.5,
    "review": 2.0,
    "view": 1.0,
}

OUTFIT_COMPLETION_RULES: dict[str, list[str]] = {
    "tops": ["bottoms", "shoes", "accessories"],
    "bottoms": ["tops", "shoes", "accessories"],
    "dresses": ["shoes", "accessories"],
    "shoes": ["tops", "bottoms", "accessories"],
    "accessories": ["tops", "bottoms"],
}

DEFAULT_STYLE_WEIGHT: float = 1.0
FRESHNESS_DECAY_HALFLIFE = timedelta(days=30)
OUTFIT_SCORE_FLOOR: float = 0.25
MAX_STYLE_TAGS: int = 20

