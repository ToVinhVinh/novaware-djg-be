"""Mongo-native GNN-inspired recommendation engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import time
from typing import Any, Iterable

from bson import ObjectId

from apps.products.mongo_models import (
    Color as MongoColor,
    Product as MongoProduct,
)
from apps.recommendations.common.gender_utils import normalize_gender
from apps.users.mongo_models import User as MongoUser, UserInteraction as MongoInteraction


INTERACTION_WEIGHTS: dict[str, float] = {
    "view": 0.5,
    "like": 1.0,
    "cart": 1.5,
    "review": 1.2,
    "purchase": 3.0,
}

_COLOR_NAME_CACHE: dict[str, str] = {}

_GNN_ARTIFACTS_CACHE: dict[str, Any] | None = None
_CACHE_TTL_SECONDS = 3600  # 1 hour cache TTL

# Only process interactions from the last N days to improve performance
_INTERACTION_LOOKBACK_DAYS = 90  # Process last 90 days of interactions only


def _style_tokens(product: MongoProduct) -> Iterable[str]:
    tokens: list[str] = []
    if isinstance(getattr(product, "style_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.style_tags if tag)
    if isinstance(getattr(product, "outfit_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.outfit_tags if tag)
    if getattr(product, "category_type", None):
        tokens.append(str(product.category_type).lower())
    if getattr(product, "category_id", None):
        # We don't dereference category here; name may not be available in product doc
        tokens.append("category")
    return tokens


def _product_color_tokens(product: MongoProduct, cache: dict[str, list[str]]) -> list[str]:
    product_id = str(product.id)
    if product_id in cache:
        return cache[product_id]
    colors: list[str] = []
    color_ids = getattr(product, "color_ids", []) or []
    for cid in color_ids:
        key = str(cid)
        name = _COLOR_NAME_CACHE.get(key)
        if name is None:
            color = MongoColor.objects(id=cid).first()
            name = (color.name.lower() if color and getattr(color, "name", None) else "")
            _COLOR_NAME_CACHE[key] = name
        if name:
            colors.append(name)
    cache[product_id] = colors
    return colors


@dataclass
class MongoRecommendationContext:
    user: MongoUser
    current_product: MongoProduct
    top_k_personal: int
    top_k_outfit: int
    interactions: list[MongoInteraction]
    candidate_products: list[MongoProduct]
    brand_weights: dict[str, float]
    style_weights: dict[str, float]
    color_weights: dict[str, float]
    excluded_product_ids: set[ObjectId]
    color_cache: dict[str, list[str]]


def _resolve_user(user_id: str | ObjectId) -> MongoUser:
    oid = ObjectId(str(user_id))
    user = MongoUser.objects(id=oid).first()
    if not user:
        raise ValueError("Mongo user not found")
    return user


def _resolve_product(product_id: str | ObjectId) -> MongoProduct:
    oid = ObjectId(str(product_id))
    product = MongoProduct.objects(id=oid).first()
    if not product:
        raise ValueError("Mongo product not found")
    return product


def _load_interactions(user: MongoUser) -> list[MongoInteraction]:
    return list(MongoInteraction.objects(user_id=user.id).order_by("+timestamp"))


def _build_context(
    *,
    user_id: str | ObjectId,
    current_product_id: str | ObjectId, 
    top_k_personal: int,
    top_k_outfit: int,
) -> MongoRecommendationContext:
    user = _resolve_user(user_id)
    current_product = _resolve_product(current_product_id)
    interactions = _load_interactions(user)
    color_cache: dict[str, list[str]] = {}

    # Exclude current and history products
    excluded_ids: set[ObjectId] = {current_product.id}
    for it in interactions:
        excluded_ids.add(it.product_id)

    # Simple candidate pool: same gender/age when present, otherwise any
    filters: dict[str, Any] = {}
    if getattr(current_product, "gender", None):
        filters["gender"] = current_product.gender
    if getattr(current_product, "age_group", None):
        filters["age_group"] = current_product.age_group

    qs = MongoProduct.objects(**filters)
    if excluded_ids:
        qs = qs.filter(__raw__={"_id": {"$nin": list(excluded_ids)}})
    candidate_products = list(qs)
    # Fallback if empty: drop gender/age constraints, use all except excluded
    if not candidate_products:
        qs_all = MongoProduct.objects
        if excluded_ids:
            qs_all = qs_all.filter(__raw__={"_id": {"$nin": list(excluded_ids)}})
        candidate_products = list(qs_all.limit(500))

    # Build style and brand preferences from interactions
    style_weights: dict[str, float] = defaultdict(float)
    brand_weights: dict[str, float] = defaultdict(float)
    color_weights: dict[str, float] = defaultdict(float)
    product_cache: dict[ObjectId, MongoProduct] = {}

    for it in interactions:
        weight = INTERACTION_WEIGHTS.get(it.interaction_type, 1.0)
        if not it.product_id:
            continue
        prod = product_cache.get(it.product_id)
        if prod is None:
            prod = MongoProduct.objects(id=it.product_id).first()
            if prod:
                product_cache[it.product_id] = prod
        if not prod:
            continue
        for token in _style_tokens(prod):
            style_weights[token] += weight
        for color_token in _product_color_tokens(prod, color_cache):
            color_weights[color_token] += weight
        if getattr(prod, "brand_id", None):
            brand_weights[str(prod.brand_id)] += weight

    # Also leverage embedded interaction_history on user if available
    embedded_history = getattr(user, "interaction_history", []) or []
    for record in embedded_history:
        weight = float(record.get("weight", 1.0)) if isinstance(record, dict) else 1.0
        if isinstance(record, dict):
            for token in record.get("style_tags") or record.get("styleTags") or []:
                if token:
                    style_weights[str(token).lower()] += weight
            for color in record.get("colors") or []:
                if color:
                    color_weights[str(color).lower()] += weight
            product_id = record.get("product_id") or record.get("productId")
            if product_id:
                try:
                    prod = product_cache.get(ObjectId(product_id))
                    if prod is None:
                        prod = MongoProduct.objects(id=product_id).first()
                        if prod:
                            product_cache[ObjectId(product_id)] = prod
                    if prod:
                        for token in _style_tokens(prod):
                            style_weights[token] += weight * 0.5
                        for color_token in _product_color_tokens(prod, color_cache):
                            color_weights[color_token] += weight * 0.5
                except Exception:
                    pass

    # Include explicit user color preferences if present
    user_preferences = getattr(user, "preferences", {}) or {}
    preferred_colors = user_preferences.get("colors") or user_preferences.get("preferred_colors") or []
    for color in preferred_colors:
        if color:
            color_weights[str(color).lower()] += 1.5

    preferred_styles = user_preferences.get("styles") or user_preferences.get("style_tags") or []
    for token in preferred_styles:
        if token:
            style_weights[str(token).lower()] += 1.5

    return MongoRecommendationContext(
        user=user,
        current_product=current_product,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
        interactions=interactions,
        candidate_products=candidate_products,
        brand_weights=dict(brand_weights),
        style_weights=dict(style_weights),
        color_weights=dict(color_weights),
        excluded_product_ids=excluded_ids,
        color_cache=color_cache,
    )


def train_gnn_mongo(force_rebuild: bool = False) -> dict[str, Any]:
    """Build a simple co-occurrence graph from Mongo interactions.
    
    Uses in-memory cache to avoid rebuilding the graph on every request.
    Cache is invalidated after TTL expires or if interaction count changes.
    """
    global _GNN_ARTIFACTS_CACHE
    
    current_time = time()
    
    # Check if cache is valid
    if not force_rebuild and _GNN_ARTIFACTS_CACHE is not None:
        cache_age = current_time - _GNN_ARTIFACTS_CACHE.get("timestamp", 0)
        if cache_age < _CACHE_TTL_SECONDS:
            # Check if interaction count has changed (simple invalidation check)
            # Only check count if cache is relatively fresh (< 5 minutes) to avoid frequent DB queries
            if cache_age < 300:  # 5 minutes
                current_count = MongoInteraction.objects(product_id__ne=None).count()
                cached_count = _GNN_ARTIFACTS_CACHE.get("interaction_count", 0)
                if current_count == cached_count:
                    return _GNN_ARTIFACTS_CACHE["artifacts"]
            else:
                # For older cache, just return it without checking count (will refresh on next TTL)
                return _GNN_ARTIFACTS_CACHE["artifacts"]
    
    # Build artifacts
    co_occurrence: dict[str, dict[str, float]] = defaultdict(dict)
    product_frequency: dict[str, float] = defaultdict(float)

    # Limit to recent interactions only for performance
    cutoff_date = datetime.utcnow() - timedelta(days=_INTERACTION_LOOKBACK_DAYS)
    
    # Get total interaction count for cache invalidation (only count valid, recent interactions)
    interaction_count = MongoInteraction.objects(
        product_id__ne=None,
        timestamp__gte=cutoff_date
    ).count()
    
    # Only load recent interactions with product_id and only necessary fields
    # Use only() to limit fields loaded from database
    # Limit to recent interactions to improve performance significantly
    interactions = (
        MongoInteraction.objects(
            product_id__ne=None,
            timestamp__gte=cutoff_date
        )
        .only("user_id", "product_id", "interaction_type")
        .order_by("+user_id", "+timestamp")
    )
    
    # Optimized: Use dict to track last N products per user instead of full history
    # This reduces O(n²) complexity to O(n*k) where k is max history per user
    MAX_HISTORY_PER_USER = 50  # Only keep last 50 products per user
    user_histories: dict[str, list[tuple[str, float]]] = defaultdict(list)

    # Process interactions - optimized algorithm
    for it in interactions:
        if not it.product_id:
            continue
        product_id = str(it.product_id)
        weight = INTERACTION_WEIGHTS.get(it.interaction_type, 1.0)
        user_id_str = str(it.user_id)
        history = user_histories[user_id_str]
        
        # Build co-occurrence with current history (limited size)
        for other_id, other_weight in history:
            total = weight + other_weight
            co_occurrence[product_id][other_id] = co_occurrence[product_id].get(other_id, 0.0) + total
            co_occurrence[other_id][product_id] = co_occurrence[other_id].get(product_id, 0.0) + total
        
        # Add to history, but limit size to avoid memory bloat
        history.append((product_id, weight))
        if len(history) > MAX_HISTORY_PER_USER:
            history.pop(0)  # Remove oldest
        
        product_frequency[product_id] += weight

    artifacts = {
        "co_occurrence": {pid: dict(nei) for pid, nei in co_occurrence.items()},
        "product_frequency": dict(product_frequency),
    }
    
    # Update cache
    _GNN_ARTIFACTS_CACHE = {
        "artifacts": artifacts,
        "timestamp": current_time,
        "interaction_count": interaction_count,
    }
    
    return artifacts


def _as_product_object(prod: MongoProduct, *, color_cache: dict[str, list[str]]) -> dict[str, Any]:
    """Serialize minimal product fields for UI rendering."""
    return {
        "id": str(prod.id),
        "name": getattr(prod, "name", ""),
        "price": float(getattr(prod, "price", 0.0) or 0.0),
        "images": list(getattr(prod, "images", []) or []),
        "gender": getattr(prod, "gender", None),
        "age_group": getattr(prod, "age_group", None),
        "category_type": getattr(prod, "category_type", None),
        "brand_id": str(getattr(prod, "brand_id")) if getattr(prod, "brand_id", None) else None,
        "colors": _product_color_tokens(prod, color_cache),
    }


def _compose_reason(
    *,
    context: MongoRecommendationContext,
    product: MongoProduct,
    cand_id: str,
    history_ids: list[str],
    graph: dict[str, dict[str, float]],
    style_tokens: Iterable[str],
    frequency: dict[str, float],
    brand_id: str | None,
    product_colors: list[str],
) -> str:
    parts: list[str] = []

    user_gender = normalize_gender(getattr(context.user, "gender", ""))
    product_gender = normalize_gender(getattr(product, "gender", ""))
    if user_gender and product_gender:
        if product_gender == user_gender and product_gender in ("male", "female"):
            parts.append(f"fits your {user_gender} profile")
        elif product_gender == "unisex":
            parts.append("works for your gender preference")

    def _resolve_user_age_group() -> str | None:
        age = getattr(context.user, "age", None)
        if age:
            if age <= 12:
                return "kid"
            if age <= 19:
                return "teen"
            return "adult"
        return getattr(context.current_product, "age_group", None)

    user_age_group = (_resolve_user_age_group() or "").lower()
    product_age_group = (getattr(product, "age_group", "") or "").lower()
    if user_age_group and product_age_group == user_age_group:
        parts.append(f"sized for your {user_age_group} age group")

    neighbour_scores = graph.get(cand_id, {})
    co_count = sum(1 for hid in history_ids if neighbour_scores.get(hid, 0.0) > 0)
    if co_count > 0:
        parts.append(f"frequently paired with {co_count} items you've interacted with")

    style_contribs = [
        (t, context.style_weights.get(t, 0.0)) for t in style_tokens
    ]
    top_style = [t for t, w in sorted(style_contribs, key=lambda x: x[1], reverse=True) if w > 0][:3]
    if top_style:
        pretty_styles = ", ".join(t.replace("_", " ") for t in top_style)
        parts.append(f"shares styles you like: {pretty_styles}")

    color_matches = [
        color for color in product_colors if context.color_weights.get(color, 0.0) > 0
    ][:3]
    if color_matches:
        pretty_colors = ", ".join(color.title() for color in color_matches)
        parts.append(f"matches your preferred colors: {pretty_colors}")

    pop = frequency.get(cand_id, 0.0)
    if pop > 0:
        parts.append("popular with similar shoppers")

    if brand_id and context.brand_weights.get(brand_id, 0.0) > 0:
        parts.append("aligns with a brand you've engaged with")

    if not parts:
        return "similar to your preferences"
    return "; ".join(parts)


def recommend_gnn_mongo(
    *,
    user_id: str | ObjectId,
    current_product_id: str | ObjectId,
    top_k_personal: int = 5,
    top_k_outfit: int = 4,
) -> dict[str, Any]:
    # Build artifacts (now with caching for performance)
    artifacts = train_gnn_mongo()
    context = _build_context(
        user_id=user_id,
        current_product_id=current_product_id,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
    )

    graph: dict[str, dict[str, float]] = artifacts.get("co_occurrence", {})
    frequency: dict[str, float] = artifacts.get("product_frequency", {})
    current_neighbors = graph.get(str(context.current_product.id), {})

    scores: list[tuple[MongoProduct, float]] = []
    history_ids = [str(it.product_id) for it in context.interactions if it.product_id]

    for cand in context.candidate_products:
        cid = str(cand.id)
        score = 0.0
        cand_neighbors = graph.get(cid, {})
        for hid in history_ids:
            score += cand_neighbors.get(hid, 0.0)
        score += current_neighbors.get(cid, 0.0) * 1.2
        cand_style_tokens = list(_style_tokens(cand))
        score += sum(context.style_weights.get(t, 0.0) for t in cand_style_tokens)
        score += 0.1 * frequency.get(cid, 1.0)
        # brand weighting
        cand_brand_id = str(getattr(cand, "brand_id")) if getattr(cand, "brand_id", None) else None
        if cand_brand_id:
            score += 0.2 * context.brand_weights.get(cand_brand_id, 0.0)
        scores.append((cand, score))

    # If still empty or all zero, fallback to popularity-only among candidates
    if not scores:
        for cand in context.candidate_products:
            cid = str(cand.id)
            scores.append((cand, frequency.get(cid, 0.0)))
    else:
        # Check if all scores are zero-ish; then use frequency
        if all(s == 0.0 for _, s in scores):
            scores = [(p, frequency.get(str(p.id), 0.0)) for p, _ in scores]

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    top_personal = scores_sorted[:top_k_personal]

    def as_item(prod: MongoProduct, score: float) -> dict[str, Any]:
        cid = str(prod.id)
        style_tokens = list(_style_tokens(prod))
        product_colors = _product_color_tokens(prod, context.color_cache)
        reason = _compose_reason(
            context=context,
            product=prod,
            cand_id=cid,
            history_ids=history_ids,
            graph=graph,
            style_tokens=style_tokens,
            frequency=frequency,
            brand_id=str(getattr(prod, "brand_id")) if getattr(prod, "brand_id", None) else None,
            product_colors=product_colors,
        )
        return {
            "score": round(float(score), 4),
            "reason": reason,
            "product": _as_product_object(prod, color_cache=context.color_cache),
        }

    personalized = [as_item(p, s) for p, s in top_personal]

    # Build simple outfit per category_type: exactly 1 best item per category
    def category_key_for_user(cat: str) -> str | None:
        cat = (cat or "").lower()
        if cat not in {"accessories", "bottoms", "dresses", "shoes", "tops"}:
            return None
        # Dresses only for female users
        if cat == "dresses":
            gender = (getattr(context.user, "gender", "") or "").lower()
            if gender != "female":
                return None
        return cat

    required_categories = ["accessories", "bottoms", "shoes", "tops"]
    include_dresses = (getattr(context.user, "gender", "") or "").lower() == "female"
    if include_dresses:
        required_categories.insert(2, "dresses")  # order: accessories, bottoms, dresses, shoes, tops

    outfit: dict[str, dict[str, Any] | None] = {k: None for k in required_categories}

    # Rank all candidates by score map for outfit selection
    score_map = {str(p.id): s for p, s in scores_sorted}
    
    # Always include the current product in its category
    current_product = context.current_product
    current_category = getattr(current_product, "category_type", None)
    if current_category and current_category.lower() in [cat.lower() for cat in required_categories]:
        current_category_lower = current_category.lower()
        # Find the exact key in required_categories (case-insensitive match)
        matching_key = next((cat for cat in required_categories if cat.lower() == current_category_lower), None)
        if matching_key:
            current_id = str(current_product.id)
            current_score = score_map.get(current_id, 0.0)
            # If not in score_map, calculate it
            if current_id not in score_map:
                cand_neighbors = graph.get(current_id, {})
                for hid in history_ids:
                    current_score += cand_neighbors.get(hid, 0.0)
                current_neighbors_for_current = graph.get(current_id, {})
                current_score += current_neighbors_for_current.get(current_id, 0.0) * 1.2
                cand_style_tokens = list(_style_tokens(current_product))
                current_score += sum(context.style_weights.get(t, 0.0) for t in cand_style_tokens)
                current_score += 0.1 * frequency.get(current_id, 1.0)
                cand_brand_id = str(getattr(current_product, "brand_id")) if getattr(current_product, "brand_id", None) else None
                if cand_brand_id:
                    current_score += 0.2 * context.brand_weights.get(cand_brand_id, 0.0)
            outfit[matching_key] = as_item(current_product, current_score)
    by_category: dict[str, list[MongoProduct]] = defaultdict(list)
    for prod, _ in scores_sorted:
        key = category_key_for_user(getattr(prod, "category_type", None))
        if not key:
            continue
        by_category[key].append(prod)

    # Pick the best single item per category
    for key in required_categories:
        # Skip if current product is already in this category
        if outfit.get(key) is not None:
            continue
        products = by_category.get(key, [])
        if not products:
            # fallback: broaden search across Mongo collection for this category
            query = MongoProduct.objects(category_type=key)
            if context.excluded_product_ids:
                query = query.filter(__raw__={"_id": {"$nin": list(context.excluded_product_ids)}})
            # limit for performance
            fallback_candidates = list(query.limit(100))
            if fallback_candidates:
                fallback_sorted = sorted(
                    fallback_candidates,
                    key=lambda p: frequency.get(str(p.id), 0.0),
                    reverse=True,
                )
                best_fallback = fallback_sorted[0]
                outfit[key] = as_item(
                    best_fallback,
                    score_map.get(str(best_fallback.id), frequency.get(str(best_fallback.id), 0.0)),
                )
            continue
        products_sorted = sorted(products, key=lambda p: score_map.get(str(p.id), 0.0), reverse=True)
        best = products_sorted[0]
        outfit[key] = as_item(best, score_map.get(str(best.id), 0.0))

    # Backfill remaining categories with products matching the category
    used_product_ids = {
        item["product"]["id"] for item in outfit.values() if item and item.get("product")
    }
    for key in required_categories:
        if outfit.get(key) is not None:
            continue
        # Try to find a product matching the category from scored candidates
        product_found = False
        for prod, sc in scores_sorted:
            pid = str(prod.id)
            if pid in used_product_ids:
                continue
            # Only use products that match the category
            prod_category = getattr(prod, "category_type", None)
            if prod_category and prod_category.lower() == key.lower():
                fallback_item = as_item(prod, sc)
                fallback_item["reason"] += "; fallback to complete outfit"
                outfit[key] = fallback_item
                used_product_ids.add(pid)
                product_found = True
                break
        
        # If still no product found, try querying MongoDB for products with correct category
        if not product_found:
            query = MongoProduct.objects(category_type=key)
            if context.excluded_product_ids:
                query = query.filter(__raw__={"_id": {"$nin": list(context.excluded_product_ids)}})
            # Also exclude already used products
            if used_product_ids:
                used_str_ids = [str(uid) for uid in used_product_ids if uid]
                valid_used_ids = [ObjectId(uid) for uid in used_str_ids if ObjectId.is_valid(uid)]
                if valid_used_ids:
                    query = query.filter(__raw__={"_id": {"$nin": valid_used_ids}})
            fallback_prod = query.first()
            if fallback_prod:
                fallback_id = str(fallback_prod.id)
                fallback_score = score_map.get(fallback_id, 0.0)
                fallback_item = as_item(fallback_prod, fallback_score)
                fallback_item["reason"] += "; fallback to complete outfit"
                outfit[key] = fallback_item
                used_product_ids.add(fallback_id)

    # Compute completeness score: fraction of categories filled
    filled = sum(1 for k in required_categories if outfit.get(k) is not None)
    completeness = round(filled / len(required_categories), 3) if required_categories else 0.0

    payload = {
        "personalized": personalized,
        "outfit": outfit,
        "outfit_complete_score": completeness,
    }
    return payload


