"""Mongo-native GNN-inspired recommendation engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from bson import ObjectId

from apps.products.mongo_models import Product as MongoProduct
from apps.users.mongo_models import User as MongoUser, UserInteraction as MongoInteraction


INTERACTION_WEIGHTS: dict[str, float] = {
    "view": 0.5,
    "like": 1.0,
    "cart": 1.5,
    "review": 1.2,
    "purchase": 3.0,
}


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
        if getattr(prod, "brand_id", None):
            brand_weights[str(prod.brand_id)] += weight

    return MongoRecommendationContext(
        user=user,
        current_product=current_product,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
        interactions=interactions,
        candidate_products=candidate_products,
        brand_weights=dict(brand_weights),
        style_weights=dict(style_weights),
    )


def train_gnn_mongo() -> dict[str, Any]:
    """Build a simple co-occurrence graph from Mongo interactions."""
    co_occurrence: dict[str, dict[str, float]] = defaultdict(dict)
    product_frequency: dict[str, float] = defaultdict(float)

    interactions = MongoInteraction.objects().order_by("+user_id", "+timestamp")
    user_histories: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for it in interactions:
        if not it.product_id:
            continue
        product_id = str(it.product_id)
        weight = INTERACTION_WEIGHTS.get(it.interaction_type, 1.0)
        history = user_histories[str(it.user_id)]
        for other_id, other_weight in history:
            total = weight + other_weight
            co_occurrence[product_id][other_id] = co_occurrence[product_id].get(other_id, 0.0) + total
            co_occurrence[other_id][product_id] = co_occurrence[other_id].get(product_id, 0.0) + total
        history.append((product_id, weight))
        product_frequency[product_id] += weight

    return {
        "co_occurrence": {pid: dict(nei) for pid, nei in co_occurrence.items()},
        "product_frequency": dict(product_frequency),
    }


def _as_product_object(prod: MongoProduct) -> dict[str, Any]:
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
        "amazon_asin": getattr(prod, "amazon_asin", None),
    }


def _compose_reason(
    *,
    cand_id: str,
    history_ids: list[str],
    graph: dict[str, dict[str, float]],
    style_tokens: Iterable[str],
    style_weights: dict[str, float],
    frequency: dict[str, float],
    brand_id: str | None,
    brand_weights: dict[str, float],
) -> str:
    parts: list[str] = []
    # Graph overlap
    neighbour_scores = graph.get(cand_id, {})
    co_count = sum(1 for hid in history_ids if neighbour_scores.get(hid, 0.0) > 0)
    if co_count > 0:
        parts.append(f"co-occurs with your history ({co_count} items)")
    # Style match
    style_contribs = [(t, style_weights.get(t, 0.0)) for t in style_tokens]
    top_style = [t for t, w in sorted(style_contribs, key=lambda x: x[1], reverse=True) if w > 0][:3]
    if top_style:
        parts.append("matches your style: " + ", ".join(top_style))
    # Popularity
    pop = frequency.get(cand_id, 0.0)
    if pop > 0:
        parts.append("popular with users")
    # Brand affinity
    if brand_id and brand_weights.get(brand_id, 0.0) > 0:
        parts.append("aligns with your preferred brand")
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
    # Build artifacts on the fly (could be cached in Mongo later)
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

    scores.sort(key=lambda x: x[1], reverse=True)
    top_personal = scores[:top_k_personal]

    def as_item(prod: MongoProduct, score: float) -> dict[str, Any]:
        cid = str(prod.id)
        reason = _compose_reason(
            cand_id=cid,
            history_ids=history_ids,
            graph=graph,
            style_tokens=_style_tokens(prod),
            style_weights=context.style_weights,
            frequency=frequency,
            brand_id=str(getattr(prod, "brand_id")) if getattr(prod, "brand_id", None) else None,
            brand_weights=context.brand_weights,
        )
        return {
            "score": round(float(score), 4),
            "reason": reason,
            "product": _as_product_object(prod),
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
    score_map = {str(p.id): s for p, s in scores}
    by_category: dict[str, list[MongoProduct]] = defaultdict(list)
    for prod, _ in scores:
        key = category_key_for_user(getattr(prod, "category_type", None))
        if not key:
            continue
        by_category[key].append(prod)

    # Pick the best single item per category
    for key in required_categories:
        products = by_category.get(key, [])
        if not products:
            continue
        products_sorted = sorted(products, key=lambda p: score_map.get(str(p.id), 0.0), reverse=True)
        best = products_sorted[0]
        outfit[key] = as_item(best, score_map.get(str(best.id), 0.0))

    # Compute completeness score: fraction of categories filled
    filled = sum(1 for k in required_categories if outfit.get(k) is not None)
    completeness = round(filled / len(required_categories), 3) if required_categories else 0.0

    payload = {
        "personalized": personalized,
        "outfit": outfit,
        "outfit_complete_score": completeness,
    }
    return payload


