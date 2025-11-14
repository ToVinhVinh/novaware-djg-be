"""Mongo-native Hybrid recommendation engine blending graph and content signals."""

from __future__ import annotations

from typing import Any

from bson import ObjectId

from apps.recommendations.gnn import mongo_engine as gnn_mongo

# Re-export for clarity
MongoRecommendationContext = gnn_mongo.MongoRecommendationContext


def _build_context(*, user_id: str | ObjectId, current_product_id: str | ObjectId, top_k_personal: int, top_k_outfit: int) -> MongoRecommendationContext:
    return gnn_mongo._build_context(  # type: ignore[attr-defined]
        user_id=user_id,
        current_product_id=current_product_id,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
    )


def train_hybrid_mongo() -> dict[str, Any]:
    """Reuse the GNN artifact builder (co-occurrence + popularity)."""
    return gnn_mongo.train_gnn_mongo()


def _graph_score(
    *,
    candidate,
    history_ids: list[str],
    current_neighbors: dict[str, float],
    graph: dict[str, dict[str, float]],
) -> float:
    cid = str(candidate.id)
    score = 0.0
    cand_neighbors = graph.get(cid, {})
    for hid in history_ids:
        score += cand_neighbors.get(hid, 0.0)
    score += current_neighbors.get(cid, 0.0) * 1.2
    return score


def _content_score(context: MongoRecommendationContext, candidate, *, frequency: dict[str, float]) -> float:
    cid = str(candidate.id)
    style_tokens = gnn_mongo._style_tokens(candidate)  # type: ignore[attr-defined]
    style_score = sum(context.style_weights.get(token, 0.0) for token in style_tokens)
    color_tokens = gnn_mongo._product_color_tokens(candidate, context.color_cache)  # type: ignore[attr-defined]
    color_score = sum(context.color_weights.get(color, 0.0) for color in color_tokens)
    brand_score = 0.0
    if getattr(candidate, "brand_id", None):
        brand_score = context.brand_weights.get(str(candidate.brand_id), 0.0) * 0.5
    # Age / gender alignment
    alignment = 0.0
    user_gender = (getattr(context.user, "gender", "") or "").lower()
    candidate_gender = (getattr(candidate, "gender", "") or "").lower()
    if user_gender and (candidate_gender == user_gender or candidate_gender == "unisex"):
        alignment += 0.5
    user_age_group = (getattr(context.current_product, "age_group", "") or "").lower()
    candidate_age_group = (getattr(candidate, "age_group", "") or "").lower()
    if user_age_group and candidate_age_group == user_age_group:
        alignment += 0.5
    popularity = 0.1 * frequency.get(cid, 0.0)
    return style_score + color_score + brand_score + alignment + popularity


def recommend_hybrid_mongo(
    *,
    user_id: str | ObjectId,
    current_product_id: str | ObjectId,
    top_k_personal: int = 5,
    top_k_outfit: int = 4,
    alpha: float = 0.6,
) -> dict[str, Any]:
    artifacts = train_hybrid_mongo()
    context = _build_context(
        user_id=user_id,
        current_product_id=current_product_id,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
    )

    graph: dict[str, dict[str, float]] = artifacts.get("co_occurrence", {})
    frequency: dict[str, float] = artifacts.get("product_frequency", {})
    current_neighbors = graph.get(str(context.current_product.id), {})

    history_ids = [str(it.product_id) for it in context.interactions if it.product_id]

    scores: list[tuple[Any, float, float, float]] = []
    for candidate in context.candidate_products:
        cid = str(candidate.id)
        if not cid:
            continue
        g_score = _graph_score(
            candidate=candidate,
            history_ids=history_ids,
            current_neighbors=current_neighbors,
            graph=graph,
        )
        c_score = _content_score(context, candidate, frequency=frequency)
        blended = alpha * g_score + (1 - alpha) * c_score
        scores.append((candidate, blended, g_score, c_score))

    # Fallback using content score if no blended scores generated
    if not scores:
        for candidate in context.candidate_products:
            cid = str(candidate.id)
            if not cid:
                continue
            c_score = _content_score(context, candidate, frequency=frequency)
            scores.append((candidate, c_score, 0.0, c_score))

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    top_personal = scores_sorted[:top_k_personal]

    def as_item(candidate, blended_score: float, g_score: float, c_score: float) -> dict[str, Any]:
        cid = str(candidate.id)
        style_tokens = list(gnn_mongo._style_tokens(candidate))  # type: ignore[attr-defined]
        product_colors = gnn_mongo._product_color_tokens(candidate, context.color_cache)  # type: ignore[attr-defined]
        reason = gnn_mongo._compose_reason(  # type: ignore[attr-defined]
            context=context,
            product=candidate,
            cand_id=cid,
            history_ids=history_ids,
            graph=graph,
            style_tokens=style_tokens,
            frequency=frequency,
            brand_id=str(getattr(candidate, "brand_id")) if getattr(candidate, "brand_id", None) else None,
            product_colors=product_colors,
        )
        reason = f"{reason}; hybrid blend {alpha:.2f} graph / {1 - alpha:.2f} content (G={round(g_score, 2)}, C={round(c_score, 2)})"
        return {
            "score": round(float(blended_score), 4),
            "reason": reason,
            "product": gnn_mongo._as_product_object(candidate, color_cache=context.color_cache),  # type: ignore[attr-defined]
        }

    personalized = [as_item(c, s, g, c_score) for c, s, g, c_score in top_personal]

    # Outfit construction
    def category_key_for_user(cat: str) -> str | None:
        cat = (cat or "").lower()
        if cat not in {"accessories", "bottoms", "dresses", "shoes", "tops"}:
            return None
        if cat == "dresses":
            gender = (getattr(context.user, "gender", "") or "").lower()
            if gender != "female":
                return None
        return cat

    required_categories = ["accessories", "bottoms", "shoes", "tops"]
    include_dresses = (getattr(context.user, "gender", "") or "").lower() == "female"
    if include_dresses:
        required_categories.insert(2, "dresses")

    outfit: dict[str, dict[str, Any] | None] = {k: None for k in required_categories}
    score_map = {str(candidate.id): blended for candidate, blended, _, _ in scores_sorted}
    
    # Always include the current product in its category
    current_product = context.current_product
    current_category = getattr(current_product, "category_type", None)
    if current_category and current_category.lower() in [cat.lower() for cat in required_categories]:
        current_category_lower = current_category.lower()
        # Find the exact key in required_categories (case-insensitive match)
        matching_key = next((cat for cat in required_categories if cat.lower() == current_category_lower), None)
        if matching_key:
            current_id = str(current_product.id)
            current_blended = score_map.get(current_id, _content_score(context, current_product, frequency=frequency))
            current_g_score = 0.0
            current_c_score = current_blended
            # Try to get graph score if available
            current_neighbors = graph.get(current_id, {})
            for hid in history_ids:
                current_g_score += current_neighbors.get(hid, 0.0)
            current_g_score += current_neighbors.get(current_id, 0.0) * 1.2
            current_blended = alpha * current_g_score + (1 - alpha) * current_c_score
            outfit[matching_key] = as_item(current_product, current_blended, current_g_score, current_c_score)

    from collections import defaultdict as _defaultdict  # local import to avoid repetition

    by_category: dict[str, list[Any]] = _defaultdict(list)
    for candidate, blended, g_score, c_score in scores_sorted:
        candidate_category = getattr(candidate, "category_type", None)
        if not candidate_category:
            continue
        # Normalize category for comparison
        candidate_category_normalized = str(candidate_category).lower().strip()
        key = category_key_for_user(candidate_category_normalized)
        if not key:
            continue
        # Double-check: ensure the normalized category matches the key
        if candidate_category_normalized != key.lower():
            continue
        by_category[key].append((candidate, blended, g_score, c_score))

    for key in required_categories:
        # Skip if current product is already in this category
        if outfit.get(key) is not None:
            continue
        entries = by_category.get(key, [])
        if entries:
            candidate, blended, g_score, c_score = sorted(entries, key=lambda tup: tup[1], reverse=True)[0]
            outfit[key] = as_item(candidate, blended, g_score, c_score)
            continue
        # fallback within category via popularity
        query = gnn_mongo.MongoProduct.objects(category_type=key)  # type: ignore[attr-defined]
        if context.excluded_product_ids:
            query = query.filter(__raw__={"_id": {"$nin": list(context.excluded_product_ids)}})
        fallback_candidates = list(query.limit(100))
        if fallback_candidates:
            fallback_sorted = sorted(
                fallback_candidates,
                key=lambda prod: frequency.get(str(prod.id), 0.0),
                reverse=True,
            )
            best = fallback_sorted[0]
            fallback_score = score_map.get(str(best.id), _content_score(context, best, frequency=frequency))
            outfit[key] = as_item(best, fallback_score, 0.0, fallback_score)

    used_ids = {
        slot["product"]["id"] for slot in outfit.values() if slot and slot.get("product")
    }
    for key in required_categories:
        if outfit.get(key) is not None:
            continue
        # Try to find a product matching the category from scored candidates
        product_found = False
        for candidate, blended, g_score, c_score in scores_sorted:
            cid = str(candidate.id)
            if cid in used_ids:
                continue
            # Only use products that match the category (strict case-insensitive comparison)
            candidate_category = getattr(candidate, "category_type", None)
            if not candidate_category:
                continue
            candidate_category_normalized = str(candidate_category).lower().strip()
            key_normalized = str(key).lower().strip()
            if candidate_category_normalized == key_normalized:
                item = as_item(candidate, blended, g_score, c_score)
                item["reason"] += "; fallback to complete outfit"
                outfit[key] = item
                used_ids.add(cid)
                product_found = True
                break
        
        # If still no product found, try querying MongoDB for products with correct category
        if not product_found:
            # Use case-insensitive query - MongoDB doesn't support case-insensitive directly in this ORM
            # So we'll need to query and filter in Python
            query = gnn_mongo.MongoProduct.objects  # type: ignore[attr-defined]
            if context.excluded_product_ids:
                excluded_str_ids = [str(eid) for eid in context.excluded_product_ids]
                query = query.filter(__raw__={"_id": {"$nin": [ObjectId(eid) for eid in excluded_str_ids if ObjectId.is_valid(eid)]}})
            # Also exclude already used products
            if used_ids:
                used_str_ids = [str(uid) for uid in used_ids if uid]
                valid_used_ids = [ObjectId(uid) for uid in used_str_ids if ObjectId.is_valid(uid)]
                if valid_used_ids:
                    query = query.filter(__raw__={"_id": {"$nin": valid_used_ids}})
            # Filter by category case-insensitively
            key_normalized = str(key).lower().strip()
            for candidate in query.limit(500):  # Limit to avoid loading too many
                candidate_category = getattr(candidate, "category_type", None)
                if candidate_category and str(candidate_category).lower().strip() == key_normalized:
                    fallback_id = str(candidate.id)
                    fallback_score = score_map.get(fallback_id, _content_score(context, candidate, frequency=frequency))
                    outfit[key] = as_item(candidate, fallback_score, 0.0, fallback_score)
                    used_ids.add(fallback_id)
                    break

    filled = sum(1 for slot in outfit.values() if slot is not None)
    completeness = round(filled / len(required_categories), 3) if required_categories else 0.0

    return {
        "personalized": personalized,
        "outfit": outfit,
        "outfit_complete_score": completeness,
    }


