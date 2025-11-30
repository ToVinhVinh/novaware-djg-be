from __future__ import annotations

from typing import Any

from bson import ObjectId

from apps.recommendations.gnn import mongo_engine as gnn_mongo

def recommend_cbf_mongo(
    *,
    user_id: str | ObjectId,
    current_product_id: str | ObjectId,
    top_k_personal: int = 5,
    top_k_outfit: int = 4,
) -> dict[str, Any]:
    context = gnn_mongo._build_context(
        user_id=user_id,
        current_product_id=current_product_id,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
    )

    artifacts = gnn_mongo.train_gnn_mongo()
    frequency: dict[str, float] = artifacts.get("product_frequency", {})

    scores: list[tuple[Any, float]] = []
    for candidate in context.candidate_products:
        cid = str(candidate.id)
        if not cid:
            continue
        style_score = sum(context.style_weights.get(token, 0.0) for token in gnn_mongo._style_tokens(candidate))
        color_score = sum(
            context.color_weights.get(color, 0.0)
            for color in gnn_mongo._product_color_tokens(candidate, context.color_cache)
        )
        brand_score = 0.0
        if getattr(candidate, "brand_id", None):
            brand_score = context.brand_weights.get(str(candidate.brand_id), 0.0) * 0.5
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
        score = style_score + color_score + brand_score + alignment + popularity
        scores.append((candidate, score))

    if not scores:
        return {
            "personalized": [],
            "outfit": {},
            "outfit_complete_score": 0.0,
        }

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    top_personal = scores_sorted[:top_k_personal]

    def as_item(candidate, score: float) -> dict[str, Any]:
        cid = str(candidate.id)
        style_tokens = list(gnn_mongo._style_tokens(candidate))
        product_colors = gnn_mongo._product_color_tokens(candidate, context.color_cache)
        reason = gnn_mongo._compose_reason(
            context=context,
            product=candidate,
            cand_id=cid,
            history_ids=[str(it.product_id) for it in context.interactions if it.product_id],
            graph={},
            style_tokens=style_tokens,
            frequency=frequency,
            brand_id=str(getattr(candidate, "brand_id")) if getattr(candidate, "brand_id", None) else None,
            product_colors=product_colors,
        )
        reason += "; content-only scoring"
        return {
            "score": round(float(score), 4),
            "reason": reason,
            "product": gnn_mongo._as_product_object(candidate, color_cache=context.color_cache),
        }

    personalized = [as_item(candidate, score) for candidate, score in top_personal]

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

    current_product = context.current_product
    current_category = getattr(current_product, "category_type", None)
    if current_category and current_category.lower() in [cat.lower() for cat in required_categories]:
        current_category_lower = current_category.lower()
        matching_key = next((cat for cat in required_categories if cat.lower() == current_category_lower), None)
        if matching_key:
            current_id = str(current_product.id)
            style_score = sum(context.style_weights.get(token, 0.0) for token in gnn_mongo._style_tokens(current_product))
            color_score = sum(
                context.color_weights.get(color, 0.0)
                for color in gnn_mongo._product_color_tokens(current_product, context.color_cache)
            )
            brand_score = 0.0
            if getattr(current_product, "brand_id", None):
                brand_score = context.brand_weights.get(str(current_product.brand_id), 0.0) * 0.5
            alignment = 0.0
            user_gender = (getattr(context.user, "gender", "") or "").lower()
            candidate_gender = (getattr(current_product, "gender", "") or "").lower()
            if user_gender and (candidate_gender == user_gender or candidate_gender == "unisex"):
                alignment += 0.5
            user_age_group = (getattr(context.current_product, "age_group", "") or "").lower()
            candidate_age_group = (getattr(current_product, "age_group", "") or "").lower()
            if user_age_group and candidate_age_group == user_age_group:
                alignment += 0.5
            popularity = 0.1 * frequency.get(current_id, 0.0)
            current_score = style_score + color_score + brand_score + alignment + popularity
            outfit[matching_key] = as_item(current_product, current_score)

    from collections import defaultdict as _defaultdict

    by_category: dict[str, list[tuple[Any, float]]] = _defaultdict(list)
    for candidate, score in scores_sorted:
        key = category_key_for_user(getattr(candidate, "category_type", None))
        if not key:
            continue
        by_category[key].append((candidate, score))

    for key in required_categories:
        if outfit.get(key) is not None:
            continue
        entries = by_category.get(key, [])
        if entries:
            candidate, score = sorted(entries, key=lambda tup: tup[1], reverse=True)[0]
            outfit[key] = as_item(candidate, score)
            continue
        query = gnn_mongo.MongoProduct.objects(category_type=key)
        if context.excluded_product_ids:
            query = query.filter(__raw__={"_id": {"$nin": list(context.excluded_product_ids)}})
        fallback_candidates = list(query.limit(100))
        if fallback_candidates:
            fallback_sorted = sorted(fallback_candidates, key=lambda prod: frequency.get(str(prod.id), 0.0), reverse=True)
            best = fallback_sorted[0]
            fallback_score = frequency.get(str(best.id), 0.0)
            outfit[key] = as_item(best, fallback_score)

    used_ids = {
        slot["product"]["id"] for slot in outfit.values() if slot and slot.get("product")
    }
    for key in required_categories:
        if outfit.get(key) is not None:
            continue
        product_found = False
        for candidate, score in scores_sorted:
            cid = str(candidate.id)
            if cid in used_ids:
                continue
            candidate_category = getattr(candidate, "category_type", None)
            if candidate_category and candidate_category.lower() == key.lower():
                item = as_item(candidate, score)
                item["reason"] += "; fallback to complete outfit"
                outfit[key] = item
                used_ids.add(cid)
                product_found = True
                break

        if not product_found:
            query = gnn_mongo.MongoProduct.objects(category_type=key)
            if context.excluded_product_ids:
                query = query.filter(__raw__={"_id": {"$nin": list(context.excluded_product_ids)}})
            if used_ids:
                used_str_ids = [str(uid) for uid in used_ids if uid]
                valid_used_ids = [ObjectId(uid) for uid in used_str_ids if ObjectId.is_valid(uid)]
                if valid_used_ids:
                    query = query.filter(__raw__={"_id": {"$nin": valid_used_ids}})
            fallback_candidate = query.first()
            if fallback_candidate:
                fallback_id = str(fallback_candidate.id)
                fallback_score = frequency.get(fallback_id, 0.0)
                outfit[key] = as_item(fallback_candidate, fallback_score)
                used_ids.add(fallback_id)

    filled = sum(1 for slot in outfit.values() if slot is not None)
    completeness = round(filled / len(required_categories), 3) if required_categories else 0.0

    return {
        "personalized": personalized,
        "outfit": outfit,
        "outfit_complete_score": completeness,
    }

