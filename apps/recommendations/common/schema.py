from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from bson import ObjectId
    from apps.products.mongo_models import Product as MongoProduct, ProductVariant
    MONGO_AVAILABLE = True
except Exception:
    MONGO_AVAILABLE = False
    ObjectId = None
    MongoProduct = None
    ProductVariant = None

_mongo_checked = False
_mongo_actually_available = None

def _check_mongo_available() -> bool:
    global _mongo_checked, _mongo_actually_available
    if _mongo_checked:
        return _mongo_actually_available

    if not MONGO_AVAILABLE or not MongoProduct:
        _mongo_checked = True
        _mongo_actually_available = False
        return False

    try:
        _ = MongoProduct.objects.first()
        _mongo_checked = True
        _mongo_actually_available = True
        return True
    except Exception:
        _mongo_checked = True
        _mongo_actually_available = False
        return False

def _get_mongo_product_id(product: Product) -> str | None:
    if not _check_mongo_available() or not MongoProduct:
        return None

    slug = getattr(product, "slug", None)
    if slug:
        try:
            mongo_product = MongoProduct.objects(slug=slug).first()
            if mongo_product:
                return str(mongo_product.id)
        except Exception:
            pass

    return None

def _get_mongo_product_data(product: Product) -> dict[str, Any] | None:
    if not _check_mongo_available() or not MongoProduct:
        return None

    mongo_product = None

    slug = getattr(product, "slug", None)
    if slug:
        try:
            mongo_product = MongoProduct.objects(slug=slug).first()
            if mongo_product:
                return {
                    "id": str(mongo_product.id),
                    "name": mongo_product.name,
                    "images": list(mongo_product.images) if mongo_product.images else [],
                    "color_ids": [str(cid) for cid in mongo_product.color_ids] if mongo_product.color_ids else [],
                    "price": float(mongo_product.price) if mongo_product.price else None,
                }
        except Exception:
            pass

    product_name = _get_product_name(product)
    price = _get_product_price(product)
    if product_name and price is not None:
        try:
            mongo_products = MongoProduct.objects(name=product_name)
            for mp in mongo_products:
                mp_price = getattr(mp, "price", None)
                if mp_price is not None and abs(float(mp_price) - float(price)) < 0.01:
                    return {
                        "id": str(mp.id),
                        "name": mp.name,
                        "images": list(mp.images) if mp.images else [],
                        "color_ids": [str(cid) for cid in mp.color_ids] if mp.color_ids else [],
                        "price": float(mp.price) if mp.price else None,
                    }
        except Exception:
            pass

    if product_name:
        try:
            mongo_product = MongoProduct.objects(name=product_name).first()
            if mongo_product:
                return {
                    "id": str(mongo_product.id),
                    "name": mongo_product.name,
                    "images": list(mongo_product.images) if mongo_product.images else [],
                    "color_ids": [str(cid) for cid in mongo_product.color_ids] if mongo_product.color_ids else [],
                    "price": float(mongo_product.price) if mongo_product.price else None,
                }
        except Exception:
            pass

    product_category = _get_category_type(product)
    product_gender = getattr(product, "gender", None)
    product_age_group = _get_age_group(product)
    if product_category and product_gender and product_age_group and price is not None:
        try:
            mongo_products = MongoProduct.objects(
                category_type=product_category,
                gender=product_gender,
                age_group=product_age_group,
            )
            best_match = None
            min_price_diff = float('inf')
            for mp in mongo_products:
                if mp.price:
                    price_diff = abs(float(mp.price) - float(price))
                    if price_diff < min_price_diff:
                        min_price_diff = price_diff
                        best_match = mp
            if best_match and min_price_diff < 1000:
                return {
                    "id": str(best_match.id),
                    "name": best_match.name,
                    "images": list(best_match.images) if best_match.images else [],
                    "color_ids": [str(cid) for cid in best_match.color_ids] if best_match.color_ids else [],
                    "price": float(best_match.price) if best_match.price else None,
                }
        except Exception:
            pass

    return None

def _serialize_product(product: MongoProduct, original_mongo_id: str | None = None) -> dict[str, Any]:
    product_id = getattr(product, "id", None)

    variants = []
    if ProductVariant is not None:
        try:
            product_variants = ProductVariant.objects(product_id=product_id)
            for variant in product_variants:
                variants.append({
                    "id": str(variant.id),
                    "color": variant.color,
                    "size": variant.size,
                    "price": float(variant.price) if variant.price is not None else None,
                    "stock": variant.stock if variant.stock is not None else 0,
                })
        except Exception:
            variants = []

    return {
        "id": int(product_id) if product_id is not None else None,
        "gender": getattr(product, "gender", None),
        "masterCategory": getattr(product, "masterCategory", None),
        "subCategory": getattr(product, "subCategory", None),
        "articleType": getattr(product, "articleType", None),
        "baseColour": getattr(product, "baseColour", None),
        "season": getattr(product, "season", None),
        "year": getattr(product, "year", None),
        "usage": getattr(product, "usage", None),
        "productDisplayName": getattr(product, "productDisplayName", getattr(product, "name", None)),

        "images": list(getattr(product, "images", [])) or [],

        "rating": float(getattr(product, "rating", 0.0)) if getattr(product, "rating", None) is not None else None,
        "sale": float(getattr(product, "sale", 0.0)) if getattr(product, "sale", None) is not None else None,

        "reviews": [],
        "variants": variants,

        "created_at": getattr(product, "created_at", None).isoformat() if getattr(product, "created_at", None) else None,
        "updated_at": getattr(product, "updated_at", None).isoformat() if getattr(product, "updated_at", None) else None,
    }

@dataclass(slots=True)
class PersonalizedRecommendation:
    product: MongoProduct
    score: float
    reason: str
    context: Any = None

    def as_dict(self) -> dict[str, Any]:
        mongo_id = None
        if self.context and hasattr(self.context, 'product_id_to_mongo_id'):
            mongo_id = self.context.product_id_to_mongo_id.get(self.product.id)
        return {
            "score": float(self.score),
            "reason": self.reason,
            "product": _serialize_product(self.product, original_mongo_id=mongo_id),
        }

@dataclass(slots=True)
class OutfitRecommendation:
    category: str
    product: MongoProduct
    score: float
    reason: str = ""
    context: Any = None

    def as_dict(self) -> dict[str, Any]:
        mongo_id = None
        if self.context and hasattr(self.context, 'product_id_to_mongo_id'):
            mongo_id = self.context.product_id_to_mongo_id.get(self.product.id)
        return {
            "score": float(self.score),
            "reason": self.reason,
            "product": _serialize_product(self.product, original_mongo_id=mongo_id),
        }

@dataclass(slots=True)
class RecommendationPayload:
    personalized: List[PersonalizedRecommendation]
    outfit: Dict[str, Any]
    outfit_complete_score: float

    def as_dict(self) -> dict[str, Any]:
        outfit_dict = {}
        for category, entry in self.outfit.items():
            if isinstance(entry, dict) and "items" in entry:
                outfit_items = {}
                for item_key, outfit_rec in entry["items"].items():
                    if hasattr(outfit_rec, 'as_dict'):
                        outfit_items[item_key] = outfit_rec.as_dict()
                    else:
                        outfit_items[item_key] = outfit_rec
                outfit_dict[category] = outfit_items
            elif isinstance(entry, list):
                outfit_dict[category] = entry[0].as_dict() if entry else {}
            elif hasattr(entry, 'as_dict'):
                outfit_dict[category] = entry.as_dict()
            else:
                outfit_dict[category] = entry
        return {
            "personalized": [item.as_dict() for item in self.personalized],
            "outfit": outfit_dict,
            "outfit_complete_score": round(self.outfit_complete_score, 4),
        }

