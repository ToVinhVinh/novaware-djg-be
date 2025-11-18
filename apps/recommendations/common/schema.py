"""Serializable schema objects for recommendation responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List



try:
    from bson import ObjectId
    from apps.products.mongo_models import Product as MongoProduct
    from apps.brands.mongo_models import Brand as MongoBrand
    MONGO_AVAILABLE = True
except Exception:
    MONGO_AVAILABLE = False
    ObjectId = None
    MongoProduct = None
    MongoBrand = None

# Cache for MongoDB availability check
_mongo_checked = False
_mongo_actually_available = None




def _check_mongo_available() -> bool:
    """Check if MongoDB is actually available (lazy check)."""
    global _mongo_checked, _mongo_actually_available
    if _mongo_checked:
        return _mongo_actually_available
    
    if not MONGO_AVAILABLE or not MongoProduct:
        _mongo_checked = True
        _mongo_actually_available = False
        return False
    
    try:
        _ = MongoProduct.objects.first()  # Test query
        _mongo_checked = True
        _mongo_actually_available = True
        return True
    except Exception:
        _mongo_checked = True
        _mongo_actually_available = False
        return False


def _get_mongo_product_id(product: Product) -> str | None:
    """Try to get MongoDB ObjectId for a Django Product."""
    if not _check_mongo_available() or not MongoProduct:
        return None
    
    # Try to find by slug first (most reliable)
    slug = getattr(product, "slug", None)
    if slug:
        try:
            mongo_product = MongoProduct.objects(slug=slug).first()
            if mongo_product:
                return str(mongo_product.id)
        except Exception:
            pass
    
    # Try to find by amazon_asin
    if hasattr(product, "amazon_asin") and product.amazon_asin:
        try:
            mongo_product = MongoProduct.objects(amazon_asin=product.amazon_asin).first()
            if mongo_product:
                return str(mongo_product.id)
        except Exception:
            pass
    
    return None


def _get_mongo_product_data(product: Product) -> dict[str, Any] | None:
    """Get MongoDB product data if available."""
    if not _check_mongo_available() or not MongoProduct:
        return None
    
    mongo_product = None
    
    # Try to find by slug first (most reliable)
    slug = getattr(product, "slug", None)
    if slug:
        try:
            mongo_product = MongoProduct.objects(slug=slug).first()
            if mongo_product:
                return {
                    "id": str(mongo_product.id),
                    "name": mongo_product.name,
                    "images": list(mongo_product.images) if mongo_product.images else [],
                    "brand_id": str(mongo_product.brand_id) if mongo_product.brand_id else None,
                    "color_ids": [str(cid) for cid in mongo_product.color_ids] if mongo_product.color_ids else [],
                    "price": float(mongo_product.price) if mongo_product.price else None,
                }
        except Exception:
            pass
    
    # Try by amazon_asin if not found
    if hasattr(product, "amazon_asin") and product.amazon_asin:
        try:
            mongo_product = MongoProduct.objects(amazon_asin=product.amazon_asin).first()
            if mongo_product:
                return {
                    "id": str(mongo_product.id),
                    "name": mongo_product.name,
                    "images": list(mongo_product.images) if mongo_product.images else [],
                    "brand_id": str(mongo_product.brand_id) if mongo_product.brand_id else None,
                    "color_ids": [str(cid) for cid in mongo_product.color_ids] if mongo_product.color_ids else [],
                    "price": float(mongo_product.price) if mongo_product.price else None,
                }
        except Exception:
            pass
    
    # Try by name and price combination for better matching
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
                        "brand_id": str(mp.brand_id) if mp.brand_id else None,
                        "color_ids": [str(cid) for cid in mp.color_ids] if mp.color_ids else [],
                        "price": float(mp.price) if mp.price else None,
                    }
        except Exception:
            pass
    
    # Try by name only (last resort)
    if product_name:
        try:
            mongo_product = MongoProduct.objects(name=product_name).first()
            if mongo_product:
                return {
                    "id": str(mongo_product.id),
                    "name": mongo_product.name,
                    "images": list(mongo_product.images) if mongo_product.images else [],
                    "brand_id": str(mongo_product.brand_id) if mongo_product.brand_id else None,
                    "color_ids": [str(cid) for cid in mongo_product.color_ids] if mongo_product.color_ids else [],
                    "price": float(mongo_product.price) if mongo_product.price else None,
                }
        except Exception:
            pass
    
    # Final attempt: Try to find by category_type, gender, age_group, and similar price
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
            # Find the closest price match
            best_match = None
            min_price_diff = float('inf')
            for mp in mongo_products:
                if mp.price:
                    price_diff = abs(float(mp.price) - float(price))
                    if price_diff < min_price_diff:
                        min_price_diff = price_diff
                        best_match = mp
            if best_match and min_price_diff < 1000:  # Reasonable price difference
                return {
                    "id": str(best_match.id),
                    "name": best_match.name,
                    "images": list(best_match.images) if best_match.images else [],
                    "brand_id": str(best_match.brand_id) if best_match.brand_id else None,
                    "color_ids": [str(cid) for cid in best_match.color_ids] if best_match.color_ids else [],
                    "price": float(best_match.price) if best_match.price else None,
                }
        except Exception:
            pass
    
    return None


def _serialize_product(product: MongoProduct, original_mongo_id: str | None = None) -> dict[str, Any]:
    """Serialize a MongoProduct object to a dictionary for the API response."""
    # The 'product' argument is a MongoProduct instance from the recommendation engine.
    # This function now correctly handles it, removing all legacy Django logic.
    
    product_id = getattr(product, "id", None)
    
    # Directly access attributes from the MongoProduct object.
    # Use getattr to provide default values for fields that might be missing.
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
        
        # Always return the product's own images, even if the list is empty.
        "images": list(getattr(product, "images", [])) or [],
        
        "rating": float(getattr(product, "rating", 0.0)) if getattr(product, "rating", None) is not None else None,
        "sale": float(getattr(product, "sale", 0.0)) if getattr(product, "sale", None) is not None else None,
        
        # Default empty lists for fields not currently in the model.
        "reviews": [],
        "variants": [],
        
        "created_at": getattr(product, "created_at", None).isoformat() if getattr(product, "created_at", None) else None,
        "updated_at": getattr(product, "updated_at", None).isoformat() if getattr(product, "updated_at", None) else None,
    }


@dataclass(slots=True)
class PersonalizedRecommendation:
    product: MongoProduct
    score: float
    reason: str
    context: Any = None  # Optional context for MongoDB ID mapping

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
    product: MongoProduct  # Changed to MongoProduct for consistency
    score: float
    reason: str = ""
    context: Any = None  # Optional context for MongoDB ID mapping

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
    outfit: Dict[str, Any]  # Changed to handle new outfit structure
    outfit_complete_score: float

    def as_dict(self) -> dict[str, Any]:
        outfit_dict = {}
        for category, entry in self.outfit.items():
            if isinstance(entry, dict) and "items" in entry:
                # New outfit structure with multiple outfits
                outfit_items = {}
                for item_key, outfit_rec in entry["items"].items():
                    if hasattr(outfit_rec, 'as_dict'):
                        outfit_items[item_key] = outfit_rec.as_dict()
                    else:
                        outfit_items[item_key] = outfit_rec
                outfit_dict[category] = outfit_items
            elif isinstance(entry, list):
                # Take the first item if it's a list (for backward compatibility)
                outfit_dict[category] = entry[0].as_dict() if entry else {}
            elif hasattr(entry, 'as_dict'):
                # Single OutfitRecommendation object
                outfit_dict[category] = entry.as_dict()
            else:
                # Raw dict or other structure
                outfit_dict[category] = entry
        return {
            "personalized": [item.as_dict() for item in self.personalized],
            "outfit": outfit_dict,
            "outfit_complete_score": round(self.outfit_complete_score, 4),
        }

