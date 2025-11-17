"""Serializable schema objects for recommendation responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from apps.products.models import Product

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


def _get_product_name(product: Product) -> str | None:
    return getattr(product, "name", None) or getattr(product, "productDisplayName", None)


def _get_product_price(product: Product) -> float | int | None:
    return getattr(product, "price", None)


def _get_category_type(product: Product) -> str | None:
    for attr in ("category_type", "subCategory", "masterCategory", "articleType"):
        value = getattr(product, attr, None)
        if value:
            return value
    return None


def _get_age_group(product: Product) -> str | None:
    return getattr(product, "age_group", None)

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


def _serialize_product(product: Product, original_mongo_id: str | None = None) -> dict[str, Any]:
    import logging
    logger = logging.getLogger(__name__)
    
    product_name = _get_product_name(product)
    product_price = _get_product_price(product)
    product_gender = getattr(product, "gender", None)
    product_age_group = _get_age_group(product)
    product_category = _get_category_type(product)
    product_images = getattr(product, "images", None)
    product_brand_id = getattr(product, "brand_id", None)

    # Priority 1: If we have original MongoDB ObjectId from request/context, use it directly
    if original_mongo_id and _check_mongo_available() and MongoProduct:
        try:
            mongo_product = MongoProduct.objects(id=ObjectId(original_mongo_id)).first()
            if mongo_product:
                logger.debug(f"Using MongoDB product from original_mongo_id: {original_mongo_id}")
                # Format price properly (handle potential data issues)
                mongo_price = 0.0
                if mongo_product.price:
                    try:
                        price_val = float(mongo_product.price)
                        # If price seems too high (likely data error), use Django price as fallback
                        if price_val > 1000000:
                            mongo_price = float(product_price) if product_price is not None else 0.0
                        else:
                            mongo_price = price_val
                    except (ValueError, TypeError):
                        mongo_price = float(product_price) if product_price is not None else 0.0
                
                return {
                    "id": str(mongo_product.id),
                    "name": mongo_product.name,  # Use MongoDB name
                    "price": mongo_price,  # Use MongoDB price (with validation)
                    "images": list(mongo_product.images) if mongo_product.images else [],
                    "gender": mongo_product.gender or product_gender,
                    "age_group": mongo_product.age_group or product_age_group,
                    "category_type": mongo_product.category_type or product_category,
                    "brand_id": str(mongo_product.brand_id) if mongo_product.brand_id else None,  # Use MongoDB brand_id
                    "amazon_asin": mongo_product.amazon_asin if hasattr(mongo_product, "amazon_asin") else None,
                    "colors": [str(cid) for cid in mongo_product.color_ids] if mongo_product.color_ids else [],
                }
        except Exception as e:
            logger.warning(f"Failed to get MongoDB product by ID {original_mongo_id}: {e}")
    
    # Priority 2: Try to get MongoDB data by slug/asin/name
    mongo_data = _get_mongo_product_data(product)
    
    # Use MongoDB data if available, otherwise fall back to Django data
    if mongo_data:
        logger.debug(f"Using MongoDB data from _get_mongo_product_data for product: {product_name or '<unknown>'}")
        return {
            "id": mongo_data["id"],
            "name": mongo_data.get("name", product_name),  # Prefer MongoDB name
            "price": mongo_data.get("price") if mongo_data.get("price") is not None else (float(product_price) if product_price is not None else 0.0),  # Prefer MongoDB price
            "images": mongo_data["images"] if mongo_data["images"] else [],
            "gender": product_gender,
            "age_group": product_age_group,
            "category_type": product_category,
            "brand_id": mongo_data["brand_id"],  # Use MongoDB brand_id
            "amazon_asin": product.amazon_asin if hasattr(product, "amazon_asin") else None,
            "colors": mongo_data.get("color_ids", []),
        }
    
    # Fallback: Only if we don't have MongoDB ID from mapping, try to find similar product
    # This should be rare - most products should be mapped during context building
    if not original_mongo_id and _check_mongo_available() and MongoProduct:
        # Quick lookup: try slug/asin first (fast)
        mongo_data = _get_mongo_product_data(product)
        if mongo_data:
            base_payload = {
                "id": mongo_data["id"],
                "name": mongo_data.get("name", product_name),
                "price": mongo_data.get("price") if mongo_data.get("price") is not None else (float(product_price) if product_price is not None else 0.0),
                "images": mongo_data["images"] if mongo_data["images"] else [],
                "gender": product_gender,
                "age_group": product_age_group,
                "category_type": product_category,
                "brand_id": mongo_data["brand_id"],
                "amazon_asin": product.amazon_asin if hasattr(product, "amazon_asin") else None,
                "colors": mongo_data.get("color_ids", []),
            }
            return base_payload
        
        # Last resort: get one product with images (limit to 10 for speed)
        # Use MongoDB product data completely (name, price, brand_id from MongoDB)
        try:
            for mp in MongoProduct.objects.limit(10):
                if mp.images and len(mp.images) > 0:
                    logger.debug(f"Using fallback MongoDB product for Django product '{product_name or '<unknown>'}': MongoDB product '{mp.name}' (ID: {mp.id})")
                    # Format price properly (handle potential data issues)
                    mongo_price = 0.0
                    if mp.price:
                        try:
                            price_val = float(mp.price)
                            # If price seems too high (likely data error), use Django price as fallback
                            if price_val > 1000000:
                                mongo_price = float(product_price) if product_price is not None else 0.0
                            else:
                                mongo_price = price_val
                        except (ValueError, TypeError):
                            mongo_price = float(product_price) if product_price is not None else 0.0
                    
                    return {
                        "id": str(mp.id),
                        "name": mp.name,  # Use MongoDB name
                        "price": mongo_price,  # Use MongoDB price (with validation)
                        "images": list(mp.images),
                        "gender": getattr(mp, 'gender', None) or product_gender,
                        "age_group": getattr(mp, 'age_group', None) or product_age_group,
                        "category_type": getattr(mp, 'category_type', None) or product_category,
                        "brand_id": str(mp.brand_id) if getattr(mp, 'brand_id', None) else None,  # Use MongoDB brand_id
                        "amazon_asin": getattr(mp, 'amazon_asin', None) if hasattr(mp, "amazon_asin") else None,
                        "colors": [str(cid) for cid in getattr(mp, 'color_ids', [])] if getattr(mp, 'color_ids', None) else [],
                    }
            # If no product with images, just get first one
            mp = MongoProduct.objects.first()
            if mp:
                logger.debug(f"Using fallback MongoDB product (no images) for Django product '{product_name or '<unknown>'}': MongoDB product '{mp.name}' (ID: {mp.id})")
                # Format price properly (handle potential data issues)
                mongo_price = 0.0
                if mp.price:
                    try:
                        price_val = float(mp.price)
                        # If price seems too high (likely data error), use Django price as fallback
                        if price_val > 1000000:
                            mongo_price = float(product_price) if product_price is not None else 0.0
                        else:
                            mongo_price = price_val
                    except (ValueError, TypeError):
                        mongo_price = float(product_price) if product_price is not None else 0.0
                
                return {
                    "id": str(mp.id),
                    "name": mp.name,  # Use MongoDB name
                    "price": mongo_price,  # Use MongoDB price (with validation)
                    "images": list(mp.images) if mp.images else [],
                    "gender": getattr(mp, 'gender', None) or product_gender,
                    "age_group": getattr(mp, 'age_group', None) or product_age_group,
                    "category_type": getattr(mp, 'category_type', None) or product_category,
                    "brand_id": str(mp.brand_id) if getattr(mp, 'brand_id', None) else None,  # Use MongoDB brand_id
                    "amazon_asin": getattr(mp, 'amazon_asin', None) if hasattr(mp, "amazon_asin") else None,
                    "colors": [str(cid) for cid in getattr(mp, 'color_ids', [])] if getattr(mp, 'color_ids', None) else [],
                }
        except Exception as e:
            logger.warning(f"Failed to get fallback MongoDB product: {e}")
    
    # Final fallback: Django data only
    return {
        "id": str(getattr(product, "id", "")) if getattr(product, "id", None) is not None else "",
        "name": product_name or "",
        "price": float(product_price) if product_price is not None else 0.0,
        "images": list(product_images) if isinstance(product_images, list) and product_images else [],
        "gender": product_gender,
        "age_group": product_age_group,
        "category_type": product_category,
        "brand_id": str(product_brand_id) if product_brand_id else None,
        "amazon_asin": product.amazon_asin if hasattr(product, "amazon_asin") else None,
        "colors": [str(c.id) for c in product.colors.all()] if hasattr(product, "colors") and hasattr(product.colors, "all") else [],
    }


@dataclass(slots=True)
class PersonalizedRecommendation:
    product: Product
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
    product: Product
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
    outfit: Dict[str, OutfitRecommendation | List[OutfitRecommendation]]
    outfit_complete_score: float

    def as_dict(self) -> dict[str, Any]:
        outfit_dict = {}
        for category, entry in self.outfit.items():
            if isinstance(entry, list):
                # Take the first item if it's a list (for backward compatibility)
                outfit_dict[category] = entry[0].as_dict() if entry else {}
            else:
                outfit_dict[category] = entry.as_dict()
        return {
            "personalized": [item.as_dict() for item in self.personalized],
            "outfit": outfit_dict,
            "outfit_complete_score": round(self.outfit_complete_score, 4),
        }

