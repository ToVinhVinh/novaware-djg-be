"""
Filtering utilities for product recommendations.
Implements strict filtering rules based on user gender, age, and outfit logic.
"""

from typing import List, Optional, Set

from apps.products.mongo_models import Product as MongoProduct
from apps.users.mongo_models import User as MongoUser

from .category_mapper import map_subcategory_to_tag


def normalize_gender(gender: Optional[str]) -> Optional[str]:
    """Normalize gender string to lowercase."""
    if not gender:
        return None
    gender_lower = gender.lower().strip()
    if gender_lower in ("male", "men", "m", "boy", "boys"):
        return "male"
    elif gender_lower in ("female", "women", "f", "girl", "girls"):
        return "female"
    elif gender_lower in ("unisex", "u"):
        return "unisex"
    return None


import logging
logger = logging.getLogger(__name__)

def filter_by_age_gender(
    products: List[MongoProduct],
    user: MongoUser,
    exclude_product_ids: Optional[Set[str]] = None
) -> List[MongoProduct]:
    """
    Filter products by user's gender and age.
    
    Rules:
    - Always filter by user.gender → never recommend opposite gender or child items
    - Filter by age appropriateness
    
    Args:
        products: List of products to filter
        user: User object with gender and age
        exclude_product_ids: Set of product IDs to exclude
        
    Returns:
        Filtered list of products
    """
    if exclude_product_ids is None:
        exclude_product_ids = set()
    
    user_gender = normalize_gender(getattr(user, "gender", None))
    user_age = getattr(user, "age", None)
    
    filtered = []
    logger.debug(f"Filtering {len(products)} products for user {user.id} (gender: {user_gender}, age: {user_age})")

    for product in products:
        product_id = str(product.id)
        if product_id in exclude_product_ids:
            logger.debug(f"Product {product_id} skipped: in exclude list.")
            continue
        
        # Gender filtering
        product_gender = normalize_gender(getattr(product, "gender", None))
        if user_gender and product_gender and product_gender != 'unisex':
            if product_gender != user_gender:
                logger.debug(f"Product {product_id} filtered out: gender mismatch (user: {user_gender}, product: {product_gender}).")
                continue
        
        # Age filtering
        if user_age:
            product_usage = getattr(product, "usage", "")
            if product_usage and "kid" in product_usage.lower() and user_age >= 18:
                logger.debug(f"Product {product_id} filtered out: age mismatch (user age: {user_age}, product usage: {product_usage}).")
                continue
        
        logger.debug(f"Product {product_id} passed filters.")
        filtered.append(product)
    
    return filtered


def get_outfit_categories(current_product_tag: str, user_gender: Optional[str] = None) -> List[str]:
    """
    Get complementary outfit categories based on current product tag.
    
    Outfit logic rules:
    - If viewing "tops" → outfit = bottoms + shoes + accessories (exclude dresses)
    - If viewing "bottoms" → outfit = tops + shoes + accessories
    - If viewing "dresses" → outfit = shoes + accessories only
    - If viewing "shoes" or "accessories" → outfit = tops + bottoms
    
    Args:
        current_product_tag: Tag of current product ("tops", "bottoms", "dresses", "shoes", "accessories")
        user_gender: User's gender (to determine if dresses should be included)
        
    Returns:
        List of complementary category tags
    """
    current_tag = current_product_tag.lower().strip()
    user_gender_normalized = normalize_gender(user_gender)
    
    # Determine if user can wear dresses (typically female users)
    can_wear_dresses = user_gender_normalized == "female"
    
    if current_tag == "tops":
        # Viewing tops → recommend bottoms, shoes, accessories (no dresses)
        return ["bottoms", "shoes", "accessories"]
    
    elif current_tag == "bottoms":
        # Viewing bottoms → recommend tops, shoes, accessories
        return ["tops", "shoes", "accessories"]
    
    elif current_tag == "dresses":
        # Viewing dresses → recommend shoes, accessories only
        return ["shoes", "accessories"]
    
    elif current_tag in ("shoes", "accessories"):
        # Viewing shoes/accessories → recommend tops, bottoms
        # Include dresses for female users
        if can_wear_dresses:
            return ["tops", "bottoms", "dresses"]
        else:
            return ["tops", "bottoms"]
    
    # Default: return all other categories
    all_categories = ["tops", "bottoms", "shoes", "accessories"]
    if can_wear_dresses:
        all_categories.append("dresses")
    
    # Remove current category
    return [cat for cat in all_categories if cat != current_tag]


def deduplicate_products(products: List[MongoProduct], current_product_id: str, similarity_threshold: float = 0.95) -> List[MongoProduct]:
    """
    Remove products that are too similar to the current product.
    
    For now, we use simple deduplication (exact match removal).
    Can be enhanced with embedding similarity if needed.
    
    Args:
        products: List of products
        current_product_id: ID of current product
        similarity_threshold: Threshold for similarity (not used yet)
        
    Returns:
        Deduplicated list
    """
    return [p for p in products if str(p.id) != current_product_id]

