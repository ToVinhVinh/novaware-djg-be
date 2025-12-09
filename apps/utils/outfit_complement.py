"""
Utility functions for complement-based outfit recommendation.
Replaces usage-based filtering with Item-Item compatibility matrix.
"""

from typing import Dict, List, Set, Optional
from collections import defaultdict

COMPLEMENT_DICT = {
    'Trousers': ['Tshirts', 'Shirts', 'Jackets', 'Formal Shoes', 'Casual Shoes', 'Sports Shoes'],
    'Tshirts': ['Trousers', 'Shorts', 'Jeans', 'Jackets', 'Formal Shoes', 'Casual Shoes', 'Sports Shoes', 'Flip Flops'],
    'Shirts': ['Trousers', 'Jeans', 'Formal Shoes', 'Casual Shoes'],
    'Dresses': ['Jackets', 'Sweaters', 'Heels', 'Flats', 'Handbags'],
    'Skirts': ['Tshirts', 'Tops', 'Shirts', 'Jackets', 'Heels', 'Flats', 'Handbags'],
    'Shorts': ['Tshirts', 'Sweatshirts', 'Sports Shoes', 'Flip Flops', 'Sandals'],
    'Jeans': ['Tshirts', 'Shirts', 'Jackets', 'Casual Shoes', 'Sports Shoes'],
    'Jackets': ['Trousers', 'Dresses', 'Tshirts', 'Shirts', 'Jeans'],
    'Sweaters': ['Trousers', 'Jeans', 'Dresses'],
    'Sweatshirts': ['Trousers', 'Shorts', 'Jeans', 'Sports Shoes'],
    'Tops': ['Skirts', 'Shorts', 'Jeans', 'Heels', 'Flats'],
    'Formal Shoes': ['Trousers', 'Shirts', 'Formal Shoes'],
    'Casual Shoes': ['Trousers', 'Tshirts', 'Shirts', 'Jeans', 'Shorts'],
    'Sports Shoes': ['Tshirts', 'Shorts', 'Track Pants', 'Capris'],
    'Heels': ['Dresses', 'Skirts', 'Tops'],
    'Flats': ['Dresses', 'Skirts', 'Tops'],
    'Flip Flops': ['Tshirts', 'Shorts', 'Casual Shoes'],
    'Sandals': ['Tshirts', 'Shorts', 'Casual Shoes'],
    'Handbags': ['Dresses', 'Skirts', 'Tops'],
    'Watches': ['Tshirts', 'Shirts', 'Formal Shoes', 'Casual Shoes'],
    'Belts': ['Trousers', 'Jeans', 'Shorts'],
    'Caps': ['Tshirts', 'Sports Shoes'],
    'Track Pants': ['Tshirts', 'Sports Shoes', 'Sweatshirts'],
    'Capris': ['Tshirts', 'Tops', 'Sports Shoes'],
    'Jerseys': ['Shorts', 'Track Pants', 'Sports Shoes'],
    'Backpacks': ['Tshirts', 'Sports Shoes', 'Track Pants'],
}

# Reverse mapping: for each article type, what can it complement?
COMPLEMENT_REVERSE = defaultdict(list)
for article_type, complements in COMPLEMENT_DICT.items():
    for complement in complements:
        COMPLEMENT_REVERSE[complement].append(article_type)
        # Also add reverse relationship (bidirectional for some items)
        if complement in COMPLEMENT_DICT:
            if article_type not in COMPLEMENT_DICT[complement]:
                COMPLEMENT_DICT[complement] = COMPLEMENT_DICT[complement] + [article_type]


def normalize_article_type(article_type: str) -> str:
    """
    Normalize article type to match keys in COMPLEMENT_DICT.
    Handles variations like 'Tshirts' -> 'T-shirt', 'T-Shirts' -> 'T-shirt'
    """
    if not article_type:
        return ""
    
    article_type = str(article_type).strip()
    
    # Common mappings - normalize variations to actual articleType values
    mappings = {
        'T-shirt': 'Tshirts',
        'T-Shirts': 'Tshirts',
        'T Shirts': 'Tshirts',
        'Tshirt': 'Tshirts',
        'T-Shirt': 'Tshirts',
        'Bag': 'Handbags',
        'Bags': 'Handbags',
        'Dress': 'Dresses',
        'Sandal': 'Sandals',
        'Shoes': 'Casual Shoes',  # Default to Casual Shoes if just "Shoes"
        'Hoodie': 'Sweatshirts',
        'Cardigan': 'Sweaters',
        'Blouse': 'Tops',
    }
    
    # Check exact match first
    if article_type in COMPLEMENT_DICT:
        return article_type
    
    # Check mappings
    if article_type in mappings:
        return mappings[article_type]
    
    # Try case-insensitive match
    article_lower = article_type.lower()
    for key in COMPLEMENT_DICT.keys():
        if key.lower() == article_lower:
            return key
    
    # Try partial match (e.g., "T-shirt" in "Tshirts")
    for key in COMPLEMENT_DICT.keys():
        if key.lower() in article_lower or article_lower in key.lower():
            return key
    
    return article_type


def get_compatible_items(article_type: str) -> List[str]:
    """
    Get list of article types that are compatible with the given article type.
    
    Args:
        article_type: The article type to find complements for
        
    Returns:
        List of compatible article types
    """
    normalized = normalize_article_type(article_type)
    
    # Get direct complements
    direct_complements = COMPLEMENT_DICT.get(normalized, [])
    
    # Get reverse complements (items that complement this one)
    reverse_complements = COMPLEMENT_REVERSE.get(normalized, [])
    
    # Combine and deduplicate
    all_complements = list(set(direct_complements + reverse_complements))
    
    return all_complements


def compute_complement_score(item1_type: str, item2_type: str) -> float:
    """
    Compute compatibility score between two article types.
    
    Args:
        item1_type: First article type
        item2_type: Second article type
        
    Returns:
        Compatibility score (1.0 if directly compatible, 0.5 if reverse compatible, 0.0 otherwise)
    """
    normalized1 = normalize_article_type(item1_type)
    normalized2 = normalize_article_type(item2_type)
    
    if not normalized1 or not normalized2:
        return 0.0
    
    # Direct complement (item1 complements item2)
    if normalized1 in COMPLEMENT_DICT.get(normalized2, []):
        return 1.0
    
    # Reverse complement (item2 complements item1)
    if normalized2 in COMPLEMENT_DICT.get(normalized1, []):
        return 1.0
    
    # Check reverse mapping
    if normalized1 in COMPLEMENT_REVERSE.get(normalized2, []):
        return 0.8
    
    if normalized2 in COMPLEMENT_REVERSE.get(normalized1, []):
        return 0.8
    
    # Same type (can be compatible in some cases, but lower score)
    if normalized1 == normalized2:
        return 0.3
    
    return 0.0


def build_complement_matrix(article_types: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Build a compatibility matrix for a list of article types.
    
    Args:
        article_types: List of unique article types
        
    Returns:
        Dictionary mapping (article_type1, article_type2) -> compatibility score
    """
    matrix = {}
    unique_types = list(set(article_types))
    
    for type1 in unique_types:
        matrix[type1] = {}
        for type2 in unique_types:
            if type1 == type2:
                matrix[type1][type2] = 0.3  # Same type, low compatibility
            else:
                matrix[type1][type2] = compute_complement_score(type1, type2)
    
    return matrix


def get_outfit_categories_for_article_type(article_type: str) -> Dict[str, bool]:
    """
    Determine which outfit categories are needed/optional for a given article type.
    
    Returns:
        Dictionary with keys: 'needs_topwear', 'needs_bottomwear', 'needs_footwear', 
        'needs_accessory', 'is_dress', etc.
    """
    normalized = normalize_article_type(article_type)
    
    result = {
        'needs_topwear': False,
        'needs_bottomwear': False,
        'needs_footwear': True,  # Most outfits need shoes
        'needs_accessory': False,
        'is_dress': False,
        'is_topwear': False,
        'is_bottomwear': False,
    }
    
    # Topwear items
    topwear_types = ['Tshirts', 'Shirts', 'Tops', 'Sweatshirts', 'Sweaters', 'Jerseys']
    if normalized in topwear_types:
        result['is_topwear'] = True
        result['needs_bottomwear'] = True
        result['needs_footwear'] = True
    
    # Bottomwear items
    bottomwear_types = ['Trousers', 'Shorts', 'Skirts', 'Jeans', 'Track Pants', 'Capris']
    if normalized in bottomwear_types:
        result['is_bottomwear'] = True
        result['needs_topwear'] = True
        result['needs_footwear'] = True
    
    # Dress (replaces both topwear and bottomwear)
    if normalized == 'Dresses':
        result['is_dress'] = True
        result['needs_topwear'] = False
        result['needs_bottomwear'] = False
        result['needs_footwear'] = True
        result['needs_accessory'] = True  # Dresses often go with bags
    
    # Footwear
    footwear_types = ['Formal Shoes', 'Casual Shoes', 'Sports Shoes', 'Heels', 'Flats', 'Flip Flops', 'Sandals']
    if normalized in footwear_types:
        result['needs_topwear'] = True
        result['needs_bottomwear'] = True
    
    # Outerwear (can complement many things)
    if normalized in ['Jackets', 'Sweaters']:
        result['needs_topwear'] = True
        result['needs_bottomwear'] = True
    
    # Accessories
    if normalized in ['Handbags', 'Watches', 'Belts', 'Caps', 'Backpacks']:
        result['needs_accessory'] = False  # These are the accessories
        result['needs_topwear'] = True
        result['needs_bottomwear'] = True
    
    return result


def filter_products_by_compatibility(
    products_df,
    target_article_type: str,
    category: str = None
) -> 'pd.DataFrame':
    """
    Filter products that are compatible with the target article type.
    
    Args:
        products_df: DataFrame with products
        target_article_type: The article type to find complements for
        category: Optional category filter (e.g., 'topwear', 'bottomwear')
        
    Returns:
        Filtered DataFrame
    """
    if products_df is None or products_df.empty:
        return products_df
    
    compatible_types = get_compatible_items(target_article_type)
    
    if not compatible_types:
        return products_df
    
    # Filter by compatible article types
    mask = products_df['articleType'].astype(str).apply(
        lambda x: normalize_article_type(x) in compatible_types
    )
    
    filtered = products_df[mask].copy()
    
    # Additional category filter if provided
    if category:
        if category == 'topwear':
            filtered = filtered[
                filtered['subCategory'].astype(str).str.lower() == 'topwear'
            ]
        elif category == 'bottomwear':
            filtered = filtered[
                filtered['subCategory'].astype(str).str.lower() == 'bottomwear'
            ]
        elif category == 'footwear':
            filtered = filtered[
                filtered['masterCategory'].astype(str).str.lower() == 'footwear'
            ]
        elif category == 'accessory':
            filtered = filtered[
                filtered['masterCategory'].astype(str).str.lower() == 'accessories'
            ]
    
    return filtered

