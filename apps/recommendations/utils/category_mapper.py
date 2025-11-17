"""
Category mapping utilities for outfit recommendations.
Maps subCategory to standardized tags (tops, bottoms, shoes, accessories, dresses).
"""

from typing import Optional


class CategoryMapper:
    """
    Maps product categories to standardized outfit tags.
    Based on the category hierarchy provided in the requirements.
    """
    
    # Mapping from subCategory to standardized tag
    CATEGORY_MAP = {
        # Apparel - Topwear
        "topwear": "tops",
        # Apparel - Bottomwear
        "bottomwear": "bottoms",
        # Apparel - Dress
        "dress": "dresses",
        # Footwear
        "shoes": "shoes",
        "flip flops": "shoes",
        "sandal": "shoes",
        # Accessories
        "bags": "accessories",
        "belts": "accessories",
        "headwear": "accessories",
        "watches": "accessories",
    }
    
    # ArticleType to tag mapping (more specific)
    ARTICLE_TYPE_MAP = {
        # Topwear
        "jackets": "tops",
        "shirts": "tops",
        "sweaters": "tops",
        "sweatshirts": "tops",
        "tops": "tops",
        "tshirts": "tops",
        "tunics": "tops",
        # Bottomwear
        "capris": "bottoms",
        "jeans": "bottoms",
        "shorts": "bottoms",
        "skirts": "bottoms",
        "track pants": "bottoms",
        "tracksuits": "bottoms",
        "trousers": "bottoms",
        # Dresses
        "dresses": "dresses",
        # Footwear
        "casual shoes": "shoes",
        "flats": "shoes",
        "formal shoes": "shoes",
        "heels": "shoes",
        "sandals": "shoes",
        "sports shoes": "shoes",
        "flip flops": "shoes",
        "sports sandals": "shoes",
        # Accessories
        "backpacks": "accessories",
        "handbags": "accessories",
        "belts": "accessories",
        "caps": "accessories",
        "watches": "accessories",
    }
    
    @classmethod
    def map_product(cls, sub_category: Optional[str] = None, article_type: Optional[str] = None) -> Optional[str]:
        """
        Map a product to its outfit tag.
        
        Args:
            sub_category: Product subCategory (e.g., "Topwear", "Bottomwear")
            article_type: Product articleType (e.g., "Shirts", "Jeans")
            
        Returns:
            Standardized tag: "tops", "bottoms", "dresses", "shoes", or "accessories"
        """
        # Try article type first (more specific)
        if article_type:
            article_lower = article_type.lower().strip()
            if article_lower in cls.ARTICLE_TYPE_MAP:
                return cls.ARTICLE_TYPE_MAP[article_lower]
        
        # Fall back to subCategory
        if sub_category:
            sub_lower = sub_category.lower().strip()
            if sub_lower in cls.CATEGORY_MAP:
                return cls.CATEGORY_MAP[sub_lower]
        
        return None


def map_subcategory_to_tag(sub_category: Optional[str] = None, article_type: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to map product category to outfit tag.
    
    Args:
        sub_category: Product subCategory
        article_type: Product articleType
        
    Returns:
        Outfit tag or None
    """
    return CategoryMapper.map_product(sub_category, article_type)

