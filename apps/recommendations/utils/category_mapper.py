from typing import Optional

class CategoryMapper:

    CATEGORY_MAP = {
        "topwear": "tops",
        "bottomwear": "bottoms",
        "dress": "dresses",
        "shoes": "shoes",
        "flip flops": "shoes",
        "sandal": "shoes",
        "bags": "accessories",
        "belts": "accessories",
        "headwear": "accessories",
        "watches": "accessories",
    }

    ARTICLE_TYPE_MAP = {
        "jackets": "tops",
        "shirts": "tops",
        "sweaters": "tops",
        "sweatshirts": "tops",
        "tops": "tops",
        "tshirts": "tops",
        "tunics": "tops",
        "capris": "bottoms",
        "jeans": "bottoms",
        "shorts": "bottoms",
        "skirts": "bottoms",
        "track pants": "bottoms",
        "tracksuits": "bottoms",
        "trousers": "bottoms",
        "dresses": "dresses",
        "casual shoes": "shoes",
        "flats": "shoes",
        "formal shoes": "shoes",
        "heels": "shoes",
        "sandals": "shoes",
        "sports shoes": "shoes",
        "flip flops": "shoes",
        "sports sandals": "shoes",
        "backpacks": "accessories",
        "handbags": "accessories",
        "belts": "accessories",
        "caps": "accessories",
        "watches": "accessories",
    }

    @classmethod
    def map_product(cls, sub_category: Optional[str] = None, article_type: Optional[str] = None) -> Optional[str]:

        if article_type:
            article_lower = article_type.lower().strip()
            if article_lower in cls.ARTICLE_TYPE_MAP:
                return cls.ARTICLE_TYPE_MAP[article_lower]

        if sub_category:
            sub_lower = sub_category.lower().strip()
            if sub_lower in cls.CATEGORY_MAP:
                return cls.CATEGORY_MAP[sub_lower]

        return None

def map_subcategory_to_tag(sub_category: Optional[str] = None, article_type: Optional[str] = None) -> Optional[str]:

    return CategoryMapper.map_product(sub_category, article_type)

