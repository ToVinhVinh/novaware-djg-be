"""
Vietnamese reason generation for recommendations.
Generates specific, personalized reasons in Vietnamese mentioning age, gender, past interactions, style, and color.
"""

from typing import Dict, List, Optional

from apps.products.mongo_models import Product as MongoProduct
from apps.users.mongo_models import User as MongoUser

from .filters import normalize_gender


def generate_vietnamese_reason(
    product: MongoProduct,
    user: MongoUser,
    reason_type: str = "personalized",
    interaction_history: Optional[List[Dict]] = None,
    style_weights: Optional[Dict[str, float]] = None,
    color_weights: Optional[Dict[str, float]] = None,
    current_product: Optional[MongoProduct] = None,
) -> str:
    """
    Generate a specific Vietnamese reason for recommendation.
    
    Args:
        product: Recommended product
        user: User object
        reason_type: "personalized" or "outfit"
        interaction_history: User's interaction history
        style_weights: User's style preferences with weights
        color_weights: User's color preferences with weights
        current_product: Current product being viewed (for outfit recommendations)
        
    Returns:
        Vietnamese reason string
    """
    parts = []
    
    # Get user info
    user_age = getattr(user, "age", None)
    user_gender = normalize_gender(getattr(user, "gender", None))
    
    # Get product info
    product_gender = normalize_gender(getattr(product, "gender", None))
    product_color = getattr(product, "baseColour", None)
    product_usage = getattr(product, "usage", None)
    product_article = getattr(product, "articleType", None)
    product_subcategory = getattr(product, "subCategory", None)
    
    if reason_type == "personalized":
        # Start with age and gender context
        if user_age and user_gender:
            gender_vn = "nam" if user_gender == "male" else "nữ" if user_gender == "female" else "unisex"
            parts.append(f"Dựa trên độ tuổi {user_age} và giới tính {gender_vn}")
        elif user_age:
            parts.append(f"Phù hợp với độ tuổi {user_age}")
        elif user_gender:
            gender_vn = "nam" if user_gender == "male" else "nữ" if user_gender == "female" else "unisex"
            parts.append(f"Phù hợp với giới tính {gender_vn}")
        
        # Add interaction history context
        if interaction_history and len(interaction_history) > 0:
            # Find most common style from history
            style_counts = {}
            color_counts = {}
            
            for interaction in interaction_history[:10]:  # Last 10 interactions
                if isinstance(interaction, dict):
                    # Count styles
                    style = interaction.get("usage") or interaction.get("style")
                    if style:
                        style_counts[style] = style_counts.get(style, 0) + 1
                    
                    # Count colors
                    color = interaction.get("baseColour") or interaction.get("color")
                    if color:
                        color_counts[color] = color_counts.get(color, 0) + 1
            
            # Get top style
            if style_counts:
                top_style = max(style_counts, key=style_counts.get)
                article_vn = _translate_article_type(product_article or product_subcategory)
                parts.append(f"bạn từng tương tác nhiều với {article_vn} {top_style.lower()}")
            
            # Get top color
            if color_counts:
                top_color = max(color_counts, key=color_counts.get)
                if product_color and product_color.lower() == top_color.lower():
                    parts.append(f"màu {product_color} là màu bạn yêu thích")
                elif top_color:
                    parts.append(f"phù hợp với sở thích màu {top_color} của bạn")
        
        # Add style preference from weights
        if style_weights and product_usage:
            usage_lower = product_usage.lower()
            if usage_lower in style_weights and style_weights[usage_lower] > 0:
                parts.append(f"phong cách {product_usage.lower()} bạn thường chọn")
        
        # Add color preference from weights
        if color_weights and product_color:
            color_lower = product_color.lower()
            if color_lower in color_weights and color_weights[color_lower] > 0:
                parts.append(f"màu {product_color} nằm trong sở thích của bạn")
        
        # Default if no parts
        if not parts:
            parts.append("Sản phẩm phù hợp với phong cách của bạn")
    
    elif reason_type == "outfit":
        # Outfit-specific reasons
        if current_product:
            current_article = getattr(current_product, "articleType", "")
            current_color = getattr(current_product, "baseColour", "")
            product_article_vn = _translate_article_type(product_article or product_subcategory)
            current_article_vn = _translate_article_type(current_article)
            
            # Build outfit combination message
            if product_subcategory and product_subcategory.lower() == "bottomwear":
                parts.append(f"Phối hợp hoàn hảo với {current_article_vn}: {product_article_vn}")
            elif product_subcategory and product_subcategory.lower() == "shoes":
                parts.append(f"Giày phù hợp để hoàn thiện outfit với {current_article_vn}")
            elif product_subcategory and "accessories" in product_subcategory.lower():
                parts.append(f"Phụ kiện tô điểm cho {current_article_vn}")
            else:
                parts.append(f"Kết hợp tốt với {current_article_vn}")
            
            # Add color harmony
            if current_color and product_color:
                if current_color.lower() == product_color.lower():
                    parts.append(f"cùng tông màu {product_color}")
                else:
                    parts.append(f"màu {product_color} hài hòa với {current_color}")
        else:
            parts.append("Sản phẩm phù hợp để hoàn thiện trang phục")
    
    # Join all parts
    if parts:
        return ", ".join(parts)
    else:
        return "Sản phẩm được gợi ý dựa trên sở thích của bạn"


def _translate_article_type(article_type: Optional[str]) -> str:
    """Translate article type to Vietnamese."""
    if not article_type:
        return "sản phẩm"
    
    article_lower = article_type.lower()
    
    translations = {
        # Topwear
        "shirts": "áo sơ mi",
        "tshirts": "áo thun",
        "tops": "áo",
        "jackets": "áo khoác",
        "sweaters": "áo len",
        "sweatshirts": "áo nỉ",
        "tunics": "áo dài",
        "topwear": "áo",
        
        # Bottomwear
        "jeans": "quần jeans",
        "trousers": "quần tây",
        "shorts": "quần short",
        "skirts": "váy",
        "capris": "quần lửng",
        "track pants": "quần thể thao",
        "bottomwear": "quần",
        
        # Dresses
        "dresses": "váy liền",
        "dress": "váy",
        
        # Footwear
        "casual shoes": "giày casual",
        "formal shoes": "giày tây",
        "sports shoes": "giày thể thao",
        "sneakers": "giày sneakers",
        "heels": "giày cao gót",
        "flats": "giày bệt",
        "sandals": "dép sandal",
        "flip flops": "dép xỏ ngón",
        "shoes": "giày",
        
        # Accessories
        "watches": "đồng hồ",
        "belts": "thắt lưng",
        "bags": "túi xách",
        "backpacks": "ba lô",
        "handbags": "túi xách tay",
        "caps": "mũ",
        "accessories": "phụ kiện",
        
        # Styles
        "casual": "casual",
        "formal": "lịch sự",
        "sports": "thể thao",
        "ethnic": "dân tộc",
        "party": "dự tiệc",
    }
    
    return translations.get(article_lower, article_type)


def generate_outfit_reason(
    outfit_items: List[MongoProduct],
    current_product: MongoProduct,
) -> str:
    """
    Generate a Vietnamese reason for outfit recommendation.
    
    Args:
        outfit_items: List of products in the outfit
        current_product: Current product being viewed
        
    Returns:
        Vietnamese reason string
    """
    current_article = getattr(current_product, "articleType", "")
    current_article_vn = _translate_article_type(current_article)
    
    # Build outfit description
    item_descriptions = []
    for item in outfit_items:
        article = getattr(item, "articleType", "")
        color = getattr(item, "baseColour", "")
        article_vn = _translate_article_type(article)
        
        if color:
            item_descriptions.append(f"{article_vn} {color.lower()}")
        else:
            item_descriptions.append(article_vn)
    
    if item_descriptions:
        items_str = " + ".join(item_descriptions)
        return f"Phối hợp hoàn hảo với {current_article_vn}: {items_str}"
    else:
        return f"Outfit hoàn chỉnh cho {current_article_vn}"

