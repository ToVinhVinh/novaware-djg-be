from typing import Dict, List, Optional

from apps.products.mongo_models import Product as MongoProduct
from apps.users.mongo_models import User as MongoUser

from .filters import normalize_gender

def generate_english_reason(
    product: MongoProduct,
    user: MongoUser,
    reason_type: str = "personalized",
    interaction_history: Optional[List[Dict]] = None,
    style_weights: Optional[Dict[str, float]] = None,
    color_weights: Optional[Dict[str, float]] = None,
    current_product: Optional[MongoProduct] = None,
) -> str:

    parts = []

    user_age = getattr(user, "age", None)
    user_gender = normalize_gender(getattr(user, "gender", None))

    product_gender = normalize_gender(getattr(product, "gender", None))
    product_color = getattr(product, "baseColour", None)
    product_usage = getattr(product, "usage", None)
    product_article = getattr(product, "articleType", None)
    product_subcategory = getattr(product, "subCategory", None)

    if reason_type == "personalized":
        if user_age and user_gender:
            gender_en = "male" if user_gender == "male" else "female" if user_gender == "female" else "unisex"
            parts.append(f"Based on age {user_age} and gender {gender_en}")
        elif user_age:
            parts.append(f"Suitable for age {user_age}")
        elif user_gender:
            gender_en = "male" if user_gender == "male" else "female" if user_gender == "female" else "unisex"
            parts.append(f"Suitable for gender {gender_en}")

        if interaction_history and len(interaction_history) > 0:
            style_counts = {}
            color_counts = {}

            for interaction in interaction_history[:10]:
                if isinstance(interaction, dict):
                    style = interaction.get("usage") or interaction.get("style")
                    if style:
                        style_counts[style] = style_counts.get(style, 0) + 1

                    color = interaction.get("baseColour") or interaction.get("color")
                    if color:
                        color_counts[color] = color_counts.get(color, 0) + 1

            if style_counts:
                top_style = max(style_counts, key=style_counts.get)
                article_en = _translate_article_type(product_article or product_subcategory)
                parts.append(f"You have interacted with {article_en} {top_style.lower()}")

            if color_counts:
                top_color = max(color_counts, key=color_counts.get)
                if product_color and product_color.lower() == top_color.lower():
                    parts.append(f"Color {product_color} your favorite")
                elif top_color:
                    parts.append(f"Suitable for color {top_color} your preference")

        if style_weights and product_usage:
            usage_lower = product_usage.lower()
            if usage_lower in style_weights and style_weights[usage_lower] > 0:
                parts.append(f"Style {product_usage.lower()} you often choose")

        if color_weights and product_color:
            color_lower = product_color.lower()
            if color_lower in color_weights and color_weights[color_lower] > 0:
                parts.append(f"Color {product_color} in your preference")

        if not parts:
            parts.append("Product suitable for your style")

    elif reason_type == "outfit":
        if current_product:
            current_article = getattr(current_product, "articleType", "")
            current_color = getattr(current_product, "baseColour", "")
            product_article_en = _translate_article_type(product_article or product_subcategory)
            current_article_en = _translate_article_type(current_article)

            if product_subcategory and product_subcategory.lower() == "bottomwear":
                parts.append(f"Perfect combination with {current_article_en}: {product_article_en}")
            elif product_subcategory and product_subcategory.lower() == "shoes":
                parts.append(f"Shoes suitable to complete outfit with {current_article_en}")
            elif product_subcategory and "accessories" in product_subcategory.lower():
                parts.append(f"Accessories to enhance {current_article_en}")
            else:
                parts.append(f"Good combination with {current_article_en}")

            if current_color and product_color:
                if current_color.lower() == product_color.lower():
                    parts.append(f"Same color tone {product_color}")
                else:
                    parts.append(f"Color {product_color} harmonizes with {current_color}")
        else:
            parts.append("Product suitable to complete the outfit")

    if parts:
        return ", ".join(parts)
    else:
        return "Product suggested based on your preference"

def _translate_article_type(article_type: Optional[str]) -> str:
    if not article_type:
        return "product"

    article_lower = article_type.lower()

    translations = {
        "shirts": "Shirt",
        "tshirts": "T-shirt",
        "tops": "Top",
        "jackets": "Jacket",
        "sweaters": "Sweater",
        "sweatshirts": "Sweatshirt",
        "tunics": "Tunic",
        "topwear": "Top",

        "jeans": "Jeans",
        "trousers": "Trousers",
        "shorts": "Shorts",
        "skirts": "Skirt",
        "capris": "Capris",
        "track pants": "Track Pants",
        "bottomwear": "Bottom",

        "dresses": "Dress",
        "dress": "Dress",

        "casual shoes": "Casual Shoes",
        "formal shoes": "Formal Shoes",
        "sports shoes": "Sports Shoes",
        "sneakers": "Sneakers",
        "heels": "Heels",
        "flats": "Flats",
        "sandals": "Sandals",
        "flip flops": "Flip Flops",
        "shoes": "Shoes",

        "watches": "Watches",
        "belts": "Belts",
        "bags": "Bags",
        "backpacks": "Backpacks",
        "handbags": "Handbags",
        "caps": "Caps",
        "accessories": "Accessories",

        "casual": "Casual",
        "formal": "Formal",
        "sports": "Sports",
        "ethnic": "Ethnic",
        "party": "Party",
    }

    return translations.get(article_lower, article_type)

def generate_outfit_reason(
    outfit_items: List[MongoProduct],
    current_product: MongoProduct,
) -> str:

    current_article = getattr(current_product, "articleType", "")
    current_article_en = _translate_article_type(current_article)

    item_descriptions = []
    for item in outfit_items:
        article = getattr(item, "articleType", "")
        color = getattr(item, "baseColour", "")
        article_en = _translate_article_type(article)

        if color:
            item_descriptions.append(f"{article_en} {color.lower()}")
        else:
            item_descriptions.append(article_en)

    if item_descriptions:
        items_str = " + ".join(item_descriptions)
        return f"Perfect combination with {current_article_en}: {items_str}"
    else:
        return f"Complete outfit for {current_article_en}"

