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
        if user_gender and product_gender:
            if product_gender == user_gender and product_gender in ("male", "female"):
                parts.append(f"Suitable for your {user_gender} gender")
            elif product_gender == "unisex":
                parts.append("Suitable for all genders")

        if user_age:
            if 18 <= user_age <= 35:
                parts.append("Suitable for your young age")
            elif 36 <= user_age <= 50:
                parts.append("Suitable for your middle age")
            elif user_age > 50:
                parts.append("Suitable for your age")

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

            if color_counts:
                top_color = max(color_counts, key=color_counts.get)
                if product_color and product_color.lower() == top_color.lower():
                    parts.append(f"Similar to products you've viewed: {product_color}")
                elif top_color:
                    parts.append(f"Similar to products you've viewed: {top_color}")

        product_article_type = getattr(product, "articleType", "") if hasattr(product, "articleType") else ""
        if product_article_type:
            parts.append(f"You have shown interest in {product_article_type.lower()}")

        product_season = getattr(product, "season", "") if hasattr(product, "season") else ""
        if product_season:
            parts.append(f"Suitable for {product_season.lower()} season")

        if not parts:
            parts.append("Recommended based on your preferences")

    elif reason_type == "outfit":
        if current_product:
            current_article = getattr(current_product, "articleType", "")
            current_color = getattr(current_product, "baseColour", "")
            product_article_en = _translate_article_type_to_english(product_article or product_subcategory)
            current_article_en = _translate_article_type_to_english(current_article)

            if product_subcategory and product_subcategory.lower() == "bottomwear":
                parts.append(f"Perfect match with {current_article_en}: {product_article_en}")
            elif product_subcategory and product_subcategory.lower() == "shoes":
                parts.append(f"Suitable shoes to complete outfit with {current_article_en}")
            elif product_subcategory and "accessories" in product_subcategory.lower():
                parts.append(f"Accessories to enhance {current_article_en}")
            else:
                parts.append(f"Good combination with {current_article_en}")

            if current_color and product_color:
                if current_color.lower() == product_color.lower():
                    parts.append(f"Same color tone {product_color}")
                else:
                    parts.append(f"{product_color} color harmonizes with {current_color}")
        else:
            parts.append("Suitable product to complete your outfit")

    if parts:
        capitalized_parts = [part[0].upper() + part[1:] if part else part for part in parts]
        return "; ".join(capitalized_parts)
    else:
        return "Recommended based on your preferences"

def _translate_article_type_to_english(article_type: Optional[str]) -> str:
    if not article_type:
        return "product"

    article_lower = article_type.lower()

    translations = {
        "shirts": "shirts",
        "tshirts": "t-shirts",
        "tops": "tops",
        "jackets": "jackets",
        "sweaters": "sweaters",
        "sweatshirts": "sweatshirts",
        "tunics": "tunics",
        "topwear": "tops",

        "jeans": "jeans",
        "trousers": "trousers",
        "shorts": "shorts",
        "skirts": "skirts",
        "capris": "capris",
        "track pants": "track pants",
        "bottomwear": "bottoms",

        "dresses": "dresses",
        "dress": "dress",

        "casual shoes": "casual shoes",
        "formal shoes": "formal shoes",
        "sports shoes": "sports shoes",
        "sneakers": "sneakers",
        "heels": "heels",
        "flats": "flats",
        "sandals": "sandals",
        "flip flops": "flip flops",
        "shoes": "shoes",

        "watches": "watches",
        "belts": "belts",
        "bags": "bags",
        "backpacks": "backpacks",
        "handbags": "handbags",
        "caps": "caps",
        "accessories": "accessories",
    }

    return translations.get(article_lower, article_type)

def generate_english_outfit_reason(
    outfit_items: List[MongoProduct],
    current_product: MongoProduct,
) -> str:

    current_article = getattr(current_product, "articleType", "")
    current_article_en = _translate_article_type_to_english(current_article)

    item_descriptions = []
    for item in outfit_items:
        article = getattr(item, "articleType", "")
        color = getattr(item, "baseColour", "")
        article_en = _translate_article_type_to_english(article)

        if color:
            item_descriptions.append(f"{color.lower()} {article_en}")
        else:
            item_descriptions.append(article_en)

    if item_descriptions:
        items_str = " + ".join(item_descriptions)
        return f"Perfect combination with {current_article_en}: {items_str}"
    else:
        return f"Complete outfit for {current_article_en}"

def build_english_reason_from_context(product, context, model_name: str = "") -> str:
    parts = []

    user_gender = normalize_gender(getattr(context.user, "gender", "")) if hasattr(context.user, "gender") else ""
    product_gender = normalize_gender(getattr(product, "gender", "")) if hasattr(product, "gender") else ""

    if user_gender and product_gender:
        if product_gender == user_gender and product_gender in ("male", "female"):
            parts.append(f"Suitable for your {user_gender} gender")
        elif product_gender == "unisex":
            parts.append("Suitable for all genders")

    user_age = getattr(context.user, "age", None) if hasattr(context.user, "age") else None
    if user_age:
        if 18 <= user_age <= 35:
            parts.append("Suitable for your young age")
        elif 36 <= user_age <= 50:
            parts.append("Suitable for your middle age")
        elif user_age > 50:
            parts.append("Suitable for your age")

    user_preferences = getattr(context.user, "preferences", {}) or {}
    user_style = user_preferences.get("style", "").lower()
    product_usage = getattr(product, "usage", "").lower() if hasattr(product, "usage") else ""

    if user_style and product_usage and user_style == product_usage:
        parts.append(f"Matches your {user_style} style preference")

    color_preferences = user_preferences.get("colorPreferences", []) or []
    product_color = getattr(product, "baseColour", "") if hasattr(product, "baseColour") else ""

    if product_color and color_preferences:
        for pref_color in color_preferences:
            if pref_color.lower() in product_color.lower() or product_color.lower() in pref_color.lower():
                parts.append(f"{product_color} color matches your preferences")
                break

    if hasattr(context, 'style_weight'):
        style_tokens = []
        if hasattr(product, 'style_tags') and isinstance(getattr(product, "style_tags", None), list):
            style_tokens.extend(str(tag).lower() for tag in product.style_tags if tag)
        if hasattr(product, 'baseColour') and getattr(product, "baseColour", None):
            style_tokens.append(str(product.baseColour).lower())

        matched_styles = [token for token in style_tokens if context.style_weight(token) > 0]
        if matched_styles:
            parts.append(f"Similar to products you've viewed: {', '.join(matched_styles[:2])}")

    product_article_type = getattr(product, "articleType", "") if hasattr(product, "articleType") else ""
    if product_article_type and hasattr(context, 'style_weight') and context.style_weight(product_article_type.lower()) > 0:
        parts.append(f"You have shown interest in {product_article_type.lower()}")

    product_season = getattr(product, "season", "") if hasattr(product, "season") else ""
    if product_season:
        parts.append(f"Suitable for {product_season.lower()} season")

    if model_name == "cbf":
        parts.append("Content similarity based on product features (Sentence-BERT + FAISS)")
    elif model_name == "gnn":
        parts.append("GNN (LightGCN) recommendation based on user interaction graph")
    elif model_name == "hybrid":
        parts.append("Hybrid approach combining GNN (LightGCN) and Content-based Filtering (Sentence-BERT + FAISS)")

    if not parts:
        parts.append("Recommended based on your preferences and interaction history")

    capitalized_parts = [part[0].upper() + part[1:] if part else part for part in parts]
    return "; ".join(capitalized_parts)
