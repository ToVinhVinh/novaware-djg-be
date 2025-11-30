from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .constants import OUTFIT_COMPLETION_RULES, OUTFIT_SCORE_FLOOR
from .context import RecommendationContext
from .gender_utils import gender_filter_values, normalize_gender
from .schema import OutfitRecommendation

class OutfitBuilder:

    @classmethod
    def required_categories(cls, current_category: str) -> List[str]:
        return OUTFIT_COMPLETION_RULES.get(current_category or "", [])

    @classmethod
    def build(
        cls,
        context: RecommendationContext,
        scored_candidates: dict[int, float],
        top_k: int,
    ) -> tuple[dict[str, OutfitRecommendation], float]:

        outfits = []

        for outfit_idx in range(context.top_k_outfit):
            outfit = cls._build_single_outfit(context, scored_candidates, outfit_idx)
            if outfit:
                outfits.append(outfit)

        if not outfits:
            return {}, 0.0

        total_score = sum(outfit.get('completeness_score', 0.0) for outfit in outfits)
        avg_score = total_score / len(outfits) if outfits else 0.0

        outfit_payload = {}
        for i, outfit in enumerate(outfits):
            outfit_payload[f"outfit_{i+1}"] = outfit

        return outfit_payload, avg_score

    @classmethod
    def _build_single_outfit(cls, context: RecommendationContext, scored_candidates: dict[int, float], outfit_index: int) -> dict:

        user_product_gender = cls._get_user_product_gender(context.user)

        required_categories = {
            "accessories": cls._get_accessories_subcategory(),
            "apparel_bottomwear": "Bottomwear",
            "apparel_topwear": "Topwear",
            "footwear": cls._get_footwear_subcategory()
        }

        if hasattr(context.user, 'gender') and normalize_gender(context.user.gender) == 'female':
            required_categories["apparel_dress"] = "Dress"

        if hasattr(context.user, 'gender') and normalize_gender(context.user.gender) == 'female':
            required_categories["apparel_innerwear"] = "Innerwear"

        outfit_items = {}
        used_product_ids = set()

        if context.current_product and hasattr(context.current_product, 'id'):
            current_product = context.current_product
            current_category_key = cls._get_category_key_for_product(current_product)

            if current_category_key:
                subcategory_name = required_categories.get(current_category_key, current_category_key.replace("apparel_", "").replace("_", " ").title())

                current_score = scored_candidates.get(current_product.id, 1.0) if hasattr(current_product, 'id') else 1.0
                current_reason = cls._build_outfit_reason(current_product, context, subcategory_name)
                current_reason = f"Selected product: {current_reason}"

                outfit_items[current_category_key] = OutfitRecommendation(
                    current_category_key, current_product, current_score, current_reason, context=context
                )
                used_product_ids.add(current_product.id)

        for category_key, subcategory in required_categories.items():
            if category_key in outfit_items:
                continue
            product = cls._find_product_for_category(
                subcategory=subcategory,
                user_product_gender=user_product_gender,
                context=context,
                used_product_ids=used_product_ids,
                scored_candidates=scored_candidates,
                outfit_index=outfit_index
            )

            if product:
                score = scored_candidates.get(product.id, 0.0) if hasattr(product, 'id') else 0.0
                reason = cls._build_outfit_reason(product, context, subcategory)

                outfit_items[category_key] = OutfitRecommendation(
                    category_key, product, score, reason, context=context
                )

                if hasattr(product, 'id'):
                    used_product_ids.add(product.id)

        required_core = ["accessories", "apparel_bottomwear", "apparel_topwear", "footwear"]
        core_items = sum(1 for key in required_core if key in outfit_items)

        if core_items < len(required_core):
            return None

        completeness_score = len(outfit_items) / len(required_categories)

        return {
            "items": outfit_items,
            "completeness_score": completeness_score
        }

    @classmethod
    def _get_user_product_gender(cls, user) -> str:
        if not hasattr(user, 'age') or not hasattr(user, 'gender'):
            return "Unisex"

        user_gender = normalize_gender(user.gender)
        age = user.age

        if user_gender == 'male':
            if age <= 12:
                return "Boys"
            else:
                return "Men"
        elif user_gender == 'female':
            if age <= 12:
                return "Girls"
            else:
                return "Women"
        else:
            return "Unisex"

    @classmethod
    def _get_accessories_subcategory(cls) -> str:
        import random
        accessories_options = ["Bags", "Belts", "Headwear", "Watches"]
        return random.choice(accessories_options)

    @classmethod
    def _get_footwear_subcategory(cls) -> str:
        import random
        footwear_options = ["Shoes", "Sandal", "Flip Flops"]
        return random.choice(footwear_options)

    @classmethod
    def _find_product_for_category(cls, subcategory: str, user_product_gender: str, context: RecommendationContext,
                                 used_product_ids: set, scored_candidates: dict, outfit_index: int):

        try:
            from apps.products.mongo_models import Product as MongoProduct
        except ImportError:
            return None

        query_filters = {
            'subCategory': subcategory,
            'gender': user_product_gender
        }

        if used_product_ids:
            exclude_ids = []
            for pid in used_product_ids:
                try:
                    if hasattr(pid, '__str__'):
                        exclude_ids.append(pid)
                except:
                    pass
            if exclude_ids:
                query_filters['id__nin'] = exclude_ids

        article_types = cls._get_article_types_for_category(subcategory, normalize_gender(getattr(context.user, 'gender', '')))
        if article_types:
            query_filters['articleType__in'] = article_types

        try:
            products = list(MongoProduct.objects(**query_filters).limit(10))

            if not products:
                query_filters['gender'] = 'Unisex'
                products = list(MongoProduct.objects(**query_filters).limit(10))

            if not products:
                del query_filters['gender']
                products = list(MongoProduct.objects(**query_filters).limit(10))

            if products:
                def get_sort_score(product):
                    if hasattr(product, 'id') and product.id in scored_candidates:
                        return scored_candidates[product.id]
                    return getattr(product, 'rating', 0.0) / 5.0

                products.sort(key=get_sort_score, reverse=True)

                index = min(outfit_index, len(products) - 1)
                return products[index]

            return None

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error finding product for category {subcategory}: {e}")
            return None

    @classmethod
    def _get_article_types_for_category(cls, subcategory: str, user_gender: str) -> list:

        category_mappings = {
            "Bags": ["Backpacks", "Handbags"],
            "Belts": ["Belts"],
            "Headwear": ["Caps"],
            "Watches": ["Watches"],
            "Bottomwear": ["Capris", "Jeans", "Shorts", "Skirts", "Track Pants", "Tracksuits", "Trousers"],
            "Dress": ["Dresses"],
            "Innerwear": ["Bra"],
            "Topwear": ["Jackets", "Shirts", "Sweaters", "Sweatshirts", "Tops", "Tshirts", "Tunics"],
            "Flip Flops": ["Flip Flops"],
            "Sandal": ["Sandals", "Sports Sandals"],
            "Shoes": ["Casual Shoes", "Flats", "Formal Shoes", "Heels", "Sandals", "Sports Shoes"]
        }

        article_types = category_mappings.get(subcategory, [])

        if user_gender == 'male':
            filtered = [item for item in article_types if item not in ["Skirts", "Dresses", "Bra", "Heels", "Flats"]]
            return filtered if filtered else article_types
        elif user_gender == 'female':
            return article_types
        else:
            filtered = [item for item in article_types if item not in ["Skirts", "Dresses", "Bra", "Heels"]]
            return filtered if filtered else article_types

    @classmethod
    def _build_outfit_reason(cls, product, context: RecommendationContext, category: str) -> str:
        parts = []

        user_gender = normalize_gender(getattr(context.user, "gender", "")) if hasattr(context.user, "gender") else ""
        product_gender = normalize_gender(getattr(product, "gender", "")) if hasattr(product, "gender") else ""

        if user_gender and product_gender:
            if product_gender.lower() == user_gender:
                parts.append(f"Suitable for your {user_gender} gender")
            elif product_gender.lower() == "unisex":
                parts.append("Suitable for all genders")

        parts.append(f"Complete the outfit with {category.lower()}")

        if hasattr(product, "baseColour") and product.baseColour:
            parts.append(f"Color {product.baseColour.lower()} harmonizes")

        if not parts:
            parts.append("Suitable to complete the outfit")

        return "; ".join(parts)

    @classmethod
    def _fallback_from_database(cls, category: str, context: RecommendationContext, used_ids: set[int]):
        from apps.products.models import Product

        try:
            excluded = used_ids | context.excluded_product_ids
            allowed_genders = cls._allowed_genders(context.resolved_gender)

            query = Product.objects.filter(
                gender__in=allowed_genders,
                age_group=context.resolved_age_group,
            ).exclude(id__in=excluded)

            for product in query:
                if cls._product_matches_category(product, category):
                    product_gender = (getattr(product, "gender", "") or "").lower()
                    product_age = (getattr(product, "age_group", "") or "").lower()
                    user_gender = context.resolved_gender.lower()
                    user_age = context.resolved_age_group.lower()

                    if product_gender not in allowed_genders:
                        continue

                    if product_age != user_age:
                        continue

                    return product
            return None
        except Exception:
            return None

    @classmethod
    def _fallback_from_database_relaxed(cls, category: str, context: RecommendationContext, used_ids: set[int]):
        from apps.products.models import Product

        try:
            excluded = used_ids | context.excluded_product_ids
            category_lower = str(category).lower().strip()

            query = Product.objects.exclude(id__in=excluded)
            for product in query:
                if cls._product_matches_category(product, category_lower):
                    return product

            allowed_genders = cls._allowed_genders(context.resolved_gender)
            query = Product.objects.filter(
                gender__in=allowed_genders,
            ).exclude(id__in=excluded)
            for product in query:
                if cls._product_matches_category(product, category_lower):
                    return product

            query = Product.objects.exclude(id__in=excluded)
            for product in query:
                if cls._product_matches_category(product, category_lower):
                    return product

            return None
        except Exception:
            return None

    @staticmethod
    def _allowed_genders(gender: str) -> list[str]:
        return gender_filter_values(gender)

    @staticmethod
    def _build_reason(product, context: RecommendationContext, category: str, is_fallback: bool = False) -> str:
        from .base_engine import _extract_style_tokens

        parts = []

        user_gender = normalize_gender(getattr(context.user, "gender", "")) if hasattr(context.user, "gender") else ""
        product_gender = normalize_gender(getattr(product, "gender", "")) if hasattr(product, "gender") else ""

        if user_gender and product_gender:
            if product_gender == user_gender and product_gender in ("male", "female"):
                parts.append(f"Suitable for your {user_gender} gender")
            elif product_gender == "unisex":
                parts.append("Suitable for all genders")

        user_age = getattr(context.user, "age", None) if hasattr(context.user, "age") else None
        product_age_group = getattr(product, "age_group", "").lower() if hasattr(product, "age_group") else ""

        if user_age:
            if user_age <= 12:
                user_age_group = "kid"
            elif user_age <= 19:
                user_age_group = "teen"
            else:
                user_age_group = "adult"

            if product_age_group == user_age_group:
                parts.append(f"Suitable for your {user_age_group} age")

        tags = _extract_style_tokens(product)
        matched = [token for token in tags if context.style_weight(token) > 0]
        if matched:
            parts.append(f"Has similar style: {', '.join(matched[:3])}")

        if hasattr(product, "baseColour") and product.baseColour:
            color = product.baseColour.lower()
            if context.style_weight(color) > 0:
                parts.append(f"Color {color} suitable for your preference")
        current_sub_category = (getattr(context.current_product, "subCategory", "") or "").lower()
        current_article_type = (getattr(context.current_product, "articleType", "") or "").lower()

        if category != current_sub_category and category != current_article_type:
            if current_sub_category:
                parts.append(f"Add for {current_sub_category} you are viewing")
            elif current_article_type:
                parts.append(f"Add for {current_article_type} you are viewing")

        if not parts:
            parts.append("Suitable to complete the outfit")

        if is_fallback:
            parts.append("(Suggestion to add)")

        return "; ".join(parts)

    @classmethod
    def _rank_for_category(
        cls,
        category: str,
        scored_candidates: dict[int, float],
        candidate_map: dict[int, object],
        remaining_ids: set[int],
        context: RecommendationContext,
    ) -> list[tuple[int, float]]:
        ranked: list[tuple[int, float]] = []
        category_lower = str(category).lower().strip()
        for product_id in list(remaining_ids):
            product = candidate_map.get(product_id)
            if not product:
                continue

            if not cls._product_matches_category(product, category_lower):
                continue
            base_score = scored_candidates.get(product_id, 0.0)
            style_bonus = sum(context.style_weight(tag) for tag in _extract_style_tokens(product))
            brand_bonus = 0.0
            total_score = base_score + style_bonus + brand_bonus
            ranked.append((product_id, total_score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    @classmethod
    def _get_category_key_for_product(cls, product) -> str:
        if not product:
            return None

        sub_category = (getattr(product, "subCategory", "") or "").lower().strip()
        article_type = (getattr(product, "articleType", "") or "").lower().strip()

        if sub_category == "topwear" or article_type in ["jackets", "shirts", "sweaters", "sweatshirts", "tops", "tshirts", "tunics"]:
            return "apparel_topwear"

        if sub_category == "bottomwear" or article_type in ["capris", "jeans", "shorts", "skirts", "track pants", "tracksuits", "trousers"]:
            return "apparel_bottomwear"

        if sub_category in ["shoes", "sandal", "flip flops"] or article_type in ["flip flops", "sandals", "sports sandals", "casual shoes", "flats", "formal shoes", "heels", "sports shoes"]:
            return "footwear"

        if sub_category in ["bags", "belts", "headwear", "watches"] or article_type in ["backpacks", "handbags", "belts", "caps", "watches"]:
            return "accessories"

        if sub_category == "dress" or article_type == "dresses":
            return "apparel_dress"

        if sub_category == "innerwear" or article_type == "bra":
            return "apparel_innerwear"

        return None

    @staticmethod
    def _product_matches_category(product, target_category: str) -> bool:
        product_sub_category = (getattr(product, "subCategory", "") or "").lower().strip()
        product_article_type = (getattr(product, "articleType", "") or "").lower().strip()
        target_category = target_category.lower().strip()

        topwear_articles = ["jackets", "shirts", "sweaters", "sweatshirts", "tops", "tshirts", "tunics"]
        bottomwear_articles = ["capris", "jeans", "shorts", "skirts", "track pants", "tracksuits", "trousers"]
        footwear_articles = ["flip flops", "sandals", "sports sandals", "casual shoes", "flats", "formal shoes", "heels", "sports shoes"]
        accessories_articles = ["backpacks", "handbags", "belts", "caps", "watches"]

        if target_category == "topwear":
            return product_sub_category == "topwear" or product_article_type in topwear_articles
        if target_category == "bottomwear":
            return product_sub_category == "bottomwear" or product_article_type in bottomwear_articles
        if target_category == "footwear":
            return product_sub_category in ["shoes", "sandal", "flip flops"] or product_article_type in footwear_articles
        if target_category == "accessories":
            return product_sub_category in ["bags", "belts", "headwear", "watches"] or product_article_type in accessories_articles

        return False

def _extract_style_tokens(product) -> Iterable[str]:
    style_tags = []
    if isinstance(getattr(product, "style_tags", None), list):
        style_tags.extend(product.style_tags)
    if isinstance(getattr(product, "outfit_tags", None), list):
        style_tags.extend(product.outfit_tags)
    if getattr(product, "articleType", None):
        style_tags.append(product.articleType)
    if getattr(product, "subCategory", None):
        style_tags.append(product.subCategory)
    return {str(token).lower() for token in style_tags if token}

