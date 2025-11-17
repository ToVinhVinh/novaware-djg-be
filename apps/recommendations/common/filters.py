"""Candidate filtering utilities for recommendation engines."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from bson import ObjectId
from django.core.exceptions import ValidationError
try:
    # Optional: available only when MongoDB connection is configured
    from apps.users.mongo_models import User as MongoUser  # type: ignore
    from apps.products.mongo_models import Product as MongoProduct  # type: ignore
    from apps.products.mongo_models import Category as MongoCategory  # type: ignore
    from apps.brands.mongo_models import Brand as MongoBrand  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    MongoUser = None  # type: ignore
    MongoProduct = None  # type: ignore
    MongoCategory = None  # type: ignore
    MongoBrand = None  # type: ignore

from .constants import INTERACTION_WEIGHTS, MAX_STYLE_TAGS
from .gender_utils import gender_filter_values, normalize_gender
from .context import RecommendationContext

logger = logging.getLogger(__name__)


class CandidateFilter:
    """Build candidate pools that satisfy strict business constraints."""

    _product_field_names_cache: set[str] | None = None

    @classmethod
    def build_context(
        cls,
        *,
        user_id: str | int,
        current_product_id: str | int,
        top_k_personal: int,
        top_k_outfit: int,
        request_params: dict | None = None,
    ) -> RecommendationContext:
        user = cls._resolve_user(user_id)
        current_product, product_id_to_mongo_id = cls._resolve_product_with_mapping(current_product_id, owner_user=user)

        resolved_gender = cls._resolve_gender(user, current_product)
        resolved_age_group = cls._resolve_age_group(user, current_product)

        interactions = cls._load_interactions(user)
        history_products = [getattr(interaction, 'product', None) for interaction in interactions if getattr(interaction, 'product_id', None)]
        history_products = [p for p in history_products if p is not None]
        
        current_product_id = getattr(current_product, 'id', None)
        excluded_ids = {current_product_id} if current_product_id else set()
        for product in history_products:
            product_id = getattr(product, 'id', None)
            if product_id:
                excluded_ids.add(product_id)

        candidate_products = cls._build_candidate_pool(
            gender=resolved_gender,
            age_group=resolved_age_group,
            excluded_ids=excluded_ids,
        )

        if not candidate_products:
            # Relaxed fallback: allow same gender but drop age restriction only if nothing matches.
            fallback_products = cls._fallback_candidates(
                gender=resolved_gender,
                excluded_ids=excluded_ids,
            )
            candidate_products = fallback_products

        # Build MongoDB ID mapping for all candidate products (one-time batch operation)
        product_id_to_mongo_id = cls._build_mongo_mapping(candidate_products, product_id_to_mongo_id)
        # Also map current product and history products
        product_id_to_mongo_id = cls._build_mongo_mapping([current_product], product_id_to_mongo_id)
        product_id_to_mongo_id = cls._build_mongo_mapping(history_products, product_id_to_mongo_id)

        style_counter = cls._build_style_profile(interactions, history_products, user, current_product)
        brand_counter = cls._build_brand_profile(interactions)
        interaction_weight_map = defaultdict(float)
        for interaction in interactions:
            product_id = getattr(interaction, 'product_id', None)
            if not product_id:
                continue
            interaction_type = getattr(interaction, 'interaction_type', None)
            interaction_weight_map[product_id] += INTERACTION_WEIGHTS.get(
                interaction_type,
                1.0,
            )

        return RecommendationContext(
            user=user,
            current_product=current_product,
            top_k_personal=top_k_personal,
            top_k_outfit=top_k_outfit,
            interactions=interactions,
            history_products=history_products,
            candidate_products=candidate_products,
            excluded_product_ids=excluded_ids,
            style_counter=dict(style_counter),
            brand_counter=dict(brand_counter),
            interaction_weights=dict(interaction_weight_map),
            resolved_gender=resolved_gender,
            resolved_age_group=resolved_age_group,
            request_params=request_params or {},
            product_id_to_mongo_id=product_id_to_mongo_id,
        )

    @staticmethod
    def _looks_like_object_id(value: str | int) -> bool:
        if not isinstance(value, str):
            return False
        return len(value) == 24 and all(c in "0123456789abcdefABCDEF" for c in value)

    @classmethod
    def _resolve_user(cls, user_id: str | int):
        """Resolve user from MongoDB only."""
        if MongoUser is None:
            raise ValidationError({"user_id": "MongoDB not configured"})
        
        # Try as integer ID first
        if isinstance(user_id, int) or (isinstance(user_id, str) and user_id.isdigit()):
            try:
                mongo_user = MongoUser.objects(id=int(user_id)).first()
                if mongo_user:
                    return mongo_user
            except Exception:
                pass
        
        # Try as ObjectId
        if cls._looks_like_object_id(user_id):
            try:
                mongo_user = MongoUser.objects(id=ObjectId(str(user_id))).first()
                if mongo_user:
                    return mongo_user
            except Exception:
                pass
        
        raise ValidationError({"user_id": "User not found"})

    @classmethod
    def _resolve_product_with_mapping(cls, product_id: str | int, owner_user) -> tuple:
        """Resolve product from MongoDB only."""
        if MongoProduct is None:
            raise ValidationError({"current_product_id": "MongoDB not configured"})
        
        mongo_product, original_mongo_id = cls._fetch_mongo_product(product_id)
        if mongo_product:
            product_id_to_mongo_id = {}
            if original_mongo_id:
                # Use a simple numeric key for mapping (can be the mongo ID itself)
                product_id_to_mongo_id[hash(original_mongo_id) % (10 ** 8)] = original_mongo_id
            return mongo_product, product_id_to_mongo_id
        
        raise ValidationError({"current_product_id": "Product not found"})
    
    @classmethod
    def _resolve_product(cls, product_id: str | int, owner_user):
        """Backward compatibility wrapper."""
        product, _ = cls._resolve_product_with_mapping(product_id, owner_user)
        return product

    @staticmethod
    def _build_mongo_mapping(products: list, existing_mapping: dict[int, str]) -> dict[int, str]:
        """Build MongoDB ObjectId mapping for a list of Mongo products."""
        if not products or not MongoProduct:
            return existing_mapping
        
        mapping = existing_mapping.copy()
        
        try:
            for product in products:
                product_id = getattr(product, 'id', None)
                if product_id:
                    # Create a numeric hash key for the mapping
                    key = hash(str(product_id)) % (10 ** 8)
                    mapping[key] = str(product_id)
        except Exception:
            pass
        
        return mapping

    @staticmethod
    def _load_interactions(user) -> list:
        """Load user interactions from MongoDB."""
        # TODO: Implement MongoDB-based interaction loading
        # For now, return empty list to avoid SQL queries
        return []

    @classmethod
    def _build_candidate_pool(
        cls,
        *,
        gender: str,
        age_group: str,
        excluded_ids: set[int],
    ) -> list:
        """Build candidate pool with strict gender and age filtering from MongoDB."""
        if MongoProduct is None:
            return []
        
        allowed_genders = cls._allowed_genders(gender)
        gender_filters = cls._gender_query_values(allowed_genders)
        allowed_gender_lower = {g.lower() for g in allowed_genders if g}
        normalized_age_group = (age_group or "").strip().lower()
        
        try:
            query_filters: dict[str, Any] = {}
            if gender_filters:
                query_filters['gender__in'] = gender_filters
            if normalized_age_group and cls._product_has_field("age_group"):
                query_filters['age_group__iexact'] = normalized_age_group
            
            products = list(MongoProduct.objects(**query_filters))
            
            filtered_products = []
            for product in products:
                product_id = getattr(product, 'id', None)
                if not product_id or product_id in excluded_ids:
                    continue
                
                product_gender = (getattr(product, "gender", "") or "").strip().lower()
                product_age = (getattr(product, "age_group", "") or "").strip().lower()
                
                if product_gender and product_gender not in allowed_gender_lower:
                    continue
                if normalized_age_group and product_age and product_age != normalized_age_group:
                    continue
                
                filtered_products.append(product)
            
            return cls._deduplicate(filtered_products)
        except Exception:
            return []

    @classmethod
    def _fallback_candidates(cls, *, gender: str, excluded_ids: set[int]) -> list:
        """Fallback candidate pool with relaxed age but strict gender filtering from MongoDB."""
        if MongoProduct is None:
            return []
        
        allowed_genders = cls._allowed_genders(gender)
        gender_filters = cls._gender_query_values(allowed_genders)
        allowed_gender_lower = {g.lower() for g in allowed_genders if g}
        
        try:
            query_filters: dict[str, Any] = {}
            if gender_filters:
                query_filters['gender__in'] = gender_filters
            
            products = list(MongoProduct.objects(**query_filters))
            
            # Filter out excluded IDs and check gender
            filtered_products = []
            for product in products:
                product_id = getattr(product, 'id', None)
                if not product_id or product_id in excluded_ids:
                    continue
                product_gender = (getattr(product, "gender", "") or "").strip().lower()
                
                if product_gender and product_gender not in allowed_gender_lower:
                    continue
                
                filtered_products.append(product)
            
            return cls._deduplicate(filtered_products)
        except Exception:
            return []

    @classmethod
    def _product_field_names(cls) -> set[str]:
        """Get MongoDB product field names."""
        if cls._product_field_names_cache is None and MongoProduct:
            # Get field names from MongoEngine document
            cls._product_field_names_cache = set(MongoProduct._fields.keys()) if hasattr(MongoProduct, '_fields') else set()
        return cls._product_field_names_cache or set()

    @classmethod
    def _product_has_field(cls, field_name: str) -> bool:
        return field_name in cls._product_field_names()
    
    @staticmethod
    def _gender_query_values(values: Iterable[str]) -> list[str]:
        """Build a case-insensitive gender filter list for Mongo queries."""
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            if not value:
                continue
            variants = {
                value,
                value.lower(),
                value.upper(),
                value.title(),
            }
            for variant in variants:
                if not variant:
                    continue
                if variant not in seen:
                    seen.add(variant)
                    normalized.append(variant)
        return normalized

    @classmethod
    def _fetch_mongo_product(cls, product_id: str | int):
        if MongoProduct is None:
            return None, None
        candidate_ids: list[int] = []
        if isinstance(product_id, int):
            candidate_ids.append(product_id)
        elif isinstance(product_id, str):
            stripped = product_id.strip()
            if stripped.isdigit():
                try:
                    candidate_ids.append(int(stripped))
                except ValueError:
                    pass
        for cid in candidate_ids:
            try:
                mongo_product = MongoProduct.objects(id=cid).first()
                if mongo_product:
                    return mongo_product, str(cid)
            except Exception:
                pass
        if cls._looks_like_object_id(product_id):
            try:
                oid = ObjectId(str(product_id))
                mongo_product = MongoProduct.objects(id=oid).first()
                if mongo_product:
                    return mongo_product, str(oid)
            except Exception:
                pass
        return None, None



    @staticmethod
    def _deduplicate(products: Sequence) -> list:
        """Deduplicate products by ID."""
        seen: set = set()
        unique_products: list = []
        for product in products:
            product_id = getattr(product, 'id', None)
            if product_id and product_id in seen:
                continue
            if product_id:
                seen.add(product_id)
            unique_products.append(product)
        return unique_products

    @staticmethod
    def _allowed_genders(gender: str) -> list[str]:
        return gender_filter_values(gender)

    @staticmethod
    def _resolve_gender(user, current_product) -> str:
        # Always prioritize user's gender over product gender
        user_gender = normalize_gender(getattr(user, "gender", ""))
        if user_gender in ("male", "female", "unisex"):
            return user_gender
        
        # If user gender is not specified, use product gender as fallback
        product_gender = normalize_gender(getattr(current_product, "gender", ""))
        if product_gender in ("male", "female", "unisex"):
            return product_gender
        
        return "unisex"

    @staticmethod
    def _resolve_age_group(user, current_product) -> str:
        if hasattr(user, "age") and user.age:
            age = user.age
            if age <= 12:
                return "kid"
            if age <= 19:
                return "teen"
            return "adult"
        prod_age = getattr(current_product, "age_group", None)
        if prod_age in ("kid", "teen", "adult"):
            return prod_age
        return "adult"

    @staticmethod
    def _build_style_profile(
        interactions: Iterable,
        history_products: Iterable,
        user,
        current_product,
    ) -> Counter:
        counter: Counter = Counter()
        product_weights: defaultdict = defaultdict(float)
        for interaction in interactions:
            product_id = getattr(interaction, 'product_id', None)
            if not product_id:
                continue
            interaction_type = getattr(interaction, 'interaction_type', None)
            weight = INTERACTION_WEIGHTS.get(interaction_type, 1.0)
            product_weights[product_id] += weight
        for product in history_products:
            product_id = getattr(product, 'id', None)
            weight = product_weights.get(product_id, 1.0)
            for token in _collect_style_tokens(product):
                counter[token] += weight
        preference_styles = _extract_user_preference_styles(user)
        for token in preference_styles:
            counter[token] += 1.5
        for token in _collect_style_tokens(current_product):
            counter[token] += 0.5
        most_common = counter.most_common(MAX_STYLE_TAGS)
        return Counter(dict(most_common))

    @staticmethod
    def _build_brand_profile(interactions: Iterable) -> Counter:
        brand_counter: Counter = Counter()
        for interaction in interactions:
            product = getattr(interaction, 'product', None)
            if not product:
                continue
            # Brand field removed from Product model - returning empty counter for now
            # TODO: Implement brand tracking if needed
        return brand_counter


def _collect_style_tokens(product) -> list[str]:
    tokens: list[str] = []
    if isinstance(getattr(product, "style_tags", None), list):
        tokens.extend(token.lower() for token in product.style_tags if token)
    if isinstance(getattr(product, "outfit_tags", None), list):
        tokens.extend(token.lower() for token in product.outfit_tags if token)
    
    # Use new Product fields
    if getattr(product, "articleType", None):
        tokens.append(product.articleType.lower())
    if getattr(product, "subCategory", None):
        tokens.append(product.subCategory.lower())
    if getattr(product, "masterCategory", None):
        tokens.append(product.masterCategory.lower())
    if getattr(product, "baseColour", None):
        tokens.append(product.baseColour.lower())
    if getattr(product, "usage", None):
        tokens.append(product.usage.lower())
    if getattr(product, "season", None):
        tokens.append(product.season.lower())
    
    return tokens


def _extract_user_preference_styles(user) -> list[str]:
    preferences = getattr(user, "preferences", {}) or {}
    styles = preferences.get("styles") or preferences.get("style_tags") or []
    normalized: list[str] = []
    for token in styles:
        if not token:
            continue
        normalized.append(str(token).lower())
    return normalized

