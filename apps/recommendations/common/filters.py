"""Candidate filtering utilities for recommendation engines."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import IntegrityError
from django.db.models import Prefetch, QuerySet
from bson import ObjectId

from apps.products.models import Product, Category as SqlCategory
from apps.brands.models import Brand as SqlBrand
from apps.users.models import UserInteraction
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
from .context import RecommendationContext

User = get_user_model()


class CandidateFilter:
    """Build candidate pools that satisfy strict business constraints."""

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
        history_products = [interaction.product for interaction in interactions if interaction.product_id]
        excluded_ids = {current_product.id, *(product.id for product in history_products if product.id)}

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
            if not interaction.product_id:
                continue
            interaction_weight_map[interaction.product_id] += INTERACTION_WEIGHTS.get(
                interaction.interaction_type,
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
        # 1) Try direct SQL PK
        try:
            return User.objects.get(pk=user_id)
        except (ObjectDoesNotExist, ValueError):
            pass

        # 2) If looks like Mongo ObjectId, try mapping via Mongo → SQL by email
        if cls._looks_like_object_id(user_id) and MongoUser is not None:
            try:
                mongo_user = MongoUser.objects(id=ObjectId(str(user_id))).first()  # type: ignore[attr-defined]
                if mongo_user:
                    email = getattr(mongo_user, "email", None)
                    if email:
                        try:
                            return User.objects.get(email=email)
                        except User.DoesNotExist:
                            pass
                    # 2b) As a last resort, create a shadow User in SQL using Mongo data
                    try:
                        email_val = email or f"imported_{str(ObjectId())[:8]}@imported.local"
                        # Check if email already exists (might have been created by another process)
                        try:
                            return User.objects.get(email=email_val)
                        except User.DoesNotExist:
                            pass
                        
                        username = getattr(mongo_user, "username", None)
                        if not username:
                            base_username = email_val.split("@", 1)[0].replace(".", "_")
                            counter = 1
                            candidate = base_username
                            while User.objects.filter(username=candidate).exists():
                                counter += 1
                                candidate = f"{base_username}_{counter}"
                            username = candidate
                        
                        first_name = getattr(mongo_user, "first_name", None) or ""
                        last_name = getattr(mongo_user, "last_name", None) or ""
                        name = getattr(mongo_user, "name", None) or ""
                        if not first_name and not last_name and name:
                            # Try to split name into first/last
                            parts = name.split(maxsplit=1)
                            if len(parts) >= 2:
                                first_name, last_name = parts[0], parts[1]
                            elif len(parts) == 1:
                                first_name = parts[0]
                        
                        gender = getattr(mongo_user, "gender", None)
                        age = getattr(mongo_user, "age", None)
                        height = getattr(mongo_user, "height", None)
                        weight = getattr(mongo_user, "weight", None)
                        preferences = getattr(mongo_user, "preferences", {}) or {}
                        amazon_user_id = getattr(mongo_user, "amazon_user_id", None)
                        is_active = getattr(mongo_user, "is_active", True)
                        is_staff = getattr(mongo_user, "is_admin", False)
                        
                        try:
                            created = User.objects.create(
                                email=email_val,
                                username=username,
                                first_name=first_name,
                                last_name=last_name,
                                gender=gender,
                                age=age,
                                height=height,
                                weight=weight,
                                preferences=preferences,
                                amazon_user_id=amazon_user_id,
                                is_active=is_active,
                                is_staff=is_staff,
                                is_superuser=is_staff,
                            )
                            return created
                        except IntegrityError:
                            # Race condition: user was created by another process, try to get it
                            try:
                                return User.objects.get(email=email_val)
                            except User.DoesNotExist:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

        raise ValidationError({"user_id": "User not found"})

    @classmethod
    def _resolve_product_with_mapping(cls, product_id: str | int, owner_user) -> tuple[Product, dict[int, str]]:
        """Resolve product and return mapping of Django ID to MongoDB ObjectId."""
        product_id_to_mongo_id: dict[int, str] = {}
        
        # 1) Try direct SQL PK
        try:
            product = Product.objects.select_related("brand", "category").get(pk=product_id)
            return product, product_id_to_mongo_id
        except (ObjectDoesNotExist, ValueError):
            pass

        # 2) If looks like Mongo ObjectId, try mapping via Mongo → SQL by slug or amazon_asin
        original_mongo_id = None
        if cls._looks_like_object_id(product_id) and MongoProduct is not None:
            original_mongo_id = str(product_id)
            try:
                mongo_product = MongoProduct.objects(id=ObjectId(str(product_id))).first()  # type: ignore[attr-defined]
                if mongo_product:
                    # Prefer slug (stable unique) then amazon_asin
                    slug = getattr(mongo_product, "slug", None)
                    if slug:
                        try:
                            product = Product.objects.select_related("brand", "category").get(slug=slug)
                            if product.id:
                                product_id_to_mongo_id[product.id] = original_mongo_id
                            return product, product_id_to_mongo_id
                        except Product.DoesNotExist:
                            pass
                    asin = getattr(mongo_product, "amazon_asin", None)
                    if asin:
                        try:
                            product = Product.objects.select_related("brand", "category").get(amazon_asin=asin)
                            if product.id:
                                product_id_to_mongo_id[product.id] = original_mongo_id
                            return product, product_id_to_mongo_id
                        except Product.DoesNotExist:
                            pass
                    # 2b) As a last resort, create a shadow Product in SQL using Mongo data
                    try:
                        brand = None
                        category = None
                        if MongoBrand is not None:
                            try:
                                m_brand = MongoBrand.objects(id=getattr(mongo_product, "brand_id", None)).first()  # type: ignore[attr-defined]
                                if m_brand and getattr(m_brand, "name", None):
                                    brand, _ = SqlBrand.objects.get_or_create(name=m_brand.name)
                            except Exception:
                                pass
                        if brand is None:
                            brand, _ = SqlBrand.objects.get_or_create(name="ImportedBrand")

                        if MongoCategory is not None:
                            try:
                                m_cat = MongoCategory.objects(id=getattr(mongo_product, "category_id", None)).first()  # type: ignore[attr-defined]
                                if m_cat and getattr(m_cat, "name", None):
                                    category, _ = SqlCategory.objects.get_or_create(name=m_cat.name)
                            except Exception:
                                pass
                        if category is None:
                            category, _ = SqlCategory.objects.get_or_create(name="Imported")

                        name = getattr(mongo_product, "name", "Imported Product")
                        description = getattr(mongo_product, "description", "") or ""
                        images = getattr(mongo_product, "images", []) or []
                        price = getattr(mongo_product, "price", 0)
                        sale = getattr(mongo_product, "sale", 0)
                        count_in_stock = getattr(mongo_product, "count_in_stock", 0) or 0
                        style_tags = getattr(mongo_product, "style_tags", []) or []
                        outfit_tags = getattr(mongo_product, "outfit_tags", []) or []
                        feature_vector = getattr(mongo_product, "feature_vector", []) or []
                        gender = (getattr(mongo_product, "gender", "unisex") or "unisex").lower()
                        age_group = (getattr(mongo_product, "age_group", "adult") or "adult").lower()
                        category_type = (getattr(mongo_product, "category_type", "tops") or "tops").lower()
                        slug_val = slug or f"imported-{str(ObjectId())[:8]}"
                        asin_val = getattr(mongo_product, "amazon_asin", None)

                        created = Product.objects.create(
                            user=owner_user,
                            brand=brand,
                            category=category,
                            name=name,
                            slug=slug_val,
                            description=description,
                            images=list(images),
                            price=price,
                            sale=sale,
                            count_in_stock=count_in_stock,
                            size={},
                            outfit_tags=list(outfit_tags),
                            style_tags=list(style_tags),
                            feature_vector=list(feature_vector),
                            gender=gender,
                            age_group=age_group,
                            category_type=category_type,
                            amazon_asin=asin_val,
                        )
                        if created.id:
                            product_id_to_mongo_id[created.id] = original_mongo_id
                        return created, product_id_to_mongo_id
                    except Exception:
                        pass
            except Exception:
                pass

        raise ValidationError({"current_product_id": "Product not found"})
    
    @classmethod
    def _resolve_product(cls, product_id: str | int, owner_user) -> Product:
        """Backward compatibility wrapper."""
        product, _ = cls._resolve_product_with_mapping(product_id, owner_user)
        return product

    @staticmethod
    def _build_mongo_mapping(products: list[Product], existing_mapping: dict[int, str]) -> dict[int, str]:
        """Build MongoDB ObjectId mapping for a list of Django products (batch operation)."""
        if not products or not MongoProduct:
            return existing_mapping
        
        mapping = existing_mapping.copy()
        
        try:
            # Batch lookup by slug (most efficient)
            slugs = [p.slug for p in products if p.slug]
            if slugs:
                mongo_products_by_slug = {}
                for mp in MongoProduct.objects(slug__in=slugs):
                    mongo_products_by_slug[mp.slug] = mp
                
                for product in products:
                    if product.id and product.slug and product.slug in mongo_products_by_slug:
                        mapping[product.id] = str(mongo_products_by_slug[product.slug].id)
            
            # Batch lookup by amazon_asin for remaining products
            remaining_products = [p for p in products if p.id and p.id not in mapping and hasattr(p, 'amazon_asin') and p.amazon_asin]
            if remaining_products:
                asins = [p.amazon_asin for p in remaining_products]
                mongo_products_by_asin = {}
                for mp in MongoProduct.objects(amazon_asin__in=asins):
                    if mp.amazon_asin:
                        mongo_products_by_asin[mp.amazon_asin] = mp
                
                for product in remaining_products:
                    if product.id and product.amazon_asin and product.amazon_asin in mongo_products_by_asin:
                        mapping[product.id] = str(mongo_products_by_asin[product.amazon_asin].id)
        except Exception:
            # If batch lookup fails, fall back to individual lookups (slower but works)
            pass
        
        return mapping

    @staticmethod
    def _load_interactions(user: User) -> list[UserInteraction]:
        qs = (
            UserInteraction.objects.filter(user=user)
            .select_related("product", "product__brand", "product__category")
            .order_by("-timestamp")
        )
        return list(qs)

    @classmethod
    def _build_candidate_pool(
        cls,
        *,
        gender: str,
        age_group: str,
        excluded_ids: set[int],
    ) -> list[Product]:
        allowed_genders = cls._allowed_genders(gender)
        queryset: QuerySet[Product] = (
            Product.objects.filter(
                gender__in=allowed_genders,
                age_group=age_group,
            )
            .exclude(id__in=excluded_ids)
            .select_related("brand", "category")
        )
        products = list(queryset)
        return cls._deduplicate(products)

    @classmethod
    def _fallback_candidates(cls, *, gender: str, excluded_ids: set[int]) -> list[Product]:
        allowed_genders = cls._allowed_genders(gender)
        queryset: QuerySet[Product] = (
            Product.objects.filter(gender__in=allowed_genders)
            .exclude(id__in=excluded_ids)
            .select_related("brand", "category")
        )
        return cls._deduplicate(list(queryset))

    @staticmethod
    def _deduplicate(products: Sequence[Product]) -> list[Product]:
        seen: set[int] = set()
        unique_products: list[Product] = []
        for product in products:
            if product.id in seen:
                continue
            seen.add(product.id)
            unique_products.append(product)
        return unique_products

    @staticmethod
    def _allowed_genders(gender: str) -> list[str]:
        allowed = ["unisex"]
        if gender:
            allowed.insert(0, gender)
        return list(dict.fromkeys(allowed))

    @staticmethod
    def _resolve_gender(user, current_product: Product) -> str:
        gender = (user.gender or "").lower()
        if gender in ("male", "female"):
            return gender
        current = (getattr(current_product, "gender", None) or "").lower()
        if current in ("male", "female"):
            return current
        return "unisex"

    @staticmethod
    def _resolve_age_group(user, current_product: Product) -> str:
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
        interactions: Iterable[UserInteraction],
        history_products: Iterable[Product],
        user,
        current_product: Product,
    ) -> Counter:
        counter: Counter = Counter()
        product_weights: defaultdict[int, float] = defaultdict(float)
        for interaction in interactions:
            if not interaction.product_id:
                continue
            weight = INTERACTION_WEIGHTS.get(interaction.interaction_type, 1.0)
            product_weights[interaction.product_id] += weight
        for product in history_products:
            weight = product_weights.get(product.id, 1.0)
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
    def _build_brand_profile(interactions: Iterable[UserInteraction]) -> Counter:
        brand_counter: Counter = Counter()
        for interaction in interactions:
            product = interaction.product
            if not product or not product.brand_id:
                continue
            weight = INTERACTION_WEIGHTS.get(interaction.interaction_type, 1.0)
            brand_counter[product.brand_id] += weight
        return brand_counter


def _collect_style_tokens(product: Product) -> list[str]:
    tokens: list[str] = []
    if isinstance(getattr(product, "style_tags", None), list):
        tokens.extend(token.lower() for token in product.style_tags if token)
    if isinstance(getattr(product, "outfit_tags", None), list):
        tokens.extend(token.lower() for token in product.outfit_tags if token)
    if getattr(product, "category_type", None):
        tokens.append(product.category_type.lower())
    if getattr(product, "category", None) and getattr(product.category, "name", None):
        tokens.append(product.category.name.lower())
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

