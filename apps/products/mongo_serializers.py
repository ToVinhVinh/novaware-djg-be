"""Serializers cho MongoEngine Product models."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from bson import ObjectId
from bson.errors import InvalidId
from django.conf import settings
from django.utils.text import slugify
from rest_framework import serializers

from .mongo_models import (
    Category,
    Color,
    ContentSection,
    Product,
    ProductReview,
    ProductVariant,
    Size,
)
from apps.brands.mongo_models import Brand
from apps.brands.mongo_serializers import BrandSerializer


class ColorSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    name = serializers.CharField()
    hex_code = serializers.CharField()
    
    def get_id(self, obj):
        return str(obj.id)


class SizeSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    name = serializers.CharField()
    code = serializers.CharField()
    
    def get_id(self, obj):
        return str(obj.id)


class ProductVariantSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    color = ColorSerializer(read_only=True)
    color_id = serializers.CharField(write_only=True, required=False)
    size = SizeSerializer(read_only=True)
    size_id = serializers.CharField(write_only=True, required=False)
    price = serializers.DecimalField(max_digits=10, decimal_places=2)
    stock = serializers.IntegerField()
    
    def get_id(self, obj):
        return str(obj.id)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        data = {
            "id": str(instance.id),
            "price": float(instance.price),
            "stock": instance.stock,
        }
        
        # Load color and size
        try:
            color = Color.objects.get(id=instance.color_id)
            data["color"] = ColorSerializer(color).data
            data["color_id"] = str(instance.color_id)
        except Color.DoesNotExist:
            data["color"] = None
            data["color_id"] = str(instance.color_id)
        
        try:
            size = Size.objects.get(id=instance.size_id)
            data["size"] = SizeSerializer(size).data
            data["size_id"] = str(instance.size_id)
        except Size.DoesNotExist:
            data["size"] = None
            data["size_id"] = str(instance.size_id)
        
        return data


class CategorySerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    name = serializers.CharField()
    
    def get_id(self, obj):
        return str(obj.id)


class ProductReviewSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    name = serializers.CharField()
    rating = serializers.IntegerField()
    comment = serializers.CharField()
    user_id = serializers.SerializerMethodField()
    user_name = serializers.SerializerMethodField()
    created_at = serializers.DateTimeField()
    
    def get_id(self, obj):
        return str(obj.id)
    
    def get_user_id(self, obj):
        return str(obj.user_id)
    
    def get_user_name(self, obj):
        from apps.users.mongo_models import User
        try:
            user = User.objects.get(id=obj.user_id)
            return user.username or user.email
        except User.DoesNotExist:
            return None


class ProductSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    user_id = serializers.CharField(write_only=True, required=False)
    brand_id = serializers.CharField()
    brand_name = serializers.SerializerMethodField()
    category_id = serializers.CharField()
    category_detail = CategorySerializer(read_only=True, source="category")
    name = serializers.CharField()
    slug = serializers.SlugField()
    description = serializers.CharField()
    images = serializers.ListField(child=serializers.CharField())
    rating = serializers.FloatField(read_only=True)
    num_reviews = serializers.IntegerField(read_only=True)
    price = serializers.DecimalField(max_digits=10, decimal_places=2)
    sale = serializers.DecimalField(max_digits=5, decimal_places=2)
    count_in_stock = serializers.IntegerField(read_only=True)
    size = serializers.DictField()
    color_ids = serializers.ListField(child=serializers.CharField(), required=False)
    variants = ProductVariantSerializer(many=True, read_only=True)
    variants_payload = ProductVariantSerializer(many=True, write_only=True, required=False)
    outfit_tags = serializers.ListField(child=serializers.CharField())
    compatible_product_ids = serializers.ListField(child=serializers.CharField(), required=False)
    feature_vector = serializers.ListField(child=serializers.FloatField())
    amazon_asin = serializers.CharField(required=False, allow_null=True)
    amazon_parent_asin = serializers.CharField(required=False, allow_null=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def get_brand_name(self, obj):
        from apps.brands.mongo_models import Brand
        try:
            brand = Brand.objects.get(id=obj.brand_id)
            return brand.name
        except Brand.DoesNotExist:
            return None
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _load_sample_product():
        """Load sample product payload to use as a fallback for empty fields."""
        sample_path = os.path.join(settings.BASE_DIR, "sample-product.txt")
        if not os.path.exists(sample_path):
            return {}
        try:
            with open(sample_path, "r", encoding="utf-8") as sample_file:
                data = json.load(sample_file)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    @staticmethod
    @lru_cache(maxsize=1)
    def _brand_catalog() -> List[Dict[str, object]]:
        """Cache toàn bộ brand để reuse khi mapping dữ liệu."""
        try:
            brands = Brand.objects.only("id", "name").order_by("name")
        except Exception:
            return []

        catalog: List[Dict[str, object]] = []
        for brand in brands:
            name = getattr(brand, "name", "") or ""
            serializer_data = BrandSerializer(brand).data
            catalog.append(
                {
                    "id": str(brand.id),
                    "name": name,
                    "name_lower": name.lower(),
                    "slug": slugify(name) if name else "",
                    "data": serializer_data,
                }
            )
        return catalog

    @staticmethod
    @lru_cache(maxsize=1)
    def _category_catalog() -> Dict[str, Dict[str, object]]:
        """Cache category theo slug/name để map nhanh."""
        try:
            categories = Category.objects.only("id", "name")
        except Exception:
            return {}

        catalog: Dict[str, Dict[str, object]] = {}
        for category in categories:
            name = getattr(category, "name", "") or ""
            slug_value = slugify(name) if name else ""
            serializer_data = CategorySerializer(category).data
            payload = {
                "id": str(category.id),
                "name": name,
                "slug": slug_value,
                "data": serializer_data,
            }
            keys = {name.lower(), slug_value}
            for key in filter(None, keys):
                catalog[key] = payload
        return catalog

    @staticmethod
    @lru_cache(maxsize=1)
    def _default_color_entries() -> List[Dict[str, object]]:
        """Lấy danh sách màu mặc định (fallback)."""
        try:
            colors = list(Color.objects.only("id", "name", "hex_code").order_by("name")[:6])
        except Exception:
            return []

        entries: List[Dict[str, object]] = []
        for color in colors:
            mongo_doc = color.to_mongo().to_dict()
            entries.append(
                {
                    "id": str(color.id),
                    "name": mongo_doc.get("name") or getattr(color, "name", None),
                    "hex": mongo_doc.get("hex_code")
                    or mongo_doc.get("hexCode")
                    or getattr(color, "hex_code", None),
                }
            )
        return entries

    @staticmethod
    @lru_cache(maxsize=1)
    def _size_catalog() -> Dict[str, Dict[str, object]]:
        """Cache size theo code để build variants fallback."""
        try:
            sizes = Size.objects.only("id", "name", "code")
        except Exception:
            return {}

        catalog: Dict[str, Dict[str, object]] = {}
        for size in sizes:
            mongo_doc = size.to_mongo().to_dict()
            code = (mongo_doc.get("code") or getattr(size, "code", "")).upper()
            if not code:
                continue
            catalog[code.lower()] = {
                "id": str(size.id),
                "code": code,
                "name": mongo_doc.get("name") or getattr(size, "name", code),
            }
        return catalog

    @staticmethod
    @lru_cache(maxsize=16)
    def _category_product_pool(category_type: str) -> List[str]:
        """Danh sách product id theo category để làm compatible fallback."""
        if not category_type:
            return []
        try:
            queryset = Product.objects(category_type=category_type).only("id").order_by("-rating", "-created_at")
        except Exception:
            return []
        return [str(product.id) for product in queryset]

    @staticmethod
    def _pseudo_object_id(seed: str) -> str:
        """Sinh ObjectId giả định từ seed để đảm bảo tính ổn định."""
        normalized = (seed or "novaware-fallback").encode("utf-8")
        digest = hashlib.md5(normalized).hexdigest()[:24]
        try:
            return str(ObjectId(digest))
        except InvalidId:
            return digest.zfill(24)[:24]

    @staticmethod
    def _infer_brand_name(product_name: str) -> str:
        if not product_name:
            return "Unknown Brand"
        cleaned = product_name.strip().strip("\"' ")
        separators = [" - ", " | ", ": ", " – ", " — "]
        for separator in separators:
            if separator in cleaned:
                cleaned = cleaned.split(separator)[0]
                break
        tokens = cleaned.split()
        if not tokens:
            return "Unknown Brand"
        if tokens[0].startswith("-") and len(tokens) > 1:
            tokens[0] = tokens[0].lstrip("-")
        if len(tokens[0]) <= 2 and len(tokens) > 1:
            return " ".join(tokens[:2]).strip()
        return tokens[0].strip() if len(tokens) == 1 else " ".join(tokens[:2]).strip()

    @classmethod
    def _resolve_brand(cls, product_name: str, product_slug: str) -> Dict[str, object]:
        catalog = cls._brand_catalog()
        lower_name = (product_name or "").lower()
        slug_value = product_slug or ""

        for entry in catalog:
            if entry["name_lower"] and entry["name_lower"] in lower_name:
                return entry
        for entry in catalog:
            if entry["slug"] and entry["slug"] in slug_value:
                return entry

        inferred_name = cls._infer_brand_name(product_name)
        fallback_id = cls._pseudo_object_id(f"{inferred_name}:{product_slug}")
        return {
            "id": fallback_id,
            "name": inferred_name or "Unknown Brand",
            "slug": slugify(inferred_name) if inferred_name else "",
            "data": {
                "id": fallback_id,
                "name": inferred_name or "Unknown Brand",
            },
        }

    @classmethod
    def _resolve_category(cls, category_type: str, product_name: str) -> Dict[str, object]:
        catalog = cls._category_catalog()
        keys = []
        if category_type:
            keys.append(category_type.lower())
            keys.append(slugify(category_type))
        if product_name:
            tokens = slugify(product_name).split("-")
            if tokens:
                keys.append(tokens[0])

        for key in keys:
            if key and key in catalog:
                return catalog[key]

        if catalog:
            return next(iter(catalog.values()))

        fallback_name = (category_type or "Other").title()
        fallback_id = cls._pseudo_object_id(f"category:{fallback_name}")
        return {
            "id": fallback_id,
            "name": fallback_name,
            "slug": slugify(fallback_name),
            "data": {
                "id": fallback_id,
                "name": fallback_name,
            },
        }

    @staticmethod
    def _ensure_positive_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def to_representation(self, instance):
        """Convert MongoEngine document to dict (chuẩn hoá theo sample-product)."""

        def _stringify_id(value):
            if not value:
                return None
            if isinstance(value, ObjectId):
                return str(value)
            return str(value)

        def _id_string(value):
            return _stringify_id(value)

        def _iso_or_none(dt):
            if isinstance(dt, datetime):
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            return None

        def _ensure_list(value):
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return list(value)

        def _unwrap_extended_json(value):
            if isinstance(value, dict):
                if "$oid" in value:
                    return value["$oid"]
                if "$numberInt" in value:
                    return int(value["$numberInt"])
                if "$numberLong" in value:
                    return int(value["$numberLong"])
                if "$numberDouble" in value:
                    return float(value["$numberDouble"])
                if "$date" in value:
                    inner = value["$date"]
                    millis = None
                    if isinstance(inner, dict) and "$numberLong" in inner:
                        millis = int(inner["$numberLong"])
                    elif isinstance(inner, int):
                        millis = inner
                    if millis is not None:
                        return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).isoformat()
                    return _unwrap_extended_json(inner)
                return {k: _unwrap_extended_json(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_unwrap_extended_json(v) for v in value]
            return value

        name = getattr(instance, "name", "") or ""
        slug = getattr(instance, "slug", "") or ""
        description = getattr(instance, "description", "") or ""
        images = _ensure_list(getattr(instance, "images", []))

        price = getattr(instance, "price", None)
        if price in (None, ""):
            price = getattr(instance, "original_price", None) or getattr(instance, "originalPrice", None) or 0.0
        sale = getattr(instance, "sale", 0.0)
        count_in_stock = getattr(instance, "count_in_stock", 0) or 0
        if not count_in_stock:
            count_in_stock = getattr(instance, "countInStock", 0) or 0
        size_info = getattr(instance, "size", None) or {}
        if not size_info:
            size_info = getattr(instance, "sizes", None) or {}
        color_ids = [cid for cid in getattr(instance, "color_ids", []) if cid]
        if not color_ids:
            color_ids = getattr(instance, "colorIds", []) or []
        outfit_tags_raw = getattr(instance, "outfit_tags", None)
        if outfit_tags_raw in (None, [], ()):
            outfit_tags_raw = getattr(instance, "outfitTags", None)
        outfit_tags = _ensure_list(outfit_tags_raw)
        style_tags_raw = getattr(instance, "style_tags", None)
        if style_tags_raw in (None, [], ()):
            style_tags_raw = getattr(instance, "styleTags", None)
        style_tags = _ensure_list(style_tags_raw)

        gender_value = getattr(instance, "gender", None) or getattr(instance, "Gender", None) or "unisex"
        age_group_value = getattr(instance, "age_group", None) or getattr(instance, "ageGroup", None) or "adult"
        category_type_value = getattr(instance, "category_type", None) or getattr(instance, "categoryType", None)

        def _normalize_identifier(value):
            if isinstance(value, dict):
                if "$oid" in value:
                    return _normalize_identifier(value["$oid"])
                if "_id" in value:
                    return _normalize_identifier(value["_id"])
            if isinstance(value, ObjectId):
                return str(value), value
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None, None
                try:
                    coerced = ObjectId(cleaned)
                except (InvalidId, TypeError):
                    return cleaned, None
                return str(coerced), coerced
            return None, None

        def _normalize_sequence(values):
            normalized_strings = []
            normalized_objects = []
            for item in _ensure_list(values):
                stringified, obj_id = _normalize_identifier(item)
                if stringified:
                    normalized_strings.append(stringified)
                if obj_id:
                    normalized_objects.append(obj_id)
            return normalized_strings, normalized_objects

        brand_identifier = getattr(instance, "brand_id", None) or getattr(instance, "brandId", None)
        brand_id_str, brand_object_id = _normalize_identifier(brand_identifier)

        category_identifier = getattr(instance, "category_id", None) or getattr(instance, "categoryId", None)
        category_id_str, category_object_id = _normalize_identifier(category_identifier)

        user_identifier = (
            getattr(instance, "user_id", None)
            or getattr(instance, "userId", None)
            or getattr(instance, "user", None)
        )
        user_string, user_object_id = _normalize_identifier(user_identifier)

        color_ids_strings, _ = _normalize_sequence(color_ids)

        compatible_ids_source = getattr(instance, "compatible_product_ids", None) or getattr(
            instance, "compatibleProductIds", None
        )
        compatible_ids_list, _ = _normalize_sequence(compatible_ids_source)

        compatible_products_raw = getattr(instance, "compatibleProducts", None)
        compatible_products_strings, _ = _normalize_sequence(compatible_products_raw)
        if not compatible_products_strings and compatible_ids_list:
            compatible_products_strings = list(compatible_ids_list)

        feature_vector = list(
            getattr(instance, "feature_vector", None)
            or getattr(instance, "featureVector", None)
            or []
        )
        user_oid = user_object_id if user_object_id else getattr(instance, "user_id", None)

        amazon_parent_asin = (
            getattr(instance, "amazonParentAsin", None)
            or getattr(instance, "amazon_parent_asin", None)
        )
        amazon_asin = getattr(instance, "amazon_asin", None) or getattr(instance, "amazonAsin", None)

        created_at = getattr(instance, "created_at", None)
        updated_at = getattr(instance, "updated_at", None)

        resolved_brand = None
        if brand_object_id:
            try:
                brand_document = Brand.objects.get(id=brand_object_id)
                resolved_brand = {
                    "id": str(brand_document.id),
                    "name": getattr(brand_document, "name", None),
                    "slug": slugify(getattr(brand_document, "name", "") or ""),
                    "data": BrandSerializer(brand_document).data,
                }
            except Brand.DoesNotExist:
                resolved_brand = None
        if not resolved_brand:
            resolved_brand = self._resolve_brand(name, slug)

        brand_name = resolved_brand.get("name")
        brand_detail_payload = resolved_brand.get("data") or {}

        category_payload = None
        if category_object_id:
            try:
                category = Category.objects.get(id=category_object_id)
                category_payload = {
                    "id": str(category.id),
                    "name": getattr(category, "name", None),
                    "slug": slugify(getattr(category, "name", "") or ""),
                    "data": CategorySerializer(category).data,
                }
            except Category.DoesNotExist:
                category_payload = None
        if not category_payload:
            category_payload = self._resolve_category(category_type_value or "", name)

        category_detail = category_payload.get("data")
        category_id_value = category_payload.get("id")
        category_fallback = getattr(instance, "category", None) or category_payload.get("name")

        # Reviews
        reviews_payload = []
        reviews_queryset = ProductReview.objects(product_id=instance.id)
        if reviews_queryset:
            for review in reviews_queryset:
                reviews_payload.append(
                    {
                        "_id": _id_string(review.id),
                        "name": review.name,
                        "rating": review.rating,
                        "comment": review.comment,
                        "user": _id_string(review.user_id),
                        "createdAt": _iso_or_none(review.created_at),
                        "updatedAt": _iso_or_none(review.updated_at),
                    }
                )
        else:
            embedded_reviews = getattr(instance, "reviews", None) or []
            for review in embedded_reviews:
                unwrapped_review = _unwrap_extended_json(review)
                reviews_payload.append(
                    {
                        "_id": _id_string(unwrapped_review.get("_id")),
                        "name": unwrapped_review.get("name"),
                        "rating": unwrapped_review.get("rating"),
                        "comment": unwrapped_review.get("comment"),
                        "user": _id_string(unwrapped_review.get("user")),
                        "createdAt": unwrapped_review.get("createdAt"),
                        "updatedAt": unwrapped_review.get("updatedAt"),
                    }
                )

        num_reviews = len(reviews_payload) if reviews_payload else getattr(instance, "num_reviews", 0)
        rating = getattr(instance, "rating", 0.0) or 0.0
        if reviews_payload:
            valid_ratings = [r["rating"] for r in reviews_payload if isinstance(r.get("rating"), (int, float))]
            if valid_ratings:
                rating = sum(valid_ratings) / len(valid_ratings)

        # Variants
        variants_payload = []
        variant_documents = list(ProductVariant.objects(product_id=instance.id))
        if variant_documents:
            color_ids_for_lookup = {variant.color_id for variant in variant_documents if getattr(variant, "color_id", None)}
            size_ids_for_lookup = {variant.size_id for variant in variant_documents if getattr(variant, "size_id", None)}

            color_map = {}
            if color_ids_for_lookup:
                colors = Color.objects(id__in=list(color_ids_for_lookup))
                color_map = {color.id: color for color in colors}

            size_map = {}
            if size_ids_for_lookup:
                sizes = Size.objects(id__in=list(size_ids_for_lookup))
                size_map = {size.id: size for size in sizes}

            for variant in variant_documents:
                color_value = getattr(variant, "color", None)
                if not color_value:
                    color_doc = color_map.get(getattr(variant, "color_id", None))
                    if color_doc:
                        color_value = getattr(color_doc, "hex_code", None) or getattr(color_doc, "name", None)

                size_value = getattr(variant, "size", None)
                if not size_value:
                    size_doc = size_map.get(getattr(variant, "size_id", None))
                    if size_doc:
                        size_value = getattr(size_doc, "code", None) or getattr(size_doc, "name", None)

                variants_payload.append(
                    {
                        "_id": _id_string(variant.id),
                        "stock": variant.stock,
                        "color": color_value,
                        "size": size_value,
                        "price": float(variant.price) if getattr(variant, "price", None) is not None else None,
                        "colorId": _stringify_id(getattr(variant, "color_id", None)),
                        "sizeId": _stringify_id(getattr(variant, "size_id", None)),
                    }
                )
        else:
            embedded_variants = getattr(instance, "variants", None) or []
            for variant in embedded_variants:
                unwrapped_variant = _unwrap_extended_json(variant)
                variants_payload.append(
                    {
                        "_id": _id_string(unwrapped_variant.get("_id")),
                        "stock": unwrapped_variant.get("stock"),
                        "color": unwrapped_variant.get("color"),
                        "size": unwrapped_variant.get("size"),
                        "price": unwrapped_variant.get("price"),
                        "colorId": unwrapped_variant.get("colorId") or unwrapped_variant.get("color_id"),
                        "sizeId": unwrapped_variant.get("sizeId") or unwrapped_variant.get("size_id"),
                    }
                )

        compatible_products_payload = list(compatible_products_strings)

        data: dict[str, object] = {
            "_id": _id_string(instance.id),
            "id": str(instance.id),
            "brand_id": brand_id_str or resolved_brand.get("id"),
            "category_id": category_id_str or category_id_value,
            "name": name,
            "slug": slug,
            "description": description,
            "images": images,
            "rating": rating,
            "num_reviews": num_reviews,
            "numReviews": num_reviews,
            "price": float(price) if price is not None else 0.0,
            "sale": float(sale) if sale is not None else 0.0,
            "count_in_stock": count_in_stock,
            "size": size_info,
            "color_ids": color_ids_strings,
            "outfit_tags": outfit_tags,
            "outfitTags": outfit_tags,
            "style_tags": style_tags,
            "styleTags": style_tags,
            "gender": gender_value,
            "age_group": age_group_value,
            "ageGroup": age_group_value,
            "category_type": category_type_value,
            "categoryType": category_type_value,
            "compatible_product_ids": [pid for pid in compatible_ids_list if pid],
            "compatibleProducts": [pid for pid in compatible_products_payload if pid],
            "feature_vector": feature_vector,
            "amazon_asin": amazon_asin,
            "amazon_parent_asin": amazon_parent_asin,
            "amazonParentAsin": amazon_parent_asin,
            "created_at": _iso_or_none(created_at),
            "updated_at": _iso_or_none(updated_at),
            "brand_name": brand_name,
            "brand": brand_name,
            "category_detail": category_detail,
            "category": category_fallback,
            "user": user_string or _id_string(user_oid),
            "reviews": reviews_payload,
            "__v": getattr(instance, "__v", 0) or 0,
            "variants": variants_payload,
            "brand_detail": brand_detail_payload,
        }

        # Fallback enrichments
        if not data["brand_id"]:
            data["brand_id"] = resolved_brand.get("id")
        if not data["brand_name"]:
            data["brand_name"] = resolved_brand.get("name")
        if not data["brand"]:
            data["brand"] = resolved_brand.get("name")
        if not data.get("brand_detail"):
            data["brand_detail"] = resolved_brand.get("data")

        if not data["category_id"]:
            data["category_id"] = category_payload.get("id")
        if not data["category_detail"]:
            data["category_detail"] = category_payload.get("data")
        if not data["category"]:
            data["category"] = category_payload.get("name")

        if not data["user"]:
            data["user"] = self._pseudo_object_id("system-user")

        default_colors = self._default_color_entries()
        if not data["color_ids"] and default_colors:
            data["color_ids"] = [entry["id"] for entry in default_colors[:2]]

        if not data["outfit_tags"]:
            data["outfit_tags"] = [category_type_value or "lifestyle"]
            data["outfitTags"] = data["outfit_tags"]

        if not data["style_tags"]:
            derived_tag = slug.split("-")[0] if slug else "style"
            data["style_tags"] = [derived_tag]
            data["styleTags"] = data["style_tags"]

        if not data["feature_vector"]:
            data["feature_vector"] = [
                float(price) if price is not None else 0.0,
                float(sale) if sale is not None else 0.0,
                float(rating or 0.0),
                float(len(images)),
            ]

        if not data["compatible_product_ids"]:
            pool = self._category_product_pool(category_type_value or "")
            randomized_pool = [pid for pid in pool if pid != data["id"]][:6]
            if not randomized_pool and pool:
                randomized_pool = pool[:6]
            data["compatible_product_ids"] = randomized_pool
            data["compatibleProducts"] = randomized_pool

        if not data["variants"]:
            size_catalog = self._size_catalog()
            variants_fallback: List[Dict[str, object]] = []
            color_rotation = default_colors or [{"id": None, "name": None}]
            if size_info:
                for index, (size_code, stock_value) in enumerate(size_info.items()):
                    size_key = str(size_code).lower()
                    size_entry = size_catalog.get(size_key)
                    color_entry = color_rotation[index % len(color_rotation)]
                    variant_id = self._pseudo_object_id(f"{data['id']}:{size_code}:{index}")
                    variants_fallback.append(
                        {
                            "_id": variant_id,
                            "stock": self._ensure_positive_int(stock_value, 0),
                            "color": color_entry.get("name"),
                            "size": size_entry["code"] if size_entry else size_code.upper(),
                            "price": float(price) if price is not None else 0.0,
                            "colorId": color_entry.get("id"),
                            "sizeId": size_entry["id"] if size_entry else None,
                        }
                    )
            if variants_fallback:
                data["variants"] = variants_fallback

        if not data["count_in_stock"] and size_info:
            data["count_in_stock"] = sum(
                self._ensure_positive_int(value, 0) for value in size_info.values()
            )

        if not data["amazon_asin"]:
            data["amazon_asin"] = (slug.replace("-", "") if slug else data["id"])[:10].upper()
        if not data["amazon_parent_asin"]:
            data["amazon_parent_asin"] = f"{data['amazon_asin']}P"
            data["amazonParentAsin"] = data["amazon_parent_asin"]

        # Duplicate frequently used camelCase fields so list/detail responses stay consistent.
        data["brandId"] = data.get("brand_id")
        data["categoryId"] = data.get("category_id")
        data["brandName"] = data.get("brand_name")
        data["categoryDetail"] = data.get("category_detail")
        data["colorIds"] = data.get("color_ids") or []
        data["countInStock"] = data.get("count_in_stock", 0)
        data["featureVector"] = data.get("feature_vector") or []
        data["compatibleProductIds"] = data.get("compatible_product_ids") or []
        data["compatibleProducts"] = data.get("compatibleProducts") or []
        data["createdAt"] = data.get("created_at")
        data["updatedAt"] = data.get("updated_at")
        data["amazonAsin"] = data.get("amazon_asin")
        data["amazonParentAsin"] = data.get("amazon_parent_asin")
        data["brandDetail"] = data.get("brand_detail") or data.get("brand")
        data["categoryDetail"] = data.get("category_detail")
        data["userId"] = data.get("user")

        sample_payload = self._load_sample_product()
        excluded_sample_fields = {
            "size",
            "price",
            "countInStock",
            "featureVector",
            "colors",
            "createdAt",
            "updatedAt",
        }

        def _get_sample_value(key: str, default=None):
            if not sample_payload or key in excluded_sample_fields:
                return default
            return sample_payload.get(key, default)

        if not data.get("brand"):
            sample_brand = _unwrap_extended_json(_get_sample_value("brand"))
            if sample_brand:
                data["brand"] = sample_brand
        if not data.get("brand_name"):
            sample_brand = _unwrap_extended_json(_get_sample_value("brand"))
            if sample_brand:
                data["brand_name"] = sample_brand

        if not data.get("category"):
            sample_category = _unwrap_extended_json(_get_sample_value("category"))
            if sample_category:
                data["category"] = sample_category

        if not data.get("category_detail"):
            category_detail = _unwrap_extended_json(_get_sample_value("category_detail"))
            if category_detail:
                data["category_detail"] = category_detail

        if not data.get("amazon_parent_asin") and not data.get("amazonParentAsin"):
            sample_parent_asin = _unwrap_extended_json(_get_sample_value("amazonParentAsin"))
            if sample_parent_asin:
                data["amazon_parent_asin"] = sample_parent_asin
                data["amazonParentAsin"] = sample_parent_asin

        if not data.get("amazon_asin"):
            sample_asin = _unwrap_extended_json(_get_sample_value("amazonAsin"))
            if sample_asin:
                data["amazon_asin"] = sample_asin

        if not data.get("user"):
            sample_user = _unwrap_extended_json(_get_sample_value("user"))
            if sample_user:
                data["user"] = sample_user

        if not data.get("compatibleProducts"):
            sample_compatible = _get_sample_value("compatibleProducts", [])
            if sample_compatible:
                unwrapped = _unwrap_extended_json(sample_compatible)
                data["compatibleProducts"] = unwrapped
                data["compatible_product_ids"] = [pid for pid in unwrapped if pid]

        if not data.get("variants"):
            sample_variants = _get_sample_value("variants", [])
            if sample_variants:
                data["variants"] = _unwrap_extended_json(sample_variants)

        if not data.get("reviews"):
            sample_reviews = _get_sample_value("reviews", [])
            if sample_reviews:
                unwrapped_reviews = _unwrap_extended_json(sample_reviews)
                data["reviews"] = unwrapped_reviews
                data["num_reviews"] = len(unwrapped_reviews)
                data["numReviews"] = len(unwrapped_reviews)
                ratings_from_sample = [
                    review.get("rating")
                    for review in unwrapped_reviews
                ]
                numeric_ratings = []
                for value in ratings_from_sample:
                    try:
                        numeric_ratings.append(float(value))
                    except (TypeError, ValueError):
                        continue
                if numeric_ratings:
                    data["rating"] = sum(numeric_ratings) / len(numeric_ratings)

        if data.get("reviews"):
            data["reviews"] = [
                {
                    "_id": _id_string(review.get("_id")),
                    "name": review.get("name"),
                    "rating": review.get("rating"),
                    "comment": review.get("comment"),
                    "user": _id_string(review.get("user")),
                    "createdAt": review.get("createdAt"),
                    "updatedAt": review.get("updatedAt"),
                }
                for review in data["reviews"]
            ]

        if not data.get("__v"):
            sample_version = _get_sample_value("__v")
            if isinstance(sample_version, dict) and "$numberInt" in sample_version:
                data["__v"] = int(sample_version["$numberInt"])
            elif sample_version is not None:
                data["__v"] = sample_version

        return data
    
    def create(self, validated_data):
        """Create new product."""
        variants_data = validated_data.pop("variants_payload", [])
        color_ids = validated_data.pop("color_ids", [])
        
        # Convert string IDs to ObjectId
        if "brand_id" in validated_data:
            validated_data["brand_id"] = ObjectId(validated_data["brand_id"])
        if "category_id" in validated_data:
            validated_data["category_id"] = ObjectId(validated_data["category_id"])
        if "user_id" in validated_data:
            validated_data["user_id"] = ObjectId(validated_data["user_id"])
        
        validated_data["color_ids"] = [ObjectId(cid) for cid in color_ids]
        
        product = Product(**validated_data)
        product.save()
        
        # Create variants
        for variant_data in variants_data:
            variant_data["product_id"] = product.id
            variant_data["color_id"] = ObjectId(variant_data["color_id"])
            variant_data["size_id"] = ObjectId(variant_data["size_id"])
            variant = ProductVariant(**variant_data)
            variant.save()
        
        product.update_stock_from_variants()
        return product
    
    def update(self, instance, validated_data):
        """Update product."""
        variants_data = validated_data.pop("variants_payload", None)
        color_ids = validated_data.pop("color_ids", None)
        
        # Convert string IDs to ObjectId
        if "brand_id" in validated_data:
            validated_data["brand_id"] = ObjectId(validated_data["brand_id"])
        if "category_id" in validated_data:
            validated_data["category_id"] = ObjectId(validated_data["category_id"])
        
        if color_ids is not None:
            validated_data["color_ids"] = [ObjectId(cid) for cid in color_ids]
        
        for key, value in validated_data.items():
            setattr(instance, key, value)
        instance.save()
        
        if variants_data is not None:
            # Delete old variants
            ProductVariant.objects(product_id=instance.id).delete()
            # Create new variants
            for variant_data in variants_data:
                variant_data["product_id"] = instance.id
                variant_data["color_id"] = ObjectId(variant_data["color_id"])
                variant_data["size_id"] = ObjectId(variant_data["size_id"])
                variant = ProductVariant(**variant_data)
                variant.save()
            instance.update_stock_from_variants()
        
        return instance


class ContentSectionSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    type = serializers.CharField()
    images = serializers.ListField(child=serializers.CharField())
    image = serializers.CharField(required=False, allow_blank=True)
    subtitle = serializers.CharField(required=False, allow_blank=True)
    title = serializers.CharField(required=False, allow_blank=True)
    button_text = serializers.CharField(required=False, allow_blank=True)
    button_link = serializers.CharField(required=False, allow_blank=True)
    position = serializers.CharField(required=False, allow_blank=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def create(self, validated_data):
        return ContentSection(**validated_data).save()
    
    def update(self, instance, validated_data):
        for key, value in validated_data.items():
            setattr(instance, key, value)
        instance.save()
        return instance

