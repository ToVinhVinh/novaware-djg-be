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
        """Convert MongoEngine document to dict - đọc trực tiếp từ MongoDB document."""
        
        # Get raw MongoDB document directly from collection (not from MongoEngine instance)
        # because MongoEngine model may not have all fields defined
        from mongoengine import connection
        db = connection.get_db()
        mongo_doc = db.products.find_one({"_id": instance.id})
        if not mongo_doc:
            # Fallback to to_mongo() if direct query fails
            mongo_doc = instance.to_mongo().to_dict()

        def _stringify_id(value):
            if not value:
                return None
            if isinstance(value, ObjectId):
                return str(value)
            if isinstance(value, dict):
                if "$oid" in value:
                    return value["$oid"]
                if "_id" in value:
                    return _stringify_id(value["_id"])
            return str(value)

        def _id_string(value):
            return _stringify_id(value)

        def _iso_or_none(dt):
            if isinstance(dt, datetime):
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            return None

        def _unwrap_extended_json(value):
            """Unwrap MongoDB Extended JSON format."""
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

        def _get_field(doc, *keys, default=None):
            """Get field from document, trying multiple key names."""
            for key in keys:
                if key in doc:
                    value = doc[key]
                    return _unwrap_extended_json(value) if isinstance(value, (dict, list)) else value
            return default

        def _get_list_field(doc, *keys, default=None):
            """Get list field, ensuring it's a list."""
            value = _get_field(doc, *keys, default=default or [])
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return [value] if value else []

        # Read fields directly from MongoDB document (prioritize camelCase as in MongoDB)
        product_id = str(instance.id)
        
        # Basic fields
        name = _get_field(mongo_doc, "name", default="")
        slug = _get_field(mongo_doc, "slug", default="")
        description = _get_field(mongo_doc, "description", default="")
        images = _get_list_field(mongo_doc, "images")
        
        # Numeric fields - check both camelCase and snake_case
        price = _get_field(mongo_doc, "price", default=0.0)
        if isinstance(price, (dict, str)):
            price = float(_unwrap_extended_json(price)) if price else 0.0
        else:
            price = float(price) if price else 0.0
            
        sale = _get_field(mongo_doc, "sale", default=0.0)
        if isinstance(sale, (dict, str)):
            sale = float(_unwrap_extended_json(sale)) if sale else 0.0
        else:
            sale = float(sale) if sale else 0.0
        
        # Count in stock - prioritize camelCase from MongoDB
        count_in_stock = _get_field(mongo_doc, "countInStock", "count_in_stock", default=0)
        if isinstance(count_in_stock, dict):
            count_in_stock = _unwrap_extended_json(count_in_stock)
        count_in_stock = int(count_in_stock) if count_in_stock else 0
        
        # Num reviews - prioritize camelCase from MongoDB
        num_reviews = _get_field(mongo_doc, "numReviews", "num_reviews", default=0)
        if isinstance(num_reviews, dict):
            num_reviews = _unwrap_extended_json(num_reviews)
        num_reviews = int(num_reviews) if num_reviews else 0
        
        # Rating
        rating = _get_field(mongo_doc, "rating", default=0.0)
        if isinstance(rating, dict):
            rating = _unwrap_extended_json(rating)
        rating = float(rating) if rating else 0.0
        
        # Size info
        size_info = _get_field(mongo_doc, "size", default={})
        if isinstance(size_info, dict):
            # Unwrap nested number values in size dict
            unwrapped_size = {}
            for k, v in size_info.items():
                if isinstance(v, dict):
                    unwrapped_size[k] = _unwrap_extended_json(v)
                else:
                    unwrapped_size[k] = v
            size_info = unwrapped_size
        
        # Tags - prioritize camelCase from MongoDB
        outfit_tags = _get_list_field(mongo_doc, "outfitTags", "outfit_tags")
        style_tags = _get_list_field(mongo_doc, "styleTags", "style_tags")
        
        # Other fields - only get if they exist in MongoDB (don't use defaults)
        gender_value = _get_field(mongo_doc, "gender")
        age_group_value = _get_field(mongo_doc, "ageGroup", "age_group")
        category_type_value = _get_field(mongo_doc, "categoryType", "category_type")
        
        # Brand and Category - use direct string fields from MongoDB
        # Check if field exists in document (not just if it has value)
        brand_string = mongo_doc.get("brand") if "brand" in mongo_doc else None
        category_string = mongo_doc.get("category") if "category" in mongo_doc else None
        
        # User - prioritize camelCase, then snake_case
        user_value = _get_field(mongo_doc, "user", "user_id", "userId")
        user_string = _stringify_id(user_value) if user_value else None
        
        # Brand ID and Category ID - try to get from references
        brand_id_value = _get_field(mongo_doc, "brand_id", "brandId")
        brand_id_str = _stringify_id(brand_id_value) if brand_id_value else None
        
        category_id_value = _get_field(mongo_doc, "category_id", "categoryId")
        category_id_str = _stringify_id(category_id_value) if category_id_value else None
        
        # Color IDs
        color_ids_raw = _get_list_field(mongo_doc, "colorIds", "color_ids")
        color_ids_strings = [_stringify_id(cid) for cid in color_ids_raw if cid]
        
        # Compatible products - prioritize camelCase from MongoDB
        compatible_products_raw = _get_list_field(mongo_doc, "compatibleProducts", "compatible_product_ids", "compatibleProductIds")
        compatible_products_strings = [_stringify_id(pid) for pid in compatible_products_raw if pid]
        
        # Feature vector - prioritize camelCase from MongoDB
        feature_vector_raw = _get_list_field(mongo_doc, "featureVector", "feature_vector")
        feature_vector = []
        for item in feature_vector_raw:
            if isinstance(item, dict):
                feature_vector.append(_unwrap_extended_json(item))
            else:
                feature_vector.append(float(item) if item is not None else 0.0)
        
        # Amazon ASINs
        amazon_parent_asin = _get_field(mongo_doc, "amazonParentAsin", "amazon_parent_asin")
        amazon_asin = _get_field(mongo_doc, "amazonAsin", "amazon_asin")
        
        # Timestamps - handle Extended JSON format from MongoDB
        created_at_raw = _get_field(mongo_doc, "createdAt", "created_at")
        created_at = None
        if created_at_raw:
            if isinstance(created_at_raw, datetime):
                created_at = _iso_or_none(created_at_raw)
            elif isinstance(created_at_raw, dict):
                # Handle Extended JSON date format
                created_at = _unwrap_extended_json(created_at_raw)
                if isinstance(created_at, (int, float)):
                    created_at = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).isoformat()
            elif isinstance(created_at_raw, (int, float)):
                created_at = datetime.fromtimestamp(created_at_raw / 1000, tz=timezone.utc).isoformat()
        
        updated_at_raw = _get_field(mongo_doc, "updatedAt", "updated_at")
        updated_at = None
        if updated_at_raw:
            if isinstance(updated_at_raw, datetime):
                updated_at = _iso_or_none(updated_at_raw)
            elif isinstance(updated_at_raw, dict):
                # Handle Extended JSON date format
                updated_at = _unwrap_extended_json(updated_at_raw)
                if isinstance(updated_at, (int, float)):
                    updated_at = datetime.fromtimestamp(updated_at / 1000, tz=timezone.utc).isoformat()
            elif isinstance(updated_at_raw, (int, float)):
                updated_at = datetime.fromtimestamp(updated_at_raw / 1000, tz=timezone.utc).isoformat()
        
        # __v field
        __v = _get_field(mongo_doc, "__v", default=0)
        if isinstance(__v, dict):
            __v = _unwrap_extended_json(__v)
        __v = int(__v) if __v else 0
        
        # Reviews - check embedded reviews first, then separate collection
        reviews_payload = []
        embedded_reviews = _get_list_field(mongo_doc, "reviews")
        if embedded_reviews:
            for review in embedded_reviews:
                unwrapped = _unwrap_extended_json(review) if isinstance(review, dict) else review
                if isinstance(unwrapped, dict):
                    # Handle rating which might be in Extended JSON format
                    rating_value = unwrapped.get("rating", 0)
                    if isinstance(rating_value, dict):
                        rating_value = _unwrap_extended_json(rating_value)
                    rating_value = int(rating_value) if rating_value else 0
                    
                    # Handle createdAt and updatedAt
                    created_at_review = unwrapped.get("createdAt") or unwrapped.get("created_at")
                    if created_at_review and isinstance(created_at_review, dict):
                        created_at_review = _unwrap_extended_json(created_at_review)
                        if isinstance(created_at_review, (int, float)):
                            created_at_review = datetime.fromtimestamp(created_at_review / 1000, tz=timezone.utc).isoformat()
                    
                    updated_at_review = unwrapped.get("updatedAt") or unwrapped.get("updated_at")
                    if updated_at_review and isinstance(updated_at_review, dict):
                        updated_at_review = _unwrap_extended_json(updated_at_review)
                        if isinstance(updated_at_review, (int, float)):
                            updated_at_review = datetime.fromtimestamp(updated_at_review / 1000, tz=timezone.utc).isoformat()
                    
                    reviews_payload.append({
                        "_id": _id_string(unwrapped.get("_id") or unwrapped.get("id")),
                        "name": unwrapped.get("name", ""),
                        "rating": rating_value,
                        "comment": unwrapped.get("comment", ""),
                        "user": _id_string(unwrapped.get("user") or unwrapped.get("user_id")),
                        "createdAt": created_at_review,
                        "updatedAt": updated_at_review,
                    })
        else:
            # Try separate collection
            reviews_queryset = ProductReview.objects(product_id=instance.id)
            for review in reviews_queryset:
                reviews_payload.append({
                    "_id": _id_string(review.id),
                    "name": review.name,
                    "rating": review.rating,
                    "comment": review.comment,
                    "user": _id_string(review.user_id),
                    "createdAt": _iso_or_none(review.created_at),
                    "updatedAt": _iso_or_none(review.updated_at),
                })
        
        # Update num_reviews and rating from embedded reviews if available
        if embedded_reviews and num_reviews == 0:
            num_reviews = len(embedded_reviews)
        if embedded_reviews and rating == 0.0:
            valid_ratings = [r.get("rating", 0) for r in reviews_payload if isinstance(r.get("rating"), (int, float))]
            if valid_ratings:
                rating = sum(valid_ratings) / len(valid_ratings)

        # Variants - check embedded variants first, then separate collection
        variants_payload = []
        # Check if variants field exists in MongoDB document
        if "variants" in mongo_doc:
            embedded_variants_raw = mongo_doc["variants"]
            if isinstance(embedded_variants_raw, list) and len(embedded_variants_raw) > 0:
                for variant in embedded_variants_raw:
                    # Unwrap the entire variant dict to handle Extended JSON
                    unwrapped = _unwrap_extended_json(variant) if isinstance(variant, dict) else variant
                    if isinstance(unwrapped, dict):
                        # Handle stock and price which might be in Extended JSON format
                        stock_value = unwrapped.get("stock", 0)
                        if isinstance(stock_value, dict):
                            stock_value = _unwrap_extended_json(stock_value)
                        stock_value = int(stock_value) if stock_value else 0
                        
                        price_value = unwrapped.get("price", 0.0)
                        if isinstance(price_value, dict):
                            price_value = _unwrap_extended_json(price_value)
                        price_value = float(price_value) if price_value else 0.0
                        
                        variants_payload.append({
                            "_id": _id_string(unwrapped.get("_id") or unwrapped.get("id")),
                            "stock": stock_value,
                            "color": unwrapped.get("color", ""),
                            "size": unwrapped.get("size", ""),
                            "price": price_value,
                        })
        
        # Only try separate collection if no embedded variants found
        if not variants_payload:
            # Try separate collection
            variant_documents = list(ProductVariant.objects(product_id=instance.id))
            for variant in variant_documents:
                color_value = getattr(variant, "color", None)
                size_value = getattr(variant, "size", None)
                
                # Try to get color/size from related documents
                if not color_value and getattr(variant, "color_id", None):
                    try:
                        color_doc = Color.objects.get(id=variant.color_id)
                        color_value = getattr(color_doc, "hex_code", None) or getattr(color_doc, "name", None)
                    except Color.DoesNotExist:
                        pass
                
                if not size_value and getattr(variant, "size_id", None):
                    try:
                        size_doc = Size.objects.get(id=variant.size_id)
                        size_value = getattr(size_doc, "code", None) or getattr(size_doc, "name", None)
                    except Size.DoesNotExist:
                        pass

                variants_payload.append({
                    "_id": _id_string(variant.id),
                    "stock": variant.stock,
                    "color": color_value or "",
                    "size": size_value or "",
                    "price": float(variant.price) if getattr(variant, "price", None) is not None else 0.0,
                })
        
        # Get colors field from MongoDB (if exists)
        colors_raw = _get_list_field(mongo_doc, "colors")
        colors_list = colors_raw if colors_raw else []
        
        # Build response data - ONLY include fields that exist in MongoDB
        data: dict[str, object] = {
            "_id": product_id,
            "id": product_id,  # Keep id for API compatibility
            "name": name,
            "slug": slug,
            "description": description,
            "images": images,
            "rating": rating,
            "num_reviews": num_reviews,
            "numReviews": num_reviews,
            "price": price,
            "sale": sale,
            "count_in_stock": count_in_stock,
            "countInStock": count_in_stock,
            "size": size_info,
            "outfit_tags": outfit_tags,
            "outfitTags": outfit_tags,
            "compatibleProducts": compatible_products_strings,
            "feature_vector": feature_vector,
            "featureVector": feature_vector,
            "amazon_parent_asin": amazon_parent_asin,
            "amazonParentAsin": amazon_parent_asin,
            "user": user_string,
            "reviews": reviews_payload,
            "__v": __v,
            "variants": variants_payload,
        }
        
        # Add brand field (string from MongoDB) - always add if field exists in MongoDB
        if "brand" in mongo_doc:
            data["brand"] = brand_string if brand_string is not None else ""
        
        # Add category field (string from MongoDB) - always add if field exists in MongoDB
        if "category" in mongo_doc:
            data["category"] = category_string if category_string is not None else ""
        
        # Add colors field (always add, even if empty array)
        data["colors"] = colors_list
        
        # Add timestamps (if exist in MongoDB)
        if created_at:
            data["created_at"] = created_at
            data["createdAt"] = created_at
        
        if updated_at:
            data["updated_at"] = updated_at
            data["updatedAt"] = updated_at

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

