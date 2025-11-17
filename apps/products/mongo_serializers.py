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
    brand_id = serializers.CharField(required=False)
    brand_name = serializers.SerializerMethodField()
    category_id = serializers.CharField(required=False)
    category_detail = CategorySerializer(read_only=True, source="category")
    name = serializers.CharField(required=False)
    slug = serializers.SlugField(required=False)
    description = serializers.CharField(required=False)
    images = serializers.ListField(child=serializers.CharField(), required=False)
    rating = serializers.FloatField(read_only=True)
    num_reviews = serializers.IntegerField(read_only=True)
    price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    sale = serializers.DecimalField(max_digits=5, decimal_places=2, required=False)
    count_in_stock = serializers.IntegerField(read_only=True)
    size = serializers.DictField(required=False)
    color_ids = serializers.ListField(child=serializers.CharField(), required=False)
    variants = ProductVariantSerializer(many=True, read_only=True)
    variants_payload = ProductVariantSerializer(many=True, write_only=True, required=False)
    outfit_tags = serializers.ListField(child=serializers.CharField(), required=False)
    compatible_product_ids = serializers.ListField(child=serializers.CharField(), required=False)
    feature_vector = serializers.ListField(child=serializers.FloatField(), required=False)
    amazon_asin = serializers.CharField(required=False, allow_null=True)
    amazon_parent_asin = serializers.CharField(required=False, allow_null=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    
    # Fields from CSV data that should be handled
    _id = serializers.CharField(required=False)
    productDisplayName = serializers.CharField(required=False)
    gender = serializers.CharField(required=False)
    masterCategory = serializers.CharField(required=False)
    subCategory = serializers.CharField(required=False)
    articleType = serializers.CharField(required=False)
    baseColour = serializers.CharField(required=False)
    season = serializers.CharField(required=False)
    year = serializers.IntegerField(required=False)
    usage = serializers.CharField(required=False)
    
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
        # Handle both integer ID and ObjectId
        product_id = instance.id
        if isinstance(product_id, int):
            mongo_doc = db.products.find_one({"_id": product_id})
        else:
            mongo_doc = db.products.find_one({"_id": product_id})
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

        # Read fields directly from MongoDB document
        # Handle product ID - can be integer or ObjectId
        product_id_value = mongo_doc.get("_id")
        if isinstance(product_id_value, dict):
            # Handle Extended JSON format
            if "$numberInt" in product_id_value:
                product_id = int(product_id_value["$numberInt"])
            elif "$oid" in product_id_value:
                product_id = str(product_id_value["$oid"])
            else:
                product_id = int(product_id_value) if isinstance(product_id_value, (int, str)) else str(instance.id)
        elif isinstance(product_id_value, int):
            product_id = product_id_value
        else:
            product_id = int(product_id_value) if product_id_value else int(instance.id)
        
        # Get product fields from MongoDB document
        # Fields from the user's MongoDB document structure
        gender = _get_field(mongo_doc, "gender")
        masterCategory = _get_field(mongo_doc, "masterCategory")
        subCategory = _get_field(mongo_doc, "subCategory")
        articleType = _get_field(mongo_doc, "articleType")
        baseColour = _get_field(mongo_doc, "baseColour")
        season = _get_field(mongo_doc, "season")
        year = _get_field(mongo_doc, "year")
        if isinstance(year, dict):
            year = _unwrap_extended_json(year)
        year = int(year) if year else None
        usage = _get_field(mongo_doc, "usage")
        productDisplayName = _get_field(mongo_doc, "productDisplayName")
        images = _get_list_field(mongo_doc, "images")
        
        # Rating
        rating = _get_field(mongo_doc, "rating", default=0.0)
        if isinstance(rating, dict):
            rating = _unwrap_extended_json(rating)
        rating = float(rating) if rating else 0.0
        
        # Sale
        sale = _get_field(mongo_doc, "sale", default=0.0)
        if isinstance(sale, (dict, str)):
            sale = float(_unwrap_extended_json(sale)) if sale else 0.0
        else:
            sale = float(sale) if sale else 0.0
        
        
        # Timestamps - handle Extended JSON format from MongoDB
        created_at_raw = _get_field(mongo_doc, "created_at", "createdAt")
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
        
        updated_at_raw = _get_field(mongo_doc, "updated_at", "updatedAt")
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
                    if created_at_review:
                        if isinstance(created_at_review, datetime):
                            created_at_review = _iso_or_none(created_at_review)
                        elif isinstance(created_at_review, dict):
                            created_at_review = _unwrap_extended_json(created_at_review)
                            if isinstance(created_at_review, (int, float)):
                                created_at_review = datetime.fromtimestamp(created_at_review / 1000, tz=timezone.utc).isoformat()
                        elif isinstance(created_at_review, (int, float)):
                            created_at_review = datetime.fromtimestamp(created_at_review / 1000, tz=timezone.utc).isoformat()
                        else:
                            created_at_review = str(created_at_review)
                    
                    updated_at_review = unwrapped.get("updatedAt") or unwrapped.get("updated_at")
                    if updated_at_review:
                        if isinstance(updated_at_review, datetime):
                            updated_at_review = _iso_or_none(updated_at_review)
                        elif isinstance(updated_at_review, dict):
                            updated_at_review = _unwrap_extended_json(updated_at_review)
                            if isinstance(updated_at_review, (int, float)):
                                updated_at_review = datetime.fromtimestamp(updated_at_review / 1000, tz=timezone.utc).isoformat()
                        elif isinstance(updated_at_review, (int, float)):
                            updated_at_review = datetime.fromtimestamp(updated_at_review / 1000, tz=timezone.utc).isoformat()
                        else:
                            updated_at_review = str(updated_at_review)
                    
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
            # Try separate collection - handle both integer and ObjectId product_id
            try:
                if isinstance(product_id, int):
                    reviews_queryset = ProductReview.objects(product_id=product_id)
                else:
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
            except Exception:
                # If query fails, just use empty list
                pass

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
            # Try separate collection - handle both integer and ObjectId product_id
            try:
                if isinstance(product_id, int):
                    variant_documents = list(ProductVariant.objects(product_id=product_id))
                else:
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
            except Exception:
                # If query fails, just use empty list
                pass
        
        # Build response data according to user's desired format
        data: dict[str, object] = {
            "id": product_id,  # Use integer ID, not _id
            "gender": gender,
            "masterCategory": masterCategory,
            "subCategory": subCategory,
            "articleType": articleType,
            "baseColour": baseColour,
            "season": season,
            "year": year,
            "usage": usage,
            "productDisplayName": productDisplayName,
            "images": images,
            "rating": rating,
            "sale": sale,
            "reviews": reviews_payload,
            "variants": variants_payload,
        }
        
        # Add timestamps if they exist
        if created_at:
            data["created_at"] = created_at
        if updated_at:
            data["updated_at"] = updated_at

        return data
    
    def create(self, validated_data):
        """Create new product."""
        variants_data = validated_data.pop("variants_payload", [])
        color_ids = validated_data.pop("color_ids", [])
        
        # Handle CSV-style fields mapping
        if "_id" in validated_data:
            validated_data.pop("_id")  # Remove _id as it's not needed for create
        
        # Map productDisplayName to name if name is not provided
        if "productDisplayName" in validated_data and "name" not in validated_data:
            validated_data["name"] = validated_data["productDisplayName"]
        
        # Generate slug from name if not provided
        if "name" in validated_data and "slug" not in validated_data:
            validated_data["slug"] = slugify(validated_data["name"])
        
        # Set default values for required fields if not provided
        if "brand_id" not in validated_data:
            # Try to find or create a default brand
            try:
                default_brand = Brand.objects.first()
                if default_brand:
                    validated_data["brand_id"] = default_brand.id
            except:
                pass
        
        if "category_id" not in validated_data:
            # Try to find or create a default category
            try:
                default_category = Category.objects.first()
                if default_category:
                    validated_data["category_id"] = default_category.id
            except:
                pass
        
        # Set default values for other required fields
        if "description" not in validated_data:
            validated_data["description"] = validated_data.get("productDisplayName", "No description available")
        
        if "price" not in validated_data:
            validated_data["price"] = 0.0
        
        if "size" not in validated_data:
            validated_data["size"] = {}
        
        if "outfit_tags" not in validated_data:
            validated_data["outfit_tags"] = []
        
        if "feature_vector" not in validated_data:
            # Generate a default feature vector (128 dimensions with random values)
            validated_data["feature_vector"] = [0.0] * 128
        
        # Convert string IDs to ObjectId
        if "brand_id" in validated_data and isinstance(validated_data["brand_id"], str):
            try:
                validated_data["brand_id"] = ObjectId(validated_data["brand_id"])
            except:
                pass
        if "category_id" in validated_data and isinstance(validated_data["category_id"], str):
            try:
                validated_data["category_id"] = ObjectId(validated_data["category_id"])
            except:
                pass
        if "user_id" in validated_data and isinstance(validated_data["user_id"], str):
            try:
                validated_data["user_id"] = ObjectId(validated_data["user_id"])
            except:
                pass
        
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
        
        # Handle CSV-style fields mapping
        if "_id" in validated_data:
            validated_data.pop("_id")  # Remove _id as it's not needed for update
        
        # Map productDisplayName to name if name is not provided
        if "productDisplayName" in validated_data and "name" not in validated_data:
            validated_data["name"] = validated_data["productDisplayName"]
        
        # Generate slug from name if not provided
        if "name" in validated_data and "slug" not in validated_data:
            validated_data["slug"] = slugify(validated_data["name"])
        
        # Set default values for required fields if not provided
        if "brand_id" not in validated_data:
            # Try to find or create a default brand
            try:
                default_brand = Brand.objects.first()
                if default_brand:
                    validated_data["brand_id"] = default_brand.id
            except:
                pass
        
        if "category_id" not in validated_data:
            # Try to find or create a default category
            try:
                default_category = Category.objects.first()
                if default_category:
                    validated_data["category_id"] = default_category.id
            except:
                pass
        
        # Set default values for other required fields
        if "description" not in validated_data:
            validated_data["description"] = validated_data.get("productDisplayName", "No description available")
        
        if "price" not in validated_data:
            validated_data["price"] = 0.0
        
        if "size" not in validated_data:
            validated_data["size"] = {}
        
        if "outfit_tags" not in validated_data:
            validated_data["outfit_tags"] = []
        
        if "feature_vector" not in validated_data:
            # Generate a default feature vector (128 dimensions with random values)
            validated_data["feature_vector"] = [0.0] * 128
        
        # Convert string IDs to ObjectId
        if "brand_id" in validated_data and isinstance(validated_data["brand_id"], str):
            try:
                validated_data["brand_id"] = ObjectId(validated_data["brand_id"])
            except:
                pass
        if "category_id" in validated_data and isinstance(validated_data["category_id"], str):
            try:
                validated_data["category_id"] = ObjectId(validated_data["category_id"])
            except:
                pass
        
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

