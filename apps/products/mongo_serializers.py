"""Serializers cho MongoEngine Product models."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache

from bson import ObjectId
from django.conf import settings
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

        price = getattr(instance, "price", 0.0)
        sale = getattr(instance, "sale", 0.0)
        count_in_stock = getattr(instance, "count_in_stock", 0) or 0
        size_info = getattr(instance, "size", None) or {}
        color_ids = [cid for cid in getattr(instance, "color_ids", []) if cid]
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

        compatible_ids = getattr(instance, "compatible_product_ids", None)
        compatible_products_raw = getattr(instance, "compatibleProducts", None)
        compatible_ids_list = []
        if compatible_ids:
            compatible_ids_list = [_stringify_id(pid) for pid in compatible_ids if pid]
        elif compatible_products_raw:
            for item in compatible_products_raw:
                if isinstance(item, dict) and "$oid" in item:
                    compatible_ids_list.append(item["$oid"])
                elif item:
                    compatible_ids_list.append(_stringify_id(item))

        feature_vector = list(getattr(instance, "feature_vector", []) or [])
        user_oid = getattr(instance, "user_id", None)

        amazon_parent_asin = (
            getattr(instance, "amazonParentAsin", None)
            or getattr(instance, "amazon_parent_asin", None)
        )
        amazon_asin = getattr(instance, "amazon_asin", None)

        created_at = getattr(instance, "created_at", None)
        updated_at = getattr(instance, "updated_at", None)

        brand_name = self.get_brand_name(instance)
        category_detail = None
        category_id_value = getattr(instance, "category_id", None)
        if category_id_value:
            try:
                category = Category.objects.get(id=category_id_value)
                category_detail = CategorySerializer(category).data
            except Category.DoesNotExist:
                category_detail = None
        
        brand_fallback = getattr(instance, "brand", None) or brand_name or None
        category_fallback = getattr(instance, "category", None)
        if not category_fallback and category_detail:
            category_fallback = category_detail.get("name") or ""

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

        compatible_products_payload = []
        for pid in compatible_ids_list:
            compatible_products_payload.append(_id_string(pid))

        data: dict[str, object] = {
            "_id": _id_string(instance.id),
            "id": str(instance.id),
            "brand_id": _stringify_id(getattr(instance, "brand_id", None)),
            "category_id": _stringify_id(category_id_value),
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
            "color_ids": [_stringify_id(cid) for cid in color_ids],
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
            "brand": brand_fallback,
            "category_detail": category_detail,
            "category": category_fallback,
            "user": _id_string(user_oid),
            "reviews": reviews_payload,
            "__v": getattr(instance, "__v", 0) or 0,
            "variants": variants_payload,
        }

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

