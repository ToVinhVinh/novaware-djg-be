"""Serializers cho MongoEngine Product models."""

from __future__ import annotations

from bson import ObjectId
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
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        data = {
            "id": str(instance.id),
            "brand_id": str(instance.brand_id),
            "category_id": str(instance.category_id),
            "name": instance.name,
            "slug": instance.slug,
            "description": instance.description,
            "images": instance.images,
            "rating": instance.rating,
            "num_reviews": instance.num_reviews,
            "price": float(instance.price),
            "sale": float(instance.sale),
            "count_in_stock": instance.count_in_stock,
            "size": instance.size,
            "color_ids": [str(cid) for cid in instance.color_ids],
            "outfit_tags": instance.outfit_tags,
            "compatible_product_ids": [str(pid) for pid in instance.compatible_product_ids],
            "feature_vector": instance.feature_vector,
            "amazon_asin": instance.amazon_asin,
            "amazon_parent_asin": instance.amazon_parent_asin,
            "created_at": instance.created_at,
            "updated_at": instance.updated_at,
        }
        
        # Load brand name
        data["brand_name"] = self.get_brand_name(instance)
        
        # Load category
        try:
            category = Category.objects.get(id=instance.category_id)
            data["category_detail"] = CategorySerializer(category).data
        except Category.DoesNotExist:
            data["category_detail"] = None
        
        # Load variants
        variants = ProductVariant.objects(product_id=instance.id)
        data["variants"] = ProductVariantSerializer(variants, many=True).data
        
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

