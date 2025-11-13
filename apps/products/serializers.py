"""Serializer cho module sản phẩm."""

from __future__ import annotations

from rest_framework import serializers

from .models import (
    Category,
    Color,
    ContentSection,
    Product,
    ProductReview,
    ProductVariant,
    Size,
)


class ColorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Color
        fields = ["id", "name", "hex_code"]


class SizeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Size
        fields = ["id", "name", "code"]


class ProductVariantSerializer(serializers.ModelSerializer):
    color = ColorSerializer(read_only=True)
    color_id = serializers.PrimaryKeyRelatedField(
        queryset=Color.objects.all(), source="color", write_only=True
    )
    size = SizeSerializer(read_only=True)
    size_id = serializers.PrimaryKeyRelatedField(
        queryset=Size.objects.all(), source="size", write_only=True
    )

    class Meta:
        model = ProductVariant
        fields = ["id", "color", "color_id", "size", "size_id", "price", "stock"]


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ["id", "name"]


class ProductReviewSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ProductReview
        fields = ["id", "name", "rating", "comment", "user", "user_name", "created_at"]
        read_only_fields = ["user", "created_at"]


class ProductSerializer(serializers.ModelSerializer):
    brand_name = serializers.CharField(source="brand.name", read_only=True)
    category_detail = CategorySerializer(source="category", read_only=True)
    variants = ProductVariantSerializer(many=True, read_only=True)
    variants_payload = ProductVariantSerializer(many=True, write_only=True, required=False)

    class Meta:
        model = Product
        fields = [
            "id",
            "user",
            "brand",
            "brand_name",
            "category",
            "category_detail",
            "name",
            "slug",
            "description",
            "images",
            "rating",
            "num_reviews",
            "price",
            "sale",
            "count_in_stock",
            "size",
            "colors",
            "variants",
            "variants_payload",
            "outfit_tags",
            "style_tags",
            "compatible_products",
            "feature_vector",
            "gender",
            "age_group",
            "category_type",
            "amazon_asin",
            "amazon_parent_asin",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["user", "rating", "num_reviews", "count_in_stock", "created_at", "updated_at"]

    def create(self, validated_data):
        variants_data = validated_data.pop("variants_payload", [])
        product = super().create(validated_data)
        for variant in variants_data:
            product.variants.create(**variant)
        product.count_in_stock = sum(v.stock for v in product.variants.all())
        product.save(update_fields=["count_in_stock"])
        return product

    def update(self, instance, validated_data):
        variants_data = validated_data.pop("variants_payload", None)
        product = super().update(instance, validated_data)
        if variants_data is not None:
            instance.variants.all().delete()
            for variant in variants_data:
                instance.variants.create(**variant)
            instance.count_in_stock = sum(v.stock for v in instance.variants.all())
            instance.save(update_fields=["count_in_stock"])
        return product


class ContentSectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContentSection
        fields = [
            "id",
            "type",
            "images",
            "image",
            "subtitle",
            "title",
            "button_text",
            "button_link",
            "position",
            "created_at",
            "updated_at",
        ]

