"""Serializer for product module."""

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
    _id = serializers.CharField(source="id", read_only=True)

    class Meta:
        model = ProductVariant
        fields = ["_id", "stock", "color", "size", "price"]


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ["id", "name"]


class ProductReviewSerializer(serializers.ModelSerializer):
    _id = serializers.CharField(source="id", read_only=True)
    user = serializers.CharField(source="user.id", read_only=True)
    createdAt = serializers.DateTimeField(source="created_at", read_only=True)
    updatedAt = serializers.DateTimeField(source="updated_at", read_only=True)

    class Meta:
        model = ProductReview
        fields = ["_id", "name", "rating", "comment", "user", "createdAt", "updatedAt"]
        read_only_fields = ["user", "createdAt", "updatedAt"]


class ProductSerializer(serializers.ModelSerializer):
    variants = ProductVariantSerializer(many=True, read_only=True)
    reviews = ProductReviewSerializer(many=True, read_only=True)

    class Meta:
        model = Product
        fields = [
            "id",
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "year",
            "usage",
            "productDisplayName",
            "images",
            "rating",
            "sale",
            "reviews",
            "variants",
        ]
        read_only_fields = ["id", "rating"]


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

