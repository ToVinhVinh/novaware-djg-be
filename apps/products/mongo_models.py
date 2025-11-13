"""Models sản phẩm sử dụng mongoengine cho MongoDB."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import mongoengine as me
from mongoengine import fields


class Category(me.Document):
    """Model danh mục sản phẩm."""
    
    meta = {
        "collection": "categories",
        "indexes": ["name"],
        "strict": False,
    }
    
    name = fields.StringField(required=True, unique=True, max_length=255)
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save để tự động cập nhật updated_at."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def __str__(self) -> str:
        return self.name


class Color(me.Document):
    """Model màu sắc."""
    
    meta = {
        "collection": "colors",
        "indexes": ["name", "hex_code"],
    }
    
    name = fields.StringField(required=True, unique=True, max_length=100)
    hex_code = fields.StringField(required=True, max_length=7)
    
    def __str__(self) -> str:
        return f"{self.name} ({self.hex_code})"


class Size(me.Document):
    """Model kích thước."""
    
    meta = {
        "collection": "sizes",
        "indexes": ["name", "code"],
    }
    
    name = fields.StringField(required=True, unique=True, max_length=100)
    code = fields.StringField(required=True, unique=True, max_length=10)
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save để tự động cập nhật updated_at."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def __str__(self) -> str:
        return self.code


class Product(me.Document):
    """Model sản phẩm."""
    
    meta = {
        "collection": "products",
        "indexes": [
            "name",
            "slug",
            "user_id",
            "brand_id",
            "category_id",
            "amazon_asin",
            "amazon_parent_asin",
        ],
        "strict": False,
    }
    
    # References
    user_id = fields.ObjectIdField(required=True)
    brand_id = fields.ObjectIdField(required=True)
    category_id = fields.ObjectIdField(required=True)
    
    # Basic info
    name = fields.StringField(required=True, max_length=255)
    slug = fields.StringField(required=True, unique=True, max_length=255)
    description = fields.StringField(required=True)
    images = fields.ListField(fields.StringField(), default=list)
    
    # Ratings
    rating = fields.FloatField(default=0.0, min_value=0.0, max_value=5.0)
    num_reviews = fields.IntField(default=0, min_value=0)
    gender = fields.StringField(choices=("male", "female", "unisex"), default="unisex")
    age_group = fields.StringField(choices=("kid", "teen", "adult"), default="adult")
    category_type = fields.StringField(
        choices=("tops", "bottoms", "dresses", "shoes", "accessories"),
        default="tops",
    )
    
    # Pricing
    price = fields.DecimalField(required=True, default=0.0, precision=10, decimal_places=2)
    sale = fields.DecimalField(default=0.0, precision=5, decimal_places=2)
    count_in_stock = fields.IntField(default=0, min_value=0)
    
    # Variants
    size = fields.DictField(default=dict)
    color_ids = fields.ListField(fields.ObjectIdField(), default=list)
    
    # Features
    outfit_tags = fields.ListField(fields.StringField(), default=list)
    style_tags = fields.ListField(fields.StringField(), default=list)
    compatible_product_ids = fields.ListField(fields.ObjectIdField(), default=list)
    feature_vector = fields.ListField(fields.FloatField(), default=list)
    
    # Amazon
    amazon_asin = fields.StringField(null=True, max_length=50)
    amazon_parent_asin = fields.StringField(null=True, max_length=50)
    
    # Timestamps
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save để tự động cập nhật updated_at."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def update_stock_from_variants(self) -> None:
        """Cập nhật tổng số lượng từ variants."""
        variants = ProductVariant.objects(product_id=self.id)
        total_stock = sum(variant.stock for variant in variants)
        self.count_in_stock = total_stock
        self.save()
    
    def __str__(self) -> str:
        return self.name


class ProductVariant(me.Document):
    """Model biến thể sản phẩm (màu + kích thước)."""
    
    meta = {
        "collection": "product_variants",
        "indexes": ["product_id", "color_id", "size_id"],
    }
    
    product_id = fields.ObjectIdField(required=True)
    color_id = fields.ObjectIdField(required=True)
    size_id = fields.ObjectIdField(required=True)
    price = fields.DecimalField(required=True, precision=10, decimal_places=2)
    stock = fields.IntField(default=0, min_value=0)
    
    def __str__(self) -> str:
        return f"{self.product_id} - {self.color_id} / {self.size_id}"


class ProductReview(me.Document):
    """Model đánh giá sản phẩm."""
    
    meta = {
        "collection": "product_reviews",
        "indexes": ["product_id", "user_id", "created_at"],
    }
    
    product_id = fields.ObjectIdField(required=True)
    user_id = fields.ObjectIdField(required=True)
    name = fields.StringField(required=True, max_length=255)
    rating = fields.IntField(required=True, min_value=1, max_value=5)
    comment = fields.StringField(required=True)
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save để tự động cập nhật updated_at."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def __str__(self) -> str:
        return f"Review for {self.product_id} by {self.user_id}"


class ContentSection(me.Document):
    """Model section nội dung."""
    
    meta = {
        "collection": "content_sections",
        "indexes": ["type", "created_at"],
    }
    
    type = fields.StringField(required=True, max_length=100)
    images = fields.ListField(fields.StringField(), default=list)
    image = fields.StringField(max_length=500)
    subtitle = fields.StringField(max_length=255)
    title = fields.StringField(max_length=255)
    button_text = fields.StringField(max_length=100)
    button_link = fields.StringField(max_length=500)
    position = fields.StringField(max_length=100)
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save để tự động cập nhật updated_at."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def __str__(self) -> str:
        return self.title or self.type

