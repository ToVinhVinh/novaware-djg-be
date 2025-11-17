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
        "strict": False,
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
        "strict": False,
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
            "gender",
            "masterCategory",
            "subCategory",
            "productDisplayName",
            "name",
            "slug",
        ],
        "strict": False,
    }
    
    # Các field từ CSV
    id = fields.IntField(primary_key=True)
    gender = fields.StringField(max_length=50)
    masterCategory = fields.StringField(max_length=255)
    subCategory = fields.StringField(max_length=255)
    articleType = fields.StringField(max_length=255)
    baseColour = fields.StringField(max_length=100)
    season = fields.StringField(max_length=50)
    year = fields.IntField()
    usage = fields.StringField(max_length=100)
    productDisplayName = fields.StringField(max_length=255)
    
    # Additional required fields for API compatibility
    name = fields.StringField(max_length=255)
    slug = fields.StringField(max_length=255)
    description = fields.StringField()
    brand_id = fields.ObjectIdField()
    category_id = fields.ObjectIdField()
    price = fields.DecimalField(default=0.0, precision=10, decimal_places=2)
    size = fields.DictField(default=dict)
    outfit_tags = fields.ListField(fields.StringField(), default=list)
    feature_vector = fields.ListField(fields.FloatField(), default=list)
    color_ids = fields.ListField(fields.ObjectIdField(), default=list)
    compatible_product_ids = fields.ListField(fields.StringField(), default=list)
    amazon_asin = fields.StringField()
    amazon_parent_asin = fields.StringField()
    user_id = fields.ObjectIdField()
    num_reviews = fields.IntField(default=0)
    count_in_stock = fields.IntField(default=0)
    
    # Các field giữ lại
    images = fields.ListField(fields.StringField(), default=list)
    rating = fields.FloatField(default=0.0, min_value=0.0, max_value=5.0)
    sale = fields.DecimalField(default=0.0, precision=5, decimal_places=2)
    
    # Timestamps
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save để tự động cập nhật updated_at."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def update_stock_from_variants(self):
        """Update count_in_stock from variants."""
        try:
            variants = ProductVariant.objects(product_id=self.id)
            total_stock = sum(variant.stock for variant in variants)
            self.count_in_stock = total_stock
            self.save()
        except Exception:
            pass
    
    def __str__(self) -> str:
        return self.productDisplayName or self.name or f"Product {self.id}"


class ProductVariant(me.Document):
    """Model biến thể sản phẩm (màu + kích thước)."""
    
    meta = {
        "collection": "product_variants",
        "indexes": ["product_id", "color", "size"],
        "strict": False,
    }
    
    product_id = fields.IntField(required=True)  # Reference to Product.id (int)
    color = fields.StringField(required=True, max_length=7)  # Hex color code
    size = fields.StringField(required=True, max_length=10)  # Size code
    price = fields.DecimalField(required=True, precision=10, decimal_places=2)
    stock = fields.IntField(default=0, min_value=0)
    
    def __str__(self) -> str:
        return f"{self.product_id} - {self.color} / {self.size}"


class ProductReview(me.Document):
    """Model đánh giá sản phẩm."""
    
    meta = {
        "collection": "product_reviews",
        "indexes": ["product_id", "user_id", "created_at"],
        "strict": False,
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
        "strict": False,
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

