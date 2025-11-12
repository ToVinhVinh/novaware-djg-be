"""Các model sản phẩm migrate từ MongoDB."""

from __future__ import annotations

from django.conf import settings
from django.db import models


class Category(models.Model):
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "categories"
        ordering = ["name"]
        verbose_name = "Danh mục"
        verbose_name_plural = "Danh mục"

    def __str__(self) -> str:
        return self.name


class Color(models.Model):
    name = models.CharField(max_length=100, unique=True)
    hex_code = models.CharField(max_length=7)

    class Meta:
        db_table = "colors"
        unique_together = ("name", "hex_code")

    def __str__(self) -> str:
        return f"{self.name} ({self.hex_code})"


class Size(models.Model):
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=10, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "sizes"
        ordering = ["name"]

    def __str__(self) -> str:
        return self.code


class Product(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="products")
    brand = models.ForeignKey("brands.Brand", on_delete=models.PROTECT, related_name="products")
    category = models.ForeignKey(Category, on_delete=models.PROTECT, related_name="products")
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField()
    images = models.JSONField(default=list, blank=True)
    rating = models.FloatField(default=0)
    num_reviews = models.PositiveIntegerField(default=0)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    sale = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    count_in_stock = models.PositiveIntegerField(default=0)
    size = models.JSONField(default=dict, blank=True)
    colors = models.ManyToManyField(Color, through="ProductColor", related_name="products", blank=True)
    outfit_tags = models.JSONField(default=list, blank=True)
    compatible_products = models.ManyToManyField("self", blank=True, symmetrical=False)
    feature_vector = models.JSONField(default=list, blank=True)
    amazon_asin = models.CharField(max_length=50, blank=True, null=True, db_index=True)
    amazon_parent_asin = models.CharField(max_length=50, blank=True, null=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "products"
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    def update_stock_from_variants(self) -> None:
        total_stock = sum(variant.stock for variant in self.variants.all())
        self.count_in_stock = total_stock
        self.save(update_fields=["count_in_stock"])


class ProductColor(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    color = models.ForeignKey(Color, on_delete=models.CASCADE)

    class Meta:
        db_table = "product_colors"
        unique_together = ("product", "color")


class ProductVariant(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="variants")
    color = models.ForeignKey(Color, on_delete=models.CASCADE, related_name="product_variants")
    size = models.ForeignKey(Size, on_delete=models.CASCADE, related_name="product_variants")
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField(default=0)

    class Meta:
        db_table = "product_variants"
        unique_together = ("product", "color", "size")

    def __str__(self) -> str:
        return f"{self.product.name} - {self.color.name} / {self.size.code}"


class ProductReview(models.Model):
    product = models.ForeignKey(Product, related_name="reviews", on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    rating = models.PositiveSmallIntegerField()
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "product_reviews"
        ordering = ["-created_at"]
        unique_together = ("product", "user")


class ContentSection(models.Model):
    type = models.CharField(max_length=100)
    images = models.JSONField(default=list, blank=True)
    image = models.CharField(max_length=500, blank=True)
    subtitle = models.CharField(max_length=255, blank=True)
    title = models.CharField(max_length=255, blank=True)
    button_text = models.CharField(max_length=100, blank=True)
    button_link = models.CharField(max_length=500, blank=True)
    position = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "content_sections"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title or self.type

