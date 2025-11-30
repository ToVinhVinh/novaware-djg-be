from django.contrib import admin

from .models import (
    Category,
    Color,
    ContentSection,
    Product,
    ProductColor,
    ProductReview,
    ProductVariant,
    Size,
)

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    search_fields = ("name",)
    list_display = ("name", "created_at", "updated_at")

@admin.register(Color)
class ColorAdmin(admin.ModelAdmin):
    list_display = ("name", "hex_code")
    search_fields = ("name", "hex_code")

@admin.register(Size)
class SizeAdmin(admin.ModelAdmin):
    list_display = ("name", "code", "created_at")
    search_fields = ("name", "code")

class ProductVariantInline(admin.TabularInline):
    model = ProductVariant
    extra = 1

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "productDisplayName",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "sale",
        "rating",
        "year",
    )
    list_filter = ("gender", "masterCategory", "subCategory", "season", "year")
    search_fields = ("productDisplayName", "masterCategory", "subCategory", "articleType")
    inlines = [ProductVariantInline]

@admin.register(ProductReview)
class ProductReviewAdmin(admin.ModelAdmin):
    list_display = ("product", "user", "rating", "created_at")
    search_fields = ("product__productDisplayName", "user__email", "name")
    list_filter = ("rating", "created_at")

@admin.register(ContentSection)
class ContentSectionAdmin(admin.ModelAdmin):
    list_display = ("type", "title", "position", "created_at")
    search_fields = ("type", "title")

@admin.register(ProductColor)
class ProductColorAdmin(admin.ModelAdmin):
    list_display = ("product", "color")

