"""ViewSets cho module sản phẩm."""

from __future__ import annotations

from django.db.models import Avg, Count
from django.utils.text import slugify
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_success

from .models import Category, Color, ContentSection, Product, ProductReview, ProductVariant, Size
from .serializers import (
    CategorySerializer,
    ColorSerializer,
    ContentSectionSerializer,
    ProductReviewSerializer,
    ProductSerializer,
    ProductVariantSerializer,
    SizeSerializer,
)


class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [permissions.IsAuthenticated]
    search_fields = ["name"]
    ordering_fields = ["name", "created_at"]


class ColorViewSet(viewsets.ModelViewSet):
    queryset = Color.objects.all()
    serializer_class = ColorSerializer
    permission_classes = [permissions.IsAuthenticated]
    search_fields = ["name", "hex_code"]


class SizeViewSet(viewsets.ModelViewSet):
    queryset = Size.objects.all()
    serializer_class = SizeSerializer
    permission_classes = [permissions.IsAuthenticated]
    search_fields = ["name", "code"]


class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.select_related("brand", "category", "user").prefetch_related(
        "variants__color",
        "variants__size",
        "reviews",
    )
    serializer_class = ProductSerializer
    permission_classes = [permissions.IsAuthenticated]
    filterset_fields = ["brand", "category"]
    search_fields = ["name", "slug", "description"]
    ordering_fields = ["name", "price", "rating", "created_at"]

    def perform_create(self, serializer):
        slug = serializer.validated_data.get("slug") or slugify(serializer.validated_data["name"])
        serializer.save(user=self.request.user, slug=slug)

    def perform_update(self, serializer):
        slug = serializer.validated_data.get("slug") or slugify(serializer.validated_data.get("name", serializer.instance.name))
        serializer.save(slug=slug)

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny])
    def top(self, request):
        queryset = self.get_queryset().filter(num_reviews__gt=0).order_by("-rating")[:10]
        serializer = self.get_serializer(queryset, many=True)
        return api_success(
            "Top products retrieved successfully",
            {
                "products": serializer.data,
            },
        )

    @action(detail=True, methods=["post"])
    def review(self, request, pk=None):
        product = self.get_object()
        serializer = ProductReviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        ProductReview.objects.update_or_create(
            product=product,
            user=request.user,
            defaults={
                "name": serializer.validated_data["name"],
                "rating": serializer.validated_data["rating"],
                "comment": serializer.validated_data["comment"],
            },
        )
        stats = product.reviews.aggregate(avg=Avg("rating"), count=Count("id"))
        product.rating = stats["avg"] or 0
        product.num_reviews = stats["count"] or 0
        product.save(update_fields=["rating", "num_reviews"])
        return api_success(
            "Đánh giá đã được cập nhật",
            {
                "product": ProductSerializer(product).data,
            },
        )

    @action(detail=True, methods=["get"], permission_classes=[permissions.AllowAny])
    def variants(self, request, pk=None):
        product = self.get_object()
        serializer = ProductVariantSerializer(product.variants.all(), many=True)
        return api_success(
            "Product variants retrieved successfully",
            {
                "variants": serializer.data,
            },
        )


class ContentSectionViewSet(viewsets.ModelViewSet):
    queryset = ContentSection.objects.all()
    serializer_class = ContentSectionSerializer
    permission_classes = [permissions.IsAuthenticated]
    search_fields = ["title", "type"]
    ordering_fields = ["created_at"]

