from __future__ import annotations

import random

from django.db.models import Avg, Count
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
    permission_classes = [permissions.AllowAny]
    search_fields = ["name"]
    ordering_fields = ["name", "created_at"]

class ColorViewSet(viewsets.ModelViewSet):
    queryset = Color.objects.all()
    serializer_class = ColorSerializer
    permission_classes = [permissions.AllowAny]
    search_fields = ["name", "hex_code"]

class SizeViewSet(viewsets.ModelViewSet):
    queryset = Size.objects.all()
    serializer_class = SizeSerializer
    permission_classes = [permissions.AllowAny]
    search_fields = ["name", "code"]

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.prefetch_related(
        "variants",
        "reviews",
    )
    serializer_class = ProductSerializer
    permission_classes = [permissions.AllowAny]
    search_fields = ["productDisplayName", "masterCategory", "subCategory", "articleType"]
    ordering_fields = ["id", "rating", "year", "created_at"]

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny])
    def top(self, request):
        per_page_raw = request.query_params.get("perPage") or request.query_params.get("per_page")
        try:
            per_page = int(per_page_raw) if per_page_raw is not None else 10
        except (TypeError, ValueError):
            per_page = 10
        per_page = max(1, min(per_page, 50))

        base_queryset = self.get_queryset()
        candidate_limit = max(per_page * 3, per_page)

        top_queryset = base_queryset.filter(rating__gt=0).order_by("-rating", "-created_at")
        candidates = list(top_queryset[:candidate_limit])

        if len(candidates) < per_page:
            excluded_ids = [product.pk for product in candidates]
            fallback_queryset = base_queryset.exclude(pk__in=excluded_ids).order_by("-rating", "-created_at")
            needed = max(per_page - len(candidates), candidate_limit - len(candidates))
            candidates.extend(list(fallback_queryset[:needed]))

        if not candidates:
            candidates = list(base_queryset.order_by("-created_at")[:per_page])

        random.shuffle(candidates)
        selected = candidates[:per_page]

        serializer = self.get_serializer(selected, many=True)
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
        product.save(update_fields=["rating"])
        return api_success(
            "Review has been updated",
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
    permission_classes = [permissions.AllowAny]
    search_fields = ["title", "type"]
    ordering_fields = ["created_at"]

