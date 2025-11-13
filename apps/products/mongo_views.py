"""ViewSets cho module sản phẩm sử dụng MongoEngine."""

from __future__ import annotations

import math

import random

from bson import ObjectId
from django.utils.text import slugify
from rest_framework import authentication, permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset
from apps.users.authentication import MongoEngineJWTAuthentication

from .mongo_models import (
    Category,
    Color,
    ContentSection,
    Product,
    ProductReview,
    ProductVariant,
    Size,
)
from .mongo_serializers import (
    CategorySerializer,
    ColorSerializer,
    ContentSectionSerializer,
    ProductReviewSerializer,
    ProductSerializer,
    ProductVariantSerializer,
    SizeSerializer,
)


class CategoryViewSet(viewsets.ViewSet):
    """ViewSet cho Category."""
    
    permission_classes = [permissions.AllowAny]  # Mặc định cho phép công khai
    authentication_classes = []  # Không yêu cầu authentication

    def get_permissions(self):
        # Các action cần authentication
        action = getattr(self, "action", None)
        if action in ["create", "update", "destroy", "partial_update"]:
            return [permissions.IsAuthenticated()]
        # Các action khác cho phép công khai
        return [permissions.AllowAny()]
    
    def get_authenticators(self):
        """Override để yêu cầu authentication cho các action cần thiết."""
        action = getattr(self, "action", None)
        if action in ["create", "update", "destroy", "partial_update"]:
            return [MongoEngineJWTAuthentication()]
        # Các action khác không cần authentication
        return []
    
    def list(self, request):
        """List all categories."""
        categories = Category.objects.all().order_by("name")

        page_number, per_page = get_pagination_params(request)
        paginated, total_count, total_pages, current_page, per_page = paginate_queryset(
            categories, page_number, per_page
        )

        serializer = CategorySerializer(paginated, many=True)
        return api_success(
            "Categories retrieved successfully",
            {
                "categories": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": per_page,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get category by id."""
        try:
            category = Category.objects.get(id=ObjectId(pk))
        except (Category.DoesNotExist, Exception):
            return api_error(
                "Category không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        serializer = CategorySerializer(category)
        return api_success(
            "Category retrieved successfully",
            {
                "category": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new category."""
        request_serializer = CategorySerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        category = request_serializer.create(request_serializer.validated_data)
        response_serializer = CategorySerializer(category)
        return api_success(
            "Category created successfully",
            {
                "category": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update category."""
        try:
            category = Category.objects.get(id=ObjectId(pk))
        except (Category.DoesNotExist, Exception):
            return api_error(
                "Category không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        request_serializer = CategorySerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        category = request_serializer.update(category, request_serializer.validated_data)
        response_serializer = CategorySerializer(category)
        return api_success(
            "Category updated successfully",
            {
                "category": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete category."""
        try:
            category = Category.objects.get(id=ObjectId(pk))
            category.delete()
            return api_success(
                "Category deleted successfully",
                data=None,
            )
        except (Category.DoesNotExist, Exception):
            return api_error(
                "Category không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )


class ColorViewSet(viewsets.ViewSet):
    """ViewSet cho Color."""
    
    permission_classes = [permissions.IsAuthenticated]

    def get_permissions(self):
        if getattr(self, "action", None) in ["list", "retrieve"]:
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]
    
    def list(self, request):
        """List all colors."""
        colors = Color.objects.all().order_by("name")

        page_number, per_page = get_pagination_params(request)
        paginated, total_count, total_pages, current_page, per_page = paginate_queryset(
            colors, page_number, per_page
        )
        serializer = ColorSerializer(paginated, many=True)
        return api_success(
            "Colors retrieved successfully",
            {
                "colors": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": per_page,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get color by id."""
        try:
            color = Color.objects.get(id=ObjectId(pk))
        except (Color.DoesNotExist, Exception):
            return api_error(
                "Color không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        serializer = ColorSerializer(color)
        return api_success(
            "Color retrieved successfully",
            {
                "color": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new color."""
        serializer = ColorSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        color = Color(**serializer.validated_data)
        color.save()
        response_serializer = ColorSerializer(color)
        return api_success(
            "Color created successfully",
            {
                "color": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update color."""
        try:
            color = Color.objects.get(id=ObjectId(pk))
        except (Color.DoesNotExist, Exception):
            return api_error(
                "Color không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        serializer = ColorSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        for key, value in serializer.validated_data.items():
            setattr(color, key, value)
        color.save()
        response_serializer = ColorSerializer(color)
        return api_success(
            "Color updated successfully",
            {
                "color": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete color."""
        try:
            color = Color.objects.get(id=ObjectId(pk))
            color.delete()
            return api_success(
                "Color deleted successfully",
                data=None,
            )
        except (Color.DoesNotExist, Exception):
            return api_error(
                "Color không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )


class SizeViewSet(viewsets.ViewSet):
    """ViewSet cho Size."""
    
    permission_classes = [permissions.IsAuthenticated]

    def get_permissions(self):
        if getattr(self, "action", None) in ["list", "retrieve"]:
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]
    
    def list(self, request):
        """List all sizes."""
        sizes = Size.objects.all().order_by("name")
        page_number, per_page = get_pagination_params(request)
        paginated, total_count, total_pages, current_page, per_page = paginate_queryset(
            sizes, page_number, per_page
        )
        serializer = SizeSerializer(paginated, many=True)
        return api_success(
            "Sizes retrieved successfully",
            {
                "sizes": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": per_page,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get size by id."""
        try:
            size = Size.objects.get(id=ObjectId(pk))
        except (Size.DoesNotExist, Exception):
            return api_error(
                "Size không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        serializer = SizeSerializer(size)
        return api_success(
            "Size retrieved successfully",
            {
                "size": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new size."""
        request_serializer = SizeSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        size = Size(**request_serializer.validated_data)
        size.save()
        response_serializer = SizeSerializer(size)
        return api_success(
            "Size created successfully",
            {
                "size": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update size."""
        try:
            size = Size.objects.get(id=ObjectId(pk))
        except (Size.DoesNotExist, Exception):
            return api_error(
                "Size không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        request_serializer = SizeSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        for key, value in request_serializer.validated_data.items():
            setattr(size, key, value)
        size.save()
        response_serializer = SizeSerializer(size)
        return api_success(
            "Size updated successfully",
            {
                "size": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete size."""
        try:
            size = Size.objects.get(id=ObjectId(pk))
            size.delete()
            return api_success(
                "Size deleted successfully",
                data=None,
            )
        except (Size.DoesNotExist, Exception):
            return api_error(
                "Size không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )


class ProductViewSet(viewsets.ViewSet):
    """ViewSet cho Product."""
    
    permission_classes = [permissions.AllowAny]  # Mặc định cho phép công khai
    authentication_classes = []  # Không yêu cầu authentication

    def get_permissions(self):
        # Các action cần authentication
        action = getattr(self, "action", None)
        if action in ["create", "update", "destroy", "partial_update", "review"]:
            return [permissions.IsAuthenticated()]
        # Các action khác cho phép công khai
        return [permissions.AllowAny()]
    
    def get_authenticators(self):
        """Override để yêu cầu authentication cho các action cần thiết."""
        action = getattr(self, "action", None)
        if action in ["create", "update", "destroy", "partial_update", "review"]:
            return [MongoEngineJWTAuthentication()]
        # Các action khác không cần authentication
        return []
    
    def list(self, request):
        """List products with filtering and pagination."""
        queryset = Product.objects.all()
        
        # Filter by brand
        brand_id = request.query_params.get("brand")
        if brand_id:
            try:
                queryset = queryset.filter(brand_id=ObjectId(brand_id))
            except Exception:
                pass
        
        # Filter by category
        category_id = request.query_params.get("category")
        if category_id:
            try:
                queryset = queryset.filter(category_id=ObjectId(category_id))
            except Exception:
                pass
        
        # Search
        search = request.query_params.get("search")
        if search:
            queryset = queryset.filter(
                __raw__={"$or": [
                    {"name": {"$regex": search, "$options": "i"}},
                    {"slug": {"$regex": search, "$options": "i"}},
                    {"description": {"$regex": search, "$options": "i"}},
                ]}
            )
        
        # Ordering
        ordering = request.query_params.get("ordering", "name")
        if ordering.startswith("-"):
            queryset = queryset.order_by(f"-{ordering[1:]}")
        else:
            queryset = queryset.order_by(ordering)
        
        # Pagination
        page, page_size = get_pagination_params(request)

        # Option handling (e.g., option=all to get all without pagination)
        option = request.query_params.get("option")
        if option == "all":
            products = list(queryset)
            serializer = ProductSerializer(products, many=True)
            total_count = queryset.count()
            return api_success(
                "Products retrieved successfully",
                {
                    "products": serializer.data,
                    "page": 1,
                    "pages": 1,
                    "perPage": len(products) or total_count or page_size,
                    "count": total_count,
                },
            )

        products, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = ProductSerializer(products, many=True)
        return api_success(
            "Products retrieved successfully",
            {
                "products": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get product by id."""
        try:
            product = Product.objects.get(id=ObjectId(pk))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        serializer = ProductSerializer(product)
        return api_success(
            "Product retrieved successfully",
            {
                "product": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new product."""
        request_serializer = ProductSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        
        # Add user_id and slug
        validated_data = request_serializer.validated_data.copy()
        validated_data["user_id"] = str(request.user.id)
        
        if "slug" not in validated_data or not validated_data["slug"]:
            validated_data["slug"] = slugify(validated_data["name"])
        
        product = request_serializer.create(validated_data)
        response_serializer = ProductSerializer(product)
        return api_success(
            "Product created successfully",
            {
                "product": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update product."""
        try:
            product = Product.objects.get(id=ObjectId(pk))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        request_serializer = ProductSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        
        validated_data = request_serializer.validated_data.copy()
        if "name" in validated_data and "slug" not in validated_data:
            validated_data["slug"] = slugify(validated_data["name"])
        
        product = request_serializer.update(product, validated_data)
        response_serializer = ProductSerializer(product)
        return api_success(
            "Product updated successfully",
            {
                "product": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete product."""
        try:
            product = Product.objects.get(id=ObjectId(pk))
            product.delete()
            return api_success(
                "Product deleted successfully",
                data=None,
            )
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
    
    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def top(self, request):
        """Get top rated products."""
        per_page_raw = request.query_params.get("perPage") or request.query_params.get("per_page")
        try:
            per_page = int(per_page_raw) if per_page_raw is not None else 10
        except (TypeError, ValueError):
            per_page = 10
        per_page = max(1, min(per_page, 50))

        base_queryset = Product.objects
        candidate_limit = max(per_page * 3, per_page)

        top_queryset = base_queryset.filter(num_reviews__gt=0).order_by("-rating", "-num_reviews", "-created_at")
        candidates = list(top_queryset[:candidate_limit])

        if len(candidates) < per_page:
            excluded_ids = [product.id for product in candidates]
            fallback_queryset = base_queryset.filter(id__nin=excluded_ids).order_by("-num_reviews", "-rating", "-created_at")
            needed = max(per_page - len(candidates), candidate_limit - len(candidates))
            candidates.extend(list(fallback_queryset[:needed]))

        if not candidates:
            candidates = list(base_queryset.order_by("-created_at")[:per_page])

        random.shuffle(candidates)
        selected = candidates[:per_page]

        serializer = ProductSerializer(selected, many=True)
        return api_success(
            "Top products retrieved successfully",
            {
                "products": serializer.data,
            },
        )
    
    @action(detail=True, methods=["post"])
    def review(self, request, pk=None):
        """Add or update product review."""
        try:
            product = Product.objects.get(id=ObjectId(pk))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        serializer = ProductReviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Update or create review
        review, created = ProductReview.objects.update_or_create(
            product_id=product.id,
            user_id=request.user.id,
            defaults={
                "name": serializer.validated_data["name"],
                "rating": serializer.validated_data["rating"],
                "comment": serializer.validated_data["comment"],
            },
        )
        
        # Update product rating
        reviews = ProductReview.objects(product_id=product.id)
        if reviews:
            total_rating = sum(r.rating for r in reviews)
            product.rating = total_rating / len(reviews)
            product.num_reviews = len(reviews)
        else:
            product.rating = 0
            product.num_reviews = 0
        product.save()
        
        review_serializer = ProductReviewSerializer(review)
        return api_success(
            "Đánh giá đã được cập nhật",
            {
                "review": review_serializer.data,
                "created": created,
            },
        )
    
    @action(detail=True, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def variants(self, request, pk=None):
        """Get product variants."""
        try:
            product = Product.objects.get(id=ObjectId(pk))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        variants = ProductVariant.objects(product_id=product.id)
        serializer = ProductVariantSerializer(variants, many=True)
        return api_success(
            "Product variants retrieved successfully",
            {
                "variants": serializer.data,
            },
        )

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def latest(self, request):
        """Get latest products (sorted by created_at desc)."""
        # Pagination aliases
        page, page_size = get_pagination_params(request)
        queryset = Product.objects.order_by("-created_at")
        products, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = ProductSerializer(products, many=True)
        return api_success(
            "Latest products retrieved successfully",
            {
                "products": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def sale(self, request):
        """Get products on sale (sale > 0), sorted by highest sale then newest."""
        page, page_size = get_pagination_params(request)
        # Filter sale > 0 using raw query for DecimalField
        queryset = Product.objects.filter(__raw__={"sale": {"$gt": 0}}).order_by("-sale", "-created_at")
        products, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = ProductSerializer(products, many=True)
        return api_success(
            "Sale products retrieved successfully",
            {
                "products": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )


class ContentSectionViewSet(viewsets.ViewSet):
    """ViewSet cho ContentSection."""
    
    permission_classes = [permissions.IsAuthenticated]

    def get_permissions(self):
        if getattr(self, "action", None) in ["list", "retrieve"]:
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]
    
    def list(self, request):
        """List all content sections."""
        sections = ContentSection.objects.all().order_by("-created_at")
        page_number, per_page = get_pagination_params(request)
        paginated, total_count, total_pages, current_page, per_page = paginate_queryset(
            sections, page_number, per_page
        )
        serializer = ContentSectionSerializer(paginated, many=True)
        return api_success(
            "Content sections retrieved successfully",
            {
                "contentSections": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": per_page,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get content section by id."""
        try:
            section = ContentSection.objects.get(id=ObjectId(pk))
        except (ContentSection.DoesNotExist, Exception):
            return api_error(
                "ContentSection không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        serializer = ContentSectionSerializer(section)
        return api_success(
            "Content section retrieved successfully",
            {
                "contentSection": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new content section."""
        request_serializer = ContentSectionSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        section = request_serializer.create(request_serializer.validated_data)
        response_serializer = ContentSectionSerializer(section)
        return api_success(
            "Content section created successfully",
            {
                "contentSection": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update content section."""
        try:
            section = ContentSection.objects.get(id=ObjectId(pk))
        except (ContentSection.DoesNotExist, Exception):
            return api_error(
                "ContentSection không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        request_serializer = ContentSectionSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        section = request_serializer.update(section, request_serializer.validated_data)
        response_serializer = ContentSectionSerializer(section)
        return api_success(
            "Content section updated successfully",
            {
                "contentSection": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete content section."""
        try:
            section = ContentSection.objects.get(id=ObjectId(pk))
            section.delete()
            return api_success(
                "Content section deleted successfully",
                data=None,
            )
        except (ContentSection.DoesNotExist, Exception):
            return api_error(
                "ContentSection không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

