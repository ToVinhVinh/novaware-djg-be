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
    
     
    
    def list(self, request):
        """List all categories from unique values in Product collection."""
        from collections import defaultdict
        
        # Get all products with category fields
        products = Product.objects.only("masterCategory", "subCategory", "articleType").all()
        
        # Build hierarchical structure
        hierarchy_dict = defaultdict(lambda: defaultdict(set))
        master_categories_set = set()
        sub_categories_set = set()
        article_types_set = set()
        # Track all subCategories per masterCategory (even without articleTypes)
        master_sub_mapping = defaultdict(set)
        
        for product in products:
            master_cat = getattr(product, "masterCategory", None)
            sub_cat = getattr(product, "subCategory", None)
            article_type = getattr(product, "articleType", None)
            
            # Filter out None/empty values and normalize strings
            if master_cat and str(master_cat).strip():
                master_cat = str(master_cat).strip()
                master_categories_set.add(master_cat)
                
                if sub_cat and str(sub_cat).strip():
                    sub_cat = str(sub_cat).strip()
                    sub_categories_set.add(sub_cat)
                    master_sub_mapping[master_cat].add(sub_cat)
                    
                    if article_type and str(article_type).strip():
                        article_type = str(article_type).strip()
                        hierarchy_dict[master_cat][sub_cat].add(article_type)
                        article_types_set.add(article_type)
        
        # Build hierarchy list
        hierarchy = []
        for master_cat in sorted(master_categories_set):
            sub_categories_list = []
            # Get all subCategories for this masterCategory (including those without articleTypes)
            for sub_cat in sorted(master_sub_mapping[master_cat]):
                article_types_list = sorted(list(hierarchy_dict[master_cat].get(sub_cat, set())))
                sub_categories_list.append({
                    "subCategory": sub_cat,
                    "articleTypes": article_types_list
                })
            hierarchy.append({
                "masterCategory": master_cat,
                "subCategories": sub_categories_list
            })
        
        # Return structured data
        return api_success(
            "Categories retrieved successfully",
            {
                "hierarchy": hierarchy,
                "masterCategories": sorted(master_categories_set),
                "subCategories": sorted(sub_categories_set),
                "articleTypes": sorted(article_types_set),
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
    
    def list(self, request):
        """List products with filtering and pagination."""
        queryset = Product.objects.all()
        
        # Filter by category FIRST (before brand filter)
        category_id = request.query_params.get("category")
        if category_id:
            try:
                category_object_id = ObjectId(category_id)
                
                # Try multiple filter methods and use the one that works
                # Method 1: Direct field access with ObjectId (MongoEngine's preferred way)
                queryset1 = queryset.filter(category_id=category_object_id)
                count1 = queryset1.count()
                
                if count1 > 0:
                    queryset = queryset1
                else:
                    # Method 2: __raw__ with ObjectId
                    queryset2 = Product.objects.all().filter(__raw__={"category_id": category_object_id})
                    count2 = queryset2.count()
                    
                    if count2 > 0:
                        queryset = queryset2
                    else:
                        # Method 3: __raw__ with string (in case it's stored as string)
                        queryset3 = Product.objects.all().filter(__raw__={"category_id": category_id})
                        count3 = queryset3.count()
                        
                        if count3 > 0:
                            queryset = queryset3
                        else:
                            # Method 4: Try direct field access with string (MongoEngine auto-conversion)
                            try:
                                queryset4 = Product.objects.all().filter(category_id=category_id)
                                count4 = queryset4.count()
                                if count4 > 0:
                                    queryset = queryset4
                            except:
                                pass
                            
            except Exception as e:
                # If ObjectId conversion fails, try as string
                queryset = queryset.filter(__raw__={"category_id": category_id})
        
        # Filter by brand
        brand_id = request.query_params.get("brand")
        if brand_id:
            try:
                brand_object_id = ObjectId(brand_id)
                # Use __raw__ to ensure proper filtering
                queryset = queryset.filter(__raw__={"brand_id": brand_object_id})
            except Exception as e:
                # Invalid ObjectId format, skip filter
                pass
        
        # Search
        search = request.query_params.get("search")
        if search:
            # Check if search value is numeric (for ID search)
            if search.isdigit():
                try:
                    search_id = int(search)
                    # Search by ID first, then fallback to text search
                    queryset = queryset.filter(
                        __raw__={"$or": [
                            {"_id": search_id},
                            {"id": search_id},
                            {"name": {"$regex": search, "$options": "i"}},
                            {"productDisplayName": {"$regex": search, "$options": "i"}},
                            {"slug": {"$regex": search, "$options": "i"}},
                            {"description": {"$regex": search, "$options": "i"}},
                        ]}
                    )
                except ValueError:
                    # If conversion fails, treat as text search
                    queryset = queryset.filter(
                        __raw__={"$or": [
                            {"name": {"$regex": search, "$options": "i"}},
                            {"productDisplayName": {"$regex": search, "$options": "i"}},
                            {"slug": {"$regex": search, "$options": "i"}},
                            {"description": {"$regex": search, "$options": "i"}},
                        ]}
                    )
            else:
                # Text search
                queryset = queryset.filter(
                    __raw__={"$or": [
                        {"name": {"$regex": search, "$options": "i"}},
                        {"productDisplayName": {"$regex": search, "$options": "i"}},
                        {"slug": {"$regex": search, "$options": "i"}},
                        {"description": {"$regex": search, "$options": "i"}},
                    ]}
                )
        
        ordering = request.query_params.get("ordering")
        if ordering:
            if ordering.startswith("-"):
                queryset = queryset.order_by(f"-{ordering[1:]}")
            else:
                queryset = queryset.order_by(ordering)
        else:
            queryset = queryset.order_by("id")
        
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
            # Try integer ID first (for products imported from CSV)
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                # Fallback to ObjectId if integer fails
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
            # Try integer ID first (for products imported from CSV)
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                # Fallback to ObjectId if integer fails
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
            # Try integer ID first (for products imported from CSV)
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                # Fallback to ObjectId if integer fails
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
            # Try integer ID first (for products imported from CSV)
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                # Fallback to ObjectId if integer fails
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
            # Try integer ID first (for products imported from CSV)
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                # Fallback to ObjectId if integer fails
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
    
    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def debug_category(self, request):
        """Debug endpoint to check category and products count."""
        category_id = request.query_params.get("category")
        if not category_id:
            return api_error("Category ID is required", status_code=status.HTTP_400_BAD_REQUEST)
        
        try:
            category_object_id = ObjectId(category_id)
        except Exception as e:
            return api_error(f"Invalid category ID format: {str(e)}", status_code=status.HTTP_400_BAD_REQUEST)
        
        debug_info = {
            "category_id": category_id,
            "category_object_id": str(category_object_id),
        }
        
        # Check if category exists
        try:
            category = Category.objects.get(id=category_object_id)
            debug_info["category_exists"] = True
            debug_info["category_name"] = category.name
        except Category.DoesNotExist:
            debug_info["category_exists"] = False
            debug_info["category_name"] = None
        except Exception as e:
            debug_info["category_error"] = str(e)
        
        # Test different filter methods
        total_products = Product.objects.all().count()
        debug_info["total_products"] = total_products
        
        # Method 1: Direct field access with ObjectId
        try:
            queryset1 = Product.objects.all().filter(category_id=category_object_id)
            count1 = queryset1.count()
            debug_info["method1_direct_objectid"] = {
                "count": count1,
                "works": count1 > 0
            }
            if count1 > 0:
                sample1 = queryset1.first()
                debug_info["method1_direct_objectid"]["sample_category_id_type"] = str(type(sample1.category_id))
                debug_info["method1_direct_objectid"]["sample_category_id_value"] = str(sample1.category_id)
        except Exception as e:
            debug_info["method1_direct_objectid"] = {"error": str(e)}
        
        # Method 2: __raw__ with ObjectId
        try:
            queryset2 = Product.objects.all().filter(__raw__={"category_id": category_object_id})
            count2 = queryset2.count()
            debug_info["method2_raw_objectid"] = {
                "count": count2,
                "works": count2 > 0
            }
            if count2 > 0:
                sample2 = queryset2.first()
                debug_info["method2_raw_objectid"]["sample_category_id_type"] = str(type(sample2.category_id))
                debug_info["method2_raw_objectid"]["sample_category_id_value"] = str(sample2.category_id)
        except Exception as e:
            debug_info["method2_raw_objectid"] = {"error": str(e)}
        
        # Method 3: __raw__ with string
        try:
            queryset3 = Product.objects.all().filter(__raw__={"category_id": category_id})
            count3 = queryset3.count()
            debug_info["method3_raw_string"] = {
                "count": count3,
                "works": count3 > 0
            }
            if count3 > 0:
                sample3 = queryset3.first()
                debug_info["method3_raw_string"]["sample_category_id_type"] = str(type(sample3.category_id))
                debug_info["method3_raw_string"]["sample_category_id_value"] = str(sample3.category_id)
        except Exception as e:
            debug_info["method3_raw_string"] = {"error": str(e)}
        
        # Check actual data in first few products
        try:
            products_sample = Product.objects.all()[:5]
            sample_data = []
            for p in products_sample:
                cat_id = p.category_id
                sample_data.append({
                    "product_id": str(p.id),
                    "category_id_type": str(type(cat_id)),
                    "category_id_value": str(cat_id),
                    "matches": str(cat_id) == category_id
                })
            debug_info["sample_products"] = sample_data
        except Exception as e:
            debug_info["sample_products_error"] = str(e)
        
        # Try MongoDB raw query
        try:
            from mongoengine import connection
            db = connection.get_db()
            
            # Raw query with ObjectId
            count_raw1 = db.products.count_documents({"category_id": category_object_id})
            debug_info["raw_mongodb_objectid"] = {"count": count_raw1}
            
            if count_raw1 > 0:
                doc1 = db.products.find_one({"category_id": category_object_id})
                if doc1:
                    debug_info["raw_mongodb_objectid"]["sample_type"] = str(type(doc1.get("category_id")))
                    debug_info["raw_mongodb_objectid"]["sample_value"] = str(doc1.get("category_id"))
            
            # Raw query with string
            count_raw2 = db.products.count_documents({"category_id": category_id})
            debug_info["raw_mongodb_string"] = {"count": count_raw2}
            
            if count_raw2 > 0:
                doc2 = db.products.find_one({"category_id": category_id})
                if doc2:
                    debug_info["raw_mongodb_string"]["sample_type"] = str(type(doc2.get("category_id")))
                    debug_info["raw_mongodb_string"]["sample_value"] = str(doc2.get("category_id"))
        except Exception as e:
            debug_info["raw_mongodb_error"] = str(e)
        
        return api_success("Category debug info", debug_info)


class ContentSectionViewSet(viewsets.ViewSet):
    """ViewSet cho ContentSection."""
    
     
    
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

