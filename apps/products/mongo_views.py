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

def ensure_mongodb_connection():
    try:
        from mongoengine import connection
        connection.get_connection()
    except Exception:
        from config.mongodb import connect_mongodb
        connect_mongodb()

class CategoryViewSet(viewsets.ViewSet):

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        from collections import defaultdict

        ensure_mongodb_connection()

        products = Product.objects.only("masterCategory", "subCategory", "articleType").all()

        hierarchy_dict = defaultdict(lambda: defaultdict(set))
        master_categories_set = set()
        sub_categories_set = set()
        article_types_set = set()
        master_sub_mapping = defaultdict(set)

        for product in products:
            master_cat = getattr(product, "masterCategory", None)
            sub_cat = getattr(product, "subCategory", None)
            article_type = getattr(product, "articleType", None)

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

        hierarchy = []
        for master_cat in sorted(master_categories_set):
            sub_categories_list = []
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
        try:
            category = Category.objects.get(id=ObjectId(pk))
        except (Category.DoesNotExist, Exception):
            return api_error(
                "Category does not exist.",
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
        try:
            category = Category.objects.get(id=ObjectId(pk))
        except (Category.DoesNotExist, Exception):
            return api_error(
                "Category does not exist.",
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
        try:
            category = Category.objects.get(id=ObjectId(pk))
            category.delete()
            return api_success(
                "Category deleted successfully",
                data=None,
            )
        except (Category.DoesNotExist, Exception):
            return api_error(
                "Category does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

class ColorViewSet(viewsets.ViewSet):

    def list(self, request):
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
        try:
            color = Color.objects.get(id=ObjectId(pk))
        except (Color.DoesNotExist, Exception):
            return api_error(
                "Color does not exist.",
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
        try:
            color = Color.objects.get(id=ObjectId(pk))
        except (Color.DoesNotExist, Exception):
            return api_error(
                "Color does not exist.",
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
        try:
            color = Color.objects.get(id=ObjectId(pk))
            color.delete()
            return api_success(
                "Color deleted successfully",
                data=None,
            )
        except (Color.DoesNotExist, Exception):
            return api_error(
                "Color does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

class SizeViewSet(viewsets.ViewSet):

    def list(self, request):
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
        try:
            size = Size.objects.get(id=ObjectId(pk))
        except (Size.DoesNotExist, Exception):
            return api_error(
                "Size does not exist.",
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
        try:
            size = Size.objects.get(id=ObjectId(pk))
        except (Size.DoesNotExist, Exception):
            return api_error(
                "Size does not exist.",
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
        try:
            size = Size.objects.get(id=ObjectId(pk))
            size.delete()
            return api_success(
                "Size deleted successfully",
                data=None,
            )
        except (Size.DoesNotExist, Exception):
            return api_error(
                "Size does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

class ProductViewSet(viewsets.ViewSet):

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        queryset = Product.objects.all()

        category_id = request.query_params.get("category")
        if category_id:
            try:
                category_object_id = ObjectId(category_id)

                queryset1 = queryset.filter(category_id=category_object_id)
                count1 = queryset1.count()

                if count1 > 0:
                    queryset = queryset1
                else:
                    queryset2 = Product.objects.all().filter(__raw__={"category_id": category_object_id})
                    count2 = queryset2.count()

                    if count2 > 0:
                        queryset = queryset2
                    else:
                        queryset3 = Product.objects.all().filter(__raw__={"category_id": category_id})
                        count3 = queryset3.count()

                        if count3 > 0:
                            queryset = queryset3
                        else:
                            try:
                                queryset4 = Product.objects.all().filter(category_id=category_id)
                                count4 = queryset4.count()
                                if count4 > 0:
                                    queryset = queryset4
                            except:
                                pass

            except Exception as e:
                queryset = queryset.filter(__raw__={"category_id": category_id})

        article_type = request.query_params.get("articleType")
        if article_type:
            queryset = queryset.filter(articleType__iexact=article_type)

        search = request.query_params.get("search")
        if search:
            if search.isdigit():
                try:
                    search_id = int(search)
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
                    queryset = queryset.filter(
                        __raw__={"$or": [
                            {"name": {"$regex": search, "$options": "i"}},
                            {"productDisplayName": {"$regex": search, "$options": "i"}},
                            {"slug": {"$regex": search, "$options": "i"}},
                            {"description": {"$regex": search, "$options": "i"}},
                        ]}
                    )
            else:
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

        page, page_size = get_pagination_params(request)

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
        try:
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                product = Product.objects.get(id=ObjectId(pk))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product does not exist.",
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
        request_serializer = ProductSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)

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
        try:
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                product = Product.objects.get(id=ObjectId(pk))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product does not exist.",
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
        try:
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                product = Product.objects.get(id=ObjectId(pk))
            product.delete()
            return api_success(
                "Product deleted successfully",
                data=None,
            )
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def top(self, request):
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
        try:
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                product = Product.objects.get(id=ObjectId(pk))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        serializer = ProductReviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        review, created = ProductReview.objects.update_or_create(
            product_id=product.id,
            user_id=request.user.id,
            defaults={
                "name": serializer.validated_data["name"],
                "rating": serializer.validated_data["rating"],
                "comment": serializer.validated_data["comment"],
            },
        )

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
            "Review has been updated",
            {
                "review": review_serializer.data,
                "created": created,
            },
        )

    @action(detail=True, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def variants(self, request, pk=None):
        try:
            try:
                product_id = int(pk)
                product = Product.objects.get(id=product_id)
            except (ValueError, Product.DoesNotExist):
                product = Product.objects.get(id=ObjectId(pk))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product does not exist.",
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
        page, page_size = get_pagination_params(request)
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
    def filter(self, request):
        queryset = Product.objects.all()

        category_id = request.query_params.get("category")
        if category_id:
            try:
                category_object_id = ObjectId(category_id)

                queryset1 = queryset.filter(category_id=category_object_id)
                count1 = queryset1.count()

                if count1 > 0:
                    queryset = queryset1
                else:
                    queryset2 = Product.objects.all().filter(__raw__={"category_id": category_object_id})
                    count2 = queryset2.count()

                    if count2 > 0:
                        queryset = queryset2
                    else:
                        queryset3 = Product.objects.all().filter(__raw__={"category_id": category_id})
                        count3 = queryset3.count()

                        if count3 > 0:
                            queryset = queryset3
                        else:
                            try:
                                queryset4 = Product.objects.all().filter(category_id=category_id)
                                count4 = queryset4.count()
                                if count4 > 0:
                                    queryset = queryset4
                            except:
                                pass

            except Exception as e:
                queryset = queryset.filter(__raw__={"category_id": category_id})

        gender = request.query_params.get("gender")
        if gender:
            queryset = queryset.filter(gender__iexact=gender)

        master_category = request.query_params.get("masterCategory")
        if master_category:
            queryset = queryset.filter(masterCategory__iexact=master_category)

        sub_category = request.query_params.get("subCategory")
        if sub_category:
            queryset = queryset.filter(subCategory__iexact=sub_category)

        article_type = request.query_params.get("articleType")
        if article_type:
            queryset = queryset.filter(articleType__iexact=article_type)

        base_colour = request.query_params.get("baseColour")
        if base_colour:
            queryset = queryset.filter(baseColour__iexact=base_colour)

        season = request.query_params.get("season")
        if season:
            queryset = queryset.filter(season__iexact=season)

        year = request.query_params.get("year")
        if year:
            try:
                year_int = int(year)
                queryset = queryset.filter(year=year_int)
            except ValueError:
                pass

        min_price = request.query_params.get("minPrice")
        max_price = request.query_params.get("maxPrice")
        if min_price:
            try:
                min_price_float = float(min_price)
                queryset = queryset.filter(price__gte=min_price_float)
            except ValueError:
                pass
        if max_price:
            try:
                max_price_float = float(max_price)
                queryset = queryset.filter(price__lte=max_price_float)
            except ValueError:
                pass

        search = request.query_params.get("search")
        if search:
            if search.isdigit():
                try:
                    search_id = int(search)
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
                    queryset = queryset.filter(
                        __raw__={"$or": [
                            {"name": {"$regex": search, "$options": "i"}},
                            {"productDisplayName": {"$regex": search, "$options": "i"}},
                            {"slug": {"$regex": search, "$options": "i"}},
                            {"description": {"$regex": search, "$options": "i"}},
                        ]}
                    )
            else:
                queryset = queryset.filter(
                    __raw__={"$or": [
                        {"name": {"$regex": search, "$options": "i"}},
                        {"productDisplayName": {"$regex": search, "$options": "i"}},
                        {"slug": {"$regex": search, "$options": "i"}},
                        {"description": {"$regex": search, "$options": "i"}},
                    ]}
                )

        sort_by = request.query_params.get("sort_by") or request.query_params.get("ordering")
        if sort_by:
            if sort_by == "default":
                queryset = queryset.order_by("id")
            elif sort_by == "price_low_to_high":
                queryset = queryset.order_by("price")
            elif sort_by == "price_high_to_low":
                queryset = queryset.order_by("-price")
            elif sort_by == "rating":
                queryset = queryset.order_by("-rating", "-num_reviews")
            elif sort_by == "newest":
                queryset = queryset.order_by("-created_at")
            elif sort_by == "oldest":
                queryset = queryset.order_by("created_at")
            elif sort_by.startswith("-"):
                queryset = queryset.order_by(f"-{sort_by[1:]}")
            else:
                queryset = queryset.order_by(sort_by)
        else:
            queryset = queryset.order_by("id")

        page, page_size = get_pagination_params(request)

        option = request.query_params.get("option")
        if option == "all":
            products = list(queryset)
            serializer = ProductSerializer(products, many=True)
            total_count = queryset.count()
            return api_success(
                "Products filtered successfully",
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
            "Products filtered successfully",
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

        try:
            category = Category.objects.get(id=category_object_id)
            debug_info["category_exists"] = True
            debug_info["category_name"] = category.name
        except Category.DoesNotExist:
            debug_info["category_exists"] = False
            debug_info["category_name"] = None
        except Exception as e:
            debug_info["category_error"] = str(e)

        total_products = Product.objects.all().count()
        debug_info["total_products"] = total_products

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

        try:
            from mongoengine import connection
            db = connection.get_db()

            count_raw1 = db.products.count_documents({"category_id": category_object_id})
            debug_info["raw_mongodb_objectid"] = {"count": count_raw1}

            if count_raw1 > 0:
                doc1 = db.products.find_one({"category_id": category_object_id})
                if doc1:
                    debug_info["raw_mongodb_objectid"]["sample_type"] = str(type(doc1.get("category_id")))
                    debug_info["raw_mongodb_objectid"]["sample_value"] = str(doc1.get("category_id"))

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

    def list(self, request):
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
        try:
            section = ContentSection.objects.get(id=ObjectId(pk))
        except (ContentSection.DoesNotExist, Exception):
            return api_error(
                "ContentSection does not exist.",
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
        try:
            section = ContentSection.objects.get(id=ObjectId(pk))
        except (ContentSection.DoesNotExist, Exception):
            return api_error(
                "ContentSection does not exist.",
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
        try:
            section = ContentSection.objects.get(id=ObjectId(pk))
            section.delete()
            return api_success(
                "Content section deleted successfully",
                data=None,
            )
        except (ContentSection.DoesNotExist, Exception):
            return api_error(
                "ContentSection does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

