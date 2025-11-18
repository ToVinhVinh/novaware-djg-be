"""ViewSets and user endpoints."""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db.models import Count
from django.shortcuts import get_object_or_404
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset

from apps.orders.models import Order
from apps.products.models import Product
from apps.products.serializers import ProductSerializer

from .models import OutfitHistory, PasswordResetAudit, UserInteraction
from .serializers import (
    GenderSummarySerializer,
    OutfitHistorySerializer,
    PasswordChangeSerializer,
    PurchaseHistorySummarySerializer,
    StylePreferenceSummarySerializer,
    UserDetailSerializer,
    UserForTestingSerializer,
    UserInteractionSerializer,
    UserSerializer,
)

User = get_user_model()


class IsAdminOrSelf(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return request.user.is_staff or obj == request.user


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
     
    search_fields = ["email", "username"]
    ordering_fields = ["email", "date_joined"]

    def get_serializer_class(self):
        if self.action in {"retrieve", "me"}:
            return UserDetailSerializer
        return super().get_serializer_class()

    def get_queryset(self):
        if self.request.user.is_staff:
            return User.objects.all()
        return User.objects.filter(pk=self.request.user.pk)

    # Removed permission restrictions - all actions are now public

    @action(detail=False, methods=["get"])
    def me(self, request):
        serializer = self.get_serializer(request.user)
        return api_success(
            "Current user retrieved successfully",
            {
                "user": serializer.data,
            },
        )

    @action(detail=True, methods=["get", "post", "delete"])
    def favorites(self, request, pk=None):
        user = self.get_object()
        if request.method == "GET":
            page, page_size = get_pagination_params(request)
            favorites = user.favorites.all()
            favorites_page, total_count, total_pages, current_page, page_size = paginate_queryset(
                favorites, page, page_size
            )
            serializer = ProductSerializer(favorites_page, many=True)
            return api_success(
                "Favorites retrieved successfully",
                {
                    "favorites": serializer.data,
                    "page": current_page,
                    "pages": total_pages,
                    "perPage": page_size,
                    "count": total_count,
                },
            )

        product_id = request.data.get("product") or request.query_params.get("product")
        product = get_object_or_404(Product, pk=product_id)
        if request.method == "POST":
            user.favorites.add(product)
            return api_success(
                "Product added to favorites.",
                {
                    "product": ProductSerializer(product).data,
                    "favoritesCount": user.favorites.count(),
                },
                status_code=status.HTTP_201_CREATED,
            )
        user.favorites.remove(product)
        return api_success(
            "Product removed from favorites.",
            {
                "product": ProductSerializer(product).data,
                "favoritesCount": user.favorites.count(),
            },
        )

    @action(detail=True, methods=["get"])
    def check_purchase_history(self, request, pk=None):
        user = self.get_object()
        count = Order.objects.filter(user=user, is_paid=True).count()
        serializer = PurchaseHistorySummarySerializer({
            "has_purchase_history": count > 0,
            "order_count": count,
        })
        return api_success(
            "Purchase history checked successfully",
            {
                "summary": serializer.data,
            },
        )

    @action(detail=True, methods=["get"])
    def check_gender(self, request, pk=None):
        user = self.get_object()
        serializer = GenderSummarySerializer({
            "has_gender": bool(user.gender),
            "gender": user.gender,
        })
        return api_success(
            "Gender checked successfully",
            {
                "summary": serializer.data,
            },
        )

    @action(detail=True, methods=["get"])
    def check_style_preference(self, request, pk=None):
        user = self.get_object()
        style = user.preferences.get("style") if user.preferences else None
        serializer = StylePreferenceSummarySerializer({
            "has_style_preference": bool(style),
            "style": style,
        })
        return api_success(
            "Style preference checked successfully",
            {
                "summary": serializer.data,
            },
        )

    @action(detail=False, methods=["post"])
    def change_password(self, request):
        serializer = PasswordChangeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = request.user
        if not user.check_password(serializer.validated_data["old_password"]):
            return api_error(
                "Old password is incorrect.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        user.set_password(serializer.validated_data["new_password"])
        user.save(update_fields=["password"])
        return api_success(
            "Password changed successfully.",
            data=None,
        )

    @action(detail=False, methods=["get"])
    def testing(self, request):
        query_type = request.query_params.get("type")
        if query_type not in {"personalization", "outfit-suggestions"}:
            return api_error(
                "Invalid type parameter.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        page, page_size = get_pagination_params(request)
        queryset = User.objects.annotate(order_count=Count("orders"))
        page_items, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = UserForTestingSerializer(page_items, many=True)
        return api_success(
            "Testing data retrieved successfully",
            {
                "type": query_type,
                "users": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )


class UserInteractionViewSet(viewsets.ModelViewSet):
    serializer_class = UserInteractionSerializer
     

    def get_queryset(self):
        qs = UserInteraction.objects.select_related("user", "product")
        if not self.request.user.is_staff:
            qs = qs.filter(user=self.request.user)
        return qs

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class OutfitHistoryViewSet(viewsets.ModelViewSet):
    serializer_class = OutfitHistorySerializer
     

    def get_queryset(self):
        qs = OutfitHistory.objects.select_related("user").prefetch_related("products")
        if not self.request.user.is_staff:
            qs = qs.filter(user=self.request.user)
        return qs

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

