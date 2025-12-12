from __future__ import annotations

from bson import ObjectId
from datetime import datetime
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset

from .mongo_models import Order
from .mongo_serializers import OrderSerializer

class OrderViewSet(viewsets.ViewSet):

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        queryset = Order.objects.all()

        if request.query_params.get("is_paid") == "true":
            queryset = queryset.filter(is_paid=True)
        elif request.query_params.get("is_paid") == "false":
            queryset = queryset.filter(is_paid=False)

        if request.query_params.get("is_processing") == "true":
            queryset = queryset.filter(is_processing=True)

        if request.query_params.get("is_outfit_purchase") == "true":
            queryset = queryset.filter(is_outfit_purchase=True)

        ordering = request.query_params.get("ordering", "-created_at")
        if ordering.startswith("-"):
            queryset = queryset.order_by(f"-{ordering[1:]}")
        else:
            queryset = queryset.order_by(ordering)

        page, page_size = get_pagination_params(request)
        orders, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = OrderSerializer(orders, many=True)
        return api_success(
            "Orders retrieved successfully",
            {
                "orders": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )

    def retrieve(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Order does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        serializer = OrderSerializer(order)
        return api_success(
            "Order retrieved successfully",
            {
                "order": serializer.data,
            },
        )

    def create(self, request):
        request_serializer = OrderSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)

        validated_data = request_serializer.validated_data.copy()
        if not validated_data.get("user_id"):
            if request.user and hasattr(request.user, 'id') and request.user.is_authenticated:
                validated_data["user_id"] = str(request.user.id)
            else:
                return api_error(
                    "user_id is required when not logged in.",
                    data=None,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

        order = request_serializer.create(validated_data)
        response_serializer = OrderSerializer(order)
        return api_success(
            "Order created successfully!",
            {
                "order": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )

    def update(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Order does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        request_serializer = OrderSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        order = request_serializer.update(order, request_serializer.validated_data)
        response_serializer = OrderSerializer(order)
        return api_success(
            "Order updated successfully",
            {
                "order": response_serializer.data,
            },
        )

    def destroy(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Order does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        order.delete()
        return api_success(
            "Order deleted successfully",
            data=None,
        )

    @action(detail=True, methods=["post"])
    def mark_paid(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Order does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        order.mark_paid()
        updated_serializer = OrderSerializer(order)
        return api_success(
            "Order has been marked as paid.",
            {
                "order": updated_serializer.data,
            },
        )

    @action(detail=True, methods=["post"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def mark_delivered(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Order does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        order.is_delivered = True
        order.delivered_at = datetime.utcnow()
        order.save()
        updated_serializer = OrderSerializer(order)
        return api_success(
            "Order has been delivered.",
            {
                "order": updated_serializer.data,
            },
        )

    @action(detail=True, methods=["post"])
    def cancel(self, request, pk=None):
        try:
            order = Order.objects.get(id=ObjectId(pk))
        except (Order.DoesNotExist, Exception):
            return api_error(
                "Order does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        order.is_cancelled = True
        order.save()
        updated_serializer = OrderSerializer(order)
        return api_success(
            "Order has been cancelled.",
            {
                "order": updated_serializer.data,
            },
        )

    @action(detail=False, methods=["get"], url_path="my/orders", permission_classes=[permissions.AllowAny], authentication_classes=[])
    def my_orders(self, request):
        user_id = request.query_params.get("user_id")
        if not user_id:
            if request.user and hasattr(request.user, 'id') and request.user.is_authenticated:
                user_id = str(request.user.id)
            else:
                return api_error(
                    "user_id is required when not logged in.",
                    data=None,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
        queryset = Order.objects.filter(user_id=user_id).order_by("-created_at")

        page, page_size = get_pagination_params(request)
        orders, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = OrderSerializer(orders, many=True)
        return api_success(
            "Orders retrieved successfully",
            {
                "orders": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )

