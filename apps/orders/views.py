"""ViewSets for order management."""

from __future__ import annotations

from django.utils import timezone
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_success

from .models import Order
from .serializers import OrderSerializer


class OrderViewSet(viewsets.ModelViewSet):
    serializer_class = OrderSerializer
     
    filterset_fields = ["is_paid", "is_processing", "is_outfit_purchase"]
    search_fields = ["id", "user__email"]
    ordering_fields = ["created_at", "total_price"]

    def get_queryset(self):
        queryset = Order.objects.select_related("user", "shipping_address").prefetch_related("items")
        if not self.request.user.is_staff:
            queryset = queryset.filter(user=self.request.user)
        return queryset

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["post"])
    def mark_paid(self, request, pk=None):
        order = self.get_object()
        order.mark_paid()
        return api_success(
            "Order has been marked as paid.",
            {
                "order": OrderSerializer(order).data,
            },
        )

    @action(detail=True, methods=["post"])
    def mark_delivered(self, request, pk=None):
        order = self.get_object()
        order.is_delivered = True
        order.delivered_at = timezone.now()
        order.save(update_fields=["is_delivered", "delivered_at"])
        return api_success(
            "Order has been delivered.",
            {
                "order": OrderSerializer(order).data,
            },
        )

    @action(detail=True, methods=["post"])
    def cancel(self, request, pk=None):
        order = self.get_object()
        order.is_cancelled = True
        order.save(update_fields=["is_cancelled"])
        return api_success(
            "Order has been cancelled.",
            {
                "order": OrderSerializer(order).data,
            },
        )

