"""Serializer cho đơn hàng."""

from __future__ import annotations

from rest_framework import serializers

from .models import Order, OrderItem, ShippingAddress


class OrderItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = OrderItem
        fields = [
            "id",
            "product",
            "name",
            "qty",
            "size_selected",
            "color_selected",
            "images",
            "price_sale",
        ]


class ShippingAddressSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShippingAddress
        fields = ["id", "address", "city", "postal_code", "country", "recipient_phone_number"]


class OrderSerializer(serializers.ModelSerializer):
    items = OrderItemSerializer(many=True)
    shipping_address = ShippingAddressSerializer()

    class Meta:
        model = Order
        fields = [
            "id",
            "user",
            "payment_method",
            "payment_result",
            "tax_price",
            "shipping_price",
            "total_price",
            "is_paid",
            "paid_at",
            "is_delivered",
            "delivered_at",
            "is_cancelled",
            "is_processing",
            "is_outfit_purchase",
            "created_at",
            "updated_at",
            "items",
            "shipping_address",
        ]
        read_only_fields = ["user", "created_at", "updated_at", "is_paid", "is_delivered", "is_cancelled", "is_processing"]

    def create(self, validated_data):
        items_data = validated_data.pop("items")
        shipping_data = validated_data.pop("shipping_address")
        order = Order.objects.create(**validated_data)
        ShippingAddress.objects.create(order=order, **shipping_data)
        for item in items_data:
            OrderItem.objects.create(order=order, **item)
        return order

    def update(self, instance, validated_data):
        items_data = validated_data.pop("items", None)
        shipping_data = validated_data.pop("shipping_address", None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        if shipping_data:
            ShippingAddress.objects.update_or_create(order=instance, defaults=shipping_data)
        if items_data is not None:
            instance.items.all().delete()
            for item in items_data:
                OrderItem.objects.create(order=instance, **item)
        return instance

