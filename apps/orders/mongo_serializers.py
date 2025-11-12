"""Serializers cho MongoEngine Order models."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import serializers

from .mongo_models import Order, OrderItem, ShippingAddress


class OrderItemSerializer(serializers.Serializer):
    product_id = serializers.CharField()
    name = serializers.CharField()
    qty = serializers.IntegerField()
    size_selected = serializers.CharField()
    color_selected = serializers.CharField()
    images = serializers.ListField(child=serializers.CharField())
    price_sale = serializers.DecimalField(max_digits=10, decimal_places=2)
    
    def to_representation(self, instance):
        """Convert embedded document to dict."""
        return {
            "product_id": str(instance.product_id),
            "name": instance.name,
            "qty": instance.qty,
            "size_selected": instance.size_selected,
            "color_selected": instance.color_selected,
            "images": instance.images,
            "price_sale": float(instance.price_sale),
        }
    
    def to_internal_value(self, data):
        """Convert dict to embedded document."""
        validated = super().to_internal_value(data)
        validated["product_id"] = ObjectId(validated["product_id"])
        validated["price_sale"] = validated["price_sale"]
        return validated


class ShippingAddressSerializer(serializers.Serializer):
    address = serializers.CharField()
    city = serializers.CharField()
    postal_code = serializers.CharField()
    country = serializers.CharField()
    recipient_phone_number = serializers.CharField()
    
    def to_representation(self, instance):
        """Convert embedded document to dict."""
        if instance is None:
            return None
        return {
            "address": instance.address,
            "city": instance.city,
            "postal_code": instance.postal_code,
            "country": instance.country,
            "recipient_phone_number": instance.recipient_phone_number,
        }
    
    def to_internal_value(self, data):
        """Convert dict to embedded document."""
        return super().to_internal_value(data)


class OrderSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    user_id = serializers.CharField(write_only=True, required=False)
    payment_method = serializers.CharField()
    payment_result = serializers.DictField(default=dict)
    tax_price = serializers.DecimalField(max_digits=10, decimal_places=2)
    shipping_price = serializers.DecimalField(max_digits=10, decimal_places=2)
    total_price = serializers.DecimalField(max_digits=10, decimal_places=2)
    is_paid = serializers.BooleanField(read_only=True)
    paid_at = serializers.DateTimeField(read_only=True, allow_null=True)
    is_delivered = serializers.BooleanField(read_only=True)
    delivered_at = serializers.DateTimeField(read_only=True, allow_null=True)
    is_cancelled = serializers.BooleanField(read_only=True)
    is_processing = serializers.BooleanField(read_only=True)
    is_outfit_purchase = serializers.BooleanField()
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    items = OrderItemSerializer(many=True)
    shipping_address = ShippingAddressSerializer()
    
    def get_id(self, obj):
        return str(obj.id)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        data = {
            "id": str(instance.id),
            "user_id": str(instance.user_id),
            "payment_method": instance.payment_method,
            "payment_result": instance.payment_result,
            "tax_price": float(instance.tax_price),
            "shipping_price": float(instance.shipping_price),
            "total_price": float(instance.total_price),
            "is_paid": instance.is_paid,
            "paid_at": instance.paid_at,
            "is_delivered": instance.is_delivered,
            "delivered_at": instance.delivered_at,
            "is_cancelled": instance.is_cancelled,
            "is_processing": instance.is_processing,
            "is_outfit_purchase": instance.is_outfit_purchase,
            "created_at": instance.created_at,
            "updated_at": instance.updated_at,
            "items": [OrderItemSerializer(item).to_representation(item) for item in instance.items],
            "shipping_address": ShippingAddressSerializer(instance.shipping_address).to_representation(instance.shipping_address) if instance.shipping_address else None,
        }
        return data
    
    def create(self, validated_data):
        """Create new order."""
        items_data = validated_data.pop("items")
        shipping_data = validated_data.pop("shipping_address")
        
        # Convert user_id to ObjectId
        if "user_id" in validated_data:
            validated_data["user_id"] = ObjectId(validated_data["user_id"])
        
        # Create embedded documents
        order_items = []
        for item_data in items_data:
            item_data["product_id"] = ObjectId(item_data["product_id"])
            order_items.append(OrderItem(**item_data))
        
        shipping_address = ShippingAddress(**shipping_data) if shipping_data else None
        
        order = Order(
            items=order_items,
            shipping_address=shipping_address,
            **validated_data
        )
        order.save()
        return order
    
    def update(self, instance, validated_data):
        """Update order."""
        items_data = validated_data.pop("items", None)
        shipping_data = validated_data.pop("shipping_address", None)
        
        for key, value in validated_data.items():
            setattr(instance, key, value)
        
        if shipping_data:
            instance.shipping_address = ShippingAddress(**shipping_data)
        
        if items_data is not None:
            order_items = []
            for item_data in items_data:
                item_data["product_id"] = ObjectId(item_data["product_id"])
                order_items.append(OrderItem(**item_data))
            instance.items = order_items
        
        instance.save()
        return instance

