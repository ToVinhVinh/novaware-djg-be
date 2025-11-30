from __future__ import annotations

from datetime import datetime
from typing import Optional

import mongoengine as me
from mongoengine import fields

class ShippingAddress(me.EmbeddedDocument):

    address = fields.StringField(required=True, max_length=255)
    city = fields.StringField(required=True, max_length=100)
    postal_code = fields.StringField(required=True, max_length=20)
    country = fields.StringField(required=True, max_length=100)
    recipient_phone_number = fields.StringField(required=True, max_length=20)

    def __str__(self) -> str:
        return f"{self.address}, {self.city}"

class OrderItem(me.EmbeddedDocument):

    product_id = fields.ObjectIdField(required=True)
    name = fields.StringField(required=True, max_length=255)
    qty = fields.IntField(required=True, min_value=1)
    size_selected = fields.StringField(required=True, max_length=50)
    color_selected = fields.StringField(required=True, max_length=50)
    images = fields.ListField(fields.StringField(), default=list)
    price_sale = fields.DecimalField(required=True, precision=10, decimal_places=2)

class Order(me.Document):

    meta = {
        "collection": "orders",
        "indexes": ["user_id", "created_at", "is_paid", "is_delivered"],
    }

    user_id = fields.ObjectIdField(required=True)
    payment_method = fields.StringField(required=True, max_length=100)
    payment_result = fields.DictField(default=dict)

    tax_price = fields.DecimalField(default=0.0, precision=10, decimal_places=2)
    shipping_price = fields.DecimalField(default=0.0, precision=10, decimal_places=2)
    total_price = fields.DecimalField(default=0.0, precision=10, decimal_places=2)

    is_paid = fields.BooleanField(default=False)
    paid_at = fields.DateTimeField(null=True)
    is_delivered = fields.BooleanField(default=False)
    delivered_at = fields.DateTimeField(null=True)
    is_cancelled = fields.BooleanField(default=False)
    is_processing = fields.BooleanField(default=False)
    is_outfit_purchase = fields.BooleanField(default=False)

    items = fields.ListField(fields.EmbeddedDocumentField(OrderItem), default=list)
    shipping_address = fields.EmbeddedDocumentField(ShippingAddress, null=True)

    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def mark_paid(self, timestamp: Optional[datetime] = None) -> None:
        self.is_paid = True
        self.paid_at = timestamp or datetime.utcnow()
        self.save()

    def __str__(self) -> str:
        return f"Order

