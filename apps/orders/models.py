from __future__ import annotations

from django.conf import settings
from django.db import models
from django.utils import timezone

class Order(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="orders", on_delete=models.CASCADE)
    payment_method = models.CharField(max_length=100)
    payment_result = models.JSONField(default=dict, blank=True)
    tax_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    shipping_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    is_paid = models.BooleanField(default=False)
    paid_at = models.DateTimeField(null=True, blank=True)
    is_delivered = models.BooleanField(default=False)
    delivered_at = models.DateTimeField(null=True, blank=True)
    is_cancelled = models.BooleanField(default=False)
    is_processing = models.BooleanField(default=False)
    is_outfit_purchase = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "orders"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Order {self.id}"

    def mark_paid(self, timestamp: timezone.datetime | None = None):
        self.is_paid = True
        self.paid_at = timestamp or timezone.now()
        self.save(update_fields=["is_paid", "paid_at"])

class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name="items", on_delete=models.CASCADE)
    product = models.ForeignKey("products.Product", on_delete=models.PROTECT)
    name = models.CharField(max_length=255)
    qty = models.PositiveIntegerField()
    size_selected = models.CharField(max_length=50)
    color_selected = models.CharField(max_length=50)
    images = models.JSONField(default=list, blank=True)
    price_sale = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        db_table = "order_items"

class ShippingAddress(models.Model):
    order = models.OneToOneField(Order, related_name="shipping_address", on_delete=models.CASCADE)
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    postal_code = models.CharField(max_length=20)
    country = models.CharField(max_length=100)
    recipient_phone_number = models.CharField(max_length=20)

    class Meta:
        db_table = "shipping_addresses"

    def __str__(self) -> str:
        return f"{self.address}, {self.city}"

