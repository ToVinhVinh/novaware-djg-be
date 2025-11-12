"""Đăng ký admin cho đơn hàng."""

from django.contrib import admin

from .models import Order, OrderItem, ShippingAddress


class OrderItemInline(admin.TabularInline):
    model = OrderItem
    extra = 0


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "total_price",
        "is_paid",
        "is_delivered",
        "is_processing",
        "created_at",
    )
    list_filter = ("is_paid", "is_delivered", "is_processing", "is_cancelled")
    search_fields = ("user__email", "id")
    inlines = [OrderItemInline]


@admin.register(ShippingAddress)
class ShippingAddressAdmin(admin.ModelAdmin):
    list_display = ("order", "address", "city", "country", "recipient_phone_number")


@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ("order", "product", "qty", "price_sale")

