"""Cấu hình ứng dụng thương hiệu."""

from django.apps import AppConfig


class BrandsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.brands"
    verbose_name = "Thương hiệu"

