"""Định nghĩa router cho API thương hiệu."""

from rest_framework import routers

from .mongo_views import BrandViewSet


router = routers.DefaultRouter()
router.register(r"brands", BrandViewSet, basename="brand")

