"""Router definition for brand API."""

from rest_framework import routers

from .mongo_views import BrandViewSet


router = routers.DefaultRouter(trailing_slash=False)
router.register(r"brands", BrandViewSet, basename="brand")

