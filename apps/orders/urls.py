"""Router for order API."""

from rest_framework import routers

from .mongo_views import OrderViewSet


router = routers.DefaultRouter(trailing_slash=True)
router.register(r"orders", OrderViewSet, basename="order")

