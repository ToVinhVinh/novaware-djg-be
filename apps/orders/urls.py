"""Router cho API đơn hàng."""

from rest_framework import routers

from .mongo_views import OrderViewSet


router = routers.DefaultRouter()
router.register(r"orders", OrderViewSet, basename="order")

