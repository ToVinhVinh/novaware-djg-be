"""Router cho module chat."""

from rest_framework import routers

from .mongo_views import ChatThreadViewSet


router = routers.DefaultRouter()
router.register(r"chat-threads", ChatThreadViewSet, basename="chat-thread")

