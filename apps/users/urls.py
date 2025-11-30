from rest_framework import routers

from .mongo_views import OutfitHistoryViewSet, UserInteractionViewSet, UserViewSet

router = routers.DefaultRouter(trailing_slash=False)
router.register(r"users", UserViewSet, basename="user")
router.register(r"user-interactions", UserInteractionViewSet, basename="user-interaction")
router.register(r"outfit-history", OutfitHistoryViewSet, basename="outfit-history")

