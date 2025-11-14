"""Router cho hệ thống gợi ý."""

from rest_framework import routers

from .mongo_views import (
    OutfitViewSet,
    RecommendationRequestViewSet,
    RecommendationResultViewSet,
)


router = routers.DefaultRouter(trailing_slash=False)
router.register(r"outfits", OutfitViewSet, basename="outfit")
router.register(r"recommendations", RecommendationRequestViewSet, basename="recommendation-request")
router.register(r"recommendation-results", RecommendationResultViewSet, basename="recommendation-result")

