from __future__ import annotations

from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_success

from .models import Outfit, RecommendationRequest, RecommendationResult
from .serializers import (
    OutfitSerializer,
    RecommendationRequestSerializer,
    RecommendationResultSerializer,
)
from .services import RecommendationService

class OutfitViewSet(viewsets.ModelViewSet):
    queryset = Outfit.objects.prefetch_related("products")
    serializer_class = OutfitSerializer

    search_fields = ["name", "style", "season"]
    ordering_fields = ["created_at", "compatibility_score"]

class RecommendationRequestViewSet(viewsets.ModelViewSet):
    serializer_class = RecommendationRequestSerializer

    def get_queryset(self):
        qs = RecommendationRequest.objects.select_related("user").prefetch_related("result__products", "logs")
        if not self.request.user.is_staff:
            qs = qs.filter(user=self.request.user)
        return qs

    def perform_create(self, serializer):
        request_obj = serializer.save(user=self.request.user)
        RecommendationService.enqueue_recommendation(request_obj)

    @action(detail=True, methods=["post"])
    def refresh(self, request, pk=None):
        request_obj = self.get_object()
        RecommendationService.enqueue_recommendation(request_obj)
        return api_success(
            "Request has been queued for processing.",
            {
                "request": RecommendationRequestSerializer(request_obj).data,
            },
            status_code=status.HTTP_202_ACCEPTED,
        )

class RecommendationResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RecommendationResult.objects.prefetch_related("products")
    serializer_class = RecommendationResultSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        if not self.request.user.is_staff:
            qs = qs.filter(request__user=self.request.user)
        return qs

