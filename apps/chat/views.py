"""ViewSets cho module chat."""

from __future__ import annotations

from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_success

from .models import ChatThread, Message
from .serializers import ChatThreadSerializer, MessageSerializer


class ChatThreadViewSet(viewsets.ModelViewSet):
    serializer_class = ChatThreadSerializer
     

    def get_queryset(self):
        qs = ChatThread.objects.prefetch_related("messages")
        if not self.request.user.is_staff:
            qs = qs.filter(user=self.request.user)
        return qs

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["post"])
    def add_message(self, request, pk=None):
        thread = self.get_object()
        serializer = MessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        Message.objects.create(thread=thread, **serializer.validated_data)
        thread.save(update_fields=["updated_at"])
        return api_success(
            "Tin nhắn đã được thêm.",
            {
                "thread": ChatThreadSerializer(thread).data,
            },
            status_code=status.HTTP_201_CREATED,
        )

