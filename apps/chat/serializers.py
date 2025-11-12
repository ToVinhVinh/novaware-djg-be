"""Serializer cho module chat."""

from __future__ import annotations

from rest_framework import serializers

from .models import ChatThread, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "sender", "content", "timestamp"]


class ChatThreadSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = ChatThread
        fields = ["id", "user", "created_at", "updated_at", "messages"]
        read_only_fields = ["user", "created_at", "updated_at", "messages"]

