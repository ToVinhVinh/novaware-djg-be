"""Serializers cho MongoEngine Chat models."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import serializers

from .mongo_models import ChatThread, Message


class MessageSerializer(serializers.Serializer):
    sender = serializers.CharField()
    content = serializers.CharField()
    timestamp = serializers.DateTimeField(read_only=True)
    
    def to_representation(self, instance):
        """Convert embedded document to dict."""
        return {
            "sender": instance.sender,
            "content": instance.content,
            "timestamp": instance.timestamp,
        }
    
    def to_internal_value(self, data):
        """Convert dict to embedded document."""
        return super().to_internal_value(data)


class ChatThreadSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    user_id = serializers.CharField(write_only=True, required=False)
    messages = MessageSerializer(many=True, read_only=True)
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def to_representation(self, instance):
        """Convert MongoEngine document to dict."""
        return {
            "id": str(instance.id),
            "user_id": str(instance.user_id),
            "messages": [MessageSerializer(msg).to_representation(msg) for msg in instance.messages],
            "created_at": instance.created_at,
            "updated_at": instance.updated_at,
        }
    
    def create(self, validated_data):
        """Create new chat thread."""
        if "user_id" in validated_data:
            validated_data["user_id"] = ObjectId(validated_data["user_id"])
        return ChatThread(**validated_data).save()
    
    def update(self, instance, validated_data):
        """Update chat thread."""
        for key, value in validated_data.items():
            setattr(instance, key, value)
        instance.save()
        return instance

