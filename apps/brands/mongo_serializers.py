"""Serializers cho MongoEngine Brand models."""

from __future__ import annotations

from rest_framework import serializers

from .mongo_models import Brand


class BrandSerializer(serializers.Serializer):
    id = serializers.SerializerMethodField()
    name = serializers.CharField()
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    
    def get_id(self, obj):
        return str(obj.id)
    
    def create(self, validated_data):
        return Brand(**validated_data).save()
    
    def update(self, instance, validated_data):
        for key, value in validated_data.items():
            setattr(instance, key, value)
        instance.save()
        return instance

