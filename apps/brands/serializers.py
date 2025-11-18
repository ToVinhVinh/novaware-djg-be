"""Serializer for brands."""

from rest_framework import serializers

from .models import Brand


class BrandSerializer(serializers.ModelSerializer):
    class Meta:
        model = Brand
        fields = ["id", "name", "created_at", "updated_at"]

