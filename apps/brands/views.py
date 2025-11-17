"""ViewSets cho thương hiệu."""

from rest_framework import permissions, viewsets

from .models import Brand
from .serializers import BrandSerializer


class BrandViewSet(viewsets.ModelViewSet):
    queryset = Brand.objects.all()
    serializer_class = BrandSerializer
     
    search_fields = ["name"]
    ordering_fields = ["name", "created_at"]

