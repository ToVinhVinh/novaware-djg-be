"""ViewSets for brands using MongoEngine."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import authentication, permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset
from apps.users.authentication import MongoEngineJWTAuthentication

from .mongo_models import Brand
from .mongo_serializers import BrandSerializer


class BrandViewSet(viewsets.ViewSet):
    """ViewSet for Brand."""
    
    permission_classes = [permissions.AllowAny]  
    authentication_classes = []  

    def get_permissions(self):
        action = getattr(self, "action", None)
        if action in ["create", "update", "destroy", "partial_update"]:
            return [permissions.IsAuthenticated()]
        return [permissions.AllowAny()]
    
    def get_authenticators(self):
        """Override to require authentication for necessary actions."""
        action = getattr(self, "action", None)
        if action in ["create", "update", "destroy", "partial_update"]:
            return [MongoEngineJWTAuthentication()]
        return []
    
    def list(self, request):
        """List all brands."""
        brands = Brand.objects.all().order_by("name")
        page, per_page = get_pagination_params(request)
        paginated, total_count, total_pages, current_page, per_page = paginate_queryset(
            brands, page, per_page
        )
        serializer = BrandSerializer(paginated, many=True)
        return api_success(
            "Brands retrieved successfully",
            {
                "brands": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": per_page,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get brand by id."""
        try:
            brand = Brand.objects.get(id=ObjectId(pk))
        except (Brand.DoesNotExist, Exception):
            return api_error(
                "Brand does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        serializer = BrandSerializer(brand)
        return api_success(
            "Brand retrieved successfully",
            {
                "brand": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new brand."""
        request_serializer = BrandSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        brand = request_serializer.create(request_serializer.validated_data)
        response_serializer = BrandSerializer(brand)
        return api_success(
            "Brand created successfully",
            {
                "brand": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update brand."""
        try:
            brand = Brand.objects.get(id=ObjectId(pk))
        except (Brand.DoesNotExist, Exception):
            return api_error(
                "Brand does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        request_serializer = BrandSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        brand = request_serializer.update(brand, request_serializer.validated_data)
        response_serializer = BrandSerializer(brand)
        return api_success(
            "Brand updated successfully",
            {
                "brand": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete brand."""
        try:
            brand = Brand.objects.get(id=ObjectId(pk))
            brand.delete()
            return api_success(
                "Brand deleted successfully",
                data=None,
            )
        except (Brand.DoesNotExist, Exception):
            return api_error(
                "Brand does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def grouped(self, request):
        """Return brands grouped by first letter of name."""
        brands = Brand.objects.all().order_by("name")
        groups = {}
        for b in brands:
            first = (b.name[0].upper() if b.name else "#")
            if not first.isalpha():
                first = "#"
            groups.setdefault(first, []).append(BrandSerializer(b).data)
        # Convert to list of {letter, items} sorted by letter
        grouped_list = [{"letter": k, "items": v} for k, v in sorted(groups.items(), key=lambda x: x[0])]
        return api_success(
            "Brands grouped successfully",
            {
                "groups": grouped_list,
                "count": len(grouped_list),
            },
        )

