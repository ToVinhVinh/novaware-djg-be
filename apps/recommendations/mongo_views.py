"""ViewSets cho hệ thống gợi ý sử dụng MongoEngine."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset

from .mongo_models import (
    Outfit,
    RecommendationLog,
    RecommendationRequest,
    RecommendationResult,
)
from .mongo_serializers import (
    OutfitSerializer,
    RecommendationRequestSerializer,
    RecommendationResultSerializer,
)
from .mongo_services import RecommendationService


class OutfitViewSet(viewsets.ViewSet):
    """ViewSet cho Outfit."""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def list(self, request):
        """List all outfits."""
        outfits = Outfit.objects.all().order_by("-created_at")
        
        # Search
        search = request.query_params.get("search")
        if search:
            outfits = outfits.filter(
                __raw__={"$or": [
                    {"name": {"$regex": search, "$options": "i"}},
                    {"style": {"$regex": search, "$options": "i"}},
                    {"season": {"$regex": search, "$options": "i"}},
                ]}
            )
        
        # Ordering
        ordering = request.query_params.get("ordering", "-created_at")
        if ordering.startswith("-"):
            outfits = outfits.order_by(f"-{ordering[1:]}")
        else:
            outfits = outfits.order_by(ordering)
        
        # Pagination
        page, page_size = get_pagination_params(request)
        outfit_list, total_count, total_pages, current_page, page_size = paginate_queryset(
            outfits, page, page_size
        )
        serializer = OutfitSerializer(outfit_list, many=True)
        return api_success(
            "Outfits retrieved successfully",
            {
                "outfits": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get outfit by id."""
        try:
            outfit = Outfit.objects.get(id=ObjectId(pk))
        except (Outfit.DoesNotExist, Exception):
            return api_error(
                "Outfit không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        serializer = OutfitSerializer(outfit)
        return api_success(
            "Outfit retrieved successfully",
            {
                "outfit": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new outfit."""
        request_serializer = OutfitSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        outfit = request_serializer.create(request_serializer.validated_data)
        response_serializer = OutfitSerializer(outfit)
        return api_success(
            "Outfit created successfully",
            {
                "outfit": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update outfit."""
        try:
            outfit = Outfit.objects.get(id=ObjectId(pk))
        except (Outfit.DoesNotExist, Exception):
            return api_error(
                "Outfit không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        request_serializer = OutfitSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        outfit = request_serializer.update(outfit, request_serializer.validated_data)
        response_serializer = OutfitSerializer(outfit)
        return api_success(
            "Outfit updated successfully",
            {
                "outfit": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete outfit."""
        try:
            outfit = Outfit.objects.get(id=ObjectId(pk))
            outfit.delete()
            return api_success(
                "Outfit deleted successfully",
                data=None,
            )
        except (Outfit.DoesNotExist, Exception):
            return api_error(
                "Outfit không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )


class RecommendationRequestViewSet(viewsets.ViewSet):
    """ViewSet cho RecommendationRequest."""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def list(self, request):
        """List recommendation requests."""
        queryset = RecommendationRequest.objects.all().order_by("-created_at")
        
        # Filter by user if not staff
        if not request.user.is_staff:
            queryset = queryset.filter(user_id=request.user.id)
        
        # Pagination
        page, page_size = get_pagination_params(request)
        requests, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = RecommendationRequestSerializer(requests, many=True)
        return api_success(
            "Recommendation requests retrieved successfully",
            {
                "requests": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get recommendation request by id."""
        try:
            request_obj = RecommendationRequest.objects.get(id=ObjectId(pk))
        except (RecommendationRequest.DoesNotExist, Exception):
            return api_error(
                "RecommendationRequest không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff and str(request_obj.user_id) != str(request.user.id):
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        serializer = RecommendationRequestSerializer(request_obj)
        return api_success(
            "Recommendation request retrieved successfully",
            {
                "request": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new recommendation request."""
        request_serializer = RecommendationRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        
        validated_data = request_serializer.validated_data.copy()
        validated_data["user_id"] = str(request.user.id)
        
        request_obj = request_serializer.create(validated_data)
        RecommendationService.enqueue_recommendation(request_obj)
        response_serializer = RecommendationRequestSerializer(request_obj)
        return api_success(
            "Recommendation request created successfully",
            {
                "request": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update recommendation request."""
        try:
            request_obj = RecommendationRequest.objects.get(id=ObjectId(pk))
        except (RecommendationRequest.DoesNotExist, Exception):
            return api_error(
                "RecommendationRequest không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff and str(request_obj.user_id) != str(request.user.id):
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        request_serializer = RecommendationRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        request_obj = request_serializer.update(request_obj, request_serializer.validated_data)
        response_serializer = RecommendationRequestSerializer(request_obj)
        return api_success(
            "Recommendation request updated successfully",
            {
                "request": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete recommendation request."""
        try:
            request_obj = RecommendationRequest.objects.get(id=ObjectId(pk))
        except (RecommendationRequest.DoesNotExist, Exception):
            return api_error(
                "RecommendationRequest không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff and str(request_obj.user_id) != str(request.user.id):
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        request_obj.delete()
        return api_success(
            "Recommendation request deleted successfully",
            data=None,
        )
    
    @action(detail=True, methods=["post"], permission_classes=[permissions.IsAuthenticated])
    def refresh(self, request, pk=None):
        """Refresh recommendation request."""
        try:
            request_obj = RecommendationRequest.objects.get(id=ObjectId(pk))
        except (RecommendationRequest.DoesNotExist, Exception):
            return api_error(
                "RecommendationRequest không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff and str(request_obj.user_id) != str(request.user.id):
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        RecommendationService.enqueue_recommendation(request_obj)
        response_serializer = RecommendationRequestSerializer(request_obj)
        return api_success(
            "Yêu cầu đã được đưa vào hàng xử lý.",
            {
                "request": response_serializer.data,
            },
            status_code=status.HTTP_202_ACCEPTED,
        )


class RecommendationResultViewSet(viewsets.ViewSet):
    """ViewSet cho RecommendationResult (read-only)."""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def list(self, request):
        """List recommendation results."""
        queryset = RecommendationResult.objects.all().order_by("-created_at")
        
        # Filter by user if not staff
        if not request.user.is_staff:
            # Get user's request IDs
            user_requests = RecommendationRequest.objects.filter(user_id=request.user.id)
            request_ids = [r.id for r in user_requests]
            queryset = queryset.filter(request_id__in=request_ids)
        
        # Pagination
        page, page_size = get_pagination_params(request)
        results, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = RecommendationResultSerializer(results, many=True)
        return api_success(
            "Recommendation results retrieved successfully",
            {
                "results": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get recommendation result by id."""
        try:
            result = RecommendationResult.objects.get(id=ObjectId(pk))
        except (RecommendationResult.DoesNotExist, Exception):
            return api_error(
                "RecommendationResult không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff:
            try:
                request_obj = RecommendationRequest.objects.get(id=result.request_id)
                if str(request_obj.user_id) != str(request.user.id):
                    return api_error(
                        "Không có quyền truy cập.",
                        data=None,
                        status_code=status.HTTP_403_FORBIDDEN,
                    )
            except RecommendationRequest.DoesNotExist:
                return api_error(
                    "RecommendationRequest không tồn tại.",
                    data=None,
                    status_code=status.HTTP_404_NOT_FOUND,
                )
        
        serializer = RecommendationResultSerializer(result)
        return api_success(
            "Recommendation result retrieved successfully",
            {
                "result": serializer.data,
            },
        )

