"""ViewSets cho module chat sử dụng MongoEngine."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset

from .mongo_models import ChatThread, Message
from .mongo_serializers import ChatThreadSerializer, MessageSerializer


class ChatThreadViewSet(viewsets.ViewSet):
    """ViewSet cho ChatThread."""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def list(self, request):
        """List chat threads."""
        queryset = ChatThread.objects.all().order_by("-updated_at")
        
        # Filter by user if not staff
        if not request.user.is_staff:
            queryset = queryset.filter(user_id=request.user.id)
        
        # Pagination
        page, page_size = get_pagination_params(request)
        threads, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = ChatThreadSerializer(threads, many=True)
        return api_success(
            "Chat threads retrieved successfully",
            {
                "threads": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get chat thread by id."""
        try:
            thread = ChatThread.objects.get(id=ObjectId(pk))
        except (ChatThread.DoesNotExist, Exception):
            return api_error(
                "ChatThread không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff and str(thread.user_id) != str(request.user.id):
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        serializer = ChatThreadSerializer(thread)
        return api_success(
            "Chat thread retrieved successfully",
            {
                "thread": serializer.data,
            },
        )
    
    def create(self, request):
        """Create new chat thread."""
        request_serializer = ChatThreadSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        
        validated_data = request_serializer.validated_data.copy()
        validated_data["user_id"] = str(request.user.id)
        
        thread = request_serializer.create(validated_data)
        response_serializer = ChatThreadSerializer(thread)
        return api_success(
            "Chat thread created successfully",
            {
                "thread": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )
    
    def update(self, request, pk=None):
        """Update chat thread."""
        try:
            thread = ChatThread.objects.get(id=ObjectId(pk))
        except (ChatThread.DoesNotExist, Exception):
            return api_error(
                "ChatThread không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff and str(thread.user_id) != str(request.user.id):
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        request_serializer = ChatThreadSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        thread = request_serializer.update(thread, request_serializer.validated_data)
        response_serializer = ChatThreadSerializer(thread)
        return api_success(
            "Chat thread updated successfully",
            {
                "thread": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete chat thread."""
        try:
            thread = ChatThread.objects.get(id=ObjectId(pk))
        except (ChatThread.DoesNotExist, Exception):
            return api_error(
                "ChatThread không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff and str(thread.user_id) != str(request.user.id):
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        thread.delete()
        return api_success(
            "Chat thread deleted successfully",
            data=None,
        )
    
    @action(detail=True, methods=["post"])
    def add_message(self, request, pk=None):
        """Add message to chat thread."""
        try:
            thread = ChatThread.objects.get(id=ObjectId(pk))
        except (ChatThread.DoesNotExist, Exception):
            return api_error(
                "ChatThread không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        if not request.user.is_staff and str(thread.user_id) != str(request.user.id):
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        serializer = MessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Add message using the method in ChatThread
        thread.add_message(
            sender=serializer.validated_data["sender"],
            content=serializer.validated_data["content"],
        )
        
        return api_success(
            "Tin nhắn đã được thêm.",
            {
                "thread": ChatThreadSerializer(thread).data,
            },
            status_code=status.HTTP_201_CREATED,
        )

