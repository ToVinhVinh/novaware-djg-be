"""ViewSets và endpoints người dùng sử dụng MongoEngine."""

from __future__ import annotations

from bson import ObjectId
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action

from apps.utils import api_error, api_success, get_pagination_params, paginate_queryset

from apps.orders.mongo_models import Order
from apps.products.mongo_models import Product
from apps.products.mongo_serializers import ProductSerializer

from .mongo_models import OutfitHistory, PasswordResetAudit, User, UserInteraction
from .mongo_serializers import (
    GenderSummarySerializer,
    OutfitHistorySerializer,
    PasswordChangeSerializer,
    PurchaseHistorySummarySerializer,
    StylePreferenceSummarySerializer,
    UserDetailSerializer,
    UserForTestingSerializer,
    UserInteractionSerializer,
    UserSerializer,
)


class IsAdminOrSelf(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if not request.user or not hasattr(request.user, 'id') or not request.user.is_authenticated:
            return False
        return request.user.is_staff or str(obj.id) == str(request.user.id)


class UserViewSet(viewsets.ViewSet):
    """ViewSet cho User."""
    
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    
    def list(self, request):
        """List users."""
        users = User.objects.all()
        
        # Search
        search = request.query_params.get("search")
        if search:
            users = users.filter(
                __raw__={"$or": [
                    {"email": {"$regex": search, "$options": "i"}},
                    {"username": {"$regex": search, "$options": "i"}},
                ]}
            )
        
        # Pagination
        page, page_size = get_pagination_params(request)
        user_list, total_count, total_pages, current_page, page_size = paginate_queryset(
            users, page, page_size
        )
        
        serializer = UserSerializer(user_list, many=True)
        return api_success(
            "Users retrieved successfully",
            {
                "users": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    
    def retrieve(self, request, pk=None):
        """Get user by id."""
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        serializer = UserDetailSerializer(user)
        return api_success(
            "User retrieved successfully",
            {
                "user": serializer.data,
            },
        )
    
    def update(self, request, pk=None):
        """Update user."""
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Check permission
        user_id = str(getattr(request.user, 'id', None) or '')
        if not (hasattr(request.user, 'is_staff') and request.user.is_staff) and str(user.id) != user_id:
            return api_error(
                "Không có quyền truy cập.",
                data=None,
                status_code=status.HTTP_403_FORBIDDEN,
            )
        
        request_serializer = UserSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        user = request_serializer.update(user, request_serializer.validated_data)
        response_serializer = UserDetailSerializer(user)
        return api_success(
            "User updated successfully",
            {
                "user": response_serializer.data,
            },
        )
    
    def destroy(self, request, pk=None):
        """Delete user."""
        try:
            user = User.objects.get(id=ObjectId(pk))
            user.delete()
            return api_success(
                "User deleted successfully",
                data=None,
            )
        except (User.DoesNotExist, Exception):
            return api_error(
                "User không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
    
    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def me(self, request):
        """Get current user."""
        if not request.user or not hasattr(request.user, 'id') or not request.user.is_authenticated:
            return api_error(
                "Yêu cầu đăng nhập.",
                data=None,
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        serializer = UserDetailSerializer(request.user)
        return api_success(
            "Current user retrieved successfully",
            {
                "user": serializer.data,
            },
        )
    
    @action(detail=True, methods=["get", "post", "delete"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def favorites(self, request, pk=None):
        """Manage user favorites."""
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        if request.method == "GET":
            page, page_size = get_pagination_params(request)
            favorite_products = Product.objects(id__in=user.favorites)
            paginated, total_count, total_pages, current_page, page_size = paginate_queryset(
                favorite_products, page, page_size
            )
            serializer = ProductSerializer(paginated, many=True)
            return api_success(
                "Favorites retrieved successfully",
                {
                    "favorites": serializer.data,
                    "page": current_page,
                    "pages": total_pages,
                    "perPage": page_size,
                    "count": total_count,
                },
            )
        
        product_id = request.data.get("product") or request.query_params.get("product")
        if not product_id:
            return api_error(
                "Product ID là bắt buộc.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        try:
            product = Product.objects.get(id=ObjectId(product_id))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        product_id_obj = product.id
        
        if request.method == "POST":
            if product_id_obj not in user.favorites:
                user.favorites.append(product_id_obj)
                user.save()
            return api_success(
                "Đã thêm sản phẩm vào yêu thích.",
                {
                    "product": ProductSerializer(product).data,
                    "favoritesCount": len(user.favorites),
                },
                status_code=status.HTTP_201_CREATED,
            )
        else:  # DELETE
            if product_id_obj in user.favorites:
                user.favorites.remove(product_id_obj)
                user.save()
            return api_success(
                "Đã xóa sản phẩm khỏi yêu thích.",
                {
                    "product": ProductSerializer(product).data,
                    "favoritesCount": len(user.favorites),
                },
            )
    
    @action(detail=True, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def check_purchase_history(self, request, pk=None):
        """Check if user has purchase history."""
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        count = Order.objects.filter(user_id=user.id, is_paid=True).count()
        serializer = PurchaseHistorySummarySerializer({
            "has_purchase_history": count > 0,
            "order_count": count,
        })
        return api_success(
            "Purchase history checked successfully",
            {
                "summary": serializer.data,
            },
        )
    
    @action(detail=True, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def check_gender(self, request, pk=None):
        """Check if user has gender."""
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        serializer = GenderSummarySerializer({
            "has_gender": bool(user.gender),
            "gender": user.gender,
        })
        return api_success(
            "Gender checked successfully",
            {
                "summary": serializer.data,
            },
        )
    
    @action(detail=True, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def check_style_preference(self, request, pk=None):
        """Check if user has style preference."""
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User không tồn tại.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        style = user.preferences.get("style") if user.preferences else None
        serializer = StylePreferenceSummarySerializer({
            "has_style_preference": bool(style),
            "style": style,
        })
        return api_success(
            "Style preference checked successfully",
            {
                "summary": serializer.data,
            },
        )
    
    @action(detail=False, methods=["post"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def change_password(self, request):
        """Change user password."""
        if not request.user or not hasattr(request.user, 'id') or not request.user.is_authenticated:
            return api_error(
                "Yêu cầu đăng nhập.",
                data=None,
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        serializer = PasswordChangeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = request.user
        
        if not user.check_password(serializer.validated_data["old_password"]):
            return api_error(
                "Mật khẩu cũ không chính xác.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        user.set_password(serializer.validated_data["new_password"])
        user.save()
        return api_success(
            "Đổi mật khẩu thành công.",
            data=None,
        )
    
    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def testing(self, request):
        """Testing endpoint for user data."""
        query_type = request.query_params.get("type")
        if query_type not in {"personalization", "outfit-suggestions"}:
            return api_error(
                "Tham số type không hợp lệ.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        users = User.objects.all()
        page, page_size = get_pagination_params(request)
        user_list, total_count, total_pages, current_page, page_size = paginate_queryset(
            users, page, page_size
        )
        results = []
        for user in user_list:
            order_count = Order.objects.filter(user_id=user.id, is_paid=True).count()
            user_data = {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "age": user.age,
                "gender": user.gender,
                "preferences": user.preferences,
                "order_count": order_count,
            }
            results.append(user_data)
        
        return api_success(
            "Testing data retrieved successfully",
            {
                "type": query_type,
                "users": results,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )


class UserInteractionViewSet(viewsets.ViewSet):
    """ViewSet cho UserInteraction."""
    
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    
    def list(self, request):
        """List user interactions."""
        queryset = UserInteraction.objects.all().order_by("-timestamp")
        
        # Pagination
        page, page_size = get_pagination_params(request)
        interactions, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = UserInteractionSerializer(interactions, many=True)
        return api_success(
            "User interactions retrieved successfully",
            {
                "interactions": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    
    def create(self, request):
        """Create new interaction."""
        request_serializer = UserInteractionSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        
        validated_data = request_serializer.validated_data.copy()
        # Get user_id from request data or use authenticated user if available
        if not validated_data.get("user_id"):
            if request.user and hasattr(request.user, 'id') and request.user.is_authenticated:
                validated_data["user_id"] = str(request.user.id)
            else:
                return api_error(
                    "user_id là bắt buộc khi không đăng nhập.",
                    data=None,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
        
        interaction = request_serializer.create(validated_data)
        response_serializer = UserInteractionSerializer(interaction)
        return api_success(
            "User interaction created successfully",
            {
                "interaction": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )


class OutfitHistoryViewSet(viewsets.ViewSet):
    """ViewSet cho OutfitHistory."""
    
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    
    def list(self, request):
        """List outfit history."""
        queryset = OutfitHistory.objects.all().order_by("-timestamp")
        
        # Pagination
        page, page_size = get_pagination_params(request)
        histories, total_count, total_pages, current_page, page_size = paginate_queryset(
            queryset, page, page_size
        )
        serializer = OutfitHistorySerializer(histories, many=True)
        return api_success(
            "Outfit history retrieved successfully",
            {
                "histories": serializer.data,
                "page": current_page,
                "pages": total_pages,
                "perPage": page_size,
                "count": total_count,
            },
        )
    
    def create(self, request):
        """Create new outfit history."""
        request_serializer = OutfitHistorySerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        
        validated_data = request_serializer.validated_data.copy()
        # Get user_id from request data or use authenticated user if available
        if not validated_data.get("user_id"):
            if request.user and hasattr(request.user, 'id') and request.user.is_authenticated:
                validated_data["user_id"] = str(request.user.id)
            else:
                return api_error(
                    "user_id là bắt buộc khi không đăng nhập.",
                    data=None,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
        
        history = request_serializer.create(validated_data)
        response_serializer = OutfitHistorySerializer(history)
        return api_success(
            "Outfit history created successfully",
            {
                "history": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )

