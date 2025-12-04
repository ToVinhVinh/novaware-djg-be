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

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        users = User.objects.all()

        keyword = request.query_params.get("keyword")
        if keyword:
            users = users.filter(
                __raw__={"$or": [
                    {"email": {"$regex": keyword, "$options": "i"}},
                    {"name": {"$regex": keyword, "$options": "i"}},
                    {"username": {"$regex": keyword, "$options": "i"}},
                ]}
            )

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
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
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
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
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
        try:
            user = User.objects.get(id=ObjectId(pk))
            user.delete()
            return api_success(
                "User deleted successfully",
                data=None,
            )
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def me(self, request):
        if not request.user or not hasattr(request.user, 'id') or not request.user.is_authenticated:
            return api_error(
                "Login required.",
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
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
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
                "Product ID is required.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        try:
            product = Product.objects.get(id=ObjectId(product_id))
        except (Product.DoesNotExist, Exception):
            return api_error(
                "Product does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        product_id_obj = product.id

        if request.method == "POST":
            if product_id_obj not in user.favorites:
                user.favorites.append(product_id_obj)
                user.save()
            return api_success(
                "Product added to favorites.",
                {
                    "product": ProductSerializer(product).data,
                    "favoritesCount": len(user.favorites),
                },
                status_code=status.HTTP_201_CREATED,
            )
        else:
            if product_id_obj in user.favorites:
                user.favorites.remove(product_id_obj)
                user.save()
            return api_success(
                "Product removed from favorites.",
                {
                    "product": ProductSerializer(product).data,
                    "favoritesCount": len(user.favorites),
                },
            )

    @action(detail=True, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def check_purchase_history(self, request, pk=None):
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
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
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
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
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
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

    @action(detail=False, methods=["post", "put", "patch"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def profile(self, request):
        """
        Update user profile by email.
        Expected payload: {
            "email": "user@example.com",
            "name": "User Name",
            "gender": "male",
            "height": 170,
            "weight": 55,
            "age": 21,
            ...
        }
        """
        email = request.data.get("email")
        if not email:
            return api_error(
                "Email is required.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return api_error(
                "User does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            return api_error(
                f"Error finding user: {str(e)}",
                data=None,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
        request_serializer = UserSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        
        validated_data = request_serializer.validated_data.copy()
        
        if "age" in validated_data:
            age_value = validated_data.get("age")
            if age_value is not None and (age_value <= 10 or age_value > 100):
                validated_data.pop("age")
        
        user = request_serializer.update(user, validated_data)
        
        response_serializer = UserDetailSerializer(user)
        return api_success(
            "User profile updated successfully",
            {
                "user": response_serializer.data,
            },
        )

    @action(detail=False, methods=["post"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def change_password(self, request):
        if not request.user or not hasattr(request.user, 'id') or not request.user.is_authenticated:
            return api_error(
                "Login required.",
                data=None,
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        serializer = PasswordChangeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = request.user

        if not user.check_password(serializer.validated_data["old_password"]):
            return api_error(
                "Old password is incorrect.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        user.set_password(serializer.validated_data["new_password"])
        user.save()
        return api_success(
            "Password changed successfully.",
            data=None,
        )

    @action(detail=False, methods=["get"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def testing(self, request):
        query_type = request.query_params.get("type")
        if query_type not in {"personalization", "outfit-suggestions"}:
            return api_error(
                "Invalid type parameter.",
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

    @action(detail=True, methods=["post"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def add_interaction(self, request, pk=None):
        """
        Add an interaction to user's interaction_history.
        Expected payload: {
            "product_id": int or str,
            "interaction_type": "view" | "like" | "cart" | "purchase",
            "timestamp": "ISO format datetime string" (optional, defaults to now)
        }
        """
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        product_id = request.data.get("product_id")
        interaction_type = request.data.get("interaction_type")
        timestamp_str = request.data.get("timestamp")

        if not product_id:
            return api_error(
                "product_id is required.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if not interaction_type:
            return api_error(
                "interaction_type is required.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if interaction_type not in ["view", "like", "cart", "purchase", "review"]:
            return api_error(
                "interaction_type must be one of: view, like, cart, purchase, review",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Parse timestamp or use current time
        from datetime import datetime
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except Exception:
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()

        # Create interaction entry
        interaction_entry = {
            "product_id": int(product_id) if str(product_id).isdigit() else product_id,
            "interaction_type": interaction_type,
            "timestamp": timestamp.isoformat()
        }

        # Initialize interaction_history if it doesn't exist
        if not user.interaction_history:
            user.interaction_history = []

        # Add the new interaction to the history
        user.interaction_history.append(interaction_entry)
        user.save()

        return api_success(
            "Interaction added to user history successfully",
            {
                "user_id": str(user.id),
                "interaction": interaction_entry,
                "total_interactions": len(user.interaction_history),
            },
            status_code=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=["put", "patch"], permission_classes=[permissions.AllowAny], authentication_classes=[])
    def update_interaction(self, request, pk=None):
        """
        Update interaction_type for a specific product in user's interaction_history.
        If interaction doesn't exist, it will be created.
        Expected payload: {
            "product_id": int or str,
            "interaction_type": "view" | "like" | "cart" | "purchase" | "review",
            "timestamp": "ISO format datetime string" (optional, defaults to now)
        }
        """
        try:
            user = User.objects.get(id=ObjectId(pk))
        except (User.DoesNotExist, Exception):
            return api_error(
                "User does not exist.",
                data=None,
                status_code=status.HTTP_404_NOT_FOUND,
            )

        product_id = request.data.get("product_id")
        interaction_type = request.data.get("interaction_type")
        timestamp_str = request.data.get("timestamp")

        if not product_id:
            return api_error(
                "product_id is required.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if not interaction_type:
            return api_error(
                "interaction_type is required.",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if interaction_type not in ["view", "like", "cart", "purchase", "review"]:
            return api_error(
                "interaction_type must be one of: view, like, cart, purchase, review",
                data=None,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # Parse timestamp or use current time
        from datetime import datetime
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except Exception:
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()

        # Normalize product_id to int for comparison
        try:
            product_id_int = int(product_id)
        except (ValueError, TypeError):
            product_id_int = product_id

        # Initialize interaction_history if it doesn't exist
        if not user.interaction_history:
            user.interaction_history = []

        # Find existing interaction with this product_id
        interaction_found = False
        for i, interaction in enumerate(user.interaction_history):
            # Handle different formats of product_id in history
            hist_product_id = interaction.get("product_id") if isinstance(interaction, dict) else None
            if hist_product_id is not None:
                # Convert to int for comparison if possible
                try:
                    hist_product_id_int = int(hist_product_id)
                    if hist_product_id_int == product_id_int:
                        # Update existing interaction
                        user.interaction_history[i] = {
                            "product_id": product_id_int,
                            "interaction_type": interaction_type,
                            "timestamp": timestamp.isoformat()
                        }
                        interaction_found = True
                        break
                except (ValueError, TypeError):
                    # If can't convert to int, compare as string
                    if str(hist_product_id) == str(product_id):
                        user.interaction_history[i] = {
                            "product_id": product_id_int,
                            "interaction_type": interaction_type,
                            "timestamp": timestamp.isoformat()
                        }
                        interaction_found = True
                        break

        # If interaction not found, add new one
        if not interaction_found:
            interaction_entry = {
                "product_id": product_id_int,
                "interaction_type": interaction_type,
                "timestamp": timestamp.isoformat()
            }
            user.interaction_history.append(interaction_entry)

        user.save()

        return api_success(
            "Interaction updated successfully",
            {
                "user_id": str(user.id),
                "product_id": product_id_int,
                "interaction_type": interaction_type,
                "updated": interaction_found,
                "total_interactions": len(user.interaction_history),
            },
            status_code=status.HTTP_200_OK,
        )

class UserInteractionViewSet(viewsets.ViewSet):

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        queryset = UserInteraction.objects.all().order_by("-timestamp")

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
        request_serializer = UserInteractionSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)

        validated_data = request_serializer.validated_data.copy()
        if not validated_data.get("user_id"):
            if request.user and hasattr(request.user, 'id') and request.user.is_authenticated:
                validated_data["user_id"] = str(request.user.id)
            else:
                return api_error(
                    "user_id is required when not logged in.",
                    data=None,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

        interaction = request_serializer.create(validated_data)
        
        # Update user's interaction_history
        try:
            user_id = ObjectId(validated_data["user_id"])
            user = User.objects.get(id=user_id)
            
            # Create interaction entry for user's interaction_history
            # product_id should be an integer to match CSV format
            product_id_value = interaction.product_id
            if isinstance(product_id_value, (str, ObjectId)):
                try:
                    product_id_value = int(product_id_value)
                except (ValueError, TypeError):
                    product_id_value = str(product_id_value)
            elif not isinstance(product_id_value, int):
                product_id_value = int(product_id_value) if product_id_value else None
            
            interaction_entry = {
                "product_id": product_id_value,
                "interaction_type": interaction.interaction_type,
                "timestamp": interaction.timestamp.isoformat() if interaction.timestamp else None
            }
            
            # Initialize interaction_history if it doesn't exist
            if not user.interaction_history:
                user.interaction_history = []
            
            # Check if interaction with this product_id already exists
            interaction_found = False
            for i, hist_interaction in enumerate(user.interaction_history):
                if isinstance(hist_interaction, dict):
                    hist_product_id = hist_interaction.get("product_id")
                    if hist_product_id is not None:
                        try:
                            # Compare as integers
                            if int(hist_product_id) == int(product_id_value):
                                # Update existing interaction (replace with new type)
                                user.interaction_history[i] = interaction_entry
                                interaction_found = True
                                break
                        except (ValueError, TypeError):
                            # If can't convert to int, compare as string
                            if str(hist_product_id) == str(product_id_value):
                                user.interaction_history[i] = interaction_entry
                                interaction_found = True
                                break
            
            # If interaction not found, add new one
            if not interaction_found:
                user.interaction_history.append(interaction_entry)
            
            user.save()
        except Exception as e:
            # Log error but don't fail the interaction creation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to update user interaction_history: {str(e)}")
        
        response_serializer = UserInteractionSerializer(interaction)
        return api_success(
            "User interaction created successfully",
            {
                "interaction": response_serializer.data,
            },
            status_code=status.HTTP_201_CREATED,
        )

class OutfitHistoryViewSet(viewsets.ViewSet):

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def list(self, request):
        queryset = OutfitHistory.objects.all().order_by("-timestamp")

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
        request_serializer = OutfitHistorySerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)

        validated_data = request_serializer.validated_data.copy()
        if not validated_data.get("user_id"):
            if request.user and hasattr(request.user, 'id') and request.user.is_authenticated:
                validated_data["user_id"] = str(request.user.id)
            else:
                return api_error(
                    "user_id is required when not logged in.",
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

