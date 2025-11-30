from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .models import OutfitHistory, PasswordResetAudit, User, UserInteraction

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    fieldsets = BaseUserAdmin.fieldsets + (
        (
            "Extended Information",
            {
                "fields": (
                    "height",
                    "weight",
                    "gender",
                    "age",
                    "preferences",
                    "user_embedding",
                    "content_profile",
                )
            },
        ),
        (
            "Password Reset",
            {
                "fields": (
                    "reset_password_token",
                    "reset_password_expire",
                    "unhashed_reset_password_token",
                )
            },
        ),
    )
    list_display = ("email", "username", "is_staff", "gender", "age", "last_login")
    ordering = ("email",)
    search_fields = ("email", "username")

@admin.register(UserInteraction)
class UserInteractionAdmin(admin.ModelAdmin):
    list_display = ("user", "product", "interaction_type", "rating", "timestamp")
    search_fields = ("user__email", "product__name")
    list_filter = ("interaction_type", "timestamp")

@admin.register(OutfitHistory)
class OutfitHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "outfit_id", "interaction_type", "timestamp")
    search_fields = ("user__email", "outfit_id")

@admin.register(PasswordResetAudit)
class PasswordResetAuditAdmin(admin.ModelAdmin):
    list_display = ("user", "requested_at", "ip_address", "user_agent")
    search_fields = ("user__email", "ip_address", "user_agent")

