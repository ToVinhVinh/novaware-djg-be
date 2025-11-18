"""Admin registration for recommendation system."""

from django.contrib import admin

from .models import Outfit, RecommendationLog, RecommendationRequest, RecommendationResult


@admin.register(Outfit)
class OutfitAdmin(admin.ModelAdmin):
    list_display = ("name", "style", "season", "compatibility_score", "created_at")
    search_fields = ("name", "style", "season")
    list_filter = ("style", "season")


class RecommendationLogInline(admin.TabularInline):
    model = RecommendationLog
    extra = 0


@admin.register(RecommendationRequest)
class RecommendationRequestAdmin(admin.ModelAdmin):
    list_display = ("user", "algorithm", "created_at")
    list_filter = ("algorithm", "created_at")
    search_fields = ("user__email",)
    inlines = [RecommendationLogInline]


@admin.register(RecommendationResult)
class RecommendationResultAdmin(admin.ModelAdmin):
    list_display = ("request", "created_at")

