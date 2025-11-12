"""Đăng ký admin cho module chat."""

from django.contrib import admin

from .models import ChatThread, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0


@admin.register(ChatThread)
class ChatThreadAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
    search_fields = ("user__email",)
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("thread", "sender", "timestamp")
    search_fields = ("sender", "content")

