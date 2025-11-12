"""Model lưu trữ hội thoại chat."""

from __future__ import annotations

from django.conf import settings
from django.db import models


class ChatThread(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="chat_threads", on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "chat_threads"
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"ChatThread #{self.id} - {self.user.email}"


class Message(models.Model):
    thread = models.ForeignKey(ChatThread, related_name="messages", on_delete=models.CASCADE)
    sender = models.CharField(max_length=100)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chat_messages"
        ordering = ["timestamp"]

    def __str__(self) -> str:
        return f"[{self.timestamp}] {self.sender}: {self.content[:20]}"

