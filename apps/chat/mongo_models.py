"""Models chat sử dụng mongoengine cho MongoDB."""

from __future__ import annotations

from datetime import datetime

import mongoengine as me
from mongoengine import fields


class Message(me.EmbeddedDocument):
    """Tin nhắn (embedded trong ChatThread)."""
    
    sender = fields.StringField(required=True, max_length=100)
    content = fields.StringField(required=True)
    timestamp = fields.DateTimeField(default=datetime.utcnow)
    
    def __str__(self) -> str:
        return f"[{self.timestamp}] {self.sender}: {self.content[:20]}"


class ChatThread(me.Document):
    """Model thread chat."""
    
    meta = {
        "collection": "chat_threads",
        "indexes": ["user_id", "updated_at"],
    }
    
    user_id = fields.ObjectIdField(required=True)
    messages = fields.ListField(fields.EmbeddedDocumentField(Message), default=list)
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save để tự động cập nhật updated_at."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def add_message(self, sender: str, content: str) -> None:
        """Thêm tin nhắn vào thread."""
        message = Message(sender=sender, content=content)
        self.messages.append(message)
        self.save()
    
    def __str__(self) -> str:
        return f"ChatThread #{self.id} - User {self.user_id}"

