"""Brand model using mongoengine for MongoDB."""

from __future__ import annotations

from datetime import datetime

import mongoengine as me
from mongoengine import fields


class Brand(me.Document):
    """Brand model."""
    
    meta = {
        "collection": "brands",
        "indexes": ["name"],
        "strict": False,
    }
    
    name = fields.StringField(required=True, unique=True, max_length=255)
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save to automatically update updated_at."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def __str__(self) -> str:
        return self.name

