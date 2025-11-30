from __future__ import annotations

from datetime import datetime

import mongoengine as me
from mongoengine import fields

class Outfit(me.Document):

    STYLE_CHOICES = ["casual", "formal", "sport"]

    meta = {
        "collection": "outfits",
        "indexes": ["created_at", "style", "season"],
    }

    name = fields.StringField(required=True, max_length=255)
    product_ids = fields.ListField(fields.ObjectIdField(), default=list)
    style = fields.StringField(choices=STYLE_CHOICES)
    season = fields.StringField(max_length=50)
    total_price = fields.DecimalField(null=True, precision=10, decimal_places=2)
    compatibility_score = fields.FloatField(null=True)
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)

    def __str__(self) -> str:
        return self.name

class RecommendationRequest(me.Document):

    ALGORITHM_CHOICES = ["cf", "cb", "gnn", "hybrid"]

    meta = {
        "collection": "recommendation_requests",
        "indexes": ["user_id", "created_at", "algorithm"],
    }

    user_id = fields.ObjectIdField(required=True)
    algorithm = fields.StringField(required=True, choices=ALGORITHM_CHOICES, max_length=20)
    parameters = fields.DictField(default=dict)
    created_at = fields.DateTimeField(default=datetime.utcnow)

    def __str__(self) -> str:
        return f"RecommendationRequest {self.id}"

class RecommendationResult(me.Document):

    meta = {
        "collection": "recommendation_results",
        "indexes": ["request_id", "created_at"],
    }

    request_id = fields.ObjectIdField(required=True, unique=True)
    product_ids = fields.ListField(fields.ObjectIdField(), default=list)
    metadata = fields.DictField(default=dict)
    created_at = fields.DateTimeField(default=datetime.utcnow)

    def __str__(self) -> str:
        return f"RecommendationResult for request {self.request_id}"

class RecommendationLog(me.Document):

    meta = {
        "collection": "recommendation_logs",
        "indexes": ["request_id", "created_at"],
    }

    request_id = fields.ObjectIdField(required=True)
    message = fields.StringField(required=True)
    created_at = fields.DateTimeField(default=datetime.utcnow)

    def __str__(self) -> str:
        return f"Log for request {self.request_id}"

