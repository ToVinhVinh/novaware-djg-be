from __future__ import annotations

from django.conf import settings
from django.db import models

class Outfit(models.Model):
    STYLE_CHOICES = (
        ("casual", "Casual"),
        ("formal", "Formal"),
        ("sport", "Sport"),
    )

    name = models.CharField(max_length=255)
    products = models.ManyToManyField("products.Product", related_name="outfits")
    style = models.CharField(max_length=50, choices=STYLE_CHOICES, blank=True)
    season = models.CharField(max_length=50, blank=True)
    total_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    compatibility_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "outfits"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

class RecommendationRequest(models.Model):
    ALGORITHM_CHOICES = (
        ("cf", "GNN (LightGCN) - Collaborative Filtering"),
        ("cb", "Content Based"),
        ("gnn", "Graph Neural Network"),
        ("hybrid", "Hybrid GNN+CBF"),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="recommendation_requests")
    algorithm = models.CharField(max_length=20, choices=ALGORITHM_CHOICES)
    parameters = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "recommendation_requests"
        ordering = ["-created_at"]

class RecommendationResult(models.Model):
    request = models.OneToOneField(RecommendationRequest, on_delete=models.CASCADE, related_name="result")
    products = models.ManyToManyField("products.Product", related_name="recommendation_results")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "recommendation_results"

class RecommendationLog(models.Model):
    request = models.ForeignKey(RecommendationRequest, on_delete=models.CASCADE, related_name="logs")
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "recommendation_logs"
        ordering = ["created_at"]

