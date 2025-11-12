"""Model thương hiệu được migrate từ Mongoose."""

from __future__ import annotations

from django.db import models


class Brand(models.Model):
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "brands"
        ordering = ["name"]
        verbose_name = "Thương hiệu"
        verbose_name_plural = "Thương hiệu"

    def __str__(self) -> str:
        return self.name

