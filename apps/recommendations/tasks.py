"""Expose Celery tasks for autodiscovery."""

from __future__ import annotations

# Re-export shared tasks so Celery autodiscovery picks them up.
from apps.recommendations.cbf.models import train_cbf_model  # noqa: F401
from apps.recommendations.gnn.models import train_gnn_model  # noqa: F401
from apps.recommendations.hybrid.models import train_hybrid_model  # noqa: F401

