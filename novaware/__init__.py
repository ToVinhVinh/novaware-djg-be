"""Main configuration package for Novaware Django project."""

from .celery import app as celery_app
from .mongodb import connect_mongodb
try:
    connect_mongodb()
except Exception:
    import warnings
    warnings.warn("Cannot connect to MongoDB. Some features may not work.")

__all__ = ("celery_app",)

