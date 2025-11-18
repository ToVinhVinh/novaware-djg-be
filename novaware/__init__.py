"""Main configuration package for Novaware Django project."""

from .celery import app as celery_app
from .mongodb import connect_mongodb

# Initialize MongoDB connection when Django starts
try:
    connect_mongodb()
except Exception:
    # If MongoDB connection fails, just log warning
    import warnings
    warnings.warn("Cannot connect to MongoDB. Some features may not work.")

__all__ = ("celery_app",)

