"""MongoDB connection configuration with mongoengine."""

from __future__ import annotations

import os

import mongoengine
from django.conf import settings


def connect_mongodb() -> None:
    """Initialize MongoDB connection with mongoengine."""
    mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017/novaware")
    db_name = getattr(settings, "MONGODB_DB_NAME", "novaware")
    
    try:
        mongoengine.connect(
            db=db_name,
            host=mongo_uri,
            alias="default",
        )
        print(f"Connected to MongoDB: {db_name}")
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        raise

