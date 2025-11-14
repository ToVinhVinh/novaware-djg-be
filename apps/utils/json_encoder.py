"""Custom JSON encoder for handling MongoDB ObjectId in DRF responses."""

from __future__ import annotations

from bson import ObjectId
from rest_framework.utils.encoders import JSONEncoder


class MongoJSONEncoder(JSONEncoder):
    """Custom JSON encoder that handles MongoDB ObjectId."""
    
    def default(self, obj):
        """Convert ObjectId to string for JSON serialization."""
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

_original_default = JSONEncoder.default


def _patched_default(self, obj):
    """Patched default method that handles ObjectId."""
    if isinstance(obj, ObjectId):
        return str(obj)
    return _original_default(self, obj)


# Apply the patch
JSONEncoder.default = _patched_default

