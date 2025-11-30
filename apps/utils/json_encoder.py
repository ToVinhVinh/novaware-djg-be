from __future__ import annotations

from bson import ObjectId
from rest_framework.utils.encoders import JSONEncoder

class MongoJSONEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

_original_default = JSONEncoder.default

def _patched_default(self, obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    return _original_default(self, obj)

JSONEncoder.default = _patched_default

