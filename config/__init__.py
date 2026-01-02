from .mongodb import connect_mongodb
try:
    connect_mongodb()
except Exception:
    import warnings
    warnings.warn("Cannot connect to MongoDB. Some features may not work.")

__all__ = ()

