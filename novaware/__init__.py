"""Gói cấu hình chính cho dự án Novaware Django."""

from .celery import app as celery_app
from .mongodb import connect_mongodb

# Khởi tạo kết nối MongoDB khi Django khởi động
try:
    connect_mongodb()
except Exception:
    # Nếu không kết nối được MongoDB, chỉ log warning
    import warnings
    warnings.warn("Không thể kết nối MongoDB. Một số tính năng có thể không hoạt động.")

__all__ = ("celery_app",)

