"""Models sử dụng mongoengine cho MongoDB."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import mongoengine as me
from mongoengine import fields


class User(me.Document):
    """Model người dùng sử dụng mongoengine."""
    
    meta = {
        "collection": "users",
        "indexes": ["email", "amazon_user_id"],
    }
    
    # Thông tin cơ bản
    name = fields.StringField(required=True)
    email = fields.EmailField(required=True, unique=True)
    password = fields.StringField(required=False)
    username = fields.StringField(required=False, unique=True, sparse=True)
    first_name = fields.StringField(max_length=150)
    last_name = fields.StringField(max_length=150)
    is_admin = fields.BooleanField(default=False, required=True)
    is_active = fields.BooleanField(default=True, required=True)
    
    # Thông tin cá nhân
    height = fields.FloatField(null=True)
    weight = fields.FloatField(null=True)
    gender = fields.StringField(
        choices=["male", "female", "other"],
        null=True,
    )
    age = fields.IntField(
        min_value=13,
        max_value=100,
        null=True,
    )
    
    # Reset password
    reset_password_token = fields.StringField(null=True)
    reset_password_expire = fields.DateTimeField(null=True)
    unhashed_reset_password_token = fields.StringField(null=True)
    
    # Favorites (sẽ là list ObjectId reference đến Product)
    favorites = fields.ListField(fields.ObjectIdField(), default=list)
    
    # Preferences
    preferences = fields.DictField(default=dict)
    
    # Hệ thống gợi ý
    user_embedding = fields.ListField(fields.FloatField(), default=list)
    content_profile = fields.DictField(default=dict)
    
    # Amazon identifier
    amazon_user_id = fields.StringField(null=True)
    
    # Timestamps
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save để tự động cập nhật updated_at."""
        self.updated_at = datetime.utcnow()
        if not self.username and self.email:
            base_username = self.email.split("@", 1)[0].replace(".", "_")
            counter = 1
            candidate = base_username
            while User.objects(username=candidate).exclude(id=self.id).first():
                counter += 1
                candidate = f"{base_username}_{counter}"
            self.username = candidate
        return super().save(*args, **kwargs)
    
    def check_password(self, raw_password: str) -> bool:
        """Kiểm tra mật khẩu (cần implement hash checking)."""
        # TODO: Implement bcrypt checking
        import bcrypt
        if not self.password:
            return False
        return bcrypt.checkpw(raw_password.encode("utf-8"), self.password.encode("utf-8"))
    
    def set_password(self, raw_password: str) -> None:
        """Hash và lưu mật khẩu."""
        import bcrypt
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(raw_password.encode("utf-8"), salt)
        self.password = hashed.decode("utf-8")
    
    def set_reset_password_token(self, token: str, expiry_minutes: int = 10) -> None:
        """Thiết lập token reset mật khẩu."""
        from datetime import timedelta
        self.reset_password_token = token
        self.reset_password_expire = datetime.utcnow() + timedelta(minutes=expiry_minutes)
        self.unhashed_reset_password_token = token
        self.save()
    
    def clear_reset_password_token(self) -> None:
        """Xóa token reset mật khẩu."""
        self.reset_password_token = None
        self.reset_password_expire = None
        self.unhashed_reset_password_token = None
        self.save()
    
    @property
    def is_staff(self) -> bool:
        """Tương thích với Django admin."""
        return self.is_admin
    
    @property
    def is_superuser(self) -> bool:
        """Tương thích với Django admin."""
        return self.is_admin


class UserInteraction(me.Document):
    """Lịch sử tương tác người dùng."""
    
    meta = {
        "collection": "user_interactions",
        "indexes": ["user_id", "product_id", "timestamp"],
    }
    
    user_id = fields.ObjectIdField(required=True)
    product_id = fields.ObjectIdField(required=True)
    interaction_type = fields.StringField(
        choices=["view", "like", "purchase", "cart", "review"],
        required=True,
    )
    rating = fields.IntField(min_value=1, max_value=5, null=True)
    timestamp = fields.DateTimeField(default=datetime.utcnow)


class OutfitHistory(me.Document):
    """Lịch sử outfit."""
    
    meta = {
        "collection": "outfit_history",
        "indexes": ["user_id", "timestamp"],
    }
    
    user_id = fields.ObjectIdField(required=True)
    outfit_id = fields.StringField(required=True)
    product_ids = fields.ListField(fields.ObjectIdField(), default=list)
    interaction_type = fields.StringField(
        choices=["view", "like", "purchase"],
        required=True,
    )
    timestamp = fields.DateTimeField(default=datetime.utcnow)


class PasswordResetAudit(me.Document):
    """Audit log cho password reset."""
    
    meta = {
        "collection": "password_reset_audit",
        "indexes": ["user_id", "requested_at"],
    }
    
    user_id = fields.ObjectIdField(required=True)
    requested_at = fields.DateTimeField(default=datetime.utcnow)
    ip_address = fields.StringField(null=True)
    user_agent = fields.StringField(default="")

