"""Models using mongoengine for MongoDB."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import mongoengine as me
from mongoengine import fields


class User(me.Document):
    """User model using mongoengine."""
    
    meta = {
        "collection": "users",
        "indexes": ["email", "amazon_user_id"],
        "strict": False,
    }
    
    name = fields.StringField(required=True, db_field="name")
    email = fields.EmailField(required=True, unique=True, db_field="email")
    password = fields.StringField(required=False, db_field="password")
    username = fields.StringField(required=False, unique=True, sparse=True, db_field="username")
    first_name = fields.StringField(max_length=150, db_field="firstName")
    last_name = fields.StringField(max_length=150, db_field="lastName")
    is_admin = fields.BooleanField(default=False, required=True, db_field="isAdmin")
    is_active = fields.BooleanField(default=True, required=True, db_field="isActive")
    
    height = fields.FloatField(null=True, db_field="height")
    weight = fields.FloatField(null=True, db_field="weight")
    gender = fields.StringField(
        choices=["male", "female", "other"],
        null=True,
        db_field="gender",
    )
    age = fields.IntField(
        min_value=13,
        max_value=100,
        null=True,
        db_field="age",
    )
    
    # Reset password
    reset_password_token = fields.StringField(null=True, db_field="resetPasswordToken")
    reset_password_expire = fields.DateTimeField(null=True, db_field="resetPasswordExpire")
    unhashed_reset_password_token = fields.StringField(null=True, db_field="unhashedResetPasswordToken")
    
    # Favorites (will be list of ObjectId references to Product)
    favorites = fields.ListField(fields.ObjectIdField(), default=list, db_field="favorites")
    
    # Preferences
    preferences = fields.DictField(default=dict, db_field="preferences")
    
    # Recommendation system
    user_embedding = fields.ListField(fields.FloatField(), default=list, db_field="userEmbedding")
    content_profile = fields.DictField(default=dict, db_field="contentProfile")
    interaction_history = fields.ListField(fields.DynamicField(), default=list, db_field="interactionHistory")
    outfit_history = fields.ListField(fields.DynamicField(), default=list, db_field="outfitHistory")
    
    # Amazon identifier
    amazon_user_id = fields.StringField(null=True, db_field="amazonUserId")
    version = fields.IntField(null=True, db_field="__v")
    
    # Timestamps
    created_at = fields.DateTimeField(default=datetime.utcnow, db_field="createdAt")
    updated_at = fields.DateTimeField(default=datetime.utcnow, db_field="updatedAt")
    
    def save(self, *args, **kwargs):
        """Override save to automatically update updated_at."""
        self.updated_at = datetime.utcnow()
        if not self.username and self.email:
            base_username = self.email.split("@", 1)[0].replace(".", "_")
            counter = 1
            candidate = base_username
            # Use id__ne (not equal) for MongoEngine instead of exclude()
            if self.id:
                while User.objects(username=candidate, id__ne=self.id).first():
                    counter += 1
                    candidate = f"{base_username}_{counter}"
            else:
                while User.objects(username=candidate).first():
                    counter += 1
                    candidate = f"{base_username}_{counter}"
            self.username = candidate
        return super().save(*args, **kwargs)
    
    def check_password(self, raw_password: str) -> bool:
        """Check password (need to implement hash checking)."""
        # TODO: Implement bcrypt checking
        import bcrypt
        if not self.password:
            return False
        return bcrypt.checkpw(raw_password.encode("utf-8"), self.password.encode("utf-8"))
    
    def set_password(self, raw_password: str) -> None:
        """Hash and save password."""
        import bcrypt
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(raw_password.encode("utf-8"), salt)
        self.password = hashed.decode("utf-8")
    
    def set_reset_password_token(self, token: str, expiry_minutes: int = 10) -> None:
        """Set password reset token."""
        from datetime import timedelta
        self.reset_password_token = token
        self.reset_password_expire = datetime.utcnow() + timedelta(minutes=expiry_minutes)
        self.unhashed_reset_password_token = token
        self.save()
    
    def clear_reset_password_token(self) -> None:
        """Clear password reset token."""
        self.reset_password_token = None
        self.reset_password_expire = None
        self.unhashed_reset_password_token = None
        self.save()
    
    @property
    def is_staff(self) -> bool:
        """Compatible with Django admin."""
        return self.is_admin
    
    @property
    def is_superuser(self) -> bool:
        """Compatible with Django admin."""
        return self.is_admin


class UserInteraction(me.Document):
    """User interaction history."""
    
    meta = {
        "collection": "user_interactions",
        "indexes": ["user_id", "product_id", "timestamp"],
    }
    
    user_id = fields.ObjectIdField(required=True)
    product_id = fields.IntField(required=True)
    interaction_type = fields.StringField(
        choices=["view", "like", "purchase", "cart", "review"],
        required=True,
    )
    rating = fields.IntField(min_value=1, max_value=5, null=True)
    timestamp = fields.DateTimeField(default=datetime.utcnow)


class OutfitHistory(me.Document):
    """Outfit history."""
    
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
    """Audit log for password reset."""
    
    meta = {
        "collection": "password_reset_audit",
        "indexes": ["user_id", "requested_at"],
    }
    
    user_id = fields.ObjectIdField(required=True)
    requested_at = fields.DateTimeField(default=datetime.utcnow)
    ip_address = fields.StringField(null=True)
    user_agent = fields.StringField(default="")

