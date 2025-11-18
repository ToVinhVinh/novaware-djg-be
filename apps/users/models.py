"""User models converted from Mongoose to Django ORM."""

from __future__ import annotations

from django.contrib.auth.models import AbstractUser
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

from .managers import UserManager


class User(AbstractUser):
    class Gender(models.TextChoices):
        MALE = "male", "Male"
        FEMALE = "female", "Female"
        OTHER = "other", "Other"

    username = models.CharField(
        max_length=150,
        unique=True,
        blank=True,
        help_text="Unique display name (auto-generated if not provided).",
    )
    email = models.EmailField("email address", unique=True)
    height = models.FloatField(null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=Gender.choices, null=True, blank=True)
    age = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(13), MaxValueValidator(100)],
    )
    reset_password_token = models.CharField(max_length=255, null=True, blank=True)
    reset_password_expire = models.DateTimeField(null=True, blank=True)
    unhashed_reset_password_token = models.CharField(max_length=255, null=True, blank=True)
    favorites = models.ManyToManyField(
        "products.Product",
        related_name="favorited_by",
        blank=True,
    )
    preferences = models.JSONField(blank=True, default=dict)
    user_embedding = models.JSONField(blank=True, default=list)
    content_profile = models.JSONField(blank=True, default=dict)
    amazon_user_id = models.CharField(max_length=255, blank=True, null=True, db_index=True)

    # Add related_name to avoid conflicts with default auth.User
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text=(
            'The groups this user belongs to. A user will get all permissions '
            'granted to each of their groups.'
        ),
        related_name="user_custom_set",  # Unique name
        related_query_name="user",
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name="user_custom_permissions_set",  # Unique name
        related_query_name="user",
    )

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    objects = UserManager()

    class Meta:
        db_table = "users"
        verbose_name = "User"
        verbose_name_plural = "Users"

    def save(self, *args, **kwargs):
        if not self.username:
            base_username = (self.email.split("@", 1)[0] if self.email else "user").replace(".", "_")
            counter = 1
            candidate = base_username
            while type(self).objects.filter(username=candidate).exclude(pk=self.pk).exists():
                counter += 1
                candidate = f"{base_username}_{counter}"
            self.username = candidate
        super().save(*args, **kwargs)

    @property
    def is_admin(self) -> bool:
        return self.is_staff

    def set_reset_password_token(self, token: str, expiry_minutes: int = 10) -> None:
        self.reset_password_token = token
        self.reset_password_expire = timezone.now() + timezone.timedelta(minutes=expiry_minutes)
        self.unhashed_reset_password_token = token
        self.save(update_fields=[
            "reset_password_token",
            "reset_password_expire",
            "unhashed_reset_password_token",
        ])

    def clear_reset_password_token(self) -> None:
        self.reset_password_token = None
        self.reset_password_expire = None
        self.unhashed_reset_password_token = None
        self.save(update_fields=[
            "reset_password_token",
            "reset_password_expire",
            "unhashed_reset_password_token",
        ])


class UserInteraction(models.Model):
    class InteractionType(models.TextChoices):
        VIEW = "view", "View"
        LIKE = "like", "Like"
        PURCHASE = "purchase", "Purchase"
        CART = "cart", "Cart"
        REVIEW = "review", "Review"

    user = models.ForeignKey(User, related_name="interactions", on_delete=models.CASCADE)
    product = models.ForeignKey(
        "products.Product",
        related_name="user_interactions",
        on_delete=models.CASCADE,
    )
    interaction_type = models.CharField(max_length=20, choices=InteractionType.choices)
    rating = models.PositiveSmallIntegerField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "user_interactions"
        verbose_name = "User Interaction"
        verbose_name_plural = "User Interactions"
        ordering = ["-timestamp"]


class OutfitHistory(models.Model):
    class InteractionType(models.TextChoices):
        VIEW = "view", "View"
        LIKE = "like", "Like"
        PURCHASE = "purchase", "Purchase"

    user = models.ForeignKey(User, related_name="outfit_history", on_delete=models.CASCADE)
    outfit_id = models.CharField(max_length=255)
    products = models.ManyToManyField("products.Product", related_name="outfit_histories", blank=True)
    interaction_type = models.CharField(max_length=20, choices=InteractionType.choices)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "outfit_history"
        ordering = ["-timestamp"]


class PasswordResetAudit(models.Model):
    user = models.ForeignKey(User, related_name="password_reset_audits", on_delete=models.CASCADE)
    requested_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=512, blank=True)

    class Meta:
        db_table = "password_reset_audit"
        ordering = ["-requested_at"]

