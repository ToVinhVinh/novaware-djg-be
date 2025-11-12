"""Script migrate data từ Django models sang MongoEngine models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from decimal import Decimal

# Thêm thư mục gốc vào Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import django
from django.conf import settings

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "novaware.settings")
django.setup()

from bson import ObjectId
from django.utils.text import slugify

from apps.brands.mongo_models import Brand as MongoBrand
from apps.brands.models import Brand as DjangoBrand
from apps.orders.mongo_models import Order as MongoOrder, OrderItem, ShippingAddress
from apps.orders.models import Order as DjangoOrder
from apps.products.mongo_models import (
    Category as MongoCategory,
    Color as MongoColor,
    Product as MongoProduct,
    ProductReview as MongoProductReview,
    ProductVariant as MongoProductVariant,
    Size as MongoSize,
)
from apps.products.models import (
    Category as DjangoCategory,
    Color as DjangoColor,
    Product as DjangoProduct,
    ProductReview as DjangoProductReview,
    ProductVariant as DjangoProductVariant,
    Size as DjangoSize,
)
from apps.users.mongo_models import User as MongoUser
from apps.users.models import User as DjangoUser


class DataMigrator:
    """Class để migrate data từ Django models sang MongoEngine."""
    
    def __init__(self):
        self.stats = {
            "users": {"migrated": 0, "skipped": 0, "errors": 0},
            "categories": {"migrated": 0, "skipped": 0, "errors": 0},
            "brands": {"migrated": 0, "skipped": 0, "errors": 0},
            "colors": {"migrated": 0, "skipped": 0, "errors": 0},
            "sizes": {"migrated": 0, "skipped": 0, "errors": 0},
            "products": {"migrated": 0, "skipped": 0, "errors": 0},
            "product_variants": {"migrated": 0, "skipped": 0, "errors": 0},
            "product_reviews": {"migrated": 0, "skipped": 0, "errors": 0},
            "orders": {"migrated": 0, "skipped": 0, "errors": 0},
        }
        self.id_mapping = {
            "users": {},
            "categories": {},
            "brands": {},
            "colors": {},
            "sizes": {},
            "products": {},
        }
    
    def print_section(self, title):
        """In tiêu đề section."""
        print("\n" + "=" * 60)
        print(f"🔄 {title}")
        print("=" * 60)
    
    def migrate_users(self, dry_run=False):
        """Migrate users."""
        self.print_section("MIGRATE USERS")
        
        django_users = DjangoUser.objects.all()
        total = django_users.count()
        print(f"Tổng số users: {total}")
        
        for django_user in django_users:
            try:
                # Kiểm tra xem đã tồn tại chưa
                existing = MongoUser.objects(email=django_user.email).first()
                if existing:
                    self.stats["users"]["skipped"] += 1
                    self.id_mapping["users"][str(django_user.id)] = str(existing.id)
                    continue
                
                if not dry_run:
                    mongo_user = MongoUser(
                        name=django_user.get_full_name() or django_user.username or django_user.email,
                        email=django_user.email,
                        username=django_user.username,
                        first_name=django_user.first_name,
                        last_name=django_user.last_name,
                        is_admin=django_user.is_staff or django_user.is_superuser,
                        is_active=django_user.is_active,
                        height=getattr(django_user, "height", None),
                        weight=getattr(django_user, "weight", None),
                        gender=getattr(django_user, "gender", None),
                        age=getattr(django_user, "age", None),
                        created_at=django_user.date_joined,
                        updated_at=django_user.date_joined,
                    )
                    
                    # Copy password hash nếu có
                    if django_user.password:
                        mongo_user.password = django_user.password
                    
                    mongo_user.save()
                    self.id_mapping["users"][str(django_user.id)] = str(mongo_user.id)
                    self.stats["users"]["migrated"] += 1
                else:
                    self.stats["users"]["migrated"] += 1
                
                if self.stats["users"]["migrated"] % 100 == 0:
                    print(f"  Đã migrate {self.stats['users']['migrated']} users...")
                    
            except Exception as e:
                print(f"  ❌ Lỗi migrate user {django_user.email}: {e}")
                self.stats["users"]["errors"] += 1
        
        print(f"✅ Migrated: {self.stats['users']['migrated']}")
        print(f"⏭️  Skipped: {self.stats['users']['skipped']}")
        print(f"❌ Errors: {self.stats['users']['errors']}")
    
    def migrate_categories(self, dry_run=False):
        """Migrate categories."""
        self.print_section("MIGRATE CATEGORIES")
        
        django_categories = DjangoCategory.objects.all()
        total = django_categories.count()
        print(f"Tổng số categories: {total}")
        
        for django_cat in django_categories:
            try:
                existing = MongoCategory.objects(name=django_cat.name).first()
                if existing:
                    self.stats["categories"]["skipped"] += 1
                    self.id_mapping["categories"][str(django_cat.id)] = str(existing.id)
                    continue
                
                if not dry_run:
                    mongo_cat = MongoCategory(
                        name=django_cat.name,
                        created_at=django_cat.created_at if hasattr(django_cat, "created_at") else None,
                        updated_at=django_cat.updated_at if hasattr(django_cat, "updated_at") else None,
                    )
                    mongo_cat.save()
                    self.id_mapping["categories"][str(django_cat.id)] = str(mongo_cat.id)
                    self.stats["categories"]["migrated"] += 1
                else:
                    self.stats["categories"]["migrated"] += 1
                    
            except Exception as e:
                print(f"  ❌ Lỗi migrate category {django_cat.name}: {e}")
                self.stats["categories"]["errors"] += 1
        
        print(f"✅ Migrated: {self.stats['categories']['migrated']}")
        print(f"⏭️  Skipped: {self.stats['categories']['skipped']}")
        print(f"❌ Errors: {self.stats['categories']['errors']}")
    
    def migrate_brands(self, dry_run=False):
        """Migrate brands."""
        self.print_section("MIGRATE BRANDS")
        
        django_brands = DjangoBrand.objects.all()
        total = django_brands.count()
        print(f"Tổng số brands: {total}")
        
        for django_brand in django_brands:
            try:
                existing = MongoBrand.objects(name=django_brand.name).first()
                if existing:
                    self.stats["brands"]["skipped"] += 1
                    self.id_mapping["brands"][str(django_brand.id)] = str(existing.id)
                    continue
                
                if not dry_run:
                    mongo_brand = MongoBrand(
                        name=django_brand.name,
                        created_at=django_brand.created_at if hasattr(django_brand, "created_at") else None,
                        updated_at=django_brand.updated_at if hasattr(django_brand, "updated_at") else None,
                    )
                    mongo_brand.save()
                    self.id_mapping["brands"][str(django_brand.id)] = str(mongo_brand.id)
                    self.stats["brands"]["migrated"] += 1
                else:
                    self.stats["brands"]["migrated"] += 1
                    
            except Exception as e:
                print(f"  ❌ Lỗi migrate brand {django_brand.name}: {e}")
                self.stats["brands"]["errors"] += 1
        
        print(f"✅ Migrated: {self.stats['brands']['migrated']}")
        print(f"⏭️  Skipped: {self.stats['brands']['skipped']}")
        print(f"❌ Errors: {self.stats['brands']['errors']}")
    
    def migrate_colors(self, dry_run=False):
        """Migrate colors."""
        self.print_section("MIGRATE COLORS")
        
        django_colors = DjangoColor.objects.all()
        total = django_colors.count()
        print(f"Tổng số colors: {total}")
        
        for django_color in django_colors:
            try:
                existing = MongoColor.objects(name=django_color.name).first()
                if existing:
                    self.stats["colors"]["skipped"] += 1
                    self.id_mapping["colors"][str(django_color.id)] = str(existing.id)
                    continue
                
                if not dry_run:
                    mongo_color = MongoColor(
                        name=django_color.name,
                        hex_code=getattr(django_color, "hex_code", None),
                        created_at=django_color.created_at if hasattr(django_color, "created_at") else None,
                        updated_at=django_color.updated_at if hasattr(django_color, "updated_at") else None,
                    )
                    mongo_color.save()
                    self.id_mapping["colors"][str(django_color.id)] = str(mongo_color.id)
                    self.stats["colors"]["migrated"] += 1
                else:
                    self.stats["colors"]["migrated"] += 1
                    
            except Exception as e:
                print(f"  ❌ Lỗi migrate color {django_color.name}: {e}")
                self.stats["colors"]["errors"] += 1
        
        print(f"✅ Migrated: {self.stats['colors']['migrated']}")
        print(f"⏭️  Skipped: {self.stats['colors']['skipped']}")
        print(f"❌ Errors: {self.stats['colors']['errors']}")
    
    def migrate_sizes(self, dry_run=False):
        """Migrate sizes."""
        self.print_section("MIGRATE SIZES")
        
        django_sizes = DjangoSize.objects.all()
        total = django_sizes.count()
        print(f"Tổng số sizes: {total}")
        
        for django_size in django_sizes:
            try:
                existing = MongoSize.objects(code=django_size.code).first()
                if existing:
                    self.stats["sizes"]["skipped"] += 1
                    self.id_mapping["sizes"][str(django_size.id)] = str(existing.id)
                    continue
                
                if not dry_run:
                    mongo_size = MongoSize(
                        code=django_size.code,
                        name=getattr(django_size, "name", None),
                        created_at=django_size.created_at if hasattr(django_size, "created_at") else None,
                        updated_at=django_size.updated_at if hasattr(django_size, "updated_at") else None,
                    )
                    mongo_size.save()
                    self.id_mapping["sizes"][str(django_size.id)] = str(mongo_size.id)
                    self.stats["sizes"]["migrated"] += 1
                else:
                    self.stats["sizes"]["migrated"] += 1
                    
            except Exception as e:
                print(f"  ❌ Lỗi migrate size {django_size.code}: {e}")
                self.stats["sizes"]["errors"] += 1
        
        print(f"✅ Migrated: {self.stats['sizes']['migrated']}")
        print(f"⏭️  Skipped: {self.stats['sizes']['skipped']}")
        print(f"❌ Errors: {self.stats['sizes']['errors']}")
    
    def migrate_products(self, dry_run=False):
        """Migrate products."""
        self.print_section("MIGRATE PRODUCTS")
        
        django_products = DjangoProduct.objects.all()
        total = django_products.count()
        print(f"Tổng số products: {total}")
        
        for django_product in django_products:
            try:
                existing = MongoProduct.objects(slug=django_product.slug).first()
                if existing:
                    self.stats["products"]["skipped"] += 1
                    self.id_mapping["products"][str(django_product.id)] = str(existing.id)
                    continue
                
                # Map IDs
                user_id = self.id_mapping["users"].get(str(django_product.user_id))
                brand_id = self.id_mapping["brands"].get(str(django_product.brand_id))
                category_id = self.id_mapping["categories"].get(str(django_product.category_id))
                
                if not user_id or not brand_id or not category_id:
                    self.stats["products"]["skipped"] += 1
                    continue
                
                if not dry_run:
                    mongo_product = MongoProduct(
                        user_id=ObjectId(user_id),
                        brand_id=ObjectId(brand_id),
                        category_id=ObjectId(category_id),
                        name=django_product.name,
                        slug=django_product.slug,
                        description=django_product.description or "",
                        images=list(django_product.images) if hasattr(django_product, "images") else [],
                        rating=float(django_product.rating) if hasattr(django_product, "rating") else 0.0,
                        num_reviews=int(django_product.num_reviews) if hasattr(django_product, "num_reviews") else 0,
                        price=Decimal(str(django_product.price)),
                        sale=Decimal(str(django_product.sale)) if hasattr(django_product, "sale") else Decimal("0.0"),
                        count_in_stock=int(django_product.count_in_stock) if hasattr(django_product, "count_in_stock") else 0,
                        amazon_asin=getattr(django_product, "amazon_asin", None),
                        amazon_parent_asin=getattr(django_product, "amazon_parent_asin", None),
                        created_at=django_product.created_at if hasattr(django_product, "created_at") else None,
                        updated_at=django_product.updated_at if hasattr(django_product, "updated_at") else None,
                    )
                    mongo_product.save()
                    self.id_mapping["products"][str(django_product.id)] = str(mongo_product.id)
                    self.stats["products"]["migrated"] += 1
                else:
                    self.stats["products"]["migrated"] += 1
                
                if self.stats["products"]["migrated"] % 100 == 0:
                    print(f"  Đã migrate {self.stats['products']['migrated']} products...")
                    
            except Exception as e:
                print(f"  ❌ Lỗi migrate product {django_product.name}: {e}")
                self.stats["products"]["errors"] += 1
        
        print(f"✅ Migrated: {self.stats['products']['migrated']}")
        print(f"⏭️  Skipped: {self.stats['products']['skipped']}")
        print(f"❌ Errors: {self.stats['products']['errors']}")
    
    def print_summary(self):
        """In tổng kết."""
        self.print_section("TỔNG KẾT MIGRATION")
        
        total_migrated = sum(s["migrated"] for s in self.stats.values())
        total_skipped = sum(s["skipped"] for s in self.stats.values())
        total_errors = sum(s["errors"] for s in self.stats.values())
        
        print(f"\n{'Model':<20} {'Migrated':<10} {'Skipped':<10} {'Errors':<10}")
        print("-" * 50)
        for model_name, stats in self.stats.items():
            print(f"{model_name:<20} {stats['migrated']:<10} {stats['skipped']:<10} {stats['errors']:<10}")
        
        print("-" * 50)
        print(f"{'TOTAL':<20} {total_migrated:<10} {total_skipped:<10} {total_errors:<10}")
    
    def run_migration(self, dry_run=True):
        """Chạy migration."""
        print("\n" + "=" * 60)
        print("🚀 BẮT ĐẦU MIGRATION DATA")
        print("=" * 60)
        
        if dry_run:
            print("⚠️  DRY RUN MODE - Không thực sự migrate data")
        
        # Migrate theo thứ tự dependency
        self.migrate_users(dry_run=dry_run)
        self.migrate_categories(dry_run=dry_run)
        self.migrate_brands(dry_run=dry_run)
        self.migrate_colors(dry_run=dry_run)
        self.migrate_sizes(dry_run=dry_run)
        self.migrate_products(dry_run=dry_run)
        
        self.print_summary()


def main():
    """Hàm chính."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate data từ Django models sang MongoEngine")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Thực sự migrate data (mặc định là dry-run)"
    )
    
    args = parser.parse_args()
    
    migrator = DataMigrator()
    migrator.run_migration(dry_run=not args.execute)


if __name__ == "__main__":
    main()

