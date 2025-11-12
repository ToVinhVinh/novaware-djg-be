"""Script fix các products có slug = null."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import django
from django.conf import settings
from django.utils.text import slugify

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "novaware.settings")
django.setup()

from pymongo import MongoClient


def fix_product_slugs(dry_run=True):
    """Fix các products có slug = null hoặc empty."""
    print("=" * 60)
    print("🔧 FIX PRODUCT SLUGS")
    print("=" * 60)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - Không thực sự update")
    
    # Kết nối trực tiếp với pymongo để tránh vấn đề với indexes
    mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017/novaware")
    db_name = getattr(settings, "MONGODB_DB_NAME", "novaware")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db["products"]
    
    # Tìm products có slug = null hoặc empty
    products_without_slug = list(collection.find({
        "$or": [
            {"slug": None},
            {"slug": ""},
            {"slug": {"$exists": False}},
        ]
    }))
    
    count = len(products_without_slug)
    print(f"\n📊 Tìm thấy {count} products không có slug")
    
    if count == 0:
        print("✅ Không có products nào cần fix")
        client.close()
        return
    
    fixed = 0
    errors = 0
    
    # Lấy tất cả slugs hiện có để tránh trùng
    existing_slugs = set(doc.get("slug") for doc in collection.find({"slug": {"$ne": None, "$ne": ""}}, {"slug": 1}))
    
    for product_doc in products_without_slug:
        try:
            product_id = product_doc["_id"]
            product_name = product_doc.get("name", "")
            
            if product_name:
                new_slug = slugify(product_name)
                
                # Đảm bảo slug không trùng
                base_slug = new_slug
                counter = 1
                while new_slug in existing_slugs:
                    new_slug = f"{base_slug}-{counter}"
                    counter += 1
                
                old_slug = product_doc.get("slug", "None")
                print(f"   Product: {product_name[:50]}")
                print(f"   Old slug: {old_slug}")
                print(f"   New slug: {new_slug}")
                
                if not dry_run:
                    collection.update_one(
                        {"_id": product_id},
                        {"$set": {"slug": new_slug}}
                    )
                    existing_slugs.add(new_slug)
                    print(f"   ✅ Đã update")
                else:
                    print(f"   ⏭️  Sẽ update (dry-run)")
                
                fixed += 1
                
                if fixed % 100 == 0:
                    print(f"\n   Đã fix {fixed} products...")
            else:
                print(f"   ⚠️  Product {product_id} không có name, bỏ qua")
                errors += 1
                
        except Exception as e:
            print(f"   ❌ Lỗi fix product {product_doc.get('_id')}: {e}")
            errors += 1
    
    client.close()
    
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)
    print(f"✅ Fixed: {fixed}")
    print(f"❌ Errors: {errors}")
    print(f"📦 Total: {count}")
    
    if dry_run:
        print("\n⚠️  Đây là dry-run. Chạy với --execute để thực sự fix.")


def main():
    """Hàm chính."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix products có slug = null")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Thực sự fix (mặc định là dry-run)"
    )
    
    args = parser.parse_args()
    
    fix_product_slugs(dry_run=not args.execute)


if __name__ == "__main__":
    main()

