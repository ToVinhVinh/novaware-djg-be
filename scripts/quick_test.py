"""Script test nhanh MongoDB connection và một số queries cơ bản."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import django
from django.conf import settings

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "novaware.settings")
django.setup()

from apps.products.mongo_models import Product, Category
from apps.users.mongo_models import User


def quick_test():
    """Test nhanh các chức năng cơ bản."""
    print("=" * 60)
    print("🚀 QUICK TEST")
    print("=" * 60)
    
    # Test 1: Count documents
    print("\n1️⃣  Đếm documents:")
    try:
        user_count = User.objects.count()
        product_count = Product.objects.count()
        category_count = Category.objects.count()
        
        print(f"   ✅ Users: {user_count}")
        print(f"   ✅ Products: {product_count}")
        print(f"   ✅ Categories: {category_count}")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return False
    
    # Test 2: Query một document
    print("\n2️⃣  Query một document:")
    try:
        user = User.objects.first()
        if user:
            print(f"   ✅ Tìm thấy user: {user.email}")
        else:
            print("   ⚠️  Chưa có user nào")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return False
    
    # Test 3: Create và delete test document
    print("\n3️⃣  Test create/delete:")
    try:
        test_user = User(
            email="test_quick_test@example.com",
            name="Test User",
            is_active=True,
            is_admin=False,
        )
        test_user.set_password("test123")
        test_user.save()
        print(f"   ✅ Đã tạo test user: {test_user.email}")
        
        # Delete
        test_user.delete()
        print(f"   ✅ Đã xóa test user")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ QUICK TEST PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)

