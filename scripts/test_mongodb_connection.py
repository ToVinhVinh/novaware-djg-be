"""Script kiểm tra kết nối MongoDB và indexes."""

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

import mongoengine
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from apps.brands.mongo_models import Brand
from apps.chat.mongo_models import ChatThread
from apps.orders.mongo_models import Order
from apps.products.mongo_models import Category, Color, Product, ProductReview, ProductVariant, Size
from apps.recommendations.mongo_models import Outfit, RecommendationRequest, RecommendationResult
from apps.users.mongo_models import User, UserInteraction


def test_mongodb_connection():
    """Kiểm tra kết nối MongoDB."""
    print("=" * 60)
    print("🔍 KIỂM TRA KẾT NỐI MONGODB")
    print("=" * 60)
    
    mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017/novaware")
    db_name = getattr(settings, "MONGODB_DB_NAME", "novaware")
    
    print(f"\n📌 MongoDB URI: {mongo_uri}")
    print(f"📌 Database Name: {db_name}")
    
    try:
        # Test với pymongo
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Force connection
        print("✅ Kết nối MongoDB thành công (pymongo)")
        client.close()
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"❌ Lỗi kết nối MongoDB (pymongo): {e}")
        return False
    
    try:
        # Test với mongoengine
        mongoengine.connect(
            db=db_name,
            host=mongo_uri,
            alias="default",
        )
        print("✅ Kết nối MongoDB thành công (mongoengine)")
    except Exception as e:
        print(f"❌ Lỗi kết nối MongoDB (mongoengine): {e}")
        return False
    
    return True


def check_indexes():
    """Kiểm tra indexes của các collections."""
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA INDEXES")
    print("=" * 60)
    
    models = [
        ("User", User),
        ("Product", Product),
        ("Category", Category),
        ("Brand", Brand),
        ("Color", Color),
        ("Size", Size),
        ("Order", Order),
        ("ChatThread", ChatThread),
        ("Outfit", Outfit),
        ("RecommendationRequest", RecommendationRequest),
        ("RecommendationResult", RecommendationResult),
        ("ProductReview", ProductReview),
        ("ProductVariant", ProductVariant),
        ("UserInteraction", UserInteraction),
    ]
    
    all_ok = True
    
    for model_name, model_class in models:
        print(f"\n📦 {model_name}:")
        try:
            # Lấy collection name
            collection_name = model_class._get_collection_name()
            
            # Lấy indexes từ MongoDB
            indexes = model_class._get_collection().index_information()
            
            # Lấy indexes được định nghĩa trong meta
            defined_indexes = model_class._meta.get("indexes", [])
            
            print(f"   Collection: {collection_name}")
            print(f"   Đã định nghĩa {len(defined_indexes)} indexes trong meta")
            print(f"   Có {len(indexes)} indexes trong MongoDB")
            
            # Kiểm tra từng index được định nghĩa
            for idx_def in defined_indexes:
                if isinstance(idx_def, str):
                    # Simple index
                    idx_name = idx_def
                    if idx_name in indexes:
                        print(f"   ✅ Index '{idx_name}' đã được tạo")
                    else:
                        print(f"   ⚠️  Index '{idx_name}' chưa được tạo")
                        all_ok = False
                elif isinstance(idx_def, (list, tuple)):
                    # Compound index
                    idx_fields = idx_def
                    idx_name = "_".join([str(f) for f in idx_fields])
                    found = False
                    for existing_idx_name, existing_idx_info in indexes.items():
                        if existing_idx_info.get("key") == [(f, 1) for f in idx_fields]:
                            print(f"   ✅ Compound index {idx_fields} đã được tạo ({existing_idx_name})")
                            found = True
                            break
                    if not found:
                        print(f"   ⚠️  Compound index {idx_fields} chưa được tạo")
                        all_ok = False
                elif isinstance(idx_def, dict):
                    # Index với options
                    idx_fields = idx_def.get("fields", [])
                    idx_name = idx_def.get("name") or "_".join([str(f) for f in idx_fields])
                    found = False
                    for existing_idx_name, existing_idx_info in indexes.items():
                        if existing_idx_info.get("key") == [(f, 1) for f in idx_fields]:
                            print(f"   ✅ Index {idx_fields} đã được tạo ({existing_idx_name})")
                            found = True
                            break
                    if not found:
                        print(f"   ⚠️  Index {idx_fields} chưa được tạo")
                        all_ok = False
            
            # Kiểm tra unique indexes
            for field_name, field in model_class._fields.items():
                if hasattr(field, "unique") and field.unique:
                    # Kiểm tra xem có unique index chưa
                    found_unique = False
                    for idx_name, idx_info in indexes.items():
                        if idx_name == f"{field_name}_1" or (idx_info.get("key") == [(field_name, 1)] and idx_info.get("unique")):
                            print(f"   ✅ Unique index cho '{field_name}' đã được tạo")
                            found_unique = True
                            break
                    if not found_unique:
                        print(f"   ⚠️  Unique index cho '{field_name}' chưa được tạo")
                        all_ok = False
            
        except Exception as e:
            print(f"   ❌ Lỗi khi kiểm tra indexes: {e}")
            all_ok = False
    
    return all_ok


def ensure_indexes():
    """Đảm bảo tất cả indexes được tạo."""
    print("\n" + "=" * 60)
    print("🔧 TẠO INDEXES")
    print("=" * 60)
    
    models = [
        ("User", User),
        ("Product", Product),
        ("Category", Category),
        ("Brand", Brand),
        ("Color", Color),
        ("Size", Size),
        ("Order", Order),
        ("ChatThread", ChatThread),
        ("Outfit", Outfit),
        ("RecommendationRequest", RecommendationRequest),
        ("RecommendationResult", RecommendationResult),
        ("ProductReview", ProductReview),
        ("ProductVariant", ProductVariant),
        ("UserInteraction", UserInteraction),
    ]
    
    all_ok = True
    
    for model_name, model_class in models:
        print(f"\n📦 {model_name}:")
        try:
            model_class.ensure_indexes()
            print(f"   ✅ Đã đảm bảo indexes được tạo")
        except Exception as e:
            print(f"   ❌ Lỗi khi tạo indexes: {e}")
            all_ok = False
    
    return all_ok


def check_data_counts():
    """Kiểm tra số lượng documents trong mỗi collection."""
    print("\n" + "=" * 60)
    print("📊 KIỂM TRA SỐ LƯỢNG DOCUMENTS")
    print("=" * 60)
    
    models = [
        ("User", User),
        ("Product", Product),
        ("Category", Category),
        ("Brand", Brand),
        ("Color", Color),
        ("Size", Size),
        ("Order", Order),
        ("ChatThread", ChatThread),
        ("Outfit", Outfit),
        ("RecommendationRequest", RecommendationRequest),
        ("RecommendationResult", RecommendationResult),
        ("ProductReview", ProductReview),
        ("ProductVariant", ProductVariant),
        ("UserInteraction", UserInteraction),
    ]
    
    for model_name, model_class in models:
        try:
            count = model_class.objects.count()
            print(f"   {model_name:30s}: {count:>8} documents")
        except Exception as e:
            print(f"   {model_name:30s}: ❌ Lỗi - {e}")


def main():
    """Hàm chính."""
    print("\n" + "=" * 60)
    print("🚀 KIỂM TRA MONGODB CONNECTION VÀ INDEXES")
    print("=" * 60)
    
    # Test connection
    if not test_mongodb_connection():
        print("\n❌ Không thể kết nối MongoDB. Vui lòng kiểm tra cấu hình.")
        sys.exit(1)
    
    # Check indexes
    indexes_ok = check_indexes()
    
    # Ensure indexes
    if not indexes_ok:
        print("\n⚠️  Một số indexes chưa được tạo. Đang tạo indexes...")
        ensure_indexes()
        print("\n🔍 Kiểm tra lại indexes...")
        check_indexes()
    
    # Check data counts
    check_data_counts()
    
    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT KIỂM TRA")
    print("=" * 60)


if __name__ == "__main__":
    main()

