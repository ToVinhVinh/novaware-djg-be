"""Script monitor performance và tối ưu queries."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Thêm thư mục gốc vào Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import django
from django.conf import settings

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "novaware.settings")
django.setup()

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

from apps.brands.mongo_models import Brand
from apps.orders.mongo_models import Order
from apps.products.mongo_models import Category, Product, ProductReview
from apps.users.mongo_models import User


@contextmanager
def timer():
    """Context manager để đo thời gian thực thi."""
    start = time.time()
    yield
    end = time.time()
    print(f"  ⏱️  Thời gian: {(end - start) * 1000:.2f}ms")


class PerformanceMonitor:
    """Class để monitor performance."""
    
    def __init__(self):
        mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017/novaware")
        db_name = getattr(settings, "MONGODB_DB_NAME", "novaware")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
    
    def print_section(self, title):
        """In tiêu đề section."""
        print("\n" + "=" * 60)
        print(f"📊 {title}")
        print("=" * 60)
    
    def check_collection_stats(self):
        """Kiểm tra stats của collections."""
        self.print_section("COLLECTION STATISTICS")
        
        collections = [
            ("users", User),
            ("products", Product),
            ("categories", Category),
            ("brands", Brand),
            ("orders", Order),
            ("product_reviews", ProductReview),
        ]
        
        for collection_name, model_class in collections:
            try:
                collection = self.db[collection_name]
                stats = self.db.command("collStats", collection_name)
                
                print(f"\n📦 {collection_name}:")
                print(f"   Documents: {stats.get('count', 0):,}")
                print(f"   Size: {stats.get('size', 0) / 1024 / 1024:.2f} MB")
                print(f"   Storage Size: {stats.get('storageSize', 0) / 1024 / 1024:.2f} MB")
                print(f"   Indexes: {stats.get('nindexes', 0)}")
                print(f"   Index Size: {stats.get('totalIndexSize', 0) / 1024 / 1024:.2f} MB")
                
                # Check average document size
                if stats.get('count', 0) > 0:
                    avg_size = stats.get('size', 0) / stats.get('count', 1)
                    print(f"   Avg Doc Size: {avg_size:.2f} bytes")
                
            except Exception as e:
                print(f"   ❌ Lỗi: {e}")
    
    def test_query_performance(self):
        """Test performance của các queries phổ biến."""
        self.print_section("QUERY PERFORMANCE TESTS")
        
        # Test 1: List products
        print("\n1️⃣  List Products (first 20):")
        with timer():
            products = list(Product.objects.all()[:20])
        print(f"   Kết quả: {len(products)} products")
        
        # Test 2: Get product by slug
        print("\n2️⃣  Get Product by Slug:")
        product = Product.objects.first()
        if product:
            with timer():
                found = Product.objects(slug=product.slug).first()
            print(f"   Kết quả: {'Found' if found else 'Not found'}")
        
        # Test 3: Search products
        print("\n3️⃣  Search Products (regex):")
        with timer():
            products = list(Product.objects(name__icontains="test")[:10])
        print(f"   Kết quả: {len(products)} products")
        
        # Test 4: Get products by category
        print("\n4️⃣  Get Products by Category:")
        category = Category.objects.first()
        if category:
            with timer():
                products = list(Product.objects(category_id=category.id)[:20])
            print(f"   Kết quả: {len(products)} products")
        
        # Test 5: Get user with favorites
        print("\n5️⃣  Get User with Favorites:")
        user = User.objects.first()
        if user:
            with timer():
                user = User.objects(id=user.id).first()
                favorites_count = len(user.favorites) if user.favorites else 0
            print(f"   Kết quả: User có {favorites_count} favorites")
        
        # Test 6: Get orders for user
        print("\n6️⃣  Get Orders for User:")
        user = User.objects.first()
        if user:
            with timer():
                orders = list(Order.objects(user_id=user.id)[:10])
            print(f"   Kết quả: {len(orders)} orders")
        
        # Test 7: Aggregate products by category
        print("\n7️⃣  Count Products by Category:")
        with timer():
            pipeline = [
                {"$group": {"_id": "$category_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            results = list(Product._get_collection().aggregate(pipeline))
        print(f"   Kết quả: {len(results)} categories")
    
    def check_slow_queries(self):
        """Kiểm tra slow queries trong MongoDB."""
        self.print_section("SLOW QUERIES CHECK")
        
        try:
            # Enable profiling
            self.db.set_profiling_level(1, slow_ms=100)  # Log queries > 100ms
            
            # Get slow queries
            profile_collection = self.db["system.profile"]
            slow_queries = list(profile_collection.find().sort("ts", -1).limit(10))
            
            if slow_queries:
                print(f"\n⚠️  Tìm thấy {len(slow_queries)} slow queries:")
                for query in slow_queries:
                    duration = query.get("millis", 0)
                    op = query.get("op", "unknown")
                    ns = query.get("ns", "unknown")
                    print(f"\n   Collection: {ns}")
                    print(f"   Operation: {op}")
                    print(f"   Duration: {duration}ms")
                    if "command" in query:
                        print(f"   Command: {query['command']}")
            else:
                print("\n✅ Không có slow queries")
            
            # Disable profiling
            self.db.set_profiling_level(0)
            
        except Exception as e:
            print(f"❌ Lỗi kiểm tra slow queries: {e}")
    
    def check_index_usage(self):
        """Kiểm tra index usage."""
        self.print_section("INDEX USAGE")
        
        collections = [
            ("users", User),
            ("products", Product),
            ("categories", Category),
            ("orders", Order),
        ]
        
        for collection_name, model_class in collections:
            try:
                collection = self.db[collection_name]
                
                # Lấy index information
                indexes = collection.index_information()
                
                print(f"\n📦 {collection_name}:")
                print(f"   Có {len(indexes)} indexes:")
                for index_name, index_info in indexes.items():
                    keys = index_info.get("key", [])
                    key_str = ", ".join([f"{k[0]}({k[1]})" for k in keys])
                    unique = "UNIQUE" if index_info.get("unique") else ""
                    print(f"   - {index_name}: {key_str} {unique}")
                
                # Thử lấy index stats (có thể không có trong một số MongoDB versions)
                try:
                    stats = collection.index_stats()
                    if stats:
                        print(f"   Index Usage Stats:")
                        for index_stat in stats:
                            index_name = index_stat.get("name", "unknown")
                            accesses = index_stat.get("accesses", {})
                            ops = accesses.get("ops", 0)
                            since = accesses.get("since", "unknown")
                            
                            if ops > 0:
                                print(f"     ✅ {index_name}: {ops} operations (since {since})")
                            else:
                                print(f"     ⚠️  {index_name}: Chưa được sử dụng")
                except Exception:
                    # Index stats không available trong MongoDB version này
                    pass
                        
            except Exception as e:
                print(f"   ❌ Lỗi: {e}")
    
    def suggest_optimizations(self):
        """Đề xuất tối ưu."""
        self.print_section("OPTIMIZATION SUGGESTIONS")
        
        suggestions = []
        
        # Check for missing indexes on frequently queried fields
        print("\n💡 Đề xuất:")
        
        # Check Product queries
        product_collection = Product._get_collection()
        indexes = product_collection.index_information()
        
        # Check if we have compound indexes for common queries
        has_user_category_index = any(
            "user_id" in str(idx) and "category_id" in str(idx)
            for idx in indexes.keys()
        )
        if not has_user_category_index:
            suggestions.append(
                "Tạo compound index cho Product: (user_id, category_id) "
                "nếu thường query products theo user và category"
            )
        
        # Check for text search index
        has_text_index = any(
            "text" in str(idx_info.get("key", []))
            for idx_info in indexes.values()
        )
        if not has_text_index:
            suggestions.append(
                "Tạo text index cho Product.name và Product.description "
                "để tối ưu search"
            )
        
        # Check Order queries
        order_collection = Order._get_collection()
        order_indexes = order_collection.index_information()
        
        has_user_status_index = any(
            "user_id" in str(idx) and "status" in str(idx)
            for idx in order_indexes.keys()
        )
        if not has_user_status_index:
            suggestions.append(
                "Tạo compound index cho Order: (user_id, status) "
                "nếu thường query orders theo user và status"
            )
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        else:
            print("   ✅ Không có đề xuất tối ưu nào")
    
    def check_n_plus_one_queries(self):
        """Kiểm tra N+1 query problems."""
        self.print_section("N+1 QUERY CHECK")
        
        # Example: Get products with their categories
        print("\n🔍 Test: Get products with categories")
        
        products = list(Product.objects.all()[:10])
        
        # Bad: N+1 queries
        print("\n❌ Bad (N+1 queries):")
        start = time.time()
        for product in products:
            category = Category.objects(id=product.category_id).first()
        bad_time = (time.time() - start) * 1000
        
        # Better: Batch query
        print("\n✅ Better (Batch query):")
        start = time.time()
        category_ids = [p.category_id for p in products]
        categories = {str(cat.id): cat for cat in Category.objects(id__in=category_ids)}
        for product in products:
            category = categories.get(str(product.category_id))
        good_time = (time.time() - start) * 1000
        
        print(f"   Bad approach: {bad_time:.2f}ms")
        print(f"   Better approach: {good_time:.2f}ms")
        print(f"   Improvement: {((bad_time - good_time) / bad_time * 100):.1f}%")
    
    def run_all_checks(self):
        """Chạy tất cả checks."""
        print("\n" + "=" * 60)
        print("🚀 BẮT ĐẦU MONITOR PERFORMANCE")
        print("=" * 60)
        
        self.check_collection_stats()
        self.test_query_performance()
        self.check_index_usage()
        self.check_slow_queries()
        self.check_n_plus_one_queries()
        self.suggest_optimizations()
        
        print("\n" + "=" * 60)
        print("✅ HOÀN TẤT MONITOR PERFORMANCE")
        print("=" * 60)
        
        self.client.close()


def main():
    """Hàm chính."""
    monitor = PerformanceMonitor()
    monitor.run_all_checks()


if __name__ == "__main__":
    main()

