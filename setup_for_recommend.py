"""
Script để setup dữ liệu và kiểm tra điều kiện sử dụng /api/v1/gnn/recommend/
Chạy: python setup_for_recommend.py
"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'novaware.settings')
django.setup()

from apps.recommendations.common.storage import ArtifactStorage
from apps.recommendations.common.exceptions import ModelNotTrainedError

print("=" * 70)
print("KIỂM TRA ĐIỀU KIỆN SỬ DỤNG /api/v1/gnn/recommend/")
print("=" * 70)

# Kiểm tra 1: Model đã được training chưa?
print("\n1. Kiểm tra Model Training:")
print("-" * 70)
storage = ArtifactStorage("gnn")
if storage.exists():
    try:
        artifacts = storage.load()
        trained_at = artifacts.get("trained_at", "Unknown")
        print(f"   ✅ Model đã được training")
        print(f"   📅 Trained at: {trained_at}")
        print(f"   📁 Artifacts: {storage.file_path}")
    except Exception as e:
        print(f"   ❌ Lỗi khi load artifacts: {e}")
else:
    print(f"   ❌ Model chưa được training!")
    print(f"   💡 Cần chạy: POST /api/v1/gnn/train")

# Kiểm tra 2: Có interactions trong database không?
print("\n2. Kiểm tra Interactions trong Database:")
print("-" * 70)
try:
    from apps.users.mongo_models import UserInteraction as MongoInteraction
    from apps.users.mongo_models import User as MongoUser
    from apps.products.mongo_models import Product as MongoProduct
    from bson import ObjectId
    from datetime import datetime, timedelta
    import random
    
    try:
        mongo_interactions = MongoInteraction.objects.all()
        mongo_users = MongoUser.objects.all()
        mongo_products = MongoProduct.objects.all()
        
        interactions_count = mongo_interactions.count()
        users_count = mongo_users.count()
        products_count = mongo_products.count()
        
        print(f"   ✅ Users: {users_count}")
        print(f"   ✅ Products: {products_count}")
        print(f"   ✅ Interactions: {interactions_count}")
        
        if interactions_count == 0:
            print(f"\n   ⚠️  KHÔNG CÓ INTERACTIONS!")
            print(f"   💡 Đang tạo interactions mẫu...")
            
            users_list = list(mongo_users[:10])  # Lấy 10 users đầu
            products_list = list(mongo_products[:20])  # Lấy 20 products đầu
            
            if len(users_list) == 0 or len(products_list) == 0:
                print(f"   ❌ Không có users hoặc products để tạo interactions")
            else:
                created = 0
                for user in users_list:
                    # Mỗi user tương tác với 3-5 products
                    num_interactions = random.randint(3, min(5, len(products_list)))
                    selected_products = random.sample(products_list, num_interactions)
                    
                    for product in selected_products:
                        try:
                            # Lấy _id từ MongoDB trực tiếp
                            from mongoengine import get_db
                            db = get_db()
                            doc = db.products.find_one({'id': product.id})
                            
                            if doc and '_id' in doc:
                                product_oid = doc['_id']
                                
                                # Tạo interaction
                                interaction_type = random.choice(["view", "like", "cart"])
                                days_ago = random.randint(0, 30)
                                interaction_time = datetime.utcnow() - timedelta(days=days_ago)
                                
                                MongoInteraction(
                                    user_id=user.id,
                                    product_id=product_oid,
                                    interaction_type=interaction_type,
                                    timestamp=interaction_time
                                ).save()
                                created += 1
                        except Exception as e:
                            # Bỏ qua lỗi
                            pass
                
                print(f"   ✅ Đã tạo {created} interactions mẫu")
                print(f"   💡 Bây giờ bạn có thể training model")
        else:
            print(f"   ✅ Đã có đủ interactions để training")
            
    except Exception as e:
        print(f"   ⚠️  Lỗi khi kiểm tra MongoDB: {e}")
        print(f"   (Có thể MongoDB chưa được cấu hình)")
        
except Exception as e:
    print(f"   ❌ Lỗi: {e}")

# Tổng kết
print("\n" + "=" * 70)
print("TỔNG KẾT:")
print("=" * 70)

can_use_recommend = True
issues = []

if not storage.exists():
    can_use_recommend = False
    issues.append("❌ Model chưa được training - cần chạy POST /api/v1/gnn/train")

try:
    interactions_count = MongoInteraction.objects.all().count()
    if interactions_count == 0:
        can_use_recommend = False
        issues.append("❌ Không có interactions - cần có dữ liệu interactions để training")
except:
    pass

if can_use_recommend:
    print("✅ BẠN CÓ THỂ SỬ DỤNG /api/v1/gnn/recommend/ NGAY BÂY GIỜ!")
    print("\n📝 Ví dụ request:")
    print("   POST /api/v1/gnn/recommend/")
    print("   {")
    print('     "user_id": "USER_ID_HERE",')
    print('     "current_product_id": "PRODUCT_ID_HERE",')
    print('     "top_k_personal": 10,')
    print('     "top_k_outfit": 5')
    print("   }")
else:
    print("⚠️  CHƯA THỂ SỬ DỤNG /api/v1/gnn/recommend/")
    print("\nCần thực hiện:")
    for issue in issues:
        print(f"   {issue}")
    print("\n📋 Các bước tiếp theo:")
    print("   1. Đảm bảo có interactions trong database")
    print("   2. Chạy training: POST /api/v1/gnn/train với {force_retrain: true}")
    print("   3. Đợi training hoàn thành")
    print("   4. Sau đó có thể sử dụng /api/v1/gnn/recommend/")

print("\n" + "=" * 70)

