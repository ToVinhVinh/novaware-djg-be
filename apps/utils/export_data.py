"""
Script để xuất dữ liệu từ MongoDB thành các file CSV
Sử dụng trong Streamlit app để export products, users, interactions
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup Django environment
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'novaware.settings')

import django
# Chỉ setup Django nếu chưa được setup
if not django.apps.apps.ready:
    django.setup()

# Import MongoDB models after Django setup
from novaware.mongodb import connect_mongodb
from apps.products.mongo_models import Product
from apps.users.mongo_models import User, UserInteraction

# Try to import ObjectId for better type checking
try:
    from bson import ObjectId
except ImportError:
    ObjectId = None


def ensure_export_directory():
    """Tạo thư mục exports nếu chưa có"""
    export_dir = BASE_DIR / 'apps' / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def export_products(export_dir: Path, mongodb_connected: bool = False) -> Dict:
    """
    Xuất dữ liệu products từ MongoDB thành CSV
    
    Fields: id, gender, masterCategory, subCategory, articleType, 
            baseColour, season, year, usage, productDisplayName, images
    """
    csv_path = export_dir / 'products.csv'
    
    # Kết nối MongoDB (chỉ nếu chưa kết nối)
    if not mongodb_connected:
        try:
            connect_mongodb()
        except Exception as e:
            return {'success': False, 'error': f'Lỗi kết nối MongoDB: {str(e)}', 'count': 0}
    
    try:
        # Query tất cả products
        products = Product.objects.all()
        
        # Chuẩn bị dữ liệu
        rows = []
        for product in products:
            # Xử lý images: chuyển list thành string (JSON hoặc comma-separated)
            images_str = json.dumps(product.images) if product.images else '[]'
            
            row = {
                'id': product.id or '',
                'gender': product.gender or '',
                'masterCategory': product.masterCategory or '',
                'subCategory': product.subCategory or '',
                'articleType': product.articleType or '',
                'baseColour': product.baseColour or '',
                'season': product.season or '',
                'year': product.year or '',
                'usage': product.usage or '',
                'productDisplayName': product.productDisplayName or '',
                'images': images_str
            }
            rows.append(row)
        
        # Ghi vào CSV
        if rows:
            fieldnames = ['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 
                         'baseColour', 'season', 'year', 'usage', 'productDisplayName', 'images']
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return {
            'success': True,
            'file_path': str(csv_path),
            'count': len(rows),
            'message': f'Đã xuất {len(rows)} sản phẩm thành công'
        }
    
    except Exception as e:
        return {'success': False, 'error': f'Lỗi khi xuất products: {str(e)}', 'count': 0}


def export_users(export_dir: Path, mongodb_connected: bool = False) -> Dict:
    """
    Xuất dữ liệu users từ MongoDB thành CSV
    
    Fields: id, name, email, age, gender, interaction_history
    """
    csv_path = export_dir / 'users.csv'
    
    # Kết nối MongoDB (chỉ nếu chưa kết nối)
    if not mongodb_connected:
        try:
            connect_mongodb()
        except Exception as e:
            return {'success': False, 'error': f'Lỗi kết nối MongoDB: {str(e)}', 'count': 0}
    
    try:
        users = User.objects.all()
        
        rows = []
        for user in users:
            if user.interaction_history:
                # Recursive function để clean nested structures (dict, list)
                def clean_for_json(obj):
                    """Recursively convert datetime, ObjectId và các object không serializable thành JSON-compatible types"""
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif ObjectId is not None and isinstance(obj, ObjectId):
                        return str(obj)
                    elif isinstance(obj, dict):
                        return {k: clean_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [clean_for_json(item) for item in obj]
                    elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool, type(None))):
                        # Convert các object khác (như Decimal, etc.) thành string
                        return str(obj)
                    else:
                        return obj
                
                try:
                    # Clean tất cả nested structures trước
                    cleaned_history = clean_for_json(user.interaction_history)
                    # Sau đó serialize
                    interaction_history_str = json.dumps(cleaned_history, ensure_ascii=False)
                except Exception as e:
                    # Last resort fallback: convert tất cả thành string representation
                    try:
                        interaction_history_clean = []
                        for item in user.interaction_history:
                            if isinstance(item, dict):
                                clean_item = {}
                                for k, v in item.items():
                                    if isinstance(v, datetime):
                                        clean_item[k] = v.isoformat()
                                    elif ObjectId is not None and isinstance(v, ObjectId):
                                        clean_item[k] = str(v)
                                    elif isinstance(v, dict):
                                        # Recursive cho nested dict
                                        clean_item[k] = {k2: (v2.isoformat() if isinstance(v2, datetime) else str(v2) if ObjectId is not None and isinstance(v2, ObjectId) else v2) for k2, v2 in v.items()}
                                    elif isinstance(v, (list, tuple)):
                                        # Recursive cho nested list
                                        clean_item[k] = [(i.isoformat() if isinstance(i, datetime) else str(i) if ObjectId is not None and isinstance(i, ObjectId) else i) for i in v]
                                    elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None))):
                                        clean_item[k] = str(v)
                                    else:
                                        clean_item[k] = v
                                interaction_history_clean.append(clean_item)
                            elif isinstance(item, datetime):
                                interaction_history_clean.append(item.isoformat())
                            elif ObjectId is not None and isinstance(item, ObjectId):
                                interaction_history_clean.append(str(item))
                            else:
                                interaction_history_clean.append(str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item)
                        interaction_history_str = json.dumps(interaction_history_clean, ensure_ascii=False)
                    except Exception as e2:
                        # Ultimate fallback: convert to string representation
                        interaction_history_str = json.dumps([str(item) for item in user.interaction_history], ensure_ascii=False)
            else:
                interaction_history_str = '[]'
            
            # Lấy user ID (có thể là ObjectId hoặc string)
            user_id = str(user.id) if user.id else ''
            
            row = {
                'id': user_id,
                'name': user.name or '',
                'email': user.email or '',
                'age': user.age or '',
                'gender': user.gender or '',
                'interaction_history': interaction_history_str
            }
            rows.append(row)
        
        # Ghi vào CSV
        if rows:
            fieldnames = ['id', 'name', 'email', 'age', 'gender', 'interaction_history']
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return {
            'success': True,
            'file_path': str(csv_path),
            'count': len(rows),
            'message': f'Đã xuất {len(rows)} users thành công'
        }
    
    except Exception as e:
        return {'success': False, 'error': f'Lỗi khi xuất users: {str(e)}', 'count': 0}


def export_interactions(export_dir: Path, mongodb_connected: bool = False) -> Dict:
    """
    Xuất dữ liệu interactions từ MongoDB thành CSV
    Chỉ lấy interactions của users có trong User model (map với users.csv)
    
    Fields: user_id, product_id, interaction_type, timestamp
    """
    csv_path = export_dir / 'interactions.csv'
    
    # Kết nối MongoDB (chỉ nếu chưa kết nối)
    if not mongodb_connected:
        try:
            connect_mongodb()
        except Exception as e:
            return {'success': False, 'error': f'Lỗi kết nối MongoDB: {str(e)}', 'count': 0}
    
    try:
        # Lấy danh sách tất cả user IDs từ User model (để filter interactions)
        from apps.users.mongo_models import User
        valid_user_ids = set()
        for user in User.objects.all():
            if user.id:
                valid_user_ids.add(str(user.id))
        
        print(f"📊 Tìm thấy {len(valid_user_ids)} users hợp lệ")
        
        # Query tất cả interactions
        all_interactions = UserInteraction.objects.all().order_by('timestamp')
        
        # Chuẩn bị dữ liệu - chỉ lấy interactions của users hợp lệ
        rows = []
        filtered_count = 0
        for interaction in all_interactions:
            user_id_str = str(interaction.user_id) if interaction.user_id else ''
            
            # Chỉ thêm interaction nếu user_id có trong danh sách users hợp lệ
            if user_id_str in valid_user_ids:
                # Chuyển timestamp thành string ISO format
                timestamp_str = interaction.timestamp.isoformat() if interaction.timestamp else ''
                
                row = {
                    'user_id': user_id_str,
                    'product_id': str(interaction.product_id) if interaction.product_id else '',
                    'interaction_type': interaction.interaction_type or '',
                    'timestamp': timestamp_str
                }
                rows.append(row)
            else:
                filtered_count += 1
        
        if filtered_count > 0:
            print(f"⚠️  Đã loại bỏ {filtered_count} interactions không map với users.csv")
        
        # Ghi vào CSV
        if rows:
            fieldnames = ['user_id', 'product_id', 'interaction_type', 'timestamp']
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return {
            'success': True,
            'file_path': str(csv_path),
            'count': len(rows),
            'message': f'Đã xuất {len(rows)} interactions thành công'
        }
    
    except Exception as e:
        return {'success': False, 'error': f'Lỗi khi xuất interactions: {str(e)}', 'count': 0}


def export_all_data() -> Dict:
    """
    Xuất tất cả dữ liệu (products, users, interactions) thành CSV files
    """
    export_dir = ensure_export_directory()
    
    # Kết nối MongoDB một lần cho tất cả exports
    try:
        connect_mongodb()
        mongodb_connected = True
    except Exception as e:
        return {
            'success': False,
            'error': f'Lỗi kết nối MongoDB: {str(e)}',
            'results': {},
            'export_dir': str(export_dir),
            'total_count': 0,
            'message': 'Không thể kết nối MongoDB'
        }
    
    results = {
        'products': export_products(export_dir, mongodb_connected=True),
        'users': export_users(export_dir, mongodb_connected=True),
        'interactions': export_interactions(export_dir, mongodb_connected=True)
    }
    
    # Tổng hợp kết quả
    total_success = all(r['success'] for r in results.values())
    total_count = sum(r.get('count', 0) for r in results.values())
    
    return {
        'success': total_success,
        'results': results,
        'export_dir': str(export_dir),
        'total_count': total_count,
        'message': f'Đã xuất {total_count} records tổng cộng' if total_success else 'Có lỗi xảy ra khi xuất dữ liệu'
    }


if __name__ == '__main__':
    # Test export
    result = export_all_data()
    print(json.dumps(result, indent=2, ensure_ascii=False))

