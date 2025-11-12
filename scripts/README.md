# Scripts Testing và Migration

Thư mục này chứa các scripts để test, kiểm tra và migrate data cho dự án Novaware Django với MongoEngine.

## Các Scripts

### 1. `test_mongodb_connection.py`

Script kiểm tra kết nối MongoDB và indexes.

**Chức năng:**
- Kiểm tra kết nối MongoDB (pymongo và mongoengine)
- Kiểm tra indexes được định nghĩa trong models
- Đảm bảo tất cả indexes được tạo
- Hiển thị số lượng documents trong mỗi collection

**Cách sử dụng:**
```bash
cd novaware_django
python scripts/test_mongodb_connection.py
```

### 2. `test_endpoints.py`

Script test các endpoints API.

**Chức năng:**
- Test authentication endpoints (register, login)
- Test user endpoints (profile, list users)
- Test product endpoints (list, search)
- Test brand endpoints
- Test order endpoints
- Test chat endpoints
- Test recommendation endpoints

**Cách sử dụng:**
```bash
cd novaware_django
python scripts/test_endpoints.py

# Hoặc với base URL khác
python scripts/test_endpoints.py --base-url http://localhost:8000
```

**Lưu ý:** Cần chạy Django server trước khi test endpoints.

### 3. `migrate_data_to_mongoengine.py`

Script migrate data từ Django models sang MongoEngine models.

**Chức năng:**
- Migrate users
- Migrate categories, brands, colors, sizes
- Migrate products
- Migrate orders (cần migrate products trước)
- Hỗ trợ dry-run mode để xem trước

**Cách sử dụng:**
```bash
cd novaware_django

# Dry-run (chỉ xem, không migrate)
python scripts/migrate_data_to_mongoengine.py

# Thực sự migrate data
python scripts/migrate_data_to_mongoengine.py --execute
```

**Lưu ý:** 
- Script sẽ skip các documents đã tồn tại
- Cần migrate theo thứ tự dependency (users -> categories/brands -> products -> orders)
- Nên chạy dry-run trước để xem kết quả

### 4. `monitor_performance.py`

Script monitor performance và tối ưu queries.

**Chức năng:**
- Kiểm tra collection statistics (size, count, indexes)
- Test performance của các queries phổ biến
- Kiểm tra index usage
- Phát hiện slow queries
- Kiểm tra N+1 query problems
- Đề xuất optimizations

**Cách sử dụng:**
```bash
cd novaware_django
python scripts/monitor_performance.py
```

### 5. `run_all_tests.py`

Script chạy tất cả các tests và checks.

**Chức năng:**
- Chạy tất cả các scripts test
- Hiển thị tổng kết kết quả

**Cách sử dụng:**
```bash
cd novaware_django
python scripts/run_all_tests.py
```

### 6. `quick_test.py`

Script test nhanh MongoDB connection và các queries cơ bản.

**Chức năng:**
- Đếm documents trong các collections
- Test query một document
- Test create và delete document

**Cách sử dụng:**
```bash
cd novaware_django
python scripts/quick_test.py
```

## Workflow Khuyến Nghị

### 1. Sau khi setup MongoDB

```bash
# Kiểm tra connection và indexes
python scripts/test_mongodb_connection.py
```

### 2. Sau khi migrate code

```bash
# Test endpoints
python scripts/test_endpoints.py

# Monitor performance
python scripts/monitor_performance.py
```

### 3. Khi cần migrate data

```bash
# Xem trước migration
python scripts/migrate_data_to_mongoengine.py

# Thực hiện migration
python scripts/migrate_data_to_mongoengine.py --execute

# Kiểm tra lại
python scripts/test_mongodb_connection.py
```

### 4. Chạy tất cả tests

```bash
python scripts/run_all_tests.py
```

## Requirements

Tất cả scripts cần:
- Django được setup đúng
- MongoDB connection được cấu hình trong `settings.py`
- Các models MongoEngine đã được import

**Dependencies cần thiết:**
- `mongoengine` - Đã có trong requirements.txt
- `pymongo` - Đã có trong requirements.txt (dependency của mongoengine)
- `requests` - Cần cho `test_endpoints.py` (cài: `pip install requests`)
- `djangorestframework` - Đã có trong requirements.txt

## Troubleshooting

### Lỗi kết nối MongoDB

Kiểm tra:
- MongoDB đang chạy
- `MONGO_URI` trong settings đúng
- Firewall không block port MongoDB

### Lỗi import models

Đảm bảo:
- Đã chạy `django.setup()`
- `DJANGO_SETTINGS_MODULE` được set đúng
- Tất cả apps đã được thêm vào `INSTALLED_APPS`

### Lỗi test endpoints

Đảm bảo:
- Django server đang chạy
- Base URL đúng
- Authentication đã được setup

