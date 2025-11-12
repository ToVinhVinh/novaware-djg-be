# Hướng Dẫn Testing và Kiểm Tra

Tài liệu này hướng dẫn cách test và kiểm tra hệ thống sau khi migration sang MongoEngine.

## Bước 1: Kiểm Tra MongoDB Connection

Trước tiên, đảm bảo MongoDB đang chạy và có thể kết nối:

```bash
cd novaware_django
python scripts/test_mongodb_connection.py
```

Script này sẽ:
- ✅ Kiểm tra kết nối MongoDB (pymongo và mongoengine)
- ✅ Kiểm tra tất cả indexes được định nghĩa
- ✅ Tạo indexes nếu chưa có
- ✅ Hiển thị số lượng documents trong mỗi collection

**Kết quả mong đợi:**
- Tất cả connections thành công
- Tất cả indexes đã được tạo
- Không có lỗi

## Bước 2: Test Nhanh (Quick Test)

Chạy test nhanh để đảm bảo các chức năng cơ bản hoạt động:

```bash
python scripts/quick_test.py
```

Script này sẽ:
- ✅ Đếm documents
- ✅ Query một document
- ✅ Test create/delete

## Bước 3: Test Endpoints API

**Lưu ý:** Cần chạy Django server trước:

```bash
# Terminal 1: Chạy server
python manage.py runserver

# Terminal 2: Chạy test
python scripts/test_endpoints.py
```

Script này sẽ test:
- ✅ Authentication endpoints (register, login)
- ✅ User endpoints (profile, list)
- ✅ Product endpoints (list, search)
- ✅ Brand endpoints
- ✅ Order endpoints
- ✅ Chat endpoints
- ✅ Recommendation endpoints

## Bước 4: Monitor Performance

Kiểm tra performance và tối ưu:

```bash
python scripts/monitor_performance.py
```

Script này sẽ:
- ✅ Hiển thị collection statistics
- ✅ Test performance của các queries phổ biến
- ✅ Kiểm tra index usage
- ✅ Phát hiện slow queries
- ✅ Kiểm tra N+1 query problems
- ✅ Đề xuất optimizations

## Bước 5: Migrate Data (Nếu Cần)

Nếu cần migrate data từ Django models sang MongoEngine:

```bash
# Xem trước (dry-run)
python scripts/migrate_data_to_mongoengine.py

# Thực hiện migration
python scripts/migrate_data_to_mongoengine.py --execute
```

**Lưu ý:**
- Luôn chạy dry-run trước
- Backup database trước khi migrate
- Migration sẽ skip các documents đã tồn tại

## Bước 6: Chạy Tất Cả Tests

Chạy tất cả tests cùng lúc:

```bash
python scripts/run_all_tests.py
```

## Checklist Hoàn Chỉnh

Sau khi chạy tất cả tests, đảm bảo:

- [ ] MongoDB connection thành công
- [ ] Tất cả indexes đã được tạo
- [ ] Quick test passed
- [ ] Tất cả endpoints hoạt động đúng
- [ ] Performance trong giới hạn chấp nhận được
- [ ] Không có slow queries nghiêm trọng
- [ ] Không có N+1 query problems
- [ ] Data đã được migrate (nếu cần)

## Troubleshooting

### Lỗi "Cannot connect to MongoDB"

1. Kiểm tra MongoDB đang chạy:
   ```bash
   # Windows
   net start MongoDB
   
   # Linux/Mac
   sudo systemctl status mongod
   ```

2. Kiểm tra `MONGO_URI` trong `.env` hoặc `settings.py`

3. Kiểm tra firewall không block port 27017

### Lỗi "Index not found"

Chạy lại script test_mongodb_connection.py để tạo indexes:
```bash
python scripts/test_mongodb_connection.py
```

### Lỗi "Module not found"

Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

### Endpoints trả về 401 Unauthorized

Đảm bảo:
- User đã đăng nhập
- Token được gửi đúng trong header: `Authorization: Bearer <token>`
- Token chưa hết hạn

### Performance chậm

1. Kiểm tra indexes đã được tạo
2. Xem đề xuất từ `monitor_performance.py`
3. Kiểm tra slow queries
4. Tối ưu queries theo đề xuất

## Kết Quả Mong Đợi

Sau khi hoàn thành tất cả tests:

✅ **MongoDB Connection:**
- Kết nối thành công
- Tất cả indexes đã được tạo
- Collections có data (nếu đã migrate)

✅ **Endpoints:**
- Authentication hoạt động
- Tất cả CRUD operations hoạt động
- Pagination hoạt động
- Search hoạt động
- Permissions hoạt động đúng

✅ **Performance:**
- Queries < 100ms cho các operations thông thường
- Không có N+1 query problems
- Indexes được sử dụng đúng

## Next Steps

Sau khi tất cả tests passed:

1. ✅ Deploy lên staging environment
2. ✅ Test lại trên staging
3. ✅ Deploy lên production
4. ✅ Monitor performance trong production
5. ✅ Tối ưu dựa trên real-world usage

