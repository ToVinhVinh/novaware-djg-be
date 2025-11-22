# Hướng dẫn Export Users từ MongoDB ra CSV

## Script Export Users

Script `export_users_csv.py` được tạo để xuất dữ liệu users từ MongoDB ra file CSV với format phù hợp cho hệ thống recommendation.

## Cài đặt

Script đã được tạo tại:
```
apps/users/management/commands/export_users_csv.py
```

## Sử dụng

### Cách 1: Sử dụng Django Management Command

```bash
# Export với format mặc định (có format interaction_history)
python manage.py export_users_csv --out ./exports/users.csv

# Export với format đơn giản (không format interaction_history)
python manage.py export_users_csv --out ./exports/users.csv --no-format-interactions

# Export vào thư mục khác
python manage.py export_users_csv --out /path/to/output/users.csv
```

### Cách 2: Sử dụng như Python script

```python
from apps.users.management.commands.export_users_csv import export_users_to_csv

# Export users
export_users_to_csv("./exports/users.csv", format_interactions=True)
```

## Format CSV Output

File CSV sẽ có format:
```csv
id,name,email,age,gender,interaction_history
690bf0f2d0c3753df0ecbdd6,Nguyen Thanh Doanh,nguyenthanhdoanh123@gmail.com,27,male,"{'product_id': 10866, 'interaction_type': 'purchase', 'timestamp': datetime.datetime(2025, 11, 17, 11, 10, 42, 24000), 'usage': 'Casual', 'baseColour': 'Red'};{...}"
```

### Các trường dữ liệu:

- **id**: ObjectId của user (string)
- **name**: Tên user
- **email**: Email của user
- **age**: Tuổi (có thể null)
- **gender**: Giới tính (male/female/other, có thể null)
- **interaction_history**: Lịch sử tương tác được format như string với:
  - `datetime.datetime(...)` cho timestamps
  - `ObjectId('...')` cho ObjectId
  - Các interactions được nối bằng dấu `;`

## Format Interaction History

### Với `--format-interactions` (mặc định):
```python
"{'product_id': 10866, 'interaction_type': 'purchase', 'timestamp': datetime.datetime(2025, 11, 17, 11, 10, 42, 24000), 'usage': 'Casual', 'baseColour': 'Red'};{'product_id': 10065, 'interaction_type': 'cart', 'timestamp': datetime.datetime(2025, 11, 16, 11, 10, 42, 24000), 'usage': 'Casual', 'baseColour': 'Blue'}"
```

### Với `--no-format-interactions`:
```python
"[{'product_id': 10866, 'interaction_type': 'purchase', ...}, {'product_id': 10065, ...}]"
```

## Xử lý các trường hợp đặc biệt

1. **User không có interaction_history**: Trường này sẽ để trống
2. **User không có age/gender**: Trường này sẽ để trống
3. **Interaction với ObjectId**: Sẽ được format thành `ObjectId('...')`
4. **Interaction với datetime**: Sẽ được format thành `datetime.datetime(...)`

## Lưu ý

1. **Kết nối MongoDB**: Script tự động kết nối MongoDB thông qua `connect_mongodb()`
2. **Encoding**: File CSV được lưu với encoding UTF-8
3. **Performance**: Script xử lý từng user một, có thể mất thời gian với database lớn
4. **Memory**: Script load tất cả users vào memory, cần đủ RAM

## Troubleshooting

### Lỗi: "MongoDB connection error"
- Kiểm tra MongoDB đang chạy
- Kiểm tra `MONGO_URI` trong settings.py
- Kiểm tra network connection

### Lỗi: "No module named 'apps.users'"
- Đảm bảo đang chạy từ thư mục gốc của Django project
- Kiểm tra PYTHONPATH

### File CSV trống hoặc thiếu dữ liệu
- Kiểm tra có users trong MongoDB
- Kiểm tra quyền truy cập database
- Kiểm tra log để xem có lỗi nào không

## Ví dụ Output

```csv
id,name,email,age,gender,interaction_history
690bf0f2d0c3753df0ecbdd6,Nguyen Thanh Doanh,nguyenthanhdoanh123@gmail.com,27,male,"{'product_id': 10866, 'interaction_type': 'purchase', 'timestamp': datetime.datetime(2025, 11, 17, 11, 10, 42, 24000), 'usage': 'Casual', 'baseColour': 'Red'};{'product_id': 10065, 'interaction_type': 'cart', 'timestamp': datetime.datetime(2025, 11, 16, 11, 10, 42, 24000), 'usage': 'Casual', 'baseColour': 'Blue'}"
690bf40623150d4eec246874,Amazon User AHV6QCNB,quangvinhvinh@gmail.com,,,"{'_id': ObjectId('690c13d8ca0e8f57201c656d'), 'productId': ObjectId('68fd617451f9562e9cca6fe9'), 'interactionType': 'review', 'rating': 5, 'timestamp': datetime.datetime(2025, 11, 6, 2, 9, 16, 731000)}"
```

## So sánh với export_eval_csv.py

Script `export_eval_csv.py` (trong `apps/recommendations/management/commands/`) export nhiều trường hơn và format khác:
- Export thêm: username, preferences_styles, created_at, updated_at
- Format interaction_history đơn giản hơn

Script `export_users_csv.py` này tập trung vào format phù hợp với recommendation system:
- Chỉ export các trường cần thiết: id, name, email, age, gender, interaction_history
- Format interaction_history với datetime và ObjectId để dễ parse lại

