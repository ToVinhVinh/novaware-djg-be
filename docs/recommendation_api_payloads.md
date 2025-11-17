# Recommendation API Payloads

Tài liệu mô tả nhanh các payload gửi tới ba endpoint khuyến nghị:

- `POST /api/v1/gnn/recommend/`
- `POST /api/v1/cbf/recommend/`
- `POST /api/v1/hybrid/recommend/`

## Trường dùng chung

Ba API kế thừa `RecommendationRequestSerializer` trong `apps/recommendations/common/api.py` nên đều chấp nhận các trường bắt buộc sau:

| Trường | Kiểu | Ràng buộc | Mặc định | Mô tả |
| --- | --- | --- | --- | --- |
| `user_id` | string | không rỗng | — | ID người dùng nguồn |
| `current_product_id` | string | không rỗng | — | Sản phẩm đang xem/đối chiếu |
| `top_k_personal` | integer | `1 ≤ value ≤ 50` | `5` | Số gợi ý cá nhân |
| `top_k_outfit` | integer | `1 ≤ value ≤ 10` | `4` | Số gợi ý outfit/phối đồ |

Ví dụ payload tối thiểu:

```json
{
  "user_id": "12345",
  "current_product_id": "SKU-7788"
}
```

## API chi tiết

### 1. GNN

- Endpoint: `POST /api/v1/gnn/recommend/`
- Serializer: `GNNRecommendationSerializer`
- Payload: chỉ dùng các trường chung ở trên.
- Ví dụ:

```json
{
  "user_id": "12345",
  "current_product_id": "SKU-7788",
  "top_k_personal": 10,
  "top_k_outfit": 4
}
```

### 2. CBF

- Endpoint: `POST /api/v1/cbf/recommend/`
- Serializer: `CBFRecommendationSerializer`
- Payload: giống GNN, không có trường bổ sung.

```json
{
  "user_id": "user-demo",
  "current_product_id": "SKU-9001",
  "top_k_personal": 8,
  "top_k_outfit": 4
}
```

### 3. Hybrid

- Endpoint: `POST /api/v1/hybrid/recommend/`
- Serializer: `HybridRecommendationSerializer`
- Thêm trường tùy chọn:

| Trường | Kiểu | Ràng buộc | Mặc định | Mô tả |
| --- | --- | --- | --- | --- |
| `alpha` | float | `0.0 ≤ value ≤ 1.0` | theo cấu hình mô hình | Trọng số pha trộn giữa tín hiệu CF (GNN) và CBF |

```json
{
  "user_id": "user-demo",
  "current_product_id": "SKU-9001",
  "top_k_personal": 8,
  "top_k_outfit": 4,
  "alpha": 0.6
}
```

## Ghi chú

- Nếu bỏ `top_k_personal` hoặc `top_k_outfit`, hệ thống dùng mặc định 5 và 4.
- Các trường khác chỉ hợp lệ khi được thêm vào serializer tương ứng.
- Hybrid `alpha` cho phép tinh chỉnh tỷ lệ giữa điểm CF và CBF mà không cần triển khai lại model.

