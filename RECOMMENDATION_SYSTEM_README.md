# Hệ thống Gợi ý Sản phẩm - Recommendation System

## Tổng quan

Hệ thống gợi ý sản phẩm với 3 mô hình:
1. **LightGCN (GNN)** - Graph Neural Network
2. **Content-Based Filtering** - Dựa trên đặc tính sản phẩm
3. **Hybrid** - Kết hợp LightGCN + Content-Based

## Yêu cầu

- Python 3.8+
- Các thư viện trong `requirements.txt`
- Dữ liệu CSV trong thư mục `exports/`:
  - `users.csv`
  - `products.csv`
  - `interactions.csv`

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run recommendation_system_app.py
```

## Tính năng

### 1. Import CSV Data
- Tự động load dữ liệu từ thư mục `exports/`
- Xử lý và chuẩn hóa dữ liệu

### 2. Training Models
- **LightGCN**: Train với BPR Loss
- **Content-Based**: Train với TF-IDF vectorization
- **Hybrid**: Kết hợp cả 2 mô hình

### 3. Step-by-Step Algorithms
Mỗi mô hình đều có giải thích chi tiết:
- **Công thức toán học**
- **Áp dụng công thức để tính số liệu**
- **Giải thích ý nghĩa của số liệu**

### 4. Evaluation Metrics
Bảng đánh giá với các metrics:
- **Recall@10, Recall@20**: Tỷ lệ sản phẩm relevant được tìm thấy
- **NDCG@10, NDCG@20**: Chất lượng ranking
- **Thời gian train**: Thời gian training
- **Thời gian inference/user**: Thời gian inference cho mỗi user

### 5. Personalized Recommendations
- Dựa trên lịch sử tương tác của user
- Lọc theo giới tính và độ tuổi
- Gợi ý sản phẩm tương tự

### 6. Outfit Recommendations
- Gợi ý các sản phẩm đi kèm
- Tạo bộ trang phục hoàn chỉnh
- Phân loại theo category (topwear, bottomwear, footwear, accessories)

## Cấu trúc Code

```
recommendation_system_app.py          # Main Streamlit app
apps/recommendations/streamlit_utils/
    data_loader.py                     # CSV data loading utilities
```

## Sử dụng

1. **Chọn mô hình** từ sidebar
2. **Chọn user** để xem recommendations
3. **Click "Train Models"** để train mô hình
4. **Xem recommendations**:
   - Personalized: Sản phẩm tương tự dựa trên lịch sử
   - Outfit: Sản phẩm đi kèm để tạo bộ
5. **So sánh models** bằng nút "Compare All Models"

## Algorithms

### LightGCN
1. Xây dựng đồ thị hai phía (Bipartite Graph)
2. Khởi tạo embeddings
3. LightGCN propagation qua các layers
4. Average embeddings từ tất cả layers
5. Dự đoán rating: r̂ = e_u^T · e_i
6. BPR Loss để train
7. Gradient Descent để update

### Content-Based Filtering
1. Tạo TF-IDF vectors cho mỗi sản phẩm
2. Xây dựng user profile: trung bình của product vectors
3. Tính cosine similarity: sim(u, i) = (u · v_i) / (||u|| * ||v_i||)
4. Ranking và recommendation

### Hybrid
1. Train LightGCN và Content-Based
2. Normalize scores về [0, 1]
3. Weighted combination: r_hybrid = α * r_gnn + (1-α) * r_cbf
4. Ranking theo combined score

## Metrics Explanation

### Recall@K
- **Công thức**: Recall@K = |R ∩ T| / |T|
- **Ý nghĩa**: Tỷ lệ sản phẩm relevant được tìm thấy trong top-K
- **Ví dụ**: User đã mua 10 sản phẩm, hệ thống recommend đúng 7 trong top-10 → Recall@10 = 0.7

### NDCG@K
- **Công thức**: NDCG@K = DCG@K / IDCG@K
- **DCG@K**: Σ (rel_i / log₂(i+1))
- **Ý nghĩa**: Đánh giá chất lượng ranking, ưu tiên items relevant ở vị trí cao
- **Ví dụ**: NDCG@10 = 0.8 nghĩa là ranking tốt 80% so với ranking lý tưởng

## Lưu ý

- Dữ liệu cần có đủ interactions để train
- Training có thể mất vài phút tùy vào kích thước dữ liệu
- Đảm bảo có đủ RAM cho việc train models

