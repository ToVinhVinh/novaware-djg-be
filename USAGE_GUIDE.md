# Hướng dẫn sử dụng Hệ thống Gợi ý Sản phẩm

## Khởi động ứng dụng

```bash
streamlit run recommendation_system_app.py
```

## Các bước sử dụng

### 1. Load dữ liệu
- Ứng dụng tự động load dữ liệu từ thư mục `exports/`
- Kiểm tra các file: `users.csv`, `products.csv`, `interactions.csv`

### 2. Chọn mô hình
Trong sidebar, chọn một trong 3 mô hình:
- **LightGCN (GNN)**: Sử dụng Graph Neural Network
- **Content-Based Filtering**: Dựa trên đặc tính sản phẩm
- **Hybrid (LightGCN + CBF)**: Kết hợp cả 2

### 3. Chọn người dùng
- Chọn user từ dropdown để xem recommendations
- Thông tin user hiển thị: tên, tuổi, giới tính

### 4. Train mô hình
- Click nút **"🚀 Train Models"**
- Xem quá trình training với progress bar
- Xem giải thích thuật toán từng bước (A-Z) trong expander

### 5. Xem Recommendations

#### Personalized Recommendations
- Dựa trên lịch sử tương tác của user
- Gợi ý sản phẩm tương tự (cùng category, màu sắc, style)
- Lọc theo giới tính và độ tuổi

#### Outfit Recommendations
- Chọn sản phẩm từ dropdown "Chọn sản phẩm"
- Hệ thống gợi ý các sản phẩm đi kèm:
  - **Topwear**: Áo, sơ mi, áo khoác
  - **Bottomwear**: Quần, váy
  - **Footwear**: Giày, dép
  - **Accessories**: Túi, đồng hồ, thắt lưng

### 6. Đánh giá mô hình
- Xem metrics: Recall@10, Recall@20, NDCG@10, NDCG@20
- Xem inference time
- Click **"📈 Compare All Models"** để so sánh 3 mô hình

## Giải thích Algorithms

### LightGCN
1. **Xây dựng đồ thị**: G = (U ∪ I, E)
2. **Khởi tạo embeddings**: e_u^(0), e_i^(0)
3. **Propagation**: e_u^(l+1) = Σ (e_i^(l) / √(deg(u) * deg(i)))
4. **Average embeddings**: e_u = (1/(L+1)) * Σ e_u^(l)
5. **Dự đoán**: r̂_ui = e_u^T · e_i
6. **BPR Loss**: L = -Σ log(σ(r̂_ui - r̂_uj))
7. **Gradient Descent**: θ ← θ - α * ∇L

### Content-Based Filtering
1. **TF-IDF Vectorization**: v_i = TF-IDF(features)
2. **User Profile**: u = (1/|I_u|) * Σ v_i
3. **Cosine Similarity**: sim(u, i) = (u · v_i) / (||u|| * ||v_i||)
4. **Ranking**: Sắp xếp theo similarity

### Hybrid
1. **Train 2 models**: LightGCN + CBF
2. **Normalize scores**: r_norm = (r - r_min) / (r_max - r_min)
3. **Combine**: r_hybrid = α * r_gnn + (1-α) * r_cbf
4. **Ranking**: Sắp xếp theo combined score

## Metrics

### Recall@K
- **Công thức**: |R ∩ T| / |T|
- **Ý nghĩa**: Tỷ lệ sản phẩm relevant được tìm thấy
- **Ví dụ**: 7/10 sản phẩm relevant trong top-10 → Recall@10 = 0.7

### NDCG@K
- **Công thức**: DCG@K / IDCG@K
- **DCG**: Σ (rel_i / log₂(i+1))
- **Ý nghĩa**: Chất lượng ranking, ưu tiên items relevant ở vị trí cao
- **Ví dụ**: NDCG@10 = 0.8 → ranking tốt 80% so với lý tưởng

## Lưu ý

1. **Training time**: Có thể mất vài phút tùy vào kích thước dữ liệu
2. **Memory**: Đảm bảo có đủ RAM (khuyến nghị 4GB+)
3. **Data quality**: Cần có đủ interactions để train (tối thiểu 100 interactions)
4. **User selection**: Chọn user có interactions để có recommendations tốt

## Troubleshooting

### Lỗi: "Không có dữ liệu để train"
- Kiểm tra file `interactions.csv` có dữ liệu
- Đảm bảo có ít nhất 10 interactions

### Lỗi: "User not found"
- Chọn user khác từ dropdown
- Đảm bảo user có trong file `users.csv`

### Recommendations trống
- User có thể chưa có đủ interactions
- Thử user khác hoặc train lại model

