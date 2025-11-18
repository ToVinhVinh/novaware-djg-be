## 2.3. Đánh giá mô hình

- **UserID chuẩn**: `690bf0f2d0c3753df0ecbdd6`.
- **ProductID test**: lấy từ Mongo `products` (ví dụ `db.products.findOne({}, {_id:1})`).  
  Ghi rõ trong tài liệu khi đã chọn được ID thật.
- Các mô hình cần đánh giá: GNN, Content-based, Hybrid.
- Thực thi thông qua API `/api/v1/{gnn|cbf|hybrid}/recommend` hoặc script `run_reco_demo`.

### 2.3.1. GNN (Graph Neural Network - LightGCN)

- **Quy trình theo yêu cầu**:
  - *Chuẩn hóa dữ liệu với Surprise*:  
    Dùng `surprise.Dataset.load_from_df(...)`, `train_test_split(test_size=0.2)` để tạo tập train/test.  
    - Ghi rõ "testsize = 0.2" trong tài liệu.  
    - Số lượng người dùng train: ghi số thực tế sau khi chạy (ví dụ `num_users` từ log training).  
    - Số lượng sản phẩm train: ghi số thực tế (ví dụ `num_products` từ log training).
    - Số lượng tương tác (interactions): ghi số thực tế (ví dụ `num_interactions` từ log training).
    - Số lượng training samples (BPR): ghi số thực tế (ví dụ `num_training_samples` từ log training).
  - *Dữ liệu mẫu*: xuất 5 dòng đầu cho `trainset` và `testset`.
  - *Pipeline 5 bước*:
    1. **Huấn luyện mô hình**: LightGCN với kiến trúc Graph Convolutional Network.
       - Thuật toán: LightGCN (Light Graph Convolution Network)
       - Framework: PyTorch + PyTorch Geometric
       - Loss function: BPR (Bayesian Personalized Ranking)
       - Negative sampling: 4 negative samples per positive interaction
       - Epochs: 50
       - Batch size: 2048
       - Embedding dimension: 64
       - Learning rate: ghi rõ giá trị (ví dụ 0.001)
       - Optimizer: Adam
       - Model file: `models/gnn_lightgcn.pkl`
    2. **Chuẩn bị dữ liệu graph**: 
       - Xây dựng bipartite graph từ `UserInteraction` collection.
       - Áp dụng trọng số tương tác theo `INTERACTION_WEIGHTS`:
         ```python
         INTERACTION_WEIGHTS = {
             'view': 1.0,
             'add_to_cart': 2.0,
             'purchase': 3.0,
             'wishlist': 1.5,
             'rating': 2.5
         }
         ```
       - Tạo edge index (user-product pairs) và edge weights.
       - Xuất ma trận adjacency mẫu (ví dụ 5x5 users x products) và kèm ảnh heatmap.
    3. **Tạo ma trận User-Item Interaction**: 
       - Dùng sparse matrix để biểu diễn tương tác user-product.
       - Xuất ma trận mẫu (ví dụ 5x5) và kèm ảnh heatmap thể hiện mật độ tương tác.
       - Tính toán sparsity: `sparsity = 1 - (num_interactions / (num_users * num_products))`
    4. **Tính cosine similarity** giữa user embeddings và product embeddings.  
       - Sau khi training, LightGCN sinh ra:
         - User embeddings: `[num_users, embed_dim]`
         - Product embeddings: `[num_products, embed_dim]`
       - Recommendation score = dot product giữa user embedding và product embedding.
       - Lưu bảng similarity matrix mẫu (kèm UserID và ProductID).
    5. **Tính toán chỉ số đánh giá**: MAPE, RMSE, Precision, Recall, F1, thời gian.
       - Dùng `apps/recommendations/common/evaluation.calculate_evaluation_metrics`.
       - Giải thích ý nghĩa:
         - *MAPE*: sai số phần trăm tuyệt đối trung bình giữa rating dự đoán và thực tế.
         - *RMSE*: độ lệch chuẩn của sai số dự đoán.
         - *Precision/Recall/F1*: độ chính xác/phủ và cân bằng giữa hai yếu tố khi so sánh recommendation với ground-truth.
         - *Thời gian*: latency suy luận hoặc toàn bộ pipeline (ghi rõ đơn vị: giây hoặc millisecond).

- **Ảnh cần thu thập**:
  1. Screenshot terminal trong quá trình training (hiển thị epoch, loss, metrics).
  2. Biểu đồ training loss theo epoch.
  3. Ma trận User-Item Interaction (heatmap 5x5 hoặc 10x10).
  4. Ma trận Adjacency của graph (heatmap).
  5. Biểu đồ phân bố embedding vectors (t-SNE hoặc PCA visualization).
  6. Kết quả recommendation mẫu cho 1 user (bảng sản phẩm gợi ý với score).

> Gợi ý thực thi:
> ```python
> # Training GNN
> from apps.recommendations.gnn.engine_lightgcn import LightGCNEngine
> 
> engine = LightGCNEngine()
> engine.train(epochs=50, batch_size=2048, embed_dim=64)
> 
> # Đánh giá
> from apps.recommendations.common.evaluation import calculate_evaluation_metrics
> metrics = calculate_evaluation_metrics(engine, testset)
> print(metrics)
> ```

| Mô hình | MAPE | RMSE | Precision | Recall | F1 | Thời gian |
|---------|------|------|-----------|--------|----|-----------|
| GNN (LightGCN) | … | … | … | … | … | … |

### 2.3.2. Content-based Filtering

- **Quy trình theo yêu cầu**:
  - *Chuẩn hóa dữ liệu với Surprise*:  
    Dùng `surprise.Dataset.load_from_df(...)`, `train_test_split(test_size=0.2)` (có thể thay `0.2` nếu bạn chọn khác).  
    - Ghi rõ “testsize = 0.2” trong tài liệu.  
    - Số lượng sản phẩm train: **770** (theo kiểm tra `check_training_data`).  
    - Số lượng người dùng test: ghi số thực tế sau khi chạy Surprise (ví dụ `len({uid for (_, uid, _) in testset})`).
  - *Dữ liệu mẫu*: xuất 5 dòng đầu cho `trainset` và `testset`.
  - *Pipeline 5 bước*:
    1. **Huấn luyện mô hình**: Sentence-BERT embedding + FAISS index.
    2. **Chuẩn bị dữ liệu văn bản**: ghép `category`, `gender`, `color`, `style_tags`, `productDisplayName`.
    3. **Tạo ma trận TF-IDF**: dùng `TfidfVectorizer` (dù model chính dùng SBERT, nhưng TF-IDF matrix vẫn dùng cho báo cáo).  
       Xuất ma trận mẫu (ví dụ 5x5) và kèm ảnh heatmap.
    4. **Tính cosine similarity** giữa các sản phẩm (SBERT embeddings).  
       Lưu bảng matrix (kèm ID).
    5. **Tính toán chỉ số đánh giá**: MAPE, RMSE, Precision, Recall, F1, thời gian.
       - Dùng `apps/recommendations/common/evaluation.calculate_evaluation_metrics`.
       - Giải thích ý nghĩa:
         - *MAPE*: sai số phần trăm tuyệt đối trung bình giữa rating dự đoán và thực tế.
         - *RMSE*: độ lệch chuẩn của sai số dự đoán.
         - *Precision/Recall/F1*: độ chính xác/phủ và cân bằng giữa hai yếu tố khi so sánh recommendation với ground-truth.
         - *Thời gian*: latency suy luận hoặc toàn bộ pipeline (ghi rõ đơn vị).

| Mô hình | MAPE | RMSE | Precision | Recall | F1 | Thời gian |
|---------|------|------|-----------|--------|----|-----------|
| Content-based Filtering | … | … | … | … | … | … |

### 2.3.3. Hybrid Content-based Filtering & Collaborative Filtering

- Áp dụng cùng format như 2.3.2:
  1. Huấn luyện GNN + CBF.
  2. Chuẩn bị dữ liệu (kết hợp embedding CF/Content).
  3. Bảng TF-IDF/Cosine có thể ghi rõ là “kế thừa từ CBF, cộng thêm trọng số CF”.
  4. Bảng chỉ số tương tự (MAPE, RMSE, Precision, Recall, F1, Time) với chú thích `alpha = 0.7` (hoặc giá trị bạn chọn).

| Mô hình | MAPE | RMSE | Precision | Recall | F1 | Thời gian |
|---------|------|------|-----------|--------|----|-----------|
| Hybrid CF+CBF | … | … | … | … | … | … |

---

# 3. Đánh giá 3 mô hình

| Mô hình | MAPE | RMSE | Precision | Recall | F1 | Thời gian |
|---------|------|------|-----------|--------|----|-----------|
| GNN (LightGCN) | … | … | … | … | … | … |
| Content-based Filtering | … | … | … | … | … | … |
| Hybrid CF+CBF | … | … | … | … | … | … |

- **Phân tích & lựa chọn**:
  - Nếu tập trung vào hành vi người dùng dày đặc ⇒ GNN thường cho Precision/Recall cao nhất.
  - Nếu cần xử lý cold-start hoặc catalog phong phú ⇒ CBF đảm bảo gợi ý vẫn hợp lý (nhờ lọc age/gender + reason theo style).
  - Hybrid là lựa chọn production mặc định vì duy trì ổn định giữa hai tình huống, có thể tinh chỉnh trọng số `alpha`.
  - Khi kết luận, nêu rõ lý do chọn mô hình (ví dụ Hybrid đạt F1 cao nhất và time chấp nhận được).
