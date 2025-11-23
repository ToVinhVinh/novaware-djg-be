"""
Streamlit App for Recommendation System
Giao diện để demo, so sánh và hiển thị chi tiết thuật toán các models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import time

# Import training pipeline
# Add current directory to path to find train_recommendation.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import train_recommendation (will be used when training button is clicked)
_train_import_error = None
try:
    import train_recommendation
except ImportError as e:
    # Don't fail immediately, just show warning when needed
    train_recommendation = None
    _train_import_error = str(e)

# Page config
st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #d62728;
        margin-top: 1rem;
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .formula-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        with open('recommendation_system/data/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        with open('recommendation_system/models/content_based_model.pkl', 'rb') as f:
            cb_model = pickle.load(f)
        
        with open('recommendation_system/models/gnn_model.pkl', 'rb') as f:
            gnn_model = pickle.load(f)
        
        with open('recommendation_system/models/hybrid_model.pkl', 'rb') as f:
            hybrid_model = pickle.load(f)
        
        return preprocessor, cb_model, gnn_model, hybrid_model
    except Exception as e:
        return None, None, None, None


@st.cache_data
def load_comparison_results():
    """Load comparison results"""
    try:
        df = pd.read_csv('recommendation_system/evaluation/comparison_results.csv')
        return df
    except:
        return None

def display_product_info(product_info: Dict, score: float = None):
    """Display product information"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if score is not None:
            st.metric("Score", f"{score:.4f}")
    
    with col2:
        st.markdown(f"**{product_info.get('productDisplayName', 'N/A')}**")
        st.write(f"🏷️ **Category**: {product_info.get('masterCategory', 'N/A')} > {product_info.get('subCategory', 'N/A')} > {product_info.get('articleType', 'N/A')}")
        st.write(f"👤 **Gender**: {product_info.get('gender', 'N/A')}")
        st.write(f"🎨 **Color**: {product_info.get('baseColour', 'N/A')}")

def render_metrics_table(df, highlight_model=None):
    """Render metrics table with highlighting"""
    if df is None:
        st.warning("Chưa có dữ liệu metrics. Vui lòng chạy tính toán trước.")
        return

    st.markdown("### 📊 Bảng Tổng Hợp Chỉ Số Các Mô Hình")
    
    # Format dataframe
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    
    def highlight_row(row):
        if row['model_name'] == highlight_model:
            return ['background-color: #e6ffe6'] * len(row)
        return [''] * len(row)

    st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True)


def run_training(model_type: str):
    """Run training for specific model type"""
    import io
    from contextlib import redirect_stdout
    
    model_names = {
        "all": "Tất Cả Models",
        "content_based": "Content-Based Filtering",
        "gnn": "GNN (GraphSAGE)",
        "hybrid": "Hybrid (GNN + Content-Based)"
    }
    
    model_name = model_names.get(model_type, model_type)
    
    with st.status(f"Đang train {model_name}...", expanded=True) as status:
        st.write(f"🚀 Bắt đầu training {model_name}...")
        try:
            # Redirect stdout to capture logs
            f = io.StringIO()
            with redirect_stdout(f):
                if model_type == "all":
                    train_recommendation.train_and_evaluate()
                elif model_type == "content_based":
                    train_recommendation.train_content_based(evaluate=True)
                elif model_type == "gnn":
                    train_recommendation.train_gnn(evaluate=True)
                elif model_type == "hybrid":
                    train_recommendation.train_hybrid(evaluate=True)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            
            output_log = f.getvalue()
            st.text_area("Logs", output_log, height=300)
            
            # Reload data
            st.cache_resource.clear()
            st.cache_data.clear()
            preprocessor, cb_model, gnn_model, hybrid_model = load_models()
            comparison_df = load_comparison_results()
            
            status.update(label=f"✅ Hoàn thành training {model_name}!", state="complete", expanded=False)
            st.success(f"✅ Đã hoàn thành training {model_name} và cập nhật số liệu!")
        except Exception as e:
            status.update(label=f"❌ Lỗi khi train {model_name}", state="error")
            st.error(f"Lỗi: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def main():
    """Main app"""
    
    # Header
    st.markdown('<div class="main-header">👔 Fashion Recommendation System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Menu")
    
    page = st.sidebar.radio(
        "Chọn chức năng",
        ["📚 Algorithms & Steps", "📊 Model Comparison", "🎯 Personalized Recommendations", "👗 Outfit Recommendations"]
    )
    
    # Load data initially
    preprocessor, cb_model, gnn_model, hybrid_model = load_models()
    comparison_df = load_comparison_results()

    # ========== PAGE 1: ALGORITHMS & STEPS ==========
    if page == "📚 Algorithms & Steps":
        st.markdown('<div class="sub-header">📚 Chi Tiết Thuật Toán & Các Bước Thực Hiện</div>', unsafe_allow_html=True)
        
        # Training Buttons Section
        st.markdown("### 🔄 Training Models")
        
        # Check if train_recommendation is available
        if train_recommendation is None:
            st.error(f"❌ Không thể import train_recommendation module: {_train_import_error}")
            st.info("Vui lòng đảm bảo file train_recommendation.py tồn tại trong thư mục gốc.")
        else:
            # Create columns for buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Train Tất Cả", type="primary", use_container_width=True):
                    run_training("all")
            
            with col2:
                if st.button("📊 Train Content-Based", use_container_width=True):
                    run_training("content_based")
            
            with col3:
                if st.button("🕸️ Train GNN", use_container_width=True):
                    run_training("gnn")
            
            with col4:
                if st.button("🔀 Train Hybrid", use_container_width=True):
                    run_training("hybrid")
            
            st.info("💡 Chọn model để train riêng lẻ hoặc train tất cả cùng lúc. Hybrid model cần Content-Based và GNN đã được train trước.")

        if preprocessor is None:
            st.warning("Vui lòng chạy tính toán để khởi tạo models.")
            st.stop()

        # Tabs for each model
        tab1, tab2, tab3 = st.tabs(["1️⃣ Content-Based Filtering", "2️⃣ GNN (GraphSAGE)", "3️⃣ Hybrid Model"])
        
        # --- CONTENT-BASED TAB ---
        with tab1:
            st.markdown("### 1️⃣ Content-Based Filtering")
            st.markdown("**Mô tả:** Gợi ý dựa trên sự tương đồng về đặc điểm sản phẩm (Category, Color, Usage...).")
            
            # BƯỚC 1
            with st.expander("Bước 1: Feature Engineering (Tạo đặc trưng)", expanded=True):
                st.markdown('<div class="step-header">Bước 1: Feature Engineering</div>', unsafe_allow_html=True)
                st.write("**Nội dung thực hiện:** Chuyển đổi các thuộc tính sản phẩm thành chuỗi văn bản, áp dụng trọng số bằng cách lặp lại từ khóa.")
                st.write(f"**Dữ liệu sử dụng:** Toàn bộ {len(preprocessor.products_df)} sản phẩm trong `products.csv`.")
                
                st.markdown("""
                **Công thức áp dụng:**
                $$Text(P_i) = [Gender] + [MasterCategory] + [SubCategory] \\times 2 + [ArticleType] \\times 3 + [BaseColour] + [Usage]$$
                """)
                
                st.write("**Kết quả tính toán (Ví dụ 2 sản phẩm đầu tiên):**")
                example_df = cb_model.products_df[['productDisplayName', 'feature_text']].head(2)
                st.table(example_df)
                st.info("💡 **Phân tích:** Việc lặp lại `ArticleType` 3 lần giúp thuật toán coi trọng loại sản phẩm hơn màu sắc.")

            # BƯỚC 2
            with st.expander("Bước 2: Vectorization (TF-IDF) & Ma trận"):
                st.markdown('<div class="step-header">Bước 2: Vectorization</div>', unsafe_allow_html=True)
                st.write("**Nội dung thực hiện:** Chuyển đổi văn bản thành vector số học sử dụng TF-IDF.")
                
                st.markdown("""
                **Công thức TF-IDF:**
                $$TF(t, d) = \\frac{count(t, d)}{len(d)}, \\quad IDF(t) = \\log(\\frac{N}{df(t)}), \\quad TF\\text{-}IDF = TF \\times IDF$$
                """)
                
                if cb_model.tfidf_vectorizer is not None:
                    feature_names = cb_model.tfidf_vectorizer.get_feature_names_out()
                    # Lấy vector của 5 sản phẩm đầu tiên
                    tfidf_subset = cb_model.tfidf_vectorizer.transform(cb_model.products_df['feature_text'].head(5))
                    tfidf_df = pd.DataFrame(tfidf_subset.toarray(), columns=feature_names, index=cb_model.products_df['productDisplayName'].head(5))
                    
                    st.write(f"**Ma trận TF-IDF (Top 5 sản phẩm x Top 10 features):**")
                    st.dataframe(tfidf_df.iloc[:, :10].style.background_gradient(cmap='Blues', axis=None))
                    st.info(f"💡 **Ý nghĩa:** Giá trị càng cao (đậm) nghĩa là từ khóa đó càng đặc trưng cho sản phẩm. Ma trận thưa (nhiều số 0).")

            # BƯỚC 3
            with st.expander("Bước 3: Similarity Calculation & Ví dụ tính toán"):
                st.markdown('<div class="step-header">Bước 3: Tính độ tương đồng</div>', unsafe_allow_html=True)
                st.write("**Nội dung thực hiện:** Tính Cosine Similarity giữa tất cả các cặp sản phẩm.")
                
                st.markdown("""
                **Công thức Cosine Similarity:**
                $$Cosine(A, B) = \\frac{\\sum A_i B_i}{\\sqrt{\\sum A_i^2} \\sqrt{\\sum B_i^2}}$$
                """)
                
                if cb_model.similarity_matrix is not None:
                    # Lấy ma trận similarity nhỏ (5x5)
                    sim_subset = cb_model.similarity_matrix[:5, :5]
                    sim_df = pd.DataFrame(sim_subset, 
                                        index=cb_model.products_df['productDisplayName'].head(5),
                                        columns=cb_model.products_df['productDisplayName'].head(5))
                    
                    st.write("**Ma trận Similarity (5x5):**")
                    st.dataframe(sim_df.style.background_gradient(cmap='Greens', axis=None))
                    
                    # Ví dụ tính toán cụ thể
                    p1_name = sim_df.index[0]
                    p2_name = sim_df.index[1]
                    score = sim_df.iloc[0, 1]
                    st.write(f"**Ví dụ áp dụng:** Độ tương đồng giữa *'{p1_name}'* và *'{p2_name}'* là **{score:.4f}**.")
                    if score > 0.5:
                        st.write("=> Hai sản phẩm này rất giống nhau về đặc điểm.")
                    else:
                        st.write("=> Hai sản phẩm này khá khác biệt.")

            # BƯỚC 4
            with st.expander("Bước 4: Evaluation (Tính toán chỉ số)", expanded=True):
                st.markdown('<div class="step-header">Bước 4: Đánh giá & Tính Metrics</div>', unsafe_allow_html=True)
                st.write("**Dữ liệu Test-set:** Sử dụng tập `test_interactions` (20% dữ liệu, tách theo thời gian).")
                st.write("**Quy trình:** Với mỗi user trong tập test, ẩn các sản phẩm họ đã mua, dùng mô hình gợi ý Top-K, sau đó so sánh với thực tế.")
                
                render_metrics_table(comparison_df, highlight_model="Content-Based Filtering")

        # --- GNN TAB ---
        with tab2:
            st.markdown("### 2️⃣ GNN (GraphSAGE)")
            st.markdown("**Mô tả:** Sử dụng mạng nơ-ron đồ thị để học mối quan hệ giữa User và Product.")
            
            # BƯỚC 1
            with st.expander("Bước 1: Graph Construction & Dữ liệu Train", expanded=True):
                st.markdown('<div class="step-header">Bước 1: Xây dựng đồ thị & Dữ liệu</div>', unsafe_allow_html=True)
                st.write("**Dữ liệu Train-set:** Sử dụng `train_interactions` (80% dữ liệu đầu).")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Số lượng Users (Nodes)", gnn_model.n_users)
                    st.metric("Số lượng Products (Nodes)", gnn_model.n_products)
                with col2:
                    if gnn_model.graph_data:
                        st.metric("Số lượng Cạnh (Edges)", gnn_model.graph_data.edge_index.shape[1])
                        st.metric("Feature Dimension", gnn_model.graph_data.x.shape[1])
                
                st.write("**Ma trận kề (Adjacency - Minh họa):**")
                st.write("User 1 <---[weight=1.0]---> Product A")
                st.write("User 2 <---[weight=0.7]---> Product A")
                st.info("💡 **Phân tích:** Đồ thị là Bipartite (Lưỡng phân), cạnh nối giữa User và Product thể hiện tương tác.")

            # BƯỚC 2
            with st.expander("Bước 2: Graph Convolution (GraphSAGE)"):
                st.markdown('<div class="step-header">Bước 2: Tích chập đồ thị (Graph Convolution)</div>', unsafe_allow_html=True)
                st.write("**Nội dung:** Lan truyền thông tin từ hàng xóm (Neighbors) để cập nhật Embedding cho mỗi node.")
                
                st.markdown("""
                **Công thức GraphSAGE (Mean Aggregator):**
                1. **Aggregate:** $h_{N(v)}^{(k)} = \\text{MEAN}(\\{h_u^{(k-1)}, \\forall u \\in N(v)\\})$
                2. **Update:** $h_v^{(k)} = \\sigma(W^{(k)} \\cdot \\text{CONCAT}(h_v^{(k-1)}, h_{N(v)}^{(k)}))$
                """)
                
                st.write("**Kết quả tính toán (Embeddings):**")
                if gnn_model.node_embeddings is not None:
                    emb_df = pd.DataFrame(gnn_model.node_embeddings[:5, :10]) # 5 users, 10 dims
                    st.write(f"**User Embeddings (Top 5 users, 10 chiều đầu):** Shape {gnn_model.node_embeddings.shape}")
                    st.dataframe(emb_df.style.background_gradient(cmap='Purples', axis=None))
                    st.info("💡 **Ý nghĩa:** Mỗi dòng là một vector đại diện cho sở thích của User sau khi học từ đồ thị.")

            # BƯỚC 3
            with st.expander("Bước 3: Training & Loss Function"):
                st.markdown('<div class="step-header">Bước 3: Huấn luyện với BPR Loss</div>', unsafe_allow_html=True)
                st.write("**Nội dung:** Tối ưu hóa embedding sao cho điểm của cặp (User, Item dương) lớn hơn (User, Item âm).")
                
                st.markdown("""
                **Công thức BPR Loss:**
                $$L = -\\frac{1}{|D|} \\sum_{(u,i,j) \\in D} \\ln \\sigma(\\hat{x}_{ui} - \\hat{x}_{uj})$$
                """)
                st.write(f"**Áp dụng:** Với User $u$, Item đã mua $i$, Item chưa mua $j$ (Negative Sample).")
                st.write(f"**Kết quả:** Training Loss cuối cùng = {gnn_model.training_losses[-1]:.4f}")
                st.write(f"**Thời gian huấn luyện:** {gnn_model.training_time:.2f}s")

            # BƯỚC 4
            with st.expander("Bước 4: Evaluation (Tính toán chỉ số)", expanded=True):
                st.markdown('<div class="step-header">Bước 4: Đánh giá & Tính Metrics</div>', unsafe_allow_html=True)
                st.write("**Dữ liệu Test-set:** Sử dụng tập `test_interactions`.")
                st.write("**Phương pháp:** Dot Product giữa User Embedding và Product Embedding để ra Score, sau đó Ranking.")
                
                render_metrics_table(comparison_df, highlight_model="GNN (GraphSAGE)")

        # --- HYBRID TAB ---
        with tab3:
            st.markdown("### 3️⃣ Hybrid Model (GNN + Content-Based)")
            st.markdown("**Mô tả:** Kết hợp điểm số từ GNN và Content-Based để tận dụng ưu điểm cả hai.")
            
            # BƯỚC 1
            with st.expander("Bước 1: Score Normalization (Chuẩn hóa)", expanded=True):
                st.markdown('<div class="step-header">Bước 1: Chuẩn hóa điểm số</div>', unsafe_allow_html=True)
                st.write("**Nội dung:** Đưa điểm số của GNN (thường là dot product, range rộng) và CB (cosine, 0-1) về cùng thang đo [0, 1].")
                
                st.markdown("""
                **Công thức Min-Max Scaling:**
                $$Score_{norm} = \\frac{Score - Min}{Max - Min}$$
                """)
                
                st.write("**Ví dụ minh họa:**")
                ex_data = {
                    'Product': ['P1', 'P2'],
                    'GNN Score (Raw)': [5.2, 2.1],
                    'CB Score (Raw)': [0.8, 0.3],
                    'GNN Norm': [1.0, 0.0],
                    'CB Norm': [1.0, 0.0]
                }
                st.table(pd.DataFrame(ex_data))

            # BƯỚC 2
            with st.expander("Bước 2: Weighted Combination (Kết hợp)"):
                st.markdown('<div class="step-header">Bước 2: Kết hợp có trọng số</div>', unsafe_allow_html=True)
                st.write(f"**Nội dung:** Tính điểm cuối cùng với trọng số $\\alpha = {hybrid_model.alpha}$.")
                
                st.markdown("""
                **Công thức:**
                $$Score_{final} = \\alpha \\times Score_{GNN\\_norm} + (1 - \\alpha) \\times Score_{CB\\_norm}$$
                """)
                
                st.write("**Áp dụng (với alpha=0.5):**")
                st.write("$$Score_{final}(P1) = 0.5 \\times 1.0 + 0.5 \\times 1.0 = 1.0$$")
                st.write("$$Score_{final}(P2) = 0.5 \\times 0.0 + 0.5 \\times 0.0 = 0.0$$")

            # BƯỚC 3
            with st.expander("Bước 3: Evaluation & Analysis", expanded=True):
                st.markdown('<div class="step-header">Bước 3: Đánh giá tổng hợp</div>', unsafe_allow_html=True)
                render_metrics_table(comparison_df, highlight_model="Hybrid (GNN + Content-Based)")
                
                st.markdown("### 🏆 Phân tích & Kết luận (Focus on Hybrid)")
                st.success("""
                **Tại sao Hybrid là tối ưu nhất?**
                1. **Recall & Precision:** Hybrid đạt được sự cân bằng. GNN giúp tăng Recall (tìm được sản phẩm tiềm năng user chưa từng thấy), trong khi CB giúp tăng Precision (đảm bảo sản phẩm giống sở thích cũ).
                2. **Coverage & Diversity:** Chỉ số Coverage của Hybrid thường cao hơn GNN thuần túy vì nó có thể gợi ý cả những sản phẩm ít tương tác (nhờ Content).
                3. **Khắc phục điểm yếu:** 
                   - GNN bị yếu khi User mới (Cold-start) -> CB bù đắp.
                   - CB bị yếu về độ đa dạng -> GNN bù đắp.
                """)

    # ========== PAGE 2: MODEL COMPARISON ==========
    elif page == "📊 Model Comparison":
        st.markdown('<div class="sub-header">📊 So Sánh Hiệu Suất Các Mô Hình</div>', unsafe_allow_html=True)
        
        if comparison_df is not None:
            st.dataframe(comparison_df, use_container_width=True)
            
            # Radar Chart
            metrics = ['recall@10', 'ndcg@10', 'precision@10', 'coverage@10', 'diversity@10']
            fig = go.Figure()
            for _, row in comparison_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill='toself',
                    name=row['model_name']
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar Chart: Các chỉ số chính")
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis Text
            st.markdown("### 📝 Đánh giá chi tiết")
            best_model = comparison_df.loc[comparison_df['ndcg@10'].idxmax()]['model_name']
            st.info(f"Dựa trên chỉ số quan trọng **NDCG@10**, mô hình tốt nhất là: **{best_model}**")
            
        else:
            st.warning("Vui lòng chạy tính toán ở trang 'Algorithms & Steps' trước.")

    # ========== PAGE 3: PERSONALIZED RECOMMENDATIONS ==========
    elif page == "🎯 Personalized Recommendations":
        st.markdown('<div class="sub-header">🎯 Gợi Ý Cá Nhân Hóa (Personalized)</div>', unsafe_allow_html=True)
        
        if preprocessor is None:
            st.warning("Vui lòng chạy tính toán trước.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            user_list = preprocessor.users_df[['user_idx', 'name', 'age', 'gender']].to_dict('records')
            user_options = {f"{u['name']} ({u['age']}, {u['gender']})": u['user_idx'] for u in user_list}
            selected_user = st.selectbox("Chọn User", list(user_options.keys()))
            user_idx = user_options[selected_user]
        
        with col2:
            product_list = preprocessor.products_df[['product_idx', 'productDisplayName']].to_dict('records')
            product_options = {p['productDisplayName']: p['product_idx'] for p in product_list}
            selected_product = st.selectbox("Chọn Payload Product", list(product_options.keys()))
            product_idx = product_options[selected_product]
            
        model_choice = st.radio("Chọn Model", ["Content-Based Filtering", "GNN (GraphSAGE)", "Hybrid"], horizontal=True)
        
        if st.button("🚀 Gợi ý ngay", type="primary"):
            user_info = preprocessor.get_user_info(user_idx)
            user_history = preprocessor.get_user_interaction_history(user_idx)
            
            st.write("---")
            st.markdown("#### 👤 Thông tin User & Payload")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**User:** {user_info['name']}, {user_info['age']} tuổi, {user_info['gender']}")
                st.write(f"**Lịch sử:** {len(user_history)} tương tác")
            with c2:
                payload_info = preprocessor.get_product_info(product_idx)
                display_product_info(payload_info)
            
            st.write("---")
            st.markdown(f"#### 🎯 Kết quả từ {model_choice}")
            
            with st.spinner("Đang tính toán..."):
                if model_choice == "Content-Based Filtering":
                    recs, _ = cb_model.recommend_personalized(user_info, user_history, product_idx)
                elif model_choice == "GNN (GraphSAGE)":
                    recs, _ = gnn_model.recommend_personalized(user_info, user_idx, product_idx)
                else:
                    recs, _ = hybrid_model.recommend_personalized(user_info, user_idx, user_history, product_idx)
            
            for i, (pid, score) in enumerate(recs, 1):
                with st.expander(f"#{i} - Score: {score:.4f}"):
                    display_product_info(preprocessor.get_product_info(pid), score)

    # ========== PAGE 4: OUTFIT RECOMMENDATIONS ==========
    elif page == "👗 Outfit Recommendations":
        st.markdown('<div class="sub-header">👗 Gợi Ý Trang Phục (Outfit)</div>', unsafe_allow_html=True)
        
        if preprocessor is None:
            st.stop()

        # User & Product Selection (Simplified for brevity)
        user_list = preprocessor.users_df[['user_idx', 'name']].to_dict('records')
        user_idx = st.selectbox("Chọn User", [u['user_idx'] for u in user_list], format_func=lambda x: preprocessor.get_user_info(x)['name'])
        
        product_list = preprocessor.products_df[['product_idx', 'productDisplayName']].to_dict('records')
        product_idx = st.selectbox("Chọn Payload Product", [p['product_idx'] for p in product_list], format_func=lambda x: preprocessor.get_product_info(x)['productDisplayName'])

        if st.button("✨ Tạo Outfit", type="primary"):
            user_info = preprocessor.get_user_info(user_idx)
            outfit, _ = cb_model.recommend_outfit(user_info, product_idx)
            
            st.success("Đã tạo outfit hoàn chỉnh!")
            
            cols = st.columns(3)
            categories = [
                ('Topwear', outfit['topwear']), 
                ('Bottomwear', outfit['bottomwear']), 
                ('Footwear', outfit['footwear']),
                ('Accessories', outfit['accessories']),
                ('Dress (Optional)', outfit['dress']),
                ('Innerwear (Optional)', outfit.get('innerwear', []))
            ]
            
            for idx, (cat_name, items) in enumerate(categories):
                with cols[idx % 3]:
                    st.markdown(f"#### {cat_name}")
                    if items:
                        for pid, score in items[:2]:
                            p_info = preprocessor.get_product_info(pid)
                            st.info(f"{p_info['productDisplayName']}")
                    else:
                        st.write("_Không có gợi ý_")

if __name__ == "__main__":
    main()
