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
import re

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

def compute_sparsity(df: pd.DataFrame) -> pd.Series:
    """Return sparsity (percentage of missing values) per column"""
    if df.empty:
        return pd.Series(dtype=float)
    non_null_counts = df.count()
    sparsity = 1 - (non_null_counts / len(df))
    return sparsity.sort_values(ascending=False)

def render_sparsity_chart(df: pd.DataFrame, title: str, key: str):
    """Plot sparsity bar chart"""
    sparsity = compute_sparsity(df)
    if sparsity.empty:
        st.info("Không đủ dữ liệu để tính độ thưa.")
        return
    sparsity_df = sparsity.reset_index()
    sparsity_df.columns = ['Column', 'Sparsity']
    fig = px.bar(
        sparsity_df,
        x='Column',
        y='Sparsity',
        title=title,
        labels={'Column': 'Cột', 'Sparsity': 'Độ thưa (tỉ lệ null)'}
    )
    st.plotly_chart(fig, use_container_width=True)

def render_distribution_chart(df: pd.DataFrame, dataset_key: str):
    """Plot distribution chart for selected column"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available_cols = categorical_cols + numeric_cols
    if not available_cols:
        st.info("Không có cột phù hợp để hiển thị biểu đồ tỉ lệ.")
        return
    selected_col = st.selectbox(
        "Chọn cột để hiển thị biểu đồ tỉ lệ",
        available_cols,
        key=f"{dataset_key}_distribution_column"
    )
    if selected_col in categorical_cols:
        value_counts = df[selected_col].fillna("N/A").value_counts().head(10)
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Tỉ lệ phân bố của '{selected_col}'"
        )
    else:
        numeric_series = df[selected_col].dropna()
        if numeric_series.empty:
            st.info("Cột đã chọn không có dữ liệu để vẽ biểu đồ.")
            return
        hist_data = pd.cut(numeric_series, bins=10).value_counts().sort_index()
        hist_df = hist_data.reset_index()
        hist_df.columns = ['Range', 'Count']
        hist_df['Range'] = hist_df['Range'].astype(str)
        fig = px.bar(
            hist_df,
            x='Range',
            y='Count',
            title=f"Phân bố giá trị của '{selected_col}'",
            labels={'Range': 'Khoảng giá trị', 'Count': 'Số lượng'}
        )
    st.plotly_chart(fig, use_container_width=True)

def render_data_statistics(df: pd.DataFrame):
    """Display descriptive statistics for numeric columns"""
    if df.empty:
        st.info("Dataset trống, không thể thống kê.")
        return
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.info("Không có cột số để thống kê.")
        return
    stats_df = numeric_df.describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    st.dataframe(stats_df, use_container_width=True)

def render_dataset_upload_section(
    dataset_key: str,
    display_name: str,
    purpose_text: str
):
    """Render upload + analytics UI for a dataset"""
    st.markdown(f"#### {display_name}")
    st.write(purpose_text)
    uploaded_file = st.file_uploader(
        f"Tải lên {display_name}",
        type=['csv'],
        key=f"{dataset_key}_file_uploader"
    )
    if uploaded_file is None:
        st.info("Chưa có file được tải lên.")
        return
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Lỗi khi đọc file CSV: {exc}")
        return
    st.success(f"Đã tải {display_name}: {len(df)} rows × {len(df.columns)} columns")
    col_rows, col_cols = st.columns(2)
    with col_rows:
        st.metric("Số dòng (rows)", len(df))
    with col_cols:
        st.metric("Số cột (columns)", len(df.columns))
    st.markdown("**👀 Xem trước dữ liệu (tối đa 100 dòng đầu):**")
    st.dataframe(df.head(100), use_container_width=True)
    st.markdown("**📉 Biểu đồ độ thưa (tỉ lệ giá trị null trên mỗi cột):**")
    render_sparsity_chart(df, f"Độ thưa - {display_name}", dataset_key)
    st.markdown("**📊 Biểu đồ tỉ lệ / phân bố:**")
    render_distribution_chart(df, dataset_key)
    st.markdown("**📈 Bảng thống kê dữ liệu (count, mean, std, min, 25%, 50%, 75%, max):**")
    render_data_statistics(df)

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
    
    # Format dataframe - chỉ lấy các cột cần thiết
    required_cols = ['model_name', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20', 
                     'precision@10', 'precision@20', 'training_time', 'avg_inference_time',
                     'coverage@10', 'diversity@10']
    
    # Tạo display_df với các cột cần thiết
    display_df = df.copy()
    available_cols = [col for col in required_cols if col in display_df.columns]
    display_df = display_df[available_cols]
    
    # Rename columns for better display
    column_mapping = {
        'model_name': 'Model',
        'recall@10': 'Recall@10',
        'recall@20': 'Recall@20',
        'ndcg@10': 'NDCG@10',
        'ndcg@20': 'NDCG@20',
        'precision@10': 'Precision@10',
        'precision@20': 'Precision@20',
        'training_time': 'Training Time (s)',
        'avg_inference_time': 'Inference Time (s)',
        'coverage@10': 'Coverage@10',
        'diversity@10': 'Diversity@10'
    }
    display_df = display_df.rename(columns=column_mapping)
    
    # Format numeric columns
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    
    def highlight_row(row):
        model_name = row.get('Model', '')
        if model_name == highlight_model:
            return ['background-color: #e6ffe6'] * len(row)
        return [''] * len(row)

    st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True)


def slugify_model_name(model_name: str) -> str:
    """Convert model name to slug used for log filenames."""
    return re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')


def load_evaluation_log(model_name: str):
    """Load evaluation log content for a model."""
    slug = slugify_model_name(model_name)
    log_path = os.path.join('recommendation_system', 'evaluation', 'logs', f'{slug}.log')
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            return slug, f.read()
    return slug, None


def parse_evaluation_log(log_text: str) -> Dict:
    """
    Parse evaluation log để extract metrics và ví dụ tính toán
    
    Returns:
        Dictionary chứa:
        - metrics: Dict với các metrics và giá trị
        - examples: Dict với các ví dụ tính toán cho từng metric
        - formulas: Dict với các công thức cho từng metric
    """
    if not log_text:
        return {'metrics': {}, 'examples': {}, 'formulas': {}}
    
    metrics = {}
    examples = {}
    formulas = {}
    
    lines = log_text.split('\n')
    i = 0
    current_metric = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and headers
        if not line or line.startswith('===') or line.startswith('[') or 'EVALUATING' in line or 'RESULTS FOR' in line:
            i += 1
            continue
        
        # Parse metric value (format: "metric_name: value")
        # Look for pattern like "recall@10: 0.0186" or "training_time: 0.0807"
        if ':' in line and not line.startswith('📐') and not line.startswith('🧮'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                metric_name = parts[0].strip()
                value_str = parts[1].strip()
                
                # Remove any trailing text after the number
                # e.g., "0.0186   📐 Công thức:" -> "0.0186"
                value_str = value_str.split()[0] if value_str.split() else value_str
                
                # Try to parse as float
                try:
                    value = float(value_str)
                    metrics[metric_name] = value
                    current_metric = metric_name
                except ValueError:
                    pass
        
        # Parse formula (format: "   📐 Công thức: ...")
        if '📐 Công thức:' in line:
            formula = line.split('📐 Công thức:', 1)[1].strip()
            if current_metric:
                formulas[current_metric] = formula
        
        if 'Ví dụ áp dụng:' in line:
            example = line.split('Ví dụ áp dụng:', 1)[1].strip()
            if current_metric:
                examples[current_metric] = example
        
        i += 1
    
    return {
        'metrics': metrics,
        'examples': examples,
        'formulas': formulas
    }


def render_metrics_in_step(
    metrics_data,
    metric_keys: List[str],
    step_title: str,
    key_suffix: str,
    model_name: str = None
):
    """
    Hiển thị metrics chi tiết trong một bước
    
    Args:
        metrics_data: Dictionary từ parse_evaluation_log hoặc pd.Series từ comparison_df
        metric_keys: List các metric keys cần hiển thị (e.g., ['recall@10', 'precision@10'])
        step_title: Tiêu đề của bước
        key_suffix: Suffix cho key của Streamlit components
        model_name: Tên model (để load log nếu cần)
    """
    # Kiểm tra metrics_data một cách an toàn (tránh lỗi với pandas Series)
    if metrics_data is None:
        st.info("Chưa có dữ liệu metrics. Vui lòng chạy train & evaluate trước.")
        return
    elif isinstance(metrics_data, pd.Series):
        if metrics_data.empty:
            st.info("Chưa có dữ liệu metrics. Vui lòng chạy train & evaluate trước.")
            return
    elif isinstance(metrics_data, dict):
        if not metrics_data or (isinstance(metrics_data, dict) and 'metrics' in metrics_data and not metrics_data['metrics']):
            st.info("Chưa có dữ liệu metrics. Vui lòng chạy train & evaluate trước.")
            return
    
    # Load parsed log để lấy formulas và examples (nếu chưa có trong metrics_data)
    parsed_log = None
    if model_name:
        _, log_text = load_evaluation_log(model_name)
        if log_text:
            parsed_log = parse_evaluation_log(log_text)
    
    # Tạo columns cho metrics (2 cột)
    n_cols = 2
    cols = st.columns(n_cols)
    
    for idx, metric_key in enumerate(metric_keys):
        col_idx = idx % n_cols
        with cols[col_idx]:
            # Get metric value, formula, and example
            value = None
            formula = ''
            example = ''
            
            if isinstance(metrics_data, dict) and 'metrics' in metrics_data:
                # From parsed log
                value = metrics_data['metrics'].get(metric_key, None)
                formula = metrics_data['formulas'].get(metric_key, '')
                example = metrics_data['examples'].get(metric_key, '')
            elif isinstance(metrics_data, pd.Series):
                # From comparison_df
                value = metrics_data.get(metric_key, None)
                # Get formula and example from parsed log if available
                if parsed_log:
                    formula = parsed_log['formulas'].get(metric_key, '')
                    example = parsed_log['examples'].get(metric_key, '')
            
            if value is not None:
                # Format metric name for display
                display_name = metric_key.replace('@', '@').replace('_', ' ').title()
                
                # Display metric
                st.metric(display_name, f"{value:.4f}")
                
                # Show formula and example in expander
                with st.expander(f"Chi tiết {display_name}", expanded=False):
                    if formula:
                        st.markdown(f"**Công thức:** {formula}")
                    
                    if example:
                        # Phân tích example để hiển thị rõ ràng hơn
                        if "| Trung bình" in example:
                            # Tách example thành các phần
                            parts = example.split(" | ")
                            user_examples = []
                            avg_formula = None
                            
                            for part in parts:
                                if "Trung bình" in part:
                                    avg_formula = part
                                else:
                                    user_examples.append(part)
                            
                            # Hiển thị ví dụ tính toán cho từng user
                            st.markdown("#### Ví dụ tính toán cho từng user:")
                            for i, user_ex in enumerate(user_examples, 1):
                                st.markdown(f"**{i}. {user_ex}**")
                            
                            if avg_formula:
                                st.markdown("#### Công thức tính trung bình:")
                                
                                # Parse công thức để hiển thị đẹp hơn
                                if "=" in avg_formula:
                                    formula_parts = avg_formula.split("=")
                                    if len(formula_parts) >= 2:
                                        left_side = formula_parts[0].strip()
                                        right_side = "=".join(formula_parts[1:]).strip()
                                        
                                        # Extract số users từ công thức
                                        import re
                                        n_users_match = re.search(r'user(\d+)', right_side)
                                        n_users = n_users_match.group(1) if n_users_match else "N"
                                        
                                        # Extract metric name từ display_name
                                        metric_var = display_name.replace(" ", "_").lower()
                                        
                                        # Hiển thị công thức dạng toán học
                                        st.markdown(f"""
                                        **Công thức:**
                                        $$\\text{{Trung bình}} = \\frac{{\\sum_{{u=1}}^{{{n_users}}} {display_name}_u}}{{{n_users}}}$$
                                        """)
                                        
                                        # Hiển thị dạng mở rộng
                                        st.markdown(f"""
                                        **Dạng mở rộng:**
                                        $$\\text{{Trung bình}} = \\frac{{{display_name}_{{user1}} + {display_name}_{{user2}} + \\ldots + {display_name}_{{user{n_users}}}}}{{{n_users}}}$$
                                        """)
                                        
                                        # Hiển thị với giá trị cụ thể từ ví dụ
                                        if len(user_examples) >= 1:
                                            # Lấy giá trị từ các ví dụ users
                                            example_values = []
                                            for ex in user_examples:
                                                # Extract giá trị từ ví dụ
                                                # Format có thể là: "User X: hits=Y, |T_u|=Z → recall=0.0186"
                                                # hoặc: "User X: DCG=Y, IDCG=Z → NDCG=0.0575"
                                                # hoặc: "User X: hits=Y, K=Z → precision=0.0100"
                                                
                                                # Tìm pattern: "→ metric_name=value" (giá trị sau dấu →)
                                                # Hoặc tìm giá trị cuối cùng trong chuỗi (sau dấu = cuối cùng)
                                                pattern1 = r'→\s*\w+\s*=\s*([\d.]+)'  # "→ recall=0.0186" hoặc "→ NDCG=0.0575"
                                                match1 = re.search(pattern1, ex)
                                                
                                                if match1:
                                                    val = float(match1.group(1))
                                                    example_values.append(val)
                                                else:
                                                    # Fallback: tìm tất cả các giá trị số và lấy giá trị cuối cùng
                                                    # (thường là metric value)
                                                    all_numbers = re.findall(r'([\d.]+)', ex)
                                                    if all_numbers:
                                                        # Lấy số cuối cùng (thường là metric value)
                                                        val = float(all_numbers[-1])
                                                        example_values.append(val)
                                            
                                            if example_values:
                                                n_examples = len(example_values)
                                                # Hiển thị ví dụ với các users
                                                example_text = f"**Ví dụ với {n_examples} user(s):**\n"
                                                for i, (ex, val) in enumerate(zip(user_examples[:3], example_values[:3]), 1):
                                                    # Extract user number
                                                    user_match = re.search(r'User\s+(\d+)', ex)
                                                    user_num = user_match.group(1) if user_match else str(i)
                                                    example_text += f"- {display_name}_user{user_num} = {val:.4f}\n"
                                                
                                                if n_examples < int(n_users):
                                                    example_text += f"- ...\n"
                                                    example_text += f"- {display_name}_user{n_users} = ...\n"
                                                
                                                # Tạo công thức với ví dụ
                                                if n_examples >= 2:
                                                    sum_example = sum(example_values[:2])
                                                    formula_example = f"{example_values[0]:.4f} + {example_values[1]:.4f}"
                                                    if n_examples > 2:
                                                        formula_example += f" + {example_values[2]:.4f}"
                                                    formula_example += f" + \\ldots"
                                                else:
                                                    formula_example = f"{example_values[0]:.4f} + \\ldots"
                                                
                                                st.markdown(example_text)
                                                st.markdown(f"""
                                                **Tính toán:**
                                                $$\\text{{Trung bình}} = \\frac{{{formula_example} + {display_name}_{{user{n_users}}}}}{{{n_users}}} = {value:.4f}$$
                                                """)
                        else:
                            st.markdown(f"**Ví dụ áp dụng:** {example}")
                    
                    if not formula and not example:
                        st.info("Chưa có chi tiết tính toán. Xem log evaluation để biết thêm.")
            else:
                # Metric không có trong data
                display_name = metric_key.replace('@', '@').replace('_', ' ').title()
                st.info(f"{display_name}: Chưa có dữ liệu")


def render_evaluation_log_section(model_name: str, key_suffix: str):
    """Display evaluation logs (if any) inside an expander."""
    slug, log_text = load_evaluation_log(model_name)
    with st.expander("📜 Evaluation Log (Raw)", expanded=False):
        if log_text:
            st.text_area(
                "Chi tiết log tính toán",
                log_text,
                height=320,
                key=f"log_text_{key_suffix}"
            )
            st.download_button(
                "⬇️ Tải log",
                log_text,
                file_name=f"{slug}.log",
                mime="text/plain",
                key=f"log_download_{key_suffix}"
            )
        else:
            st.info("Chưa có log evaluation. Hãy chạy train & evaluate để tạo log.")


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
        st.markdown("### Upload & Khám Phá Bộ Dữ Liệu")
        dataset_sections = [
            (
                "users",
                "users.csv",
                "Chứa thông tin hồ sơ người dùng (tuổi, giới tính, thị hiếu) dùng để cá nhân hóa gợi ý và theo dõi hành vi."
            ),
            (
                "products",
                "products.csv",
                "Danh sách toàn bộ sản phẩm (category, màu sắc, usage...) dùng cho Content-Based và visualization."
            ),
            (
                "interactions",
                "interactions.csv",
                "Log tương tác user-product (purchase/cart/like) làm đầu vào huấn luyện GNN & đánh giá."
            )
        ]
        for ds_key, ds_name, ds_desc in dataset_sections:
            render_dataset_upload_section(ds_key, ds_name, ds_desc)
        
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
                
                # Thông tin dữ liệu
                col_data1, col_data2 = st.columns(2)
                with col_data1:
                    st.metric("Tổng số sản phẩm", len(preprocessor.products_df))
                    st.metric("Tổng số users", len(preprocessor.users_df))
                with col_data2:
                    st.metric("Train interactions", len(preprocessor.train_interactions))
                    st.metric("Test interactions", len(preprocessor.test_interactions))
                
                st.write(f"**Dữ liệu sử dụng:** Toàn bộ {len(preprocessor.products_df)} sản phẩm trong `products.csv` (không phân chia train/test vì đây là dữ liệu sản phẩm tĩnh).")
                
                st.markdown("""
                **Công thức áp dụng:**
                $$Text(P_i) = [Gender] + [MasterCategory] + [SubCategory] \\times 2 + [ArticleType] \\times 3 + [BaseColour] + [Usage]$$
                
                **Giải thích:** Các features được kết hợp thành chuỗi văn bản, trong đó:
                - `ArticleType` được lặp lại **3 lần** (trọng số cao nhất - quan trọng nhất)
                - `SubCategory` được lặp lại **2 lần** (trọng số trung bình)
                - Các features khác (Gender, MasterCategory, BaseColour, Usage) xuất hiện **1 lần**
                
                **Lý do:** Việc lặp lại giúp TF-IDF coi trọng các features quan trọng hơn khi tính toán similarity.
                """)
                
                st.write("**Ví dụ áp dụng công thức:**")
                if len(cb_model.products_df) > 0:
                    example_product = cb_model.products_df.iloc[0]
                    st.write(f"- **Sản phẩm:** {example_product.get('productDisplayName', 'N/A')}")
                    st.write(f"- Gender: {example_product.get('gender', 'N/A')}")
                    st.write(f"- MasterCategory: {example_product.get('masterCategory', 'N/A')}")
                    st.write(f"- SubCategory: {example_product.get('subCategory', 'N/A')} (x2)")
                    st.write(f"- ArticleType: {example_product.get('articleType', 'N/A')} (x3)")
                    st.write(f"- BaseColour: {example_product.get('baseColour', 'N/A')}")
                    st.write(f"- Usage: {example_product.get('usage', 'N/A')}")
                
                st.write("**Kết quả tính toán (Ví dụ 2 sản phẩm đầu tiên):**")
                example_df = cb_model.products_df[['productDisplayName', 'feature_text']].head(2)
                st.dataframe(example_df, use_container_width=True)
                st.info("💡 **Phân tích:** Việc lặp lại `ArticleType` 3 lần giúp thuật toán coi trọng loại sản phẩm hơn màu sắc. Điều này giúp gợi ý sản phẩm cùng loại (ví dụ: áo thun với áo thun) thay vì chỉ dựa vào màu sắc.")

            # BƯỚC 2
            with st.expander("Bước 2: Vectorization (TF-IDF) & Ma trận"):
                st.markdown('<div class="step-header">Bước 2: Vectorization</div>', unsafe_allow_html=True)
                st.write("**Nội dung thực hiện:** Chuyển đổi văn bản thành vector số học sử dụng TF-IDF.")
                st.write(f"**Dữ liệu sử dụng:** Toàn bộ {len(cb_model.products_df)} sản phẩm từ Bước 1 (feature_text).")
                
                st.markdown("""
                **Công thức TF-IDF:**
                $$TF(t, d) = \\frac{count(t, d)}{len(d)}, \\quad IDF(t) = \\log(\\frac{N}{df(t)}), \\quad TF\\text{-}IDF = TF \\times IDF$$
                
                Trong đó:
                - $TF(t, d)$: Tần suất từ $t$ trong document $d$
                - $IDF(t)$: Nghịch đảo tần suất document, đo độ hiếm của từ $t$
                - $N$: Tổng số documents (sản phẩm)
                - $df(t)$: Số documents chứa từ $t$
                """)
                
                if cb_model.tfidf_vectorizer is not None:
                    feature_names = cb_model.tfidf_vectorizer.get_feature_names_out()
                    # Lấy vector của 5 sản phẩm đầu tiên
                    tfidf_subset = cb_model.tfidf_vectorizer.transform(cb_model.products_df['feature_text'].head(5))
                    tfidf_df = pd.DataFrame(tfidf_subset.toarray(), columns=feature_names, index=cb_model.products_df['productDisplayName'].head(5))
                    
                    st.write(f"**Ma trận TF-IDF (Top 5 sản phẩm x Top 10 features):**")
                    st.write(f"**Shape:** {tfidf_subset.shape[0]} sản phẩm × {tfidf_subset.shape[1]} features")
                    st.dataframe(tfidf_df.iloc[:, :10].style.background_gradient(cmap='Blues', axis=None), use_container_width=True)
                    
                    if len(tfidf_df) >= 2:
                        p1_name = tfidf_df.index[0]
                        p2_name = tfidf_df.index[1]
                        # Tìm feature có giá trị cao nhất cho mỗi sản phẩm
                        top_feature_p1 = tfidf_df.loc[p1_name].nlargest(1).index[0]
                        top_value_p1 = tfidf_df.loc[p1_name, top_feature_p1]
                        st.write(f"**Ví dụ áp dụng:** Sản phẩm *'{p1_name}'* có feature *'{top_feature_p1}'* với TF-IDF score = **{top_value_p1:.4f}** (cao nhất).")
                    
                    st.info(f"💡 **Ý nghĩa:** Giá trị càng cao (đậm) nghĩa là từ khóa đó càng đặc trưng cho sản phẩm. Ma trận thưa (nhiều số 0) vì mỗi sản phẩm chỉ có một số features nhất định.")

            # BƯỚC 3
            with st.expander("Bước 3: Similarity Calculation & Ví dụ tính toán"):
                st.markdown('<div class="step-header">Bước 3: Tính độ tương đồng</div>', unsafe_allow_html=True)
                st.write("**Nội dung thực hiện:** Tính Cosine Similarity giữa tất cả các cặp sản phẩm dựa trên TF-IDF vectors.")
                st.write(f"**Dữ liệu sử dụng:** Ma trận TF-IDF từ Bước 2 ({cb_model.similarity_matrix.shape[0]} × {cb_model.similarity_matrix.shape[1]}).")
                
                st.markdown("""
                **Công thức Cosine Similarity:**
                $$Cosine(A, B) = \\frac{A \\cdot B}{||A|| \\times ||B||} = \\frac{\\sum_{i=1}^{n} A_i B_i}{\\sqrt{\\sum_{i=1}^{n} A_i^2} \\sqrt{\\sum_{i=1}^{n} B_i^2}}$$
                
                Trong đó:
                - $A, B$: Hai vector TF-IDF của 2 sản phẩm
                - $A_i, B_i$: Giá trị TF-IDF của feature thứ $i$
                - Kết quả: Giá trị từ 0 đến 1 (1 = giống nhau hoàn toàn, 0 = khác biệt hoàn toàn)
                """)
                
                if cb_model.similarity_matrix is not None:
                    # Lấy ma trận similarity nhỏ (5x5)
                    sim_subset = cb_model.similarity_matrix[:5, :5]
                    sim_df = pd.DataFrame(sim_subset, 
                                        index=cb_model.products_df['productDisplayName'].head(5),
                                        columns=cb_model.products_df['productDisplayName'].head(5))
                    
                    st.write(f"**Ma trận Similarity (5×5 mẫu từ ma trận {cb_model.similarity_matrix.shape[0]}×{cb_model.similarity_matrix.shape[1]}):**")
                    st.dataframe(sim_df.style.background_gradient(cmap='Greens', axis=None), use_container_width=True)
                    
                    # Thống kê
                    avg_sim = cb_model.similarity_matrix.mean()
                    max_sim = cb_model.similarity_matrix.max()
                    min_sim = cb_model.similarity_matrix.min()
                    st.write(f"**Thống kê ma trận:**")
                    st.write(f"- Độ tương đồng trung bình: {avg_sim:.4f}")
                    st.write(f"- Độ tương đồng cao nhất: {max_sim:.4f} (sản phẩm với chính nó)")
                    st.write(f"- Độ tương đồng thấp nhất: {min_sim:.4f}")
                    
                    # Ví dụ tính toán cụ thể
                    p1_name = sim_df.index[0]
                    p2_name = sim_df.index[1]
                    score = sim_df.iloc[0, 1]
                    st.write(f"**Ví dụ áp dụng:** Độ tương đồng giữa *'{p1_name}'* và *'{p2_name}'* là **{score:.4f}**.")
                    if score > 0.5:
                        st.write("=> Hai sản phẩm này rất giống nhau về đặc điểm (có thể cùng loại, màu sắc, hoặc mục đích sử dụng).")
                    elif score > 0.3:
                        st.write("=> Hai sản phẩm này có một số điểm tương đồng.")
                    else:
                        st.write("=> Hai sản phẩm này khá khác biệt về đặc điểm.")

            # BƯỚC 4
            with st.expander("Bước 4: Evaluation (Tính toán chỉ số)", expanded=True):
                st.markdown('<div class="step-header">Bước 4: Đánh giá & Tính Metrics</div>', unsafe_allow_html=True)
                
                # Thông tin dữ liệu
                st.write("**Dữ liệu sử dụng:**")
                col_eval1, col_eval2 = st.columns(2)
                with col_eval1:
                    st.metric("Train-set", f"{len(preprocessor.train_interactions)} interactions")
                    st.write(f"- Users: {preprocessor.train_interactions['user_idx'].nunique()}")
                    st.write(f"- Products: {preprocessor.train_interactions['product_idx'].nunique()}")
                with col_eval2:
                    st.metric("Test-set", f"{len(preprocessor.test_interactions)} interactions")
                    st.write(f"- Users: {preprocessor.test_interactions['user_idx'].nunique()}")
                    st.write(f"- Products: {preprocessor.test_interactions['product_idx'].nunique()}")
                
                st.write("**Quy trình đánh giá:**")
                st.write("1. Với mỗi user trong test-set, ẩn các sản phẩm họ đã tương tác (purchase/cart/like)")
                st.write("2. Dùng mô hình gợi ý Top-K sản phẩm cho user đó")
                st.write("3. So sánh danh sách gợi ý với ground truth (sản phẩm thực tế user đã tương tác)")
                st.write("4. Tính các chỉ số metrics dựa trên kết quả so sánh")
                
                # Load parsed log data
                _, log_text = load_evaluation_log("Content-Based Filtering")
                parsed_log = parse_evaluation_log(log_text) if log_text else {}
                
                # Get metrics from comparison_df
                cb_metrics_row = None
                if comparison_df is not None:
                    cb_rows = comparison_df[comparison_df['model_name'] == 'Content-Based Filtering']
                    if len(cb_rows) > 0:
                        cb_metrics_row = cb_rows.iloc[0]
                
                # Hiển thị Training Time và Inference Time
                st.markdown("#### ⏱️ Thời gian Training & Inference")
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    training_time = parsed_log['metrics'].get('training_time', 
                        cb_metrics_row['training_time'] if cb_metrics_row is not None else None)
                    if training_time is not None:
                        st.metric("Training Time (s)", f"{training_time:.4f}")
                with col_time2:
                    inference_time = parsed_log['metrics'].get('avg_inference_time',
                        cb_metrics_row['avg_inference_time'] if cb_metrics_row is not None else None)
                    if inference_time is not None:
                        st.metric("Inference Time (s)", f"{inference_time:.4f}")
                
                # Hiển thị metrics @10
                st.markdown("#### 📈 Metrics @10")
                metrics_10 = ['recall@10', 'precision@10', 'ndcg@10', 'coverage@10', 'diversity@10']
                if cb_metrics_row is not None:
                    render_metrics_in_step(cb_metrics_row, metrics_10, "Bước 4", "cb_10", model_name="Content-Based Filtering")
                elif parsed_log:
                    render_metrics_in_step(parsed_log, metrics_10, "Bước 4", "cb_10", model_name="Content-Based Filtering")
                
                # Hiển thị metrics @20
                st.markdown("#### 📈 Metrics @20")
                metrics_20 = ['recall@20', 'precision@20', 'ndcg@20', 'coverage@20', 'diversity@20']
                if cb_metrics_row is not None:
                    render_metrics_in_step(cb_metrics_row, metrics_20, "Bước 4", "cb_20", model_name="Content-Based Filtering")
                elif parsed_log:
                    render_metrics_in_step(parsed_log, metrics_20, "Bước 4", "cb_20", model_name="Content-Based Filtering")
                
                st.markdown("#### 📊 Bảng Tổng Hợp")
                render_metrics_table(comparison_df, highlight_model="Content-Based Filtering")
                render_evaluation_log_section("Content-Based Filtering", "cb")

        # --- GNN TAB ---
        with tab2:
            st.markdown("### 2️⃣ GNN (GraphSAGE)")
            st.markdown("**Mô tả:** Sử dụng mạng nơ-ron đồ thị để học mối quan hệ giữa User và Product.")
            
            # BƯỚC 1
            with st.expander("Bước 1: Graph Construction & Dữ liệu Train", expanded=True):
                st.markdown('<div class="step-header">Bước 1: Xây dựng đồ thị & Dữ liệu</div>', unsafe_allow_html=True)
                st.write("**Nội dung thực hiện:** Xây dựng đồ thị lưỡng phân (Bipartite Graph) từ interactions giữa users và products.")
                
                # Thông tin dữ liệu
                st.write("**Dữ liệu Train-set:** Sử dụng `train_interactions` (80% dữ liệu đầu, tách theo thời gian).")
                col_data1, col_data2 = st.columns(2)
                with col_data1:
                    st.metric("Train interactions", len(preprocessor.train_interactions))
                    st.metric("Số lượng Users (Nodes)", gnn_model.n_users)
                    st.metric("Số lượng Products (Nodes)", gnn_model.n_products)
                with col_data2:
                    if gnn_model.graph_data:
                        st.metric("Số lượng Cạnh (Edges)", gnn_model.graph_data.edge_index.shape[1])
                        st.metric("Feature Dimension", gnn_model.graph_data.x.shape[1])
                        st.metric("Tổng số Nodes", gnn_model.graph_data.x.shape[0])
                
                st.write("**Cấu trúc đồ thị:**")
                st.write("- **Loại:** Bipartite Graph (đồ thị lưỡng phân)")
                st.write("- **Nodes:** Users + Products")
                st.write("- **Edges:** Tương tác giữa User và Product (purchase, cart, like)")
                st.write("- **Edge Weights:** Độ mạnh của tương tác (1.0 cho purchase, 0.7 cho cart, 0.5 cho like)")
                
                st.write("**Ma trận kề (Adjacency - Minh họa):**")
                st.code("""
User 1 <---[weight=1.0]---> Product A
User 2 <---[weight=0.7]---> Product A
User 1 <---[weight=0.5]---> Product B
                """)
                
                if gnn_model.graph_data:
                    # Tính toán một số thống kê
                    n_edges = gnn_model.graph_data.edge_index.shape[1]
                    n_nodes = gnn_model.graph_data.x.shape[0]
                    avg_degree = (n_edges * 2) / n_nodes if n_nodes > 0 else 0
                    st.write(f"**Thống kê đồ thị:**")
                    st.write(f"- Số cạnh trung bình mỗi node: {avg_degree:.2f}")
                    st.write(f"- Mật độ đồ thị: {(n_edges / (n_nodes * (n_nodes - 1))) * 100:.4f}%")
                
                st.info("💡 **Phân tích:** Đồ thị là Bipartite (Lưỡng phân), cạnh nối giữa User và Product thể hiện tương tác. GraphSAGE sẽ học embedding cho mỗi node dựa trên thông tin từ các hàng xóm (neighbors).")

            # BƯỚC 2
            with st.expander("Bước 2: Graph Convolution (GraphSAGE)"):
                st.markdown('<div class="step-header">Bước 2: Tích chập đồ thị (Graph Convolution)</div>', unsafe_allow_html=True)
                st.write("**Nội dung:** Lan truyền thông tin từ hàng xóm (Neighbors) để cập nhật Embedding cho mỗi node.")
                st.write(f"**Dữ liệu sử dụng:** Đồ thị từ Bước 1 với {gnn_model.graph_data.x.shape[0]} nodes và {gnn_model.graph_data.edge_index.shape[1]} edges.")
                
                st.markdown("""
                **Công thức GraphSAGE (Mean Aggregator):**
                
                **Bước 1 - Aggregate (Tổng hợp thông tin từ neighbors):**
                $$h_{N(v)}^{(k)} = \\frac{1}{|N(v)|} \\sum_{u \\in N(v)} h_u^{(k-1)}$$
                
                **Bước 2 - Update (Cập nhật embedding):**
                $$h_v^{(k)} = \\sigma\\left(W^{(k)} \\cdot \\text{CONCAT}(h_v^{(k-1)}, h_{N(v)}^{(k)})\\right)$$
                
                Trong đó:
                - $h_v^{(k)}$: Embedding của node $v$ ở layer $k$
                - $N(v)$: Tập neighbors của node $v$
                - $W^{(k)}$: Ma trận trọng số ở layer $k$
                - $\\sigma$: Hàm activation (ReLU)
                - $\\text{CONCAT}$: Nối vector hiện tại với vector tổng hợp từ neighbors
                """)
                
                st.write("**Ví dụ áp dụng:**")
                st.write("1. User A có neighbors: Product X, Product Y, Product Z")
                st.write("2. Aggregate: Lấy trung bình embeddings của X, Y, Z")
                st.write("3. Update: Nối embedding hiện tại của User A với vector tổng hợp, sau đó nhân với ma trận trọng số và áp dụng ReLU")
                st.write("4. Kết quả: Embedding mới của User A phản ánh sở thích dựa trên các sản phẩm đã tương tác")
                
                st.write("**Kết quả tính toán (Embeddings):**")
                if gnn_model.node_embeddings is not None:
                    emb_df = pd.DataFrame(gnn_model.node_embeddings[:5, :10]) # 5 users, 10 dims
                    st.write(f"**User Embeddings (Top 5 users, 10 chiều đầu):** Shape {gnn_model.node_embeddings.shape}")
                    st.write(f"- Tổng số embeddings: {gnn_model.node_embeddings.shape[0]} (Users + Products)")
                    st.write(f"- Dimension mỗi embedding: {gnn_model.node_embeddings.shape[1]}")
                    st.dataframe(emb_df.style.background_gradient(cmap='Purples', axis=None), use_container_width=True)
                    
                    # Thống kê
                    avg_emb = gnn_model.node_embeddings.mean()
                    std_emb = gnn_model.node_embeddings.std()
                    st.write(f"**Thống kê embeddings:**")
                    st.write(f"- Giá trị trung bình: {avg_emb:.4f}")
                    st.write(f"- Độ lệch chuẩn: {std_emb:.4f}")
                    
                    st.info("💡 **Ý nghĩa:** Mỗi dòng là một vector đại diện cho sở thích của User (hoặc đặc trưng của Product) sau khi học từ đồ thị. Các users có sở thích tương tự sẽ có embeddings gần nhau trong không gian vector.")

            # BƯỚC 3
            with st.expander("Bước 3: Training & Loss Function"):
                st.markdown('<div class="step-header">Bước 3: Huấn luyện với BPR Loss</div>', unsafe_allow_html=True)
                st.write("**Nội dung:** Tối ưu hóa embedding sao cho điểm của cặp (User, Item dương) lớn hơn (User, Item âm).")
                st.write(f"**Dữ liệu sử dụng:** Train-set với {len(preprocessor.train_interactions)} interactions.")
                
                st.markdown("""
                **Công thức BPR Loss (Bayesian Personalized Ranking):**
                $$L = -\\frac{1}{|D|} \\sum_{(u,i,j) \\in D} w_{ui} \\cdot \\ln \\sigma(\\hat{x}_{ui} - \\hat{x}_{uj})$$
                
                Trong đó:
                - $D$: Tập các triplets $(u, i, j)$
                - $u$: User
                - $i$: Item dương (user đã tương tác)
                - $j$: Item âm (user chưa tương tác, negative sample)
                - $w_{ui}$: Trọng số của interaction (1.0 cho purchase, 0.7 cho cart, 0.5 cho like)
                - $\\hat{x}_{ui} = h_u \\cdot h_i$: Điểm dự đoán (dot product của embeddings)
                - $\\sigma$: Sigmoid function
                
                **Ý nghĩa:** Loss càng nhỏ nghĩa là model càng phân biệt tốt giữa items user thích và không thích. Weighted loss giúp model coi trọng các tương tác mạnh hơn (purchase > cart > like).
                """)
                
                st.write("**Ví dụ áp dụng:**")
                st.write("1. User A đã mua Product X (positive)")
                st.write("2. Random chọn Product Y mà User A chưa mua (negative)")
                st.write("3. Tính: $score_{AX} = embedding_A \\cdot embedding_X$")
                st.write("4. Tính: $score_{AY} = embedding_A \\cdot embedding_Y$")
                st.write("5. Loss = $-\\ln(\\sigma(score_{AX} - score_{AY}))$")
                st.write("6. Mục tiêu: $score_{AX} > score_{AY}$ (User A thích X hơn Y)")
                
                if gnn_model.training_losses:
                    st.write(f"**Kết quả training:**")
                    st.write(f"- Training Loss cuối cùng: {gnn_model.training_losses[-1]:.4f}")
                    st.write(f"- Training Loss ban đầu: {gnn_model.training_losses[0]:.4f}")
                    st.write(f"- Cải thiện: {((gnn_model.training_losses[0] - gnn_model.training_losses[-1]) / gnn_model.training_losses[0] * 100):.2f}%")
                    st.write(f"- Thời gian huấn luyện: {gnn_model.training_time:.2f}s")
                    st.write(f"- Số epochs: {len(gnn_model.training_losses)}")
                    
                    # Vẽ biểu đồ loss nếu có
                    if len(gnn_model.training_losses) > 1:
                        loss_df = pd.DataFrame({
                            'Epoch': range(1, len(gnn_model.training_losses) + 1),
                            'Loss': gnn_model.training_losses
                        })
                        fig = px.line(loss_df, x='Epoch', y='Loss', title='Training Loss Over Time')
                        st.plotly_chart(fig, use_container_width=True)

            # BƯỚC 4
            with st.expander("Bước 4: Evaluation (Tính toán chỉ số)", expanded=True):
                st.markdown('<div class="step-header">Bước 4: Đánh giá & Tính Metrics</div>', unsafe_allow_html=True)
                
                # Thông tin dữ liệu
                st.write("**Dữ liệu sử dụng:**")
                col_eval1, col_eval2 = st.columns(2)
                with col_eval1:
                    st.metric("Train-set", f"{len(preprocessor.train_interactions)} interactions")
                    st.write(f"- Users: {preprocessor.train_interactions['user_idx'].nunique()}")
                    st.write(f"- Products: {preprocessor.train_interactions['product_idx'].nunique()}")
                with col_eval2:
                    st.metric("Test-set", f"{len(preprocessor.test_interactions)} interactions")
                    st.write(f"- Users: {preprocessor.test_interactions['user_idx'].nunique()}")
                    st.write(f"- Products: {preprocessor.test_interactions['product_idx'].nunique()}")
                
                st.write("**Phương pháp dự đoán:**")
                st.markdown("""
                **Công thức tính điểm:**
                $$\\hat{x}_{ui} = h_u \\cdot h_i$$
                
                Trong đó:
                - $h_u$: User embedding (từ node_embeddings)
                - $h_i$: Product embedding (từ node_embeddings)
                - $\\hat{x}_{ui}$: Điểm dự đoán user $u$ sẽ thích product $i$
                
                **Quy trình:**
                1. Với mỗi user trong test-set, tính điểm với tất cả products
                2. Sắp xếp products theo điểm giảm dần
                3. Lấy Top-K products làm recommendations
                4. So sánh với ground truth (products user thực tế đã tương tác)
                5. Tính các metrics: Recall, Precision, NDCG, Coverage, Diversity
                """)
                
                # Load parsed log data
                _, log_text = load_evaluation_log("GNN (GraphSAGE)")
                parsed_log = parse_evaluation_log(log_text) if log_text else {}
                
                # Get metrics from comparison_df
                gnn_metrics_row = None
                if comparison_df is not None:
                    gnn_rows = comparison_df[comparison_df['model_name'] == 'GNN (GraphSAGE)']
                    if len(gnn_rows) > 0:
                        gnn_metrics_row = gnn_rows.iloc[0]
                
                # Hiển thị Training Time và Inference Time
                st.markdown("#### ⏱️ Thời gian Training & Inference")
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    training_time = parsed_log['metrics'].get('training_time',
                        gnn_metrics_row['training_time'] if gnn_metrics_row is not None else None)
                    if training_time is not None:
                        st.metric("Training Time (s)", f"{training_time:.4f}")
                with col_time2:
                    inference_time = parsed_log['metrics'].get('avg_inference_time',
                        gnn_metrics_row['avg_inference_time'] if gnn_metrics_row is not None else None)
                    if inference_time is not None:
                        st.metric("Inference Time (s)", f"{inference_time:.4f}")
                
                # Hiển thị metrics @10
                st.markdown("#### 📈 Metrics @10")
                metrics_10 = ['recall@10', 'precision@10', 'ndcg@10', 'coverage@10', 'diversity@10']
                if gnn_metrics_row is not None:
                    render_metrics_in_step(gnn_metrics_row, metrics_10, "Bước 4", "gnn_10", model_name="GNN (GraphSAGE)")
                elif parsed_log:
                    render_metrics_in_step(parsed_log, metrics_10, "Bước 4", "gnn_10", model_name="GNN (GraphSAGE)")
                
                # Hiển thị metrics @20
                st.markdown("#### 📈 Metrics @20")
                metrics_20 = ['recall@20', 'precision@20', 'ndcg@20', 'coverage@20', 'diversity@20']
                if gnn_metrics_row is not None:
                    render_metrics_in_step(gnn_metrics_row, metrics_20, "Bước 4", "gnn_20", model_name="GNN (GraphSAGE)")
                elif parsed_log:
                    render_metrics_in_step(parsed_log, metrics_20, "Bước 4", "gnn_20", model_name="GNN (GraphSAGE)")
                
                st.markdown("#### 📐 Công thức tính các chỉ số (tương tự Content-Based):")
                
                # Recall@K
                with st.expander("Recall@K", expanded=False):
                    st.markdown("""
                    $$Recall@K = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{|R_u \\cap T_u|}{|T_u|}$$
                    """)
                
                # Precision@K
                with st.expander("Precision@K", expanded=False):
                    st.markdown("""
                    $$Precision@K = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{|R_u \\cap T_u|}{K}$$
                    """)
                
                # NDCG@K
                with st.expander("NDCG@K", expanded=False):
                    st.markdown("""
                    $$NDCG@K = \\frac{DCG@K}{IDCG@K}, \\quad DCG@K = \\sum_{i=1}^{K} \\frac{rel_i}{\\log_2(i+1)}$$
                    """)
                
                # Coverage@K
                with st.expander("Coverage@K", expanded=False):
                    st.markdown("""
                    $$Coverage@K = \\frac{|\\bigcup_{u \\in U} R_u|}{|P|}$$
                    """)
                
                # Diversity@K
                with st.expander("Diversity@K", expanded=False):
                    st.markdown("""
                    $$Diversity@K = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{|\\text{unique categories in } R_u|}{K}$$
                    """)
                
                st.markdown("#### 📊 Bảng Tổng Hợp")
                render_metrics_table(comparison_df, highlight_model="GNN (GraphSAGE)")
                render_evaluation_log_section("GNN (GraphSAGE)", "gnn")
        # --- HYBRID TAB ---
        with tab3:
            st.markdown("### 3️⃣ Hybrid Model (GNN + Content-Based)")
            st.markdown("**Mô tả:** Kết hợp điểm số từ GNN và Content-Based để tận dụng ưu điểm cả hai.")
            
            # BƯỚC 1
            with st.expander("Bước 1: Score Normalization (Chuẩn hóa)", expanded=True):
                st.markdown('<div class="step-header">Bước 1: Chuẩn hóa điểm số</div>', unsafe_allow_html=True)
                st.write("**Nội dung:** Đưa điểm số của GNN (thường là dot product, range rộng) và CB (cosine, 0-1) về cùng thang đo [0, 1].")
                st.write("**Dữ liệu sử dụng:** Scores từ GNN model và Content-Based model cho cùng một tập candidates.")
                
                st.markdown("""
                **Công thức Min-Max Scaling:**
                $$Score_{norm} = \\frac{Score - \\min(Scores)}{\\max(Scores) - \\min(Scores)}$$
                
                Trong đó:
                - $Score$: Điểm số gốc (từ GNN hoặc CB)
                - $\\min(Scores)$: Điểm số thấp nhất trong tập
                - $\\max(Scores)$: Điểm số cao nhất trong tập
                - $Score_{norm}$: Điểm số sau khi chuẩn hóa (0-1)
                
                **Lý do:** GNN và CB có thang điểm khác nhau, cần chuẩn hóa để kết hợp công bằng.
                """)
                
                st.write("**Ví dụ minh họa:**")
                ex_data = {
                    'Product': ['P1', 'P2', 'P3'],
                    'GNN Score (Raw)': [5.2, 2.1, 1.5],
                    'CB Score (Raw)': [0.8, 0.3, 0.2],
                    'GNN Norm': [1.0, 0.16, 0.0],
                    'CB Norm': [1.0, 0.17, 0.0]
                }
                ex_df = pd.DataFrame(ex_data)
                st.dataframe(ex_df, use_container_width=True)
                
                st.write("**Giải thích ví dụ:**")
                st.write("- GNN: min=1.5, max=5.2 → P1: (5.2-1.5)/(5.2-1.5)=1.0, P2: (2.1-1.5)/(5.2-1.5)=0.16")
                st.write("- CB: min=0.2, max=0.8 → P1: (0.8-0.2)/(0.8-0.2)=1.0, P2: (0.3-0.2)/(0.8-0.2)=0.17")
                st.info("💡 **Phân tích:** Sau chuẩn hóa, cả hai models đều có thang điểm [0, 1], giúp kết hợp công bằng hơn.")

            # BƯỚC 2
            with st.expander("Bước 2: Weighted Combination (Kết hợp)"):
                st.markdown('<div class="step-header">Bước 2: Kết hợp có trọng số</div>', unsafe_allow_html=True)
                st.write(f"**Nội dung:** Tính điểm cuối cùng bằng cách kết hợp có trọng số giữa GNN và Content-Based với $\\alpha = {hybrid_model.alpha}$.")
                st.write("**Dữ liệu sử dụng:** Normalized scores từ Bước 1.")
                
                st.markdown("""
                **Công thức Late Fusion:**
                $$Score_{final} = \\alpha \\times Score_{GNN\\_norm} + (1 - \\alpha) \\times Score_{CB\\_norm}$$
                
                Trong đó:
                - $\\alpha$: Trọng số cho GNN (mặc định 0.5 = cân bằng khi khởi tạo model)
                - $Score_{GNN\\_norm}$: Điểm số đã chuẩn hóa từ GNN (Min-Max scaling về [0, 1])
                - $Score_{CB\\_norm}$: Điểm số đã chuẩn hóa từ Content-Based (Min-Max scaling về [0, 1])
                - $Score_{final}$: Điểm số cuối cùng để ranking
                
                **Lưu ý quan trọng:** 
                - Khi khởi tạo, Hybrid model có thể dùng $\\alpha = 0.5$ (cân bằng)
                - **Trong thực tế khi recommend**, model sử dụng **dynamic weight** (GNN=0.8, CB=0.2) để ưu tiên GNN cao hơn vì GNN thường cho kết quả tốt hơn Content-Based
                - Công thức thực tế: $Score_{final} = 0.8 \\times Score_{GNN\\_norm} + 0.2 \\times Score_{CB\\_norm}$
                """)
                
                st.write("**Ví dụ áp dụng (với alpha=0.5):**")
                st.write("Giả sử có 3 sản phẩm sau khi chuẩn hóa:")
                ex_combine = pd.DataFrame({
                    'Product': ['P1', 'P2', 'P3'],
                    'GNN Norm': [1.0, 0.5, 0.2],
                    'CB Norm': [0.8, 0.6, 0.4],
                    'Final Score (α=0.5)': [0.9, 0.55, 0.3],
                    'Final Score (α=0.8)': [0.96, 0.52, 0.24]
                })
                st.dataframe(ex_combine, use_container_width=True)
                
                st.write("**Tính toán chi tiết cho P1 (α=0.5):**")
                st.write("$$Score_{final}(P1) = 0.5 \\times 1.0 + 0.5 \\times 0.8 = 0.5 + 0.4 = 0.9$$")
                st.write("**Tính toán chi tiết cho P1 (α=0.8 - dynamic weight):**")
                st.write("$$Score_{final}(P1) = 0.8 \\times 1.0 + 0.2 \\times 0.8 = 0.8 + 0.16 = 0.96$$")
                
                st.info(f"💡 **Phân tích:** Với $\\alpha = {hybrid_model.alpha}$, model cân bằng giữa GNN (học từ tương tác) và CB (dựa trên đặc trưng). Dynamic weight (0.8/0.2) ưu tiên GNN hơn vì nó thường tốt hơn CB.")

            # BƯỚC 3
            with st.expander("Bước 3: Evaluation & Analysis", expanded=True):
                st.markdown('<div class="step-header">Bước 3: Đánh giá tổng hợp</div>', unsafe_allow_html=True)
                
                # Thông tin dữ liệu
                st.write("**Dữ liệu sử dụng:**")
                col_eval1, col_eval2 = st.columns(2)
                with col_eval1:
                    st.metric("Train-set", f"{len(preprocessor.train_interactions)} interactions")
                    st.write(f"- Users: {preprocessor.train_interactions['user_idx'].nunique()}")
                    st.write(f"- Products: {preprocessor.train_interactions['product_idx'].nunique()}")
                with col_eval2:
                    st.metric("Test-set", f"{len(preprocessor.test_interactions)} interactions")
                    st.write(f"- Users: {preprocessor.test_interactions['user_idx'].nunique()}")
                    st.write(f"- Products: {preprocessor.test_interactions['product_idx'].nunique()}")
                
                st.write("**Quy trình đánh giá:**")
                st.write("1. Với mỗi user trong test-set, lấy candidates từ cả GNN và Content-Based")
                st.write("2. Chuẩn hóa và kết hợp scores theo công thức ở Bước 2")
                st.write("3. Sắp xếp và lấy Top-K recommendations")
                st.write("4. So sánh với ground truth và tính các metrics")
                
                # Load parsed log data
                _, log_text = load_evaluation_log("Hybrid (GNN + Content-Based)")
                parsed_log = parse_evaluation_log(log_text) if log_text else {}
                
                # Get metrics from comparison_df
                hybrid_metrics_row = None
                if comparison_df is not None:
                    hybrid_rows = comparison_df[comparison_df['model_name'] == 'Hybrid (GNN + Content-Based)']
                    if len(hybrid_rows) > 0:
                        hybrid_metrics_row = hybrid_rows.iloc[0]
                
                # Hiển thị Training Time và Inference Time
                st.markdown("#### ⏱️ Thời gian Training & Inference")
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    training_time = parsed_log['metrics'].get('training_time',
                        hybrid_metrics_row['training_time'] if hybrid_metrics_row is not None else None)
                    if training_time is not None:
                        st.metric("Training Time (s)", f"{training_time:.4f}")
                with col_time2:
                    inference_time = parsed_log['metrics'].get('avg_inference_time',
                        hybrid_metrics_row['avg_inference_time'] if hybrid_metrics_row is not None else None)
                    if inference_time is not None:
                        st.metric("Inference Time (s)", f"{inference_time:.4f}")
                
                # Hiển thị metrics @10
                st.markdown("#### 📈 Metrics @10")
                metrics_10 = ['recall@10', 'precision@10', 'ndcg@10', 'coverage@10', 'diversity@10']
                if hybrid_metrics_row is not None:
                    render_metrics_in_step(hybrid_metrics_row, metrics_10, "Bước 3", "hybrid_10", model_name="Hybrid (GNN + Content-Based)")
                elif parsed_log:
                    render_metrics_in_step(parsed_log, metrics_10, "Bước 3", "hybrid_10", model_name="Hybrid (GNN + Content-Based)")
                
                # Hiển thị metrics @20
                st.markdown("#### 📈 Metrics @20")
                metrics_20 = ['recall@20', 'precision@20', 'ndcg@20', 'coverage@20', 'diversity@20']
                if hybrid_metrics_row is not None:
                    render_metrics_in_step(hybrid_metrics_row, metrics_20, "Bước 3", "hybrid_20", model_name="Hybrid (GNN + Content-Based)")
                elif parsed_log:
                    render_metrics_in_step(parsed_log, metrics_20, "Bước 3", "hybrid_20", model_name="Hybrid (GNN + Content-Based)")
                
                st.markdown("#### 📐 Công thức tính các chỉ số (tương tự các models khác):")
                
                # Tóm tắt công thức
                st.write("**Recall@K:** Tỷ lệ sản phẩm relevant được tìm thấy")
                st.write("**Precision@K:** Tỷ lệ sản phẩm relevant trong Top-K")
                st.write("**NDCG@K:** Chất lượng ranking (coi trọng vị trí)")
                st.write("**Coverage@K:** Tỷ lệ sản phẩm trong catalog được gợi ý")
                st.write("**Diversity@K:** Độ đa dạng của danh sách gợi ý")
                
                st.markdown("#### 📊 Bảng Tổng Hợp")
                render_metrics_table(comparison_df, highlight_model="Hybrid (GNN + Content-Based)")
                render_evaluation_log_section("Hybrid (GNN + Content-Based)", "hybrid")
                st.markdown("### 🏆 Phân tích & Kết luận (Focus on Hybrid)")
                st.success("""
                **Tại sao Hybrid là tối ưu nhất?**
                
                1. **Recall & Precision:** Hybrid đạt được sự cân bằng tốt hơn:
                   - GNN giúp tăng Recall (tìm được sản phẩm tiềm năng user chưa từng thấy)
                   - CB giúp tăng Precision (đảm bảo sản phẩm giống sở thích cũ)
                
                2. **Coverage & Diversity:** 
                   - Coverage của Hybrid thường cao hơn GNN thuần túy vì có thể gợi ý cả những sản phẩm ít tương tác (nhờ Content-Based)
                   - Diversity tốt hơn CB thuần túy nhờ GNN đa dạng hóa recommendations
                
                3. **Khắc phục điểm yếu:** 
                   - GNN bị yếu khi User mới (Cold-start) → CB bù đắp bằng cách dựa vào đặc trưng sản phẩm
                   - CB bị yếu về độ đa dạng và khám phá → GNN bù đắp bằng cách học từ tương tác của users khác
                
                4. **Robustness:** Hybrid ít bị ảnh hưởng bởi dữ liệu thiếu hoặc không cân bằng hơn các model đơn lẻ
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
            st.markdown("### Đánh giá chi tiết")
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
