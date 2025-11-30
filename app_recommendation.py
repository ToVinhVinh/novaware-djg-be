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

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

apps_utils_path = os.path.join(current_dir, 'apps', 'utils')
if apps_utils_path not in sys.path:
    sys.path.insert(0, apps_utils_path)

_train_import_error = None
try:
    import train_recommendation
except ImportError as e:
    train_recommendation = None
    _train_import_error = str(e)

_export_import_error = None
try:
    from apps.utils.export_data import export_all_data, ensure_export_directory
except ImportError as e:
    export_all_data = None
    ensure_export_directory = None
    _export_import_error = str(e)

st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color:
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color:
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        color:
        margin-top: 1rem;
        background-color:
        padding: 0.5rem;
        border-radius: 5px;
    }
    .formula-box {
        background-color:
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid
        margin: 1rem 0;
    }
</style>

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
    try:
        df = pd.read_csv('recommendation_system/evaluation/comparison_results.csv')
        return df
    except:
        return None

def compute_sparsity(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    non_null_counts = df.count()
    sparsity = 1 - (non_null_counts / len(df))
    return sparsity.sort_values(ascending=False)

def render_sparsity_chart(df: pd.DataFrame, title: str, key: str):
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
    if df.empty:
        st.info("Dataset trống, không thể thống kê.")
        return
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.info("Không có cột số để thống kê.")
        return
    stats_df = numeric_df.describe().T
    st.dataframe(stats_df, use_container_width=True)

def render_dataset_upload_section(
    dataset_key: str,
    display_name: str,
    purpose_text: str
):
    st.markdown(f"
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
    if df is None:
        st.warning("Chưa có dữ liệu metrics. Vui lòng chạy tính toán trước.")
        return

    st.markdown("

    required_cols = ['model_name', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20',
                     'precision@10', 'precision@20', 'training_time', 'avg_inference_time',
                     'coverage@10', 'diversity@10']

    display_df = df.copy()
    available_cols = [col for col in required_cols if col in display_df.columns]
    display_df = display_df[available_cols]

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

    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)

    def highlight_row(row):
        model_name = row.get('Model', '')
        if model_name == highlight_model:
            return ['background-color:
        return [''] * len(row)

    st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True)

def slugify_model_name(model_name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')

def apply_5core_pruning(interactions_df: pd.DataFrame, min_interactions: int = 5) -> Dict:

    if interactions_df.empty:
        return {
            'pruned_interactions': pd.DataFrame(),
            'removed_users': 0,
            'removed_products': 0,
            'iterations': 0,
            'stats': []
        }

    df = interactions_df.copy()

    if 'user_id' not in df.columns or 'product_id' not in df.columns:
        raise ValueError("DataFrame phải có columns 'user_id' và 'product_id'")

    original_users = df['user_id'].nunique()
    original_products = df['product_id'].nunique()
    original_interactions = len(df)

    stats = [{
        'iteration': 0,
        'users': original_users,
        'products': original_products,
        'interactions': original_interactions,
        'removed_users': 0,
        'removed_products': 0
    }]

    iteration = 0
    changed = True

    while changed:
        iteration += 1
        changed = False

        user_counts = df['user_id'].value_counts()
        users_to_keep = user_counts[user_counts >= min_interactions].index

        product_counts = df['product_id'].value_counts()
        products_to_keep = product_counts[product_counts >= min_interactions].index

        before_len = len(df)
        df = df[df['user_id'].isin(users_to_keep) & df['product_id'].isin(products_to_keep)]
        after_len = len(df)

        if before_len != after_len:
            changed = True

        removed_users = original_users - df['user_id'].nunique()
        removed_products = original_products - df['product_id'].nunique()

        stats.append({
            'iteration': iteration,
            'users': df['user_id'].nunique(),
            'products': df['product_id'].nunique(),
            'interactions': len(df),
            'removed_users': removed_users,
            'removed_products': removed_products
        })

        if iteration >= 100:
            break

    total_removed_users = original_users - df['user_id'].nunique()
    total_removed_products = original_products - df['product_id'].nunique()

    return {
        'pruned_interactions': df,
        'removed_users': total_removed_users,
        'removed_products': total_removed_products,
        'iterations': iteration,
        'stats': stats,
        'original_users': original_users,
        'original_products': original_products,
        'original_interactions': original_interactions
    }

def apply_feature_encoding(products_df: pd.DataFrame, features: List[str] = None) -> Dict:

    if products_df.empty:
        return {
            'encoded_matrix': np.array([]),
            'feature_mapping': {},
            'feature_dims': {},
            'total_dims': 0,
            'feature_names': []
        }

    if features is None:
        features = ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'usage']

    available_features = [f for f in features if f in products_df.columns]

    if not available_features:
        return {
            'encoded_matrix': np.array([]),
            'feature_mapping': {},
            'feature_dims': {},
            'total_dims': 0,
            'feature_names': []
        }

    feature_mapping = {}
    feature_dims = {}
    encoded_parts = []
    feature_names = []
    start_idx = 0

    for feat in available_features:
        unique_values = sorted(products_df[feat].dropna().unique())
        n_values = len(unique_values)

        value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
        feature_mapping[feat] = {
            'value_to_idx': value_to_idx,
            'idx_to_value': {idx: val for val, idx in value_to_idx.items()},
            'start_idx': start_idx,
            'end_idx': start_idx + n_values
        }

        one_hot = np.zeros((len(products_df), n_values))
        for i, val in enumerate(products_df[feat]):
            if pd.notna(val) and val in value_to_idx:
                one_hot[i, value_to_idx[val]] = 1

        encoded_parts.append(one_hot)
        feature_dims[feat] = n_values

        for val in unique_values:
            feature_names.append(f"{feat}_{val}")

        start_idx += n_values

    if encoded_parts:
        encoded_matrix = np.hstack(encoded_parts)
    else:
        encoded_matrix = np.array([])

    return {
        'encoded_matrix': encoded_matrix,
        'feature_mapping': feature_mapping,
        'feature_dims': feature_dims,
        'total_dims': encoded_matrix.shape[1] if len(encoded_matrix.shape) > 1 else 0,
        'feature_names': feature_names,
        'product_ids': products_df.index.tolist() if hasattr(products_df.index, 'tolist') else list(range(len(products_df)))
    }

def load_evaluation_log(model_name: str):
    slug = slugify_model_name(model_name)
    log_path = os.path.join('recommendation_system', 'evaluation', 'logs', f'{slug}.log')
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            return slug, f.read()
    return slug, None

def parse_evaluation_log(log_text: str) -> Dict:

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

        if not line or line.startswith('===') or line.startswith('[') or 'EVALUATING' in line or 'RESULTS FOR' in line:
            i += 1
            continue

        if ':' in line and not line.startswith('📐') and not line.startswith('🧮'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                metric_name = parts[0].strip()
                value_str = parts[1].strip()

                value_str = value_str.split()[0] if value_str.split() else value_str

                try:
                    value = float(value_str)
                    metrics[metric_name] = value
                    current_metric = metric_name
                except ValueError:
                    pass

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

    parsed_log = None
    if model_name:
        _, log_text = load_evaluation_log(model_name)
        if log_text:
            parsed_log = parse_evaluation_log(log_text)

    n_cols = 2
    cols = st.columns(n_cols)

    for idx, metric_key in enumerate(metric_keys):
        col_idx = idx % n_cols
        with cols[col_idx]:
            value = None
            formula = ''
            example = ''

            if isinstance(metrics_data, dict) and 'metrics' in metrics_data:
                value = metrics_data['metrics'].get(metric_key, None)
                formula = metrics_data['formulas'].get(metric_key, '')
                example = metrics_data['examples'].get(metric_key, '')
            elif isinstance(metrics_data, pd.Series):
                value = metrics_data.get(metric_key, None)
                if parsed_log:
                    formula = parsed_log['formulas'].get(metric_key, '')
                    example = parsed_log['examples'].get(metric_key, '')

            if value is not None:
                display_name = metric_key.replace('@', '@').replace('_', ' ').title()

                st.metric(display_name, f"{value:.4f}")

                with st.expander(f"Chi tiết {display_name}", expanded=False):
                    if formula:
                        st.markdown(f"**Công thức:** {formula}")

                    if example:
                        if "| Trung bình" in example:
                            parts = example.split(" | ")
                            user_examples = []
                            avg_formula = None

                            for part in parts:
                                if "Trung bình" in part:
                                    avg_formula = part
                                else:
                                    user_examples.append(part)

                            st.markdown("
                            for i, user_ex in enumerate(user_examples, 1):
                                st.markdown(f"**{i}. {user_ex}**")

                            if avg_formula:
                                st.markdown("

                                if "=" in avg_formula:
                                    formula_parts = avg_formula.split("=")
                                    if len(formula_parts) >= 2:
                                        left_side = formula_parts[0].strip()
                                        right_side = "=".join(formula_parts[1:]).strip()

                                        import re
                                        n_users_match = re.search(r'user(\d+)', right_side)
                                        n_users = n_users_match.group(1) if n_users_match else "N"

                                        metric_var = display_name.replace(" ", "_").lower()

                                        st.markdown(f"""
                                        **Công thức:**
                                        $$\\text{{Trung bình}} = \\frac{{\\sum_{{u=1}}^{{{n_users}}} {display_name}_u}}{{{n_users}}}$$

                                        **Dạng mở rộng:**
                                        $$\\text{{Trung bình}} = \\frac{{{display_name}_{{user1}} + {display_name}_{{user2}} + \\ldots + {display_name}_{{user{n_users}}}}}{{{n_users}}}$$

                                                **Tính toán:**
                                                $$\\text{{Trung bình}} = \\frac{{{formula_example} + {display_name}_{{user{n_users}}}}}{{{n_users}}} = {value:.4f}$$

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

    st.markdown('<div class="main-header">👔 Fashion Recommendation System</div>', unsafe_allow_html=True)

    st.sidebar.title("⚙️ Menu")

    page = st.sidebar.radio(
        "Chọn chức năng",
        ["📚 Algorithms & Steps", "📊 Model Comparison", "🎯 Personalized Recommendations", "👗 Outfit Recommendations"]
    )

    preprocessor, cb_model, gnn_model, hybrid_model = load_models()
    comparison_df = load_comparison_results()

    if page == "📚 Algorithms & Steps":
        st.markdown("

        with st.expander("Bước 0: Xuất dữ liệu từ MongoDB thành CSV", expanded=True):
            st.write("**Nội dung thực hiện:** Xuất dữ liệu từ MongoDB (products, users, interactions) thành các file CSV để sử dụng cho training và evaluation.")

            if export_all_data is None:
                st.error(f"❌ Không thể import export_data module: {_export_import_error}")
                st.info("Vui lòng đảm bảo file apps/utils/export_data.py tồn tại và có thể import được.")
            else:
                col_export1, col_export2 = st.columns([2, 1])
                with col_export1:
                    st.write("**Các file sẽ được xuất:**")
                    st.write("- `products.csv`: id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName, images")
                    st.write("- `users.csv`: id, name, email, age, gender, interaction_history")
                    st.write("- `interactions.csv`: user_id, product_id, interaction_type, timestamp")
                    st.write("**Vị trí lưu:** `apps/exports/`")

                with col_export2:
                    export_button_clicked = st.button("📥 Xuất dữ liệu từ MongoDB", type="primary", use_container_width=True)

                if export_button_clicked:
                    with st.spinner("Đang xuất dữ liệu từ MongoDB..."):
                        try:
                            result = export_all_data()

                            if result['success']:
                                st.success(f"✅ {result['message']}")

                                st.markdown("
                                col_res1, col_res2, col_res3 = st.columns(3)

                                with col_res1:
                                    products_result = result['results']['products']
                                    if products_result['success']:
                                        st.success(f"✅ Products: {products_result['count']} records")
                                    else:
                                        st.error(f"❌ Products: {products_result.get('error', 'Lỗi')}")

                                with col_res2:
                                    users_result = result['results']['users']
                                    if users_result['success']:
                                        st.success(f"✅ Users: {users_result['count']} records")
                                    else:
                                        st.error(f"❌ Users: {users_result.get('error', 'Lỗi')}")

                                with col_res3:
                                    interactions_result = result['results']['interactions']
                                    if interactions_result['success']:
                                        st.success(f"✅ Interactions: {interactions_result['count']} records")
                                    else:
                                        st.error(f"❌ Interactions: {interactions_result.get('error', 'Lỗi')}")

                                st.markdown("---")
                                st.markdown("

                                export_dir = ensure_export_directory()

                                products_path = export_dir / 'products.csv'
                                if products_path.exists() and products_result['success']:
                                    st.markdown("
                                    try:
                                        products_df = pd.read_csv(products_path)
                                        st.success(f"✅ Đã tải products.csv: {len(products_df)} rows × {len(products_df.columns)} columns")

                                        col_p1, col_p2 = st.columns(2)
                                        with col_p1:
                                            st.metric("Số dòng (rows)", len(products_df))
                                        with col_p2:
                                            st.metric("Số cột (columns)", len(products_df.columns))

                                        st.markdown("**👀 Xem trước dữ liệu (tối đa 100 dòng đầu):**")
                                        st.dataframe(products_df.head(100), use_container_width=True)

                                        st.markdown("**📉 Biểu đồ độ thưa (tỉ lệ giá trị null trên mỗi cột):**")
                                        render_sparsity_chart(products_df, "Độ thưa - Products", "products_export")

                                        st.markdown("**📊 Biểu đồ tỉ lệ / phân bố:**")
                                        render_distribution_chart(products_df, "products_export")

                                        st.markdown("**📈 Bảng thống kê dữ liệu:**")
                                        render_data_statistics(products_df)
                                    except Exception as e:
                                        st.error(f"Lỗi khi đọc products.csv: {str(e)}")

                                st.markdown("---")

                                users_path = export_dir / 'users.csv'
                                if users_path.exists() and users_result['success']:
                                    st.markdown("
                                    try:
                                        users_df = pd.read_csv(users_path)
                                        st.success(f"✅ Đã tải users.csv: {len(users_df)} rows × {len(users_df.columns)} columns")

                                        col_u1, col_u2 = st.columns(2)
                                        with col_u1:
                                            st.metric("Số dòng (rows)", len(users_df))
                                        with col_u2:
                                            st.metric("Số cột (columns)", len(users_df.columns))

                                        st.markdown("**👀 Xem trước dữ liệu (tối đa 100 dòng đầu):**")
                                        st.dataframe(users_df.head(100), use_container_width=True)

                                        st.markdown("**📉 Biểu đồ độ thưa (tỉ lệ giá trị null trên mỗi cột):**")
                                        render_sparsity_chart(users_df, "Độ thưa - Users", "users_export")

                                        st.markdown("**📊 Biểu đồ tỉ lệ / phân bố:**")
                                        render_distribution_chart(users_df, "users_export")

                                        st.markdown("**📈 Bảng thống kê dữ liệu:**")
                                        render_data_statistics(users_df)
                                    except Exception as e:
                                        st.error(f"Lỗi khi đọc users.csv: {str(e)}")

                                st.markdown("---")

                                interactions_path = export_dir / 'interactions.csv'
                                if interactions_path.exists() and interactions_result['success']:
                                    st.markdown("
                                    try:
                                        interactions_df = pd.read_csv(interactions_path)
                                        st.success(f"✅ Đã tải interactions.csv: {len(interactions_df)} rows × {len(interactions_df.columns)} columns")

                                        col_i1, col_i2 = st.columns(2)
                                        with col_i1:
                                            st.metric("Số dòng (rows)", len(interactions_df))
                                        with col_i2:
                                            st.metric("Số cột (columns)", len(interactions_df.columns))

                                        st.markdown("**👀 Xem trước dữ liệu (tối đa 100 dòng đầu):**")
                                        st.dataframe(interactions_df.head(100), use_container_width=True)

                                        st.markdown("**📉 Biểu đồ độ thưa (tỉ lệ giá trị null trên mỗi cột):**")
                                        render_sparsity_chart(interactions_df, "Độ thưa - Interactions", "interactions_export")

                                        st.markdown("**📊 Biểu đồ tỉ lệ / phân bố:**")
                                        render_distribution_chart(interactions_df, "interactions_export")

                                        st.markdown("**📈 Bảng thống kê dữ liệu:**")
                                        render_data_statistics(interactions_df)
                                    except Exception as e:
                                        st.error(f"Lỗi khi đọc interactions.csv: {str(e)}")

                                st.session_state['exported_data'] = {
                                    'products_path': str(products_path) if products_path.exists() else None,
                                    'users_path': str(users_path) if users_path.exists() else None,
                                    'interactions_path': str(interactions_path) if interactions_path.exists() else None,
                                    'export_dir': str(export_dir)
                                }

                            else:
                                st.error(f"❌ Có lỗi xảy ra khi xuất dữ liệu")
                                for key, res in result['results'].items():
                                    if not res['success']:
                                        st.error(f"❌ {key}: {res.get('error', 'Lỗi không xác định')}")

                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

                export_dir = ensure_export_directory() if ensure_export_directory else None
                if export_dir:
                    st.info(f"💡 **Lưu ý:** Các file CSV sẽ được lưu tại: `{export_dir}`")

        st.markdown("---")

        with st.expander("Bước 1.1: Làm sạch và Lọc Dữ liệu (Pruning & Sparsity Handling)", expanded=True):
            st.write("**Nội dung thực hiện:** Áp dụng kỹ thuật 5-Core Pruning để loại bỏ đệ quy các người dùng và sản phẩm có dưới 5 tương tác nhằm giảm độ thưa thớt của dữ liệu.")
            st.write("**Dữ liệu sử dụng:** `interactions.csv`")

            st.markdown("""
            **Thuật toán 5-Core Pruning:**

            1. **Khởi tạo:** Đếm số lượng tương tác cho mỗi user và mỗi product
            2. **Lặp đệ quy:**
               - Loại bỏ tất cả users có < 5 interactions
               - Loại bỏ tất cả products có < 5 interactions
               - Cập nhật lại số lượng interactions của các users/products còn lại
               - Lặp lại cho đến khi không còn user/product nào bị loại bỏ
            3. **Kết quả:** Ma trận tương tác $R$ được làm sạch, chỉ giữ lại các users và products có đủ dữ liệu

            **Công thức:**
            $$R_{pruned} = \\{(u, i) \\in R : |I_u| \\geq 5 \\land |U_i| \\geq 5\\}$$

            Trong đó:
            - $R$: Ma trận tương tác gốc
            - $I_u$: Tập sản phẩm mà user $u$ đã tương tác
            - $U_i$: Tập users đã tương tác với sản phẩm $i$
            - $R_{pruned}$: Ma trận sau khi pruning

                        ❌ **Kết quả:** Tất cả dữ liệu đã bị loại bỏ!

                        **Nguyên nhân:**
                        - Với min_interactions = {min_interactions_used}, tất cả users và/hoặc products đều có ít hơn {min_interactions_used} interactions
                        - Điều này tạo ra hiệu ứng cascade: khi loại bỏ users/products, các interactions liên quan cũng bị loại bỏ, khiến các users/products khác cũng không đủ điều kiện

                        **Giải pháp:**
                        1. Giảm min_interactions xuống (ví dụ: 3 hoặc 2)
                        2. Thu thập thêm dữ liệu interactions
                        3. Chấp nhận dữ liệu thưa thớt và không áp dụng pruning

                    ✅ Ma trận tương tác thưa thớt $R$ được làm sạch, giảm nhiễu (noise) do tương tác ngẫu nhiên hoặc không đủ dữ liệu

                    ✅ Tăng mật độ dữ liệu tương tác cho các thuật toán cộng tác (GNN)

                    ✅ Loại bỏ các users và products có quá ít tương tác, giúp model học được patterns rõ ràng hơn

            **Phương pháp mã hóa:**

            **1. One-Hot Encoding:**
            - Mỗi giá trị phân loại được chuyển thành một vector nhị phân
            - Ví dụ: masterCategory có 3 giá trị → 3 chiều binary vector
            - Tổng số chiều = tổng số giá trị unique của tất cả các features

            **2. Categorical Embedding (Alternative):**
            - Sử dụng embedding layer để học vector đại diện
            - Kích thước nhỏ gọn hơn One-Hot
            - Có thể học được mối quan hệ giữa các categories

            **Công thức:**
            $$\\mathbf{v}_i = [\\text{OneHot}(\\text{masterCategory}_i), \\text{OneHot}(\\text{subCategory}_i), \\text{OneHot}(\\text{articleType}_i), \\text{OneHot}(\\text{baseColour}_i), \\text{OneHot}(\\text{usage}_i)]$$

            Trong đó:
            - $\\mathbf{v}_i$: Item Profile Vector của sản phẩm $i$
            - $\\text{OneHot}(x)$: Vector one-hot encoding của giá trị $x$
            - Kết quả: Vector concatenation của tất cả các features

            **Ma trận đặc trưng:**
            $$P \\in \\mathbb{R}^{|I| \\times d_c}$$

            Trong đó:
            - $|I|$: Số lượng sản phẩm
            - $d_c$: Tổng số chiều đặc trưng nội dung (tổng số giá trị unique của tất cả features)

                    ✅ Vector $\\mathbf{v}_i$ cho mỗi sản phẩm $i$ trong hệ thống, đại diện cho thuộc tính nội dung của nó

                    ✅ Ma trận đặc trưng $P \\in \\mathbb{R}^{|I| \\times d_c}$ được tạo thành

                    ✅ Các vector này là đầu vào cơ sở cho CBF (Content-Based Filtering) và Diversity (ILD) metric

                    ✅ Mỗi sản phẩm được biểu diễn dưới dạng vector số học, có thể tính toán similarity và distance

                **Công thức áp dụng:**
                $$Text(P_i) = [Gender] + [MasterCategory] + [SubCategory] \\times 2 + [ArticleType] \\times 3 + [BaseColour] + [Usage]$$

                **Giải thích:** Các features được kết hợp thành chuỗi văn bản, trong đó:
                - `ArticleType` được lặp lại **3 lần** (trọng số cao nhất - quan trọng nhất)
                - `SubCategory` được lặp lại **2 lần** (trọng số trung bình)
                - Các features khác (Gender, MasterCategory, BaseColour, Usage) xuất hiện **1 lần**

                **Lý do:** Việc lặp lại giúp TF-IDF coi trọng các features quan trọng hơn khi tính toán similarity.

                **Công thức TF-IDF:**
                $$TF(t, d) = \\frac{count(t, d)}{len(d)}, \\quad IDF(t) = \\log(\\frac{N}{df(t)}), \\quad TF\\text{-}IDF = TF \\times IDF$$

                Trong đó:
                - $TF(t, d)$: Tần suất từ $t$ trong document $d$
                - $IDF(t)$: Nghịch đảo tần suất document, đo độ hiếm của từ $t$
                - $N$: Tổng số documents (sản phẩm)
                - $df(t)$: Số documents chứa từ $t$

                **Công thức Cosine Similarity:**
                $$Cosine(A, B) = \\frac{A \\cdot B}{||A|| \\times ||B||} = \\frac{\\sum_{i=1}^{n} A_i B_i}{\\sqrt{\\sum_{i=1}^{n} A_i^2} \\sqrt{\\sum_{i=1}^{n} B_i^2}}$$

                Trong đó:
                - $A, B$: Hai vector TF-IDF của 2 sản phẩm
                - $A_i, B_i$: Giá trị TF-IDF của feature thứ $i$
                - Kết quả: Giá trị từ 0 đến 1 (1 = giống nhau hoàn toàn, 0 = khác biệt hoàn toàn)

User 1 <---[weight=1.0]---> Product A
User 2 <---[weight=0.7]---> Product A
User 1 <---[weight=0.5]---> Product B

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

                    $$Recall@K = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{|R_u \\cap T_u|}{|T_u|}$$

                    $$Precision@K = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{|R_u \\cap T_u|}{K}$$

                    $$NDCG@K = \\frac{DCG@K}{IDCG@K}, \\quad DCG@K = \\sum_{i=1}^{K} \\frac{rel_i}{\\log_2(i+1)}$$

                    $$Coverage@K = \\frac{|\\bigcup_{u \\in U} R_u|}{|P|}$$

                    $$Diversity@K = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{|\\text{unique categories in } R_u|}{K}$$

                **Công thức Min-Max Scaling:**
                $$Score_{norm} = \\frac{Score - \\min(Scores)}{\\max(Scores) - \\min(Scores)}$$

                Trong đó:
                - $Score$: Điểm số gốc (từ GNN hoặc CB)
                - $\\min(Scores)$: Điểm số thấp nhất trong tập
                - $\\max(Scores)$: Điểm số cao nhất trong tập
                - $Score_{norm}$: Điểm số sau khi chuẩn hóa (0-1)

                **Lý do:** GNN và CB có thang điểm khác nhau, cần chuẩn hóa để kết hợp công bằng.

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

