import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import re
import ast
from collections import defaultdict

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

_user_profile_import_error = None
try:
    from apps.utils.user_profile import (
        build_weighted_user_profile, 
        get_interaction_weight, 
        INTERACTION_WEIGHTS,
        compute_cbf_predictions,
        cosine_similarity
    )
except ImportError as e:
    build_weighted_user_profile = None
    get_interaction_weight = None
    INTERACTION_WEIGHTS = None
    compute_cbf_predictions = None
    cosine_similarity = None
    _user_profile_import_error = str(e)

_cbf_utils_import_error = None
try:
    from apps.utils.cbf_utils import (
        apply_personalized_filters,
        apply_articletype_filter,
        apply_age_gender_filter,
        get_allowed_genders
    )
except ImportError as e:
    apply_personalized_filters = None
    apply_articletype_filter = None
    apply_age_gender_filter = None
    get_allowed_genders = None
    _cbf_utils_import_error = str(e)

_outfit_import_error = None
try:
    from apps.utils.outfit_recommendation import (
        generate_outfit_recommendations,
        compute_outfit_score,
        compute_pairwise_compatibility,
        check_usage_compatibility
    )
except ImportError as e:
    generate_outfit_recommendations = None
    compute_outfit_score = None
    compute_pairwise_compatibility = None
    check_usage_compatibility = None
    _outfit_import_error = str(e)

_evaluation_import_error = None
try:
    from apps.utils.evaluation_metrics import (
        compute_cbf_metrics,
        recall_at_k,
        precision_at_k,
        ndcg_at_k,
        diversity,
        coverage
    )
except ImportError as e:
    compute_cbf_metrics = None
    recall_at_k = None
    precision_at_k = None
    ndcg_at_k = None
    diversity = None
    coverage = None
    _evaluation_import_error = str(e)

_gnn_utils_import_error = None
try:
    from apps.utils.gnn_utils import (
        build_graph,
        message_propagation,
        compute_gnn_predictions,
        train_gnn_model
    )
except ImportError as e:
    build_graph = None
    message_propagation = None
    compute_gnn_predictions = None
    train_gnn_model = None
    _gnn_utils_import_error = str(e)

_hybrid_utils_import_error = None
try:
    from apps.utils.hybrid_utils import (
        combine_hybrid_scores
    )
except ImportError as e:
    combine_hybrid_scores = None
    _hybrid_utils_import_error = str(e)

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


ARTIFACTS_DIR = Path(current_dir) / "artifacts"


def _ensure_artifacts_dir() -> Path:
    """Đảm bảo thư mục artifacts tồn tại."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def _load_pickle_if_exists(path: Path):
    """Load pickle nếu file tồn tại, ngược lại trả về None."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_pickle_safely(path: Path, obj) -> None:
    """Lưu pickle một cách an toàn, bỏ qua lỗi nếu có vấn đề IO."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        # Không crash app chỉ vì lỗi ghi file
        pass


def save_predictions_artifact(model_key: str, data: Dict) -> None:
    """
    Lưu predictions ra file artifacts cho từng loại mô hình.
    model_key: 'cbf' | 'gnn' | 'hybrid'
    """
    base = _ensure_artifacts_dir()
    filename = {
        "cbf": "streamlit_cbf_predictions.pkl",
        "gnn": "streamlit_gnn_predictions.pkl",
        "hybrid": "streamlit_hybrid_predictions.pkl",
    }.get(model_key)
    if not filename:
        return
    _save_pickle_safely(base / filename, data)


def save_intermediate_artifact(key: str, data) -> None:
    """
    Lưu intermediate results ra file artifacts.
    key: 'pruned_interactions' | 'feature_encoding' | 'user_profiles' | 
         'gnn_graph' | 'gnn_propagation' | 'gnn_training' |
         'cbf_evaluation_metrics' | 'gnn_evaluation_metrics' | 'hybrid_evaluation_metrics' |
         'personalized_filters' | 'training_time' | 'inference_time' | 
         'gnn_training_time' | 'gnn_inference_time'
    """
    base = _ensure_artifacts_dir()
    filename_mapping = {
        "pruned_interactions": "pruned_interactions.pkl",
        "feature_encoding": "feature_encoding.pkl",
        "user_profiles": "user_profiles.pkl",
        "gnn_graph": "gnn_graph.pkl",
        "gnn_propagation": "gnn_propagation.pkl",
        "gnn_training": "gnn_training.pkl",
        "cbf_evaluation_metrics": "cbf_evaluation_metrics.pkl",
        "gnn_evaluation_metrics": "gnn_evaluation_metrics.pkl",
        "hybrid_evaluation_metrics": "hybrid_evaluation_metrics.pkl",
        "personalized_filters": "personalized_filters.pkl",
        "training_time": "training_time.pkl",
        "inference_time": "inference_time.pkl",
        "gnn_training_time": "gnn_training_time.pkl",
        "gnn_inference_time": "gnn_inference_time.pkl",
    }
    filename = filename_mapping.get(key)
    if filename:
        _save_pickle_safely(base / filename, data)


def load_cached_predictions_into_session() -> None:
    """
    Auto-load predictions đã lưu (nếu có) vào session_state khi mở app.
    Chỉ nạp nếu session_state chưa có key tương ứng.
    """
    base = ARTIFACTS_DIR
    mappings = [
        ("cbf_predictions", "streamlit_cbf_predictions.pkl"),
        ("gnn_predictions", "streamlit_gnn_predictions.pkl"),
        ("hybrid_predictions", "streamlit_hybrid_predictions.pkl"),
    ]
    for state_key, fname in mappings:
        if state_key in st.session_state:
            continue
        path = base / fname
        cached = _load_pickle_if_exists(path)
        if cached:
            st.session_state[state_key] = cached


def _is_valid_data(data) -> bool:
    """Kiểm tra xem dữ liệu có hợp lệ không (không None, không rỗng)."""
    if data is None:
        return False
    if isinstance(data, dict):
        return len(data) > 0
    if isinstance(data, (list, tuple)):
        return len(data) > 0
    if isinstance(data, pd.DataFrame):
        return not data.empty
    # Các kiểu dữ liệu khác (int, float, str) đều hợp lệ nếu không None
    return True


def restore_all_artifacts() -> None:
    """
    Khôi phục tất cả các kết quả từ artifacts vào session_state.
    Được gọi khi cần đảm bảo không mất dữ liệu sau khi chạy các bước mới.
    Chỉ restore nếu session_state chưa có hoặc dữ liệu hiện tại không hợp lệ.
    """
    base = ARTIFACTS_DIR
    
    # Restore predictions
    predictions_mappings = [
        ("cbf_predictions", "streamlit_cbf_predictions.pkl"),
        ("gnn_predictions", "streamlit_gnn_predictions.pkl"),
        ("hybrid_predictions", "streamlit_hybrid_predictions.pkl"),
    ]
    for state_key, fname in predictions_mappings:
        path = base / fname
        cached = _load_pickle_if_exists(path)
        if cached:
            # Chỉ restore nếu chưa có hoặc dữ liệu hiện tại không hợp lệ
            if state_key not in st.session_state or not _is_valid_data(st.session_state[state_key]):
                st.session_state[state_key] = cached
    
    # Restore intermediate results (nếu có lưu)
    intermediate_mappings = [
        ("pruned_interactions", "pruned_interactions.pkl"),
        ("feature_encoding", "feature_encoding.pkl"),
        ("user_profiles", "user_profiles.pkl"),
        ("gnn_graph", "gnn_graph.pkl"),
        ("gnn_propagation", "gnn_propagation.pkl"),
        ("gnn_training", "gnn_training.pkl"),
        ("cbf_evaluation_metrics", "cbf_evaluation_metrics.pkl"),
        ("gnn_evaluation_metrics", "gnn_evaluation_metrics.pkl"),
        ("hybrid_evaluation_metrics", "hybrid_evaluation_metrics.pkl"),
        ("personalized_filters", "personalized_filters.pkl"),
        ("training_time", "training_time.pkl"),
        ("inference_time", "inference_time.pkl"),
        ("gnn_training_time", "gnn_training_time.pkl"),
        ("gnn_inference_time", "gnn_inference_time.pkl"),
    ]
    for state_key, fname in intermediate_mappings:
        path = base / fname
        cached = _load_pickle_if_exists(path)
        if cached:
            # Chỉ restore nếu chưa có hoặc dữ liệu hiện tại không hợp lệ
            if state_key not in st.session_state or not _is_valid_data(st.session_state[state_key]):
                st.session_state[state_key] = cached


def get_artifacts_status() -> Dict[str, bool]:
    """
    Kiểm tra trạng thái của các artifacts (đã lưu hay chưa).
    Trả về dict với key là tên artifact và value là True nếu đã tồn tại.
    """
    base = ARTIFACTS_DIR
    status = {}
    
    all_mappings = [
        ("cbf_predictions", "streamlit_cbf_predictions.pkl"),
        ("gnn_predictions", "streamlit_gnn_predictions.pkl"),
        ("hybrid_predictions", "streamlit_hybrid_predictions.pkl"),
        ("pruned_interactions", "pruned_interactions.pkl"),
        ("feature_encoding", "feature_encoding.pkl"),
        ("user_profiles", "user_profiles.pkl"),
        ("gnn_graph", "gnn_graph.pkl"),
        ("gnn_propagation", "gnn_propagation.pkl"),
        ("gnn_training", "gnn_training.pkl"),
        ("cbf_evaluation_metrics", "cbf_evaluation_metrics.pkl"),
        ("gnn_evaluation_metrics", "gnn_evaluation_metrics.pkl"),
        ("hybrid_evaluation_metrics", "hybrid_evaluation_metrics.pkl"),
    ]
    
    for state_key, fname in all_mappings:
        path = base / fname
        status[state_key] = path.exists()
    
    return status


def display_pruning_results(result: Dict) -> None:
    """Hiển thị kết quả Pruning từ session_state hoặc artifacts."""
    if result is None or result.get('pruned_interactions') is None or result['pruned_interactions'].empty:
        return
    
    st.success("✅ **Kết quả đã có sẵn:** Ma trận tương tác đã được làm sạch.")
    
    # Display statistics
    st.markdown("### 📊 Thống kê kết quả Pruning")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Users ban đầu", result['original_users'])
        st.metric("Users sau pruning", result['original_users'] - result['removed_users'])
    with col_stat2:
        st.metric("Products ban đầu", result['original_products'])
        st.metric("Products sau pruning", result['original_products'] - result['removed_products'])
    with col_stat3:
        st.metric("Interactions ban đầu", result['original_interactions'])
        st.metric("Interactions sau pruning", len(result['pruned_interactions']))
    with col_stat4:
        st.metric("Số lần lặp", result['iterations'])
        reduction_pct = ((result['original_interactions'] - len(result['pruned_interactions'])) / result['original_interactions'] * 100) if result['original_interactions'] > 0 else 0
        st.metric("Giảm đi", f"{reduction_pct:.2f}%")
    
    # Calculate values for tabs
    pruned_users = result['original_users'] - result['removed_users']
    pruned_products = result['original_products'] - result['removed_products']
    
    # Original sparsity
    original_density = result['original_interactions'] / (result['original_users'] * result['original_products']) if (result['original_users'] * result['original_products']) > 0 else 0
    original_sparsity = 1 - original_density
    
    # Pruned sparsity
    pruned_density = len(result['pruned_interactions']) / (pruned_users * pruned_products) if (pruned_users * pruned_products) > 0 else 0
    pruned_sparsity = 1 - pruned_density
    
    improvement = original_sparsity - pruned_sparsity
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Ma trận tương tác đã làm sạch",
        "📉 So sánh độ thưa thớt",
        "📈 Quá trình Pruning qua các lần lặp",
        "🔥 Ma trận tương tác (Heatmap)"
    ])
    
    with tab1:
        st.markdown("### 📋 Ma trận tương tác đã làm sạch $R_{pruned}$")
        st.dataframe(
            result['pruned_interactions'].head(100),
            use_container_width=True
        )
        
        # Download button
        csv = result['pruned_interactions'].to_csv(index=False)
        st.download_button(
            "⬇️ Tải xuống interactions đã làm sạch (CSV)",
            csv,
            file_name="interactions_pruned.csv",
            mime="text/csv",
            key="pruning_download_saved"
        )
    
    with tab2:
        st.markdown("### 📉 So sánh độ thưa thớt")
        
        col_sparse1, col_sparse2 = st.columns(2)
        with col_sparse1:
            st.metric("Độ thưa ban đầu", f"{original_sparsity:.4f}")
            st.metric("Mật độ ban đầu", f"{original_density:.6f}")
        with col_sparse2:
            st.metric("Độ thưa sau pruning", f"{pruned_sparsity:.4f}")
            st.metric("Mật độ sau pruning", f"{pruned_density:.6f}")
        
        if improvement > 0:
            st.success(f"✅ Độ thưa giảm {improvement:.4f} ({improvement/original_sparsity*100:.2f}%) - Mật độ dữ liệu tăng!")
        else:
            st.info("ℹ️ Mật độ dữ liệu đã được cải thiện cho các users/products còn lại.")
    
    with tab3:
        if result.get('stats'):
            st.markdown("### 📈 Quá trình Pruning qua các lần lặp")
            stats_df = pd.DataFrame(result['stats'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stats_df['iteration'],
                y=stats_df['users'],
                mode='lines+markers',
                name='Users',
                line=dict(color='#1f77b4')
            ))
            fig.add_trace(go.Scatter(
                x=stats_df['iteration'],
                y=stats_df['products'],
                mode='lines+markers',
                name='Products',
                line=dict(color='#2ca02c')
            ))
            fig.add_trace(go.Scatter(
                x=stats_df['iteration'],
                y=stats_df['interactions'],
                mode='lines+markers',
                name='Interactions',
                line=dict(color='#d62728')
            ))
            fig.update_layout(
                title="Thay đổi số lượng Users, Products và Interactions qua các lần lặp",
                xaxis_title="Số lần lặp",
                yaxis_title="Số lượng",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, key="pruning_stats_chart_saved")
        else:
            st.info("ℹ️ Không có dữ liệu thống kê quá trình pruning.")
    
    with tab4:
        if pruned_users <= 100 and pruned_products <= 100:
            st.markdown("### 🔥 Ma trận tương tác (Heatmap)")
            st.info("ℹ️ Hiển thị ma trận tương tác dưới dạng heatmap (1 = có tương tác, 0 = không có tương tác)")
            
            # Create interaction matrix
            interaction_matrix = result['pruned_interactions'].pivot_table(
                index='user_id',
                columns='product_id',
                aggfunc='size',
                fill_value=0
            )
            
            interaction_matrix = (interaction_matrix > 0).astype(int)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=interaction_matrix.values,
                x=interaction_matrix.columns,
                y=interaction_matrix.index,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Interaction")
            ))
            fig_heatmap.update_layout(
                title="Ma trận tương tác User-Product (1 = có tương tác, 0 = không có)",
                xaxis_title="Product ID",
                yaxis_title="User ID",
                width=800,
                height=600
            )
            st.plotly_chart(fig_heatmap, use_container_width=True, key="pruning_heatmap_chart_saved")
        else:
            st.info(f"ℹ️ Ma trận quá lớn ({pruned_users} users × {pruned_products} products) để hiển thị heatmap. Chỉ hiển thị dữ liệu dạng bảng.")
            st.markdown("**💡 Gợi ý:** Xem dữ liệu dạng bảng trong tab '📋 Ma trận tương tác đã làm sạch'")
    
    st.markdown("""
    **✅ Kết quả đạt được:**
    - ✅ Ma trận tương tác thưa thớt $R$ được làm sạch, giảm nhiễu (noise) do tương tác ngẫu nhiên hoặc không đủ dữ liệu
    - ✅ Tăng mật độ dữ liệu tương tác cho các thuật toán cộng tác (GNN)
    - ✅ Loại bỏ các users và products có quá ít tương tác, giúp model học được patterns rõ ràng hơn
    """)


@st.cache_data
def load_products_data(path: str = None):
    """Load products dataset from exports directory."""
    csv_path = path or os.path.join(current_dir, 'apps', 'exports', 'products.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
            df = df.set_index('id')
        return df
    except Exception:
        return None


@st.cache_data
def load_users_data(path: str = None):
    """Load users dataset from exports directory."""
    csv_path = path or os.path.join(current_dir, 'apps', 'exports', 'users.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
            df = df.set_index('id')
        return df
    except Exception:
        return None


@st.cache_data
def load_interactions_data(path: str = None):
    """Load interactions dataset from exports directory."""
    csv_path = path or os.path.join(current_dir, 'apps', 'exports', 'interactions.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype(str)
        if 'product_id' in df.columns:
            df['product_id'] = df['product_id'].astype(str)
        return df
    except Exception:
        return None


def get_user_record(user_id: str, users_df: pd.DataFrame):
    if users_df is None or user_id is None:
        return None
    try:
        if user_id in users_df.index:
            return users_df.loc[user_id]
        return users_df.loc[users_df.index.astype(str) == str(user_id)].iloc[0]
    except Exception:
        return None


def get_product_record(product_id: str, products_df: pd.DataFrame):
    if products_df is None or product_id is None:
        return None
    product_key = str(product_id)
    try:
        if products_df.index.name is not None or not isinstance(products_df.index, pd.RangeIndex):
            if product_key in products_df.index.astype(str):
                return products_df.loc[product_key]
        if 'id' in products_df.columns:
            match = products_df[products_df['id'].astype(str) == product_key]
            if not match.empty:
                return match.iloc[0]
    except Exception:
        return None
    return None


def ensure_hybrid_predictions(alpha: float, candidate_pool: int = 200):
    """
    Ensure hybrid predictions are available in session state.
    Recompute when alpha changes or cached predictions missing.
    """
    existing = st.session_state.get('hybrid_predictions')
    if existing and abs(existing.get('alpha', alpha) - alpha) < 1e-6:
        return existing

    cbf_predictions = st.session_state.get('cbf_predictions')
    gnn_predictions = (
        st.session_state.get('gnn_predictions')
        or st.session_state.get('gnn_training')
    )

    if cbf_predictions and gnn_predictions:
        combined = combine_hybrid_scores(
            cbf_predictions,
            gnn_predictions,
            alpha=alpha,
            top_k=max(candidate_pool, 50)
        )
        st.session_state['hybrid_predictions'] = combined
        # lưu lại hybrid predictions ra artifacts
        save_predictions_artifact("hybrid", combined)
        return combined

    if cbf_predictions and not gnn_predictions:
        fallback = {
            'predictions': cbf_predictions.get('predictions', {}),
            'rankings': cbf_predictions.get('rankings', {}),
            'alpha': alpha,
            'stats': {'note': 'Fallback to CBF scores (GNN predictions missing)'}
        }
        st.session_state['hybrid_predictions'] = fallback
        return fallback

    return existing


def build_user_interaction_preferences(
    user_id: str,
    interactions_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Derive normalized preference weights from user interaction history.
    Returns dict with article, usage, gender preference maps in [0,1].
    """
    preference_maps = {
        'articleType': defaultdict(float),
        'usage': defaultdict(float),
        'gender': defaultdict(float)
    }

    if (
        interactions_df is None
        or products_df is None
        or interactions_df.empty
        or user_id is None
    ):
        return {k: {} for k in preference_maps}

    user_history = interactions_df[interactions_df['user_id'] == str(user_id)]
    if user_history.empty:
        return {k: {} for k in preference_maps}

    for _, row in user_history.iterrows():
        product_id = str(row.get('product_id'))
        interaction_type = row.get('interaction_type', '').lower()
        weight = INTERACTION_WEIGHTS.get(interaction_type, 1.0)
        product_row = get_product_record(product_id, products_df)
        if product_row is None:
            continue

        article = str(product_row.get('articleType', '')).strip()
        usage = str(product_row.get('usage', '')).strip()
        gender = str(product_row.get('gender', '')).strip()

        if article:
            preference_maps['articleType'][article] += weight
        if usage:
            preference_maps['usage'][usage] += weight
        if gender:
            preference_maps['gender'][gender] += weight

    normalized = {}
    for key, counter in preference_maps.items():
        if not counter:
            normalized[key] = {}
            continue
        max_val = max(counter.values())
        if max_val == 0:
            normalized[key] = {k: 0.0 for k in counter}
        else:
            normalized[key] = {k: v / max_val for k, v in counter.items()}

    return normalized


def build_personalized_candidates(
    user_id: str,
    payload_product_id: str,
    hybrid_predictions: Dict,
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    top_k: int = 10,
    usage_bonus: float = 0.08,
    gender_primary_bonus: float = 0.06,
    gender_secondary_bonus: float = 0.03,
    interaction_weight: float = 0.05,
    usage_pref_weight: float = 0.04
) -> List[Dict]:
    """Compute prioritized personalized recommendations."""
    if (
        hybrid_predictions is None
        or products_df is None
        or payload_product_id is None
    ):
        return []

    payload_row = get_product_record(payload_product_id, products_df)
    if payload_row is None:
        return []

    payload_article = str(payload_row.get('articleType', '')).strip()
    payload_usage = str(payload_row.get('usage', '')).strip()
    payload_gender = str(payload_row.get('gender', '')).strip()
    payload_gender_lower = payload_gender.lower()

    user_record = get_user_record(user_id, users_df)
    user_age = None
    if user_record is not None:
        try:
            user_age = int(user_record.get('age')) if pd.notna(user_record.get('age')) else None
        except (ValueError, TypeError):
            user_age = None
    user_gender = user_record.get('gender') if user_record is not None else None

    allowed_genders = get_allowed_genders(user_age, user_gender)
    preference_maps = build_user_interaction_preferences(
        user_id,
        interactions_df,
        products_df
    )

    # Robustly fetch user scores regardless of user_id key type (str/int)
    predictions_by_user = hybrid_predictions.get('predictions', {}) or {}
    user_scores = None
    user_key_str = str(user_id)
    if user_key_str in predictions_by_user:
        user_scores = predictions_by_user[user_key_str]
    else:
        for key, val in predictions_by_user.items():
            if str(key) == user_key_str:
                user_scores = val
                break

    if not user_scores:
        # Không có bất kỳ dự đoán Hybrid nào cho user này
        return []

    prioritized = []
    seen_product_ids = set()  # Track để tránh duplicate
    
    for product_id, base_score in sorted(
        user_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        product_id_str = str(product_id)
        
        # Skip payload product
        if product_id_str == str(payload_product_id):
            continue
        
        # Skip nếu đã thêm product này rồi (tránh duplicate)
        if product_id_str in seen_product_ids:
            continue
        
        product_row = get_product_record(product_id, products_df)
        if product_row is None:
            continue

        article_type = str(product_row.get('articleType', '')).strip()
        if not article_type or article_type != payload_article:
            continue  # strict articleType requirement

        product_usage = str(product_row.get('usage', '')).strip()
        product_gender = str(product_row.get('gender', '')).strip() or 'Unspecified'
        product_gender_lower = product_gender.lower()
        payload_gender_match = False
        payload_unisex_fallback = False

        if payload_gender:
            if product_gender_lower == payload_gender_lower:
                payload_gender_match = True
            elif product_gender_lower == 'unisex':
                payload_gender_match = True
                payload_unisex_fallback = True
            else:
                continue  # enforce payload gender alignment

        score = float(base_score)
        reasons = []

        if payload_usage and product_usage and product_usage == payload_usage:
            score += usage_bonus
            reasons.append("Ưu tiên do cùng usage")

        if payload_gender:
            if payload_gender_match and not payload_unisex_fallback:
                score += gender_primary_bonus
                reasons.append("Phù hợp gender với sản phẩm đang xem")
            elif payload_unisex_fallback:
                score += gender_secondary_bonus
                reasons.append("Unisex phù hợp với sản phẩm đang xem")
        else:
            if product_gender in allowed_genders:
                score += gender_primary_bonus
                reasons.append("Phù hợp giới tính/độ tuổi")
            elif product_gender_lower == 'unisex' and (user_age or 0) >= 13:
                score += gender_secondary_bonus
                reasons.append("Unisex phù hợp (>=13)")
            else:
                score -= 0.01

        article_pref = preference_maps.get('articleType', {}).get(article_type, 0.0)
        if article_pref > 0:
            score += interaction_weight * article_pref
            reasons.append("Trọng số lịch sử articleType")

        usage_pref = preference_maps.get('usage', {}).get(product_usage, 0.0)
        if usage_pref > 0:
            score += usage_pref_weight * usage_pref
            reasons.append("Trọng số lịch sử usage")

        prioritized.append({
            'product_id': product_id_str,
            'score': score,
            'base_score': base_score,
            'usage_match': product_usage == payload_usage and bool(payload_usage),
            'gender_match': payload_gender_match if payload_gender else (product_gender in allowed_genders),
            'reasons': reasons,
            'product_row': product_row
        })
        
        seen_product_ids.add(product_id_str)  # Mark as seen

    # Sort by score (desc), then by product_id (asc) for deterministic ordering
    prioritized.sort(key=lambda x: (-x['score'], x['product_id']))
    return prioritized[:top_k]


def build_outfit_suggestions(
    user_id: str,
    payload_product_id: str,
    personalized_items: List[Dict],
    products_df: pd.DataFrame,
    hybrid_predictions: Dict,
    user_age: Optional[int],
    user_gender: Optional[str],
    max_outfits: int = 3
) -> List[Dict]:
    """Create outfits that include payload product and satisfy structural rules."""
    if (
        products_df is None
        or personalized_items is None
        or hybrid_predictions is None
    ):
        return []

    payload_row = get_product_record(payload_product_id, products_df)
    if payload_row is None:
        return []

    target_usage = str(payload_row.get('usage', '')).strip()
    target_gender = str(payload_row.get('gender', '')).strip()
    
    # Get allowed genders for user (cần cho trường hợp payload là Unisex)
    allowed_genders_for_user = get_allowed_genders(user_age, user_gender) if get_allowed_genders else []
    
    def gender_allowed(gender_value: str) -> bool:
        gender_clean = str(gender_value).strip()
        if not target_gender:
            return True
        if not gender_clean:
            return False
        gender_lower = gender_clean.lower()
        target_lower = target_gender.lower()
        if gender_lower == target_lower:
            return True
        return gender_lower == 'unisex'

    # Strict pool: same usage + compatible gender (hoặc cùng usage + user gender nếu payload là Unisex)
    usage_gender_filtered = products_df.copy()
    if target_usage:
        usage_gender_filtered = usage_gender_filtered[
            usage_gender_filtered['usage'].astype(str).str.strip() == target_usage
        ]
    if usage_gender_filtered.empty:
        usage_gender_filtered = products_df.copy()

    # Nếu payload là Unisex → cùng usage + phù hợp với gender của User
    is_payload_unisex = str(target_gender).strip().lower() == 'unisex'
    if is_payload_unisex:
        # Filter theo gender của User (không phải gender của payload)
        if 'gender' in usage_gender_filtered.columns and allowed_genders_for_user:
            allowed_set = {str(g).strip().lower() for g in allowed_genders_for_user + ["Unisex"]}
            usage_gender_filtered = usage_gender_filtered[
                usage_gender_filtered['gender'].astype(str).str.strip().str.lower().isin(allowed_set)
            ]
    elif 'gender' in usage_gender_filtered.columns and target_gender:
        # Bình thường: filter theo gender của payload
        usage_gender_filtered = usage_gender_filtered[usage_gender_filtered['gender'].apply(gender_allowed)]
    if usage_gender_filtered.empty:
        usage_gender_filtered = products_df.copy()

    # Relaxed pool: ignore usage, keep only gender-compatible (includes Unisex)
    gender_filtered = products_df.copy()
    if 'gender' in gender_filtered.columns and target_gender:
        gender_filtered = gender_filtered[gender_filtered['gender'].apply(gender_allowed)]
    if gender_filtered.empty:
        gender_filtered = products_df.copy()

    # User-gender pool: filter by user's allowed genders (includes Unisex)
    user_gender_filtered = products_df.copy()
    if 'gender' in user_gender_filtered.columns and allowed_genders_for_user:
        # So sánh case-insensitive và strip whitespace
        allowed_set = {str(g).strip().lower() for g in allowed_genders_for_user + ["Unisex"]}
        user_gender_filtered = user_gender_filtered[
            user_gender_filtered['gender'].astype(str).str.strip().str.lower().isin(allowed_set)
        ]
    if user_gender_filtered.empty:
        user_gender_filtered = products_df.copy()

    # Unisex pool: prioritize Unisex products when missing components
    unisex_filtered = products_df.copy()
    if 'gender' in unisex_filtered.columns:
        unisex_filtered = unisex_filtered[
            unisex_filtered['gender'].astype(str).str.strip().str.lower() == 'unisex'
        ]
    if unisex_filtered.empty:
        unisex_filtered = products_df.copy()

    score_lookup = {
        item['product_id']: item['score']
        for item in personalized_items
    }
    # Robustly fetch user scores regardless of user_id key type (str/int) - đồng bộ với API
    predictions_by_user = hybrid_predictions.get('predictions', {}) or {}
    user_scores = None
    user_key_str = str(user_id)
    if user_key_str in predictions_by_user:
        user_scores = predictions_by_user[user_key_str]
    else:
        for key, val in predictions_by_user.items():
            if str(key) == user_key_str:
                user_scores = val
                break
    if user_scores is None:
        user_scores = {}

    def get_product_score(pid: str) -> float:
        """Robust lookup product score from score_lookup or user_scores."""
        # Try score_lookup first (from personalized_items)
        if pid in score_lookup:
            return score_lookup[pid]
        # Try user_scores with robust lookup (handle both str and int keys)
        pid_str = str(pid)
        if pid_str in user_scores:
            return user_scores[pid_str]
        # Try int key if pid is numeric
        try:
            pid_int = int(pid)
            if pid_int in user_scores:
                return user_scores[pid_int]
        except (ValueError, TypeError):
            pass
        # Try string matching
        for key, val in user_scores.items():
            if str(key) == pid_str:
                return val
        return 0.0
    
    def sort_candidates(df_subset: pd.DataFrame) -> List[str]:
        if df_subset is None or df_subset.empty:
            return []
        # ensure index as string ID
        ids = df_subset.index.astype(str)
        scores = [get_product_score(pid) for pid in ids]
        # Sort by score (desc), then by product_id (asc) for deterministic ordering
        ordered = sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))
        return [pid for pid, _ in ordered]

    def subset_by(df: pd.DataFrame, master=None, subcategories=None):
        if master and 'masterCategory' in df.columns:
            df = df[df['masterCategory'].astype(str).str.lower() == master.lower()]
        if subcategories and 'subCategory' in df.columns:
            sub_values = [s.lower() for s in subcategories]
            df = df[df['subCategory'].astype(str).str.lower().isin(sub_values)]
        return df

    accessory_subs = ['bags', 'belts', 'headwear', 'watches']
    footwear_subs = ['shoes', 'sandal', 'flip flops']

    # Strict: same usage + gender; Relaxed: any usage + same gender/Unisex
    candidates_strict = {
        'accessory': sort_candidates(subset_by(usage_gender_filtered, master='Accessories', subcategories=accessory_subs)),
        'topwear': sort_candidates(subset_by(usage_gender_filtered, master='Apparel', subcategories=['topwear'])),
        'bottomwear': sort_candidates(subset_by(usage_gender_filtered, master='Apparel', subcategories=['bottomwear'])),
        'dress': sort_candidates(subset_by(usage_gender_filtered, master='Apparel', subcategories=['dress'])),
        'innerwear': sort_candidates(subset_by(usage_gender_filtered, master='Apparel', subcategories=['innerwear'])),
        'footwear': sort_candidates(subset_by(usage_gender_filtered, master='Footwear', subcategories=footwear_subs)),
    }

    candidates_relaxed = {
        'accessory': sort_candidates(subset_by(gender_filtered, master='Accessories', subcategories=accessory_subs)),
        'topwear': sort_candidates(subset_by(gender_filtered, master='Apparel', subcategories=['topwear'])),
        'bottomwear': sort_candidates(subset_by(gender_filtered, master='Apparel', subcategories=['bottomwear'])),
        'dress': sort_candidates(subset_by(gender_filtered, master='Apparel', subcategories=['dress'])),
        'innerwear': sort_candidates(subset_by(gender_filtered, master='Apparel', subcategories=['innerwear'])),
        'footwear': sort_candidates(subset_by(gender_filtered, master='Footwear', subcategories=footwear_subs)),
    }

    candidates_user_gender = {
        'accessory': sort_candidates(subset_by(user_gender_filtered, master='Accessories', subcategories=accessory_subs)),
        'topwear': sort_candidates(subset_by(user_gender_filtered, master='Apparel', subcategories=['topwear'])),
        'bottomwear': sort_candidates(subset_by(user_gender_filtered, master='Apparel', subcategories=['bottomwear'])),
        'dress': sort_candidates(subset_by(user_gender_filtered, master='Apparel', subcategories=['dress'])),
        'innerwear': sort_candidates(subset_by(user_gender_filtered, master='Apparel', subcategories=['innerwear'])),
        'footwear': sort_candidates(subset_by(user_gender_filtered, master='Footwear', subcategories=footwear_subs)),
    }

    candidates_unisex = {
        'accessory': sort_candidates(subset_by(unisex_filtered, master='Accessories', subcategories=accessory_subs)),
        'topwear': sort_candidates(subset_by(unisex_filtered, master='Apparel', subcategories=['topwear'])),
        'bottomwear': sort_candidates(subset_by(unisex_filtered, master='Apparel', subcategories=['bottomwear'])),
        'dress': sort_candidates(subset_by(unisex_filtered, master='Apparel', subcategories=['dress'])),
        'innerwear': sort_candidates(subset_by(unisex_filtered, master='Apparel', subcategories=['innerwear'])),
        'footwear': sort_candidates(subset_by(unisex_filtered, master='Footwear', subcategories=footwear_subs)),
    }

    # Fallback cuối cùng: lấy bất kỳ sản phẩm nào trong category, không quan tâm usage/gender
    candidates_any = {
        'accessory': sort_candidates(subset_by(products_df, master='Accessories', subcategories=accessory_subs)),
        'topwear': sort_candidates(subset_by(products_df, master='Apparel', subcategories=['topwear'])),
        'bottomwear': sort_candidates(subset_by(products_df, master='Apparel', subcategories=['bottomwear'])),
        'dress': sort_candidates(subset_by(products_df, master='Apparel', subcategories=['dress'])),
        'innerwear': sort_candidates(subset_by(products_df, master='Apparel', subcategories=['innerwear'])),
        'footwear': sort_candidates(subset_by(products_df, master='Footwear', subcategories=footwear_subs)),
    }

    def detect_categories(row):
        cats = set()
        sub = str(row.get('subCategory', '')).lower()
        master = str(row.get('masterCategory', '')).lower()
        if master == 'accessories' or sub in [s.lower() for s in accessory_subs]:
            cats.add('accessory')
        if sub == 'topwear':
            cats.add('topwear')
        if sub == 'bottomwear':
            cats.add('bottomwear')
        if sub == 'dress':
            cats.add('dress')
        if sub == 'innerwear':
            cats.add('innerwear')
        if master == 'footwear' or sub in [s.lower() for s in footwear_subs]:
            cats.add('footwear')
        return cats

    payload_categories = detect_categories(payload_row)

    # Kiểm tra payload có phải là Dresses không
    payload_article_type = str(payload_row.get('articleType', '')).strip().lower()
    payload_sub_category = str(payload_row.get('subCategory', '')).strip().lower()
    is_payload_dress = payload_article_type == 'dresses' or payload_sub_category == 'dress'
    
    required_categories = ['accessory', 'topwear', 'bottomwear', 'footwear']
    if is_payload_dress:
        required_categories = ['accessory', 'footwear']
    
    optional_categories = []
    if not is_payload_dress and target_gender:
        target_gender_lower = str(target_gender).strip().lower()
        if target_gender_lower in ['women', 'girls']:
            optional_categories.append('dress')
    outfits = []
    category_offsets = defaultdict(int)

    def pick_candidate(cat, used):
        """
        Ưu tiên:
        1. Strict: cùng usage + cùng gender (hoặc Unisex theo sản phẩm payload)
        2. Relaxed usage: bỏ điều kiện usage, giữ gender theo sản phẩm payload (hoặc Unisex)
        3. User-gender: bỏ điều kiện usage + gender payload, chỉ cần phù hợp giới tính user (hoặc Unisex)
        4. Unisex: ưu tiên Unisex khi thiếu thành phần (giảm điều kiện usage/gender)
        5. Any: bất kỳ sản phẩm nào trong category (fallback cuối cùng để đảm bảo có thể tạo outfit)
        """
        payload_article_type = str(payload_row.get('articleType', '')).strip().lower()
        payload_sub_category = str(payload_row.get('subCategory', '')).strip().lower()
        is_payload_dress = payload_article_type == 'dresses' or payload_sub_category == 'dress'
        is_payload_top_or_bottom = payload_sub_category in ['topwear', 'bottomwear']
        
        if is_payload_dress and cat in ['topwear', 'bottomwear']:
            return None
        
        if is_payload_top_or_bottom and cat == 'dress':
            return None
        
        # Nếu payload có gender=Unisex → chỉ dùng candidates_strict (cùng usage)
        is_payload_unisex = str(target_gender).strip().lower() == 'unisex'
        
        if is_payload_unisex:
            # Chỉ sử dụng strict pool (cùng usage) khi payload là Unisex
            pools = [
                ('strict', candidates_strict.get(cat, [])),
            ]
        else:
            # Bình thường: sử dụng tất cả các pools
            pools = [
                ('strict', candidates_strict.get(cat, [])),
                ('relaxed', candidates_relaxed.get(cat, [])),
                ('user_gender', candidates_user_gender.get(cat, [])),
                ('unisex', candidates_unisex.get(cat, [])),
                ('any', candidates_any.get(cat, [])),
            ]
        for pool_key, pool in pools:
            if not pool:
                continue
            offset_key = f"{cat}:{pool_key}"
            start = category_offsets[offset_key]
            for shift in range(len(pool)):
                idx = (start + shift) % len(pool)
                pid = pool[idx]
                if pid in used or pid == str(payload_product_id):
                    continue
                category_offsets[offset_key] = idx + 1
                return pid
        return None

    for outfit_idx in range(max_outfits):
        used = {str(payload_product_id)}
        ordered_products = [str(payload_product_id)]
        missing_required = False

        for cat in required_categories:
            if cat in payload_categories:
                continue
            candidate = pick_candidate(cat, used)
            if candidate:
                used.add(candidate)
                ordered_products.append(candidate)
            else:
                missing_required = True
                break

        if missing_required:
            continue

        for cat in optional_categories:
            if cat in payload_categories:
                continue
            candidate = pick_candidate(cat, used)
            if candidate:
                used.add(candidate)
                ordered_products.append(candidate)

        score = sum(
            get_product_score(pid)
            for pid in ordered_products
        )
        outfits.append({
            'products': ordered_products,
            'score': score
        })

    return outfits
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
    col1, col2 = st.columns([1, 3])

    with col1:
        if score is not None:
            st.metric("Score", f"{score:.4f}")
        image_url = extract_primary_image_url(product_info)
        if image_url:
            st.image(
                image_url,
                caption=product_info.get('productDisplayName', 'Product image'),
                use_container_width=True
            )

    with col2:
        st.markdown(f"**{product_info.get('productDisplayName', 'N/A')}**")
        st.write(
            f"🏷️ **Category**: "
            f"{product_info.get('masterCategory', 'N/A')} > "
            f"{product_info.get('subCategory', 'N/A')} > "
            f"{product_info.get('articleType', 'N/A')}"
        )
        st.write(f"👤 **Gender**: {product_info.get('gender', 'N/A')}")
        st.write(f"🧩 **Usage**: {product_info.get('usage', 'N/A')}")
        st.write(f"🎨 **Color**: {product_info.get('baseColour', 'N/A')}")


def extract_primary_image_url(product_info: Dict) -> Optional[str]:
    """Return first valid image URL from product record if available."""
    if not product_info:
        return None

    images_field = product_info.get('images')
    if images_field is None or (isinstance(images_field, float) and pd.isna(images_field)):
        return None

    if isinstance(images_field, list) and images_field:
        return images_field[0]

    if isinstance(images_field, str):
        stripped = images_field.strip()
        if stripped.startswith('['):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)) and parsed:
                    return parsed[0]
            except (ValueError, SyntaxError):
                pass
        if stripped.startswith('http'):
            return stripped

    return None

def render_metrics_table(df, highlight_model=None):
    if df is None:
        st.warning("Chưa có dữ liệu metrics. Vui lòng chạy tính toán trước.")
        return

    st.markdown("### 📊 Bảng Tổng Hợp Chỉ Số Các Mô Hình")
    
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
            return ['background-color: #e6ffe6'] * len(row)
        return [''] * len(row)

    st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True)

def slugify_model_name(model_name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')

def apply_5core_pruning(interactions_df: pd.DataFrame, min_interactions: int = 2) -> Dict:

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
                            
                            st.markdown("#### Ví dụ tính toán cho từng user:")
                            for i, user_ex in enumerate(user_examples, 1):
                                st.markdown(f"**{i}. {user_ex}**")
                            
                            if avg_formula:
                                st.markdown("#### Công thức tính trung bình:")
                                
                                if "=" in avg_formula:
                                    formula_parts = avg_formula.split("=")
                                    if len(formula_parts) >= 2:
                                        left_side = formula_parts[0].strip()
                                        right_side = "=".join(formula_parts[1:]).strip()
                                        
                                        import re
                                        n_users_match = re.search(r'user(\\d+)', right_side)
                                        n_users = n_users_match.group(1) if n_users_match else "N"
                                        
                                        metric_var = display_name.replace(" ", "_").lower()
                                        
                                        st.markdown(f"""
                                        **Công thức:**
                                        $$\\text{{Trung bình}} = \\frac{{\\sum_{{u=1}}^{{{n_users}}} {display_name}_u}}{{{n_users}}}$$

                                        **Dạng mở rộng:**
                                        $$\\text{{Trung bình}} = \\frac{{{display_name}_{{user1}} + {display_name}_{{user2}} + \\ldots + {display_name}_{{user{n_users}}}}}{{{n_users}}}$$
                                        """)

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
        "gnn": "GNN",
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
            
            restore_all_artifacts()
            
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
        ["📚 Algorithms & Steps", "👗 Recommendations"]
    )
    
    load_cached_predictions_into_session()
    restore_all_artifacts()

    preprocessor, cb_model, gnn_model, hybrid_model = load_models()
    comparison_df = load_comparison_results()

    if page == "📚 Algorithms & Steps":
        st.markdown("## 📚 Algorithms & Steps")
        st.markdown('<div class="sub-header">📚 PHẦN I: TIỀN XỬ LÝ DỮ LIỆU & TẠO TẬP DỮ LIỆU CHUNG (DỮ LIỆU ĐẦU VÀO)</div>', unsafe_allow_html=True)
        st.markdown("")
        with st.expander("Bước 1.1: Xuất dữ liệu từ MongoDB thành CSV", expanded=True):
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

                                st.markdown("### 📊 Kết quả xuất dữ liệu:")
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
                                st.markdown("### 📁 Xem chi tiết dữ liệu đã xuất:")

                                export_dir = ensure_export_directory()
                                
                                tab1, tab2, tab3 = st.tabs(["📦 Products Data", "👥 Users Data", "🔗 Interactions Data"])
                                
                                with tab1:
                                    products_path = export_dir / 'products.csv'
                                    if products_path.exists() and products_result['success']:
                                        st.markdown("#### 📦 Products Data:")
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
                                    else:
                                        st.info("Chưa có dữ liệu Products để hiển thị.")
                                
                                with tab2:
                                    users_path = export_dir / 'users.csv'
                                    if users_path.exists() and users_result['success']:
                                        st.markdown("#### 👥 Users Data:")
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
                                    else:
                                        st.info("Chưa có dữ liệu Users để hiển thị.")
                                
                                with tab3:
                                    interactions_path = export_dir / 'interactions.csv'
                                    if interactions_path.exists() and interactions_result['success']:
                                        st.markdown("#### 🔗 Interactions Data:")
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
                                    else:
                                        st.info("Chưa có dữ liệu Interactions để hiển thị.")
                                
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
        
        with st.expander("Bước 1.2: Làm sạch và Lọc Dữ liệu (Pruning & Sparsity Handling)", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:** Áp dụng kỹ thuật k-Core Pruning để loại bỏ đệ quy các người dùng và sản phẩm có dưới số lượng tương tác tối thiểu (có thể điều chỉnh) nhằm giảm độ thưa thớt của dữ liệu.")
            st.write("**Dữ liệu sử dụng:** `interactions.csv`")
            
            # Hiển thị kết quả đã có nếu đã chạy bước này trước đó
            if 'pruned_interactions' in st.session_state and _is_valid_data(st.session_state['pruned_interactions']):
                st.markdown("---")
                st.markdown("### 📋 Kết quả đã có sẵn (từ lần chạy trước)")
                display_pruning_results(st.session_state['pruned_interactions'])
                st.markdown("---")
                st.markdown("### 🔄 Chạy lại bước này (tùy chọn)")

            # Data source selection
            col_source1, col_source2 = st.columns([2, 1])
            with col_source1:
                use_exported = st.checkbox(
                    "Sử dụng dữ liệu đã xuất từ MongoDB (Bước 1.1)",
                    value=True,
                    key="pruning_use_exported"
                )
            
            interactions_df = None
            
            if use_exported and 'exported_data' in st.session_state and st.session_state['exported_data'].get('interactions_path'):
                interactions_path = st.session_state['exported_data']['interactions_path']
                if os.path.exists(interactions_path):
                    try:
                        interactions_df = pd.read_csv(interactions_path)
                        st.success(f"✅ Đã tải interactions.csv từ dữ liệu đã xuất: {len(interactions_df)} rows")
                    except Exception as e:
                        st.error(f"Lỗi khi đọc file: {str(e)}")
                else:
                    st.warning("File interactions.csv không tồn tại. Vui lòng tải lên file hoặc xuất dữ liệu từ MongoDB.")
            
            if interactions_df is None:
                # Auto import từ apps/exports
                export_dir = ensure_export_directory() if ensure_export_directory else None
                if export_dir:
                    interactions_path_auto = export_dir / 'interactions.csv'
                    if interactions_path_auto.exists():
                        try:
                            interactions_df = pd.read_csv(interactions_path_auto)
                            st.success(f"✅ Đã tự động tải interactions.csv từ apps/exports: {len(interactions_df)} rows × {len(interactions_df.columns)} columns")
                        except Exception as e:
                            st.error(f"Lỗi khi đọc file từ apps/exports: {str(e)}")
                    else:
                        st.info("💡 File interactions.csv không tồn tại trong apps/exports. Vui lòng xuất dữ liệu từ MongoDB (Bước 1.1) hoặc đảm bảo file tồn tại.")
                else:
                    st.info("💡 Không thể truy cập thư mục apps/exports. Vui lòng xuất dữ liệu từ MongoDB (Bước 1.1).")
            
            if interactions_df is not None:
                # Configuration
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    min_interactions = st.number_input(
                        "Số lượng tương tác tối thiểu (min_interactions)",
                        min_value=1,
                        value=2,
                        step=1,
                        key="pruning_min_interactions"
                    )
                
                with col_config2:
                    st.write("")  # Spacing
                    process_button = st.button(
                        f"🔧 Áp dụng {min_interactions}-Core Pruning",
                        type="primary",
                        use_container_width=True,
                        key="pruning_process_button"
                    )
                
                st.markdown(f"""
                **Thuật toán {min_interactions}-Core Pruning:**

                1. **Khởi tạo:** Đếm số lượng tương tác cho mỗi user và mỗi product
                2. **Lặp đệ quy:**
                   - Loại bỏ tất cả users có < {min_interactions} interactions
                   - Loại bỏ tất cả products có < {min_interactions} interactions
                   - Cập nhật lại số lượng interactions của các users/products còn lại
                   - Lặp lại cho đến khi không còn user/product nào bị loại bỏ
                3. **Kết quả:** Ma trận tương tác $R$ được làm sạch, chỉ giữ lại các users và products có đủ dữ liệu

                **Công thức:**
                $$R_{{pruned}} = \\{{(u, i) \\in R : |I_u| \\geq {min_interactions} \\land |U_i| \\geq {min_interactions}\\}}$$

                Trong đó:
                - $R$: Ma trận tương tác gốc
                - $I_u$: Tập sản phẩm mà user $u$ đã tương tác
                - $U_i$: Tập users đã tương tác với sản phẩm $i$
                - $R_{{pruned}}$: Ma trận sau khi pruning
                - ${min_interactions}$: Số lượng tương tác tối thiểu (min_interactions)
                """)
                
                if process_button:
                    with st.spinner(f"Đang áp dụng {min_interactions}-Core Pruning..."):
                        try:
                            result = apply_5core_pruning(interactions_df, min_interactions)
                            
                            if result['pruned_interactions'].empty:
                                st.error("❌ **Kết quả:** Tất cả dữ liệu đã bị loại bỏ!")
                                st.warning(f"""
                        **Nguyên nhân:**
                                - Với min_interactions = {min_interactions}, tất cả users và/hoặc products đều có ít hơn {min_interactions} interactions
                        - Điều này tạo ra hiệu ứng cascade: khi loại bỏ users/products, các interactions liên quan cũng bị loại bỏ, khiến các users/products khác cũng không đủ điều kiện

                        **Giải pháp:**
                        1. Giảm min_interactions xuống (ví dụ: {max(1, min_interactions - 1)} hoặc {max(1, min_interactions - 2)})
                        2. Thu thập thêm dữ liệu interactions
                        3. Chấp nhận dữ liệu thưa thớt và không áp dụng pruning
                                """)
                            else:
                                st.success("✅ **Hoàn thành!** Ma trận tương tác đã được làm sạch.")
                                
                                # Store in session state
                                st.session_state['pruned_interactions'] = result
                                # Lưu vào artifacts để không bị mất khi chạy bước khác
                                save_intermediate_artifact('pruned_interactions', result)
                                
                                # Display statistics
                                st.markdown("### 📊 Thống kê kết quả Pruning")
                                
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                with col_stat1:
                                    st.metric("Users ban đầu", result['original_users'])
                                    st.metric("Users sau pruning", result['original_users'] - result['removed_users'])
                                with col_stat2:
                                    st.metric("Products ban đầu", result['original_products'])
                                    st.metric("Products sau pruning", result['original_products'] - result['removed_products'])
                                with col_stat3:
                                    st.metric("Interactions ban đầu", result['original_interactions'])
                                    st.metric("Interactions sau pruning", len(result['pruned_interactions']))
                                with col_stat4:
                                    st.metric("Số lần lặp", result['iterations'])
                                    reduction_pct = ((result['original_interactions'] - len(result['pruned_interactions'])) / result['original_interactions'] * 100) if result['original_interactions'] > 0 else 0
                                    st.metric("Giảm đi", f"{reduction_pct:.2f}%")
                                
                                # Calculate values for tabs
                                pruned_users = result['original_users'] - result['removed_users']
                                pruned_products = result['original_products'] - result['removed_products']
                                
                                # Create tabs for different visualizations
                                tab1, tab2, tab3 = st.tabs([
                                    "📋 Ma trận tương tác đã làm sạch",
                                    "📈 Quá trình Pruning qua các lần lặp",
                                    "🔥 Ma trận tương tác (Heatmap)"
                                ])
                                
                                with tab1:
                                    st.markdown("### 📋 Ma trận tương tác đã làm sạch $R_{pruned}$")
                                    st.dataframe(
                                        result['pruned_interactions'].head(100),
                                        use_container_width=True
                                    )
                                    
                                    # Download button
                                    csv = result['pruned_interactions'].to_csv(index=False)
                                    st.download_button(
                                        "⬇️ Tải xuống interactions đã làm sạch (CSV)",
                                        csv,
                                        file_name="interactions_pruned.csv",
                                        mime="text/csv",
                                        key="pruning_download"
                                    )
                                
                                with tab2:
                                    if result['stats']:
                                        st.markdown("### 📈 Quá trình Pruning qua các lần lặp")
                                        stats_df = pd.DataFrame(result['stats'])
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=stats_df['iteration'],
                                            y=stats_df['users'],
                                            mode='lines+markers',
                                            name='Users',
                                            line=dict(color='#1f77b4')
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=stats_df['iteration'],
                                            y=stats_df['products'],
                                            mode='lines+markers',
                                            name='Products',
                                            line=dict(color='#2ca02c')
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=stats_df['iteration'],
                                            y=stats_df['interactions'],
                                            mode='lines+markers',
                                            name='Interactions',
                                            line=dict(color='#d62728')
                                        ))
                                        fig.update_layout(
                                            title="Thay đổi số lượng Users, Products và Interactions qua các lần lặp",
                                            xaxis_title="Số lần lặp",
                                            yaxis_title="Số lượng",
                                            hovermode='x unified'
                                        )
                                        st.plotly_chart(fig, use_container_width=True, key="pruning_stats_chart_new")
                                    else:
                                        st.info("ℹ️ Không có dữ liệu thống kê quá trình pruning.")
                                
                                with tab3:
                                    if pruned_users <= 100 and pruned_products <= 100:
                                        st.markdown("### 🔥 Ma trận tương tác (Heatmap)")
                                        st.info("ℹ️ Hiển thị ma trận tương tác dưới dạng heatmap (1 = có tương tác, 0 = không có tương tác)")
                                        
                                        # Create interaction matrix
                                        interaction_matrix = result['pruned_interactions'].pivot_table(
                                            index='user_id',
                                            columns='product_id',
                                            aggfunc='size',
                                            fill_value=0
                                        )
                                        
                                        interaction_matrix = (interaction_matrix > 0).astype(int)
                                        
                                        fig_heatmap = go.Figure(data=go.Heatmap(
                                            z=interaction_matrix.values,
                                            x=interaction_matrix.columns,
                                            y=interaction_matrix.index,
                                            colorscale='YlOrRd',
                                            showscale=True,
                                            colorbar=dict(title="Interaction")
                                        ))
                                        fig_heatmap.update_layout(
                                            title="Ma trận tương tác User-Product (1 = có tương tác, 0 = không có)",
                                            xaxis_title="Product ID",
                                            yaxis_title="User ID",
                                            width=800,
                                            height=600
                                        )
                                        st.plotly_chart(fig_heatmap, use_container_width=True, key="pruning_heatmap_chart_new")
                                    else:
                                        st.info(f"ℹ️ Ma trận quá lớn ({pruned_users} users × {pruned_products} products) để hiển thị heatmap. Chỉ hiển thị dữ liệu dạng bảng.")
                                        st.markdown("**💡 Gợi ý:** Xem dữ liệu dạng bảng trong tab '📋 Ma trận tương tác đã làm sạch'")
                                
                                st.markdown("""
                                **✅ Kết quả đạt được:**
                                - ✅ Ma trận tương tác thưa thớt $R$ được làm sạch, giảm nhiễu (noise) do tương tác ngẫu nhiên hoặc không đủ dữ liệu
                                - ✅ Tăng mật độ dữ liệu tương tác cho các thuật toán cộng tác (GNN)
                                - ✅ Loại bỏ các users và products có quá ít tương tác, giúp model học được patterns rõ ràng hơn
                                """)
                        
                        except Exception as e:
                            st.error(f"❌ Lỗi khi áp dụng pruning: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("💡 Vui lòng tải lên file interactions.csv hoặc xuất dữ liệu từ MongoDB (Bước 1.1) để tiếp tục.")

        with st.expander("Bước 1.3: Mã hóa Đặc trưng Nội dung (Feature Encoding)", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:** Chuyển đổi các đặc trưng phân loại của sản phẩm (masterCategory, subCategory, articleType, baseColour, usage) thành Item Profile Vector $\\mathbf{v}_i$ bằng One-Hot Encoding hoặc Categorical Embedding.")
            st.write("**Dữ liệu sử dụng:** `products.csv`")
            
            # Hiển thị kết quả đã có nếu đã chạy bước này trước đó
            if 'feature_encoding' in st.session_state and _is_valid_data(st.session_state['feature_encoding']):
                result = st.session_state['feature_encoding']
                st.markdown("---")
                st.markdown("### 📋 Kết quả đã có sẵn (từ lần chạy trước)")
                st.success("✅ **Đặc trưng nội dung đã được mã hóa.**")
                
                # Display statistics
                st.markdown("### 📊 Thống kê kết quả Feature Encoding")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    num_products = len(result.get('product_ids', []))
                    st.metric("Số lượng sản phẩm", num_products)
                    st.metric("Số features được mã hóa", len(result.get('feature_dims', {})))
                with col_stat2:
                    st.metric("Tổng số chiều", result.get('total_dims', 0))
                    st.metric("Kích thước ma trận", f"{num_products} × {result.get('total_dims', 0)}")
                with col_stat3:
                    memory_size_mb = (num_products * result.get('total_dims', 0) * 4) / (1024 * 1024)
                    st.metric("Kích thước bộ nhớ (ước tính)", f"{memory_size_mb:.2f} MB")
                
                st.markdown("---")
                st.markdown("### 🔄 Chạy lại bước này (tùy chọn)")

            st.markdown("""
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

            **Kết quả tính toán:**
            - Ma trận đặc trưng $P \\in \\mathbb{R}^{|I| \\times d_c}$, nơi $d_c$ là tổng số chiều đặc trưng nội dung (tổng số giá trị unique của tất cả features)
            - $|I|$: Số lượng sản phẩm

            **Kết quả mong đợi:**
            - Vector $\\mathbf{v}_i$ cho mỗi sản phẩm $i$ trong hệ thống, đại diện cho thuộc tính nội dung của nó
            - Các vector này là đầu vào cơ sở cho CBF (Content-Based Filtering) và Diversity (ILD) metric
            - Mỗi sản phẩm được biểu diễn dưới dạng vector số học, có thể tính toán similarity và distance
            """)

            # Data source selection
            col_source1, col_source2 = st.columns([2, 1])
            with col_source1:
                use_exported = st.checkbox(
                    "Sử dụng dữ liệu đã xuất từ MongoDB (Bước 1.1)",
                    value=True,
                    key="encoding_use_exported"
                )
            
            products_df = None
            
            if use_exported and 'exported_data' in st.session_state and st.session_state['exported_data'].get('products_path'):
                products_path = st.session_state['exported_data']['products_path']
                if os.path.exists(products_path):
                    try:
                        products_df = pd.read_csv(products_path)
                        # Set product_id as index if available
                        if 'id' in products_df.columns:
                            products_df = products_df.set_index('id')
                        st.success(f"✅ Đã tải products.csv từ dữ liệu đã xuất: {len(products_df)} rows")
                    except Exception as e:
                        st.error(f"Lỗi khi đọc file: {str(e)}")
                else:
                    st.warning("File products.csv không tồn tại. Vui lòng tải lên file hoặc xuất dữ liệu từ MongoDB.")
            
            if products_df is None:
                # Auto import từ apps/exports
                export_dir = ensure_export_directory() if ensure_export_directory else None
                if export_dir:
                    products_path_auto = export_dir / 'products.csv'
                    if products_path_auto.exists():
                        try:
                            products_df = pd.read_csv(products_path_auto)
                            # Set product_id as index if available
                            if 'id' in products_df.columns:
                                products_df = products_df.set_index('id')
                            st.success(f"✅ Đã tự động tải products.csv từ apps/exports: {len(products_df)} rows × {len(products_df.columns)} columns")
                        except Exception as e:
                            st.error(f"Lỗi khi đọc file từ apps/exports: {str(e)}")
                    else:
                        st.info("💡 File products.csv không tồn tại trong apps/exports. Vui lòng xuất dữ liệu từ MongoDB (Bước 1.1) hoặc đảm bảo file tồn tại.")
                else:
                    st.info("💡 Không thể truy cập thư mục apps/exports. Vui lòng xuất dữ liệu từ MongoDB (Bước 1.1).")
            
            if products_df is not None:
                # Feature selection
                default_features = ['masterCategory', 'subCategory', 'articleType', 'baseColour', 'usage']
                available_features = [f for f in default_features if f in products_df.columns]
                
                if not available_features:
                    st.warning("⚠️ Không tìm thấy các features mặc định. Vui lòng chọn features từ danh sách có sẵn.")
                    categorical_cols = products_df.select_dtypes(include=['object', 'category']).columns.tolist()
                    selected_features = st.multiselect(
                        "Chọn các features để mã hóa",
                        categorical_cols,
                        default=categorical_cols[:5] if len(categorical_cols) >= 5 else categorical_cols,
                        key="encoding_features"
                    )
                else:
                    selected_features = st.multiselect(
                        "Chọn các features để mã hóa",
                        available_features,
                        default=available_features,
                        key="encoding_features"
                    )
                
                col_config1, col_config2 = st.columns([1, 1])
                with col_config1:
                    st.write("")  # Spacing
                with col_config2:
                    process_button = st.button(
                        "🔧 Áp dụng Feature Encoding",
                        type="primary",
                        use_container_width=True,
                        key="encoding_process_button"
                    )
                
                if process_button:
                    if not selected_features:
                        st.error("❌ Vui lòng chọn ít nhất một feature để mã hóa.")
                    else:
                        with st.spinner("Đang mã hóa đặc trưng nội dung..."):
                            try:
                                result = apply_feature_encoding(products_df, selected_features)
                                
                                if result['total_dims'] == 0:
                                    st.error("❌ Không thể mã hóa. Vui lòng kiểm tra lại dữ liệu.")
                                else:
                                    st.success("✅ **Hoàn thành!** Đặc trưng nội dung đã được mã hóa.")
                                    
                                    # Store in session state
                                    st.session_state['feature_encoding'] = result
                                    # Lưu vào artifacts để không bị mất khi chạy bước khác
                                    save_intermediate_artifact('feature_encoding', result)
                                    
                                    # Display statistics
                                    st.markdown("### 📊 Thống kê kết quả Feature Encoding")
                                    
                                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                                    with col_stat1:
                                        st.metric("Số lượng sản phẩm", len(products_df))
                                        st.metric("Số features được mã hóa", len(selected_features))
                                    with col_stat2:
                                        st.metric("Tổng số chiều", result['total_dims'])
                                        st.metric("Kích thước ma trận", f"{len(products_df)} × {result['total_dims']}")
                                    with col_stat3:
                                        memory_size_mb = (len(products_df) * result['total_dims'] * 4) / (1024 * 1024)  # Assuming float32
                                        st.metric("Kích thước bộ nhớ (ước tính)", f"{memory_size_mb:.2f} MB")
                                    
                                    # Display feature dimensions
                                    st.markdown("### 📐 Chi tiết các Features")
                                    feature_dims_df = pd.DataFrame([
                                        {
                                            'Feature': feat,
                                            'Số giá trị unique': result['feature_dims'].get(feat, 0),
                                            'Start Index': result['feature_mapping'].get(feat, {}).get('start_idx', 0),
                                            'End Index': result['feature_mapping'].get(feat, {}).get('end_idx', 0)
                                        }
                                        for feat in selected_features
                                    ])
                                    st.dataframe(feature_dims_df, use_container_width=True)
                                    
                                    # Display sample encoded vectors
                                    st.markdown("### 🔢 Mẫu Vector đã mã hóa (5 sản phẩm đầu tiên)")
                                    sample_indices = min(5, len(products_df))
                                    sample_matrix = result['encoded_matrix'][:sample_indices, :]
                                    
                                    # Limit to first 20 features for display
                                    max_features_display = min(20, len(result['feature_names']))
                                    sample_matrix_display = sample_matrix[:, :max_features_display]
                                    feature_names_display = result['feature_names'][:max_features_display]
                                    
                                    # Create a more readable display
                                    sample_df = pd.DataFrame(
                                        sample_matrix_display,
                                        index=[f"Product {i+1}" for i in range(sample_indices)],
                                        columns=feature_names_display
                                    )
                                    st.dataframe(sample_df, use_container_width=True)
                                    
                                    if len(result['feature_names']) > 20:
                                        st.info(f"ℹ️ Chỉ hiển thị 20 features đầu tiên. Tổng cộng có {len(result['feature_names'])} features.")
                                    
                                    # Display feature mapping details
                                    with st.expander("📋 Chi tiết Feature Mapping", expanded=False):
                                        for feat in selected_features:
                                            if feat in result['feature_mapping']:
                                                mapping = result['feature_mapping'][feat]
                                                st.markdown(f"#### {feat}")
                                                st.write(f"- **Số giá trị unique:** {result['feature_dims'][feat]}")
                                                st.write(f"- **Chỉ số bắt đầu:** {mapping['start_idx']}")
                                                st.write(f"- **Chỉ số kết thúc:** {mapping['end_idx']}")
                                                values_str = ', '.join(list(mapping['value_to_idx'].keys())[:10])
                                                if len(mapping['value_to_idx']) > 10:
                                                    values_str += f" ... và {len(mapping['value_to_idx']) - 10} giá trị khác"
                                                st.write(f"- **Các giá trị:** {values_str}")
                                    
                                    # Visualize feature distribution
                                    st.markdown("### 📊 Phân bố số lượng giá trị unique theo Feature")
                                    dims_data = {
                                        'Feature': list(result['feature_dims'].keys()),
                                        'Số giá trị unique': list(result['feature_dims'].values())
                                    }
                                    dims_df = pd.DataFrame(dims_data)
                                    fig = px.bar(
                                        dims_df,
                                        x='Feature',
                                        y='Số giá trị unique',
                                        title="Số lượng giá trị unique của mỗi feature",
                                        labels={'Feature': 'Feature', 'Số giá trị unique': 'Số giá trị unique'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display matrix info
                                    st.markdown("### 📐 Thông tin Ma trận đặc trưng $P$")
                                    st.latex(f"P \\in \\mathbb{{R}}^{{{len(products_df)} \\times {result['total_dims']}}}")
                                    
                                    # Sparsity of encoded matrix
                                    total_elements = len(products_df) * result['total_dims']
                                    non_zero_elements = np.count_nonzero(result['encoded_matrix'])
                                    sparsity = 1 - (non_zero_elements / total_elements) if total_elements > 0 else 0
                                    
                                    col_matrix1, col_matrix2 = st.columns(2)
                                    with col_matrix1:
                                        st.metric("Tổng số phần tử", f"{total_elements:,}")
                                        st.metric("Phần tử khác 0", f"{non_zero_elements:,}")
                                    with col_matrix2:
                                        st.metric("Độ thưa", f"{sparsity:.4f}")
                                        density = 1 - sparsity
                                        st.metric("Mật độ", f"{density:.4f}")
                                    
                                    st.info("ℹ️ Ma trận One-Hot Encoding thường có độ thưa cao vì mỗi hàng chỉ có một số phần tử bằng 1 (tương ứng với các giá trị của features).")
                                    
                                    st.markdown("""
                                    **✅ Kết quả đạt được:**
                                    - ✅ Vector $\\mathbf{v}_i$ cho mỗi sản phẩm $i$ trong hệ thống, đại diện cho thuộc tính nội dung của nó
                                    - ✅ Ma trận đặc trưng $P \\in \\mathbb{R}^{|I| \\times d_c}$ được tạo thành
                                    - ✅ Các vector này là đầu vào cơ sở cho CBF (Content-Based Filtering) và Diversity (ILD) metric
                                    - ✅ Mỗi sản phẩm được biểu diễn dưới dạng vector số học, có thể tính toán similarity và distance
                                    """)
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi mã hóa features: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
            else:
                st.info("💡 Vui lòng tải lên file products.csv hoặc xuất dữ liệu từ MongoDB (Bước 1.1) để tiếp tục.")

        st.markdown('<div class="sub-header">📚 PHẦN II: MÔ HÌNH LỌC DỰA TRÊN NỘI DUNG (CONTENT-BASED FILTERING - CBF)</div>', unsafe_allow_html=True)
        st.markdown("")

        with st.expander("Bước 2.1: Xây dựng Hồ sơ Người dùng Có Trọng số (Weighted User Profile)", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:** Vector Hồ sơ Người dùng $\\mathbf{P}_u$ được xây dựng bằng cách tổng hợp có trọng số các Item Profile $\\mathbf{v}_i$ của các sản phẩm mà người dùng đã tương tác tích cực.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 1.2 (Pruned Interactions) và Bước 1.3 (Feature Encoding)")

            st.markdown("""
            **Công thức Vector Hồ sơ Người dùng:**
            $$\\mathbf{P}_u = \\frac{\\sum_{i \\in I_u^+} w_{ui} \\mathbf{v}_i}{\\sum_{i \\in I_u^+} w_{ui}}$$
                
                Trong đó:
            - $\\mathbf{P}_u$: Vector hồ sơ người dùng $u$
            - $I_u^+$: Tập hợp các sản phẩm đã tương tác của user $u$
            - $w_{ui}$: Trọng số tương tác giữa user $u$ và item $i$
            - $\\mathbf{v}_i$: Item Profile Vector của sản phẩm $i$

            **Trọng số tương tác ($w_{ui}$):**
            | interaction_type | $w_{ui}$ | Độ Ưu tiên |
            |------------------|----------|------------|
            | purchase | 5.0 | Cao nhất (sở thích rõ ràng) |
            | like | 3.0 | Sở thích mạnh mẽ |
            | cart | 2.0 | Ý định mua sắm |
            | view | 1.0 | Tương tác thụ động |

            **Kết quả mong đợi:**
            - Vector $\\mathbf{P}_u$ (Hồ sơ Người dùng) được tính toán cho mỗi người dùng
            - Đại diện cho sở thích trung bình có trọng số của họ trong không gian thuộc tính sản phẩm
            - Vector này là cơ sở để tính toán điểm tương đồng cho CBF (Content-Based Filtering)
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_pruned_interactions = 'pruned_interactions' in st.session_state
            has_feature_encoding = 'feature_encoding' in st.session_state

            if not has_pruned_interactions:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 1.2 (Pruning). Vui lòng chạy Bước 1.2 trước.")
            if not has_feature_encoding:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 1.3 (Feature Encoding). Vui lòng chạy Bước 1.3 trước.")

            if has_pruned_interactions and has_feature_encoding:
                if build_weighted_user_profile is None:
                    st.error(f"❌ Không thể import user_profile module: {_user_profile_import_error}")
                    st.info("Vui lòng đảm bảo file apps/utils/user_profile.py tồn tại và có thể import được.")
                else:
                    # Lấy dữ liệu từ session state
                    pruning_result = st.session_state['pruned_interactions']
                    encoding_result = st.session_state['feature_encoding']
                    
                    pruned_interactions_df = pruning_result['pruned_interactions']
                    encoded_matrix = encoding_result['encoded_matrix']
                    product_ids = encoding_result['product_ids']
                    
                    # Hiển thị thông tin dữ liệu
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.info(f"📋 Interactions: {len(pruned_interactions_df)} rows")
                        st.info(f"📐 Feature Matrix: {encoded_matrix.shape[0]} products × {encoded_matrix.shape[1]} features")
                    with col_info2:
                        st.info(f"👥 Users: {pruned_interactions_df['user_id'].nunique()}")
                        st.info(f"📦 Products: {pruned_interactions_df['product_id'].nunique()}")
                    
                    # Kiểm tra interaction_type
                    has_interaction_type = 'interaction_type' in pruned_interactions_df.columns
                    if not has_interaction_type:
                        st.warning("⚠️ Không tìm thấy cột 'interaction_type'. Sẽ sử dụng trọng số mặc định = 1.0 cho tất cả interactions.")
                    
                    # Hiển thị bảng trọng số
                    st.markdown("### ⚖️ Trọng số tương tác")
                    weights_df = pd.DataFrame([
                        {'Interaction Type': k, 'Weight': v, 'Mô tả': {
                            'purchase': 'Cao nhất (sở thích rõ ràng)',
                            'like': 'Sở thích mạnh mẽ',
                            'cart': 'Ý định mua sắm',
                            'view': 'Tương tác thụ động'
                        }.get(k, 'Mặc định')}
                        for k, v in INTERACTION_WEIGHTS.items()
                    ])
                    st.dataframe(weights_df, use_container_width=True)
                    
                    # Button để tính toán
                    process_button = st.button(
                        "🔧 Xây dựng Weighted User Profiles",
                        type="primary",
                        use_container_width=True,
                        key="user_profile_process_button"
                    )
                    
                    if process_button:
                        # Đo Training Time (Bước 2.1: xây dựng P_u)
                        training_start_time = time.time()
                        
                        with st.spinner("Đang xây dựng hồ sơ người dùng có trọng số..."):
                            try:
                                result = build_weighted_user_profile(
                                    pruned_interactions_df,
                                    encoded_matrix,
                                    product_ids,
                                    INTERACTION_WEIGHTS
                                )
                                
                                # Kết thúc đo Training Time
                                training_end_time = time.time()
                                training_time_measured = training_end_time - training_start_time
                                
                                # Lưu vào session state
                                st.session_state['training_time'] = training_time_measured
                                
                                if result['total_users'] == 0:
                                    st.error("❌ Không thể xây dựng user profiles. Vui lòng kiểm tra lại dữ liệu.")
                                else:
                                    st.success(f"✅ **Hoàn thành!** Đã xây dựng {result['total_users']} user profiles.")
                                    
                                    st.session_state['user_profiles'] = result
                                    # Lưu vào artifacts để không bị mất khi chạy bước khác
                                    save_intermediate_artifact('user_profiles', result)
                                    
                                    st.markdown("### 📊 Thống kê kết quả User Profiles")
                                    
                                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                                    with col_stat1:
                                        st.metric("Tổng số users", result['total_users'])
                                        st.metric("Số chiều feature vector", result['feature_dim'])
                                    with col_stat2:
                                        total_interactions = sum(stat['interaction_count'] for stat in result['user_stats'].values())
                                        avg_interactions = total_interactions / result['total_users'] if result['total_users'] > 0 else 0
                                        st.metric("Tổng interactions", total_interactions)
                                        st.metric("Trung bình interactions/user", f"{avg_interactions:.2f}")
                                    with col_stat3:
                                        total_weight = sum(stat['total_weight'] for stat in result['user_stats'].values())
                                        avg_weight = total_weight / result['total_users'] if result['total_users'] > 0 else 0
                                        st.metric("Tổng trọng số", f"{total_weight:.2f}")
                                        st.metric("Trung bình trọng số/user", f"{avg_weight:.2f}")
                                    
                                    # Hiển thị cảnh báo về skipped products nếu có
                                    if result.get('skipped_products', 0) > 0:
                                        st.warning(f"⚠️ Có {result['skipped_products']} products trong interactions không tìm thấy trong encoded matrix. Các products này sẽ bị bỏ qua khi tính toán user profiles.")
                                        if result.get('skipped_product_ids'):
                                            with st.expander("Xem danh sách products bị skip (10 đầu tiên)", expanded=False):
                                                st.write(result['skipped_product_ids'])
                                    
                                    # Create tabs for different visualizations
                                    tab1, tab2, tab3 = st.tabs([
                                        "📋 Mẫu User Profiles",
                                        "📊 Phân bố số lượng Interactions",
                                        "📈 Phân bố Trọng số"
                                    ])
                                    
                                    with tab1:
                                        st.markdown("### 📋 Mẫu User Profiles (5 users đầu tiên)")
                                        
                                        # Lấy 5 users đầu tiên
                                        sample_users = list(result['user_profiles'].keys())[:5]
                                        
                                        for idx, user_id in enumerate(sample_users, 1):
                                            profile = result['user_profiles'][user_id]
                                            stats = result['user_stats'][user_id]
                                            
                                            with st.expander(f"User {user_id} (Interactions: {stats['interaction_count']}, Total Weight: {stats['total_weight']:.2f})", expanded=False):
                                                # Hiển thị một phần vector (20 features đầu)
                                                max_features_display = min(20, len(profile))
                                                profile_display = profile[:max_features_display]
                                                
                                                profile_df = pd.DataFrame({
                                                    'Feature Index': range(max_features_display),
                                                    'Value': profile_display
                                                })
                                                st.dataframe(profile_df, use_container_width=True)
                                                
                                                if len(profile) > max_features_display:
                                                    st.info(f"ℹ️ Chỉ hiển thị {max_features_display} features đầu tiên. Tổng cộng có {len(profile)} features.")
                                                
                                                # Thống kê vector
                                                col_vec1, col_vec2, col_vec3 = st.columns(3)
                                                with col_vec1:
                                                    st.metric("Min", f"{profile.min():.4f}")
                                                with col_vec2:
                                                    st.metric("Max", f"{profile.max():.4f}")
                                                with col_vec3:
                                                    st.metric("Mean", f"{profile.mean():.4f}")
                                    
                                    with tab2:
                                        st.markdown("### 📊 Phân bố số lượng Interactions per User")
                                        
                                        interaction_counts = [stats['interaction_count'] for stats in result['user_stats'].values()]
                                        counts_df = pd.DataFrame({
                                            'User': range(len(interaction_counts)),
                                            'Interaction Count': interaction_counts
                                        })
                                        
                                        fig = px.histogram(
                                            counts_df,
                                            x='Interaction Count',
                                            nbins=20,
                                            title="Phân bố số lượng interactions của mỗi user",
                                            labels={'Interaction Count': 'Số lượng Interactions', 'count': 'Số lượng Users'}
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Thống kê
                                        col_dist1, col_dist2, col_dist3 = st.columns(3)
                                        with col_dist1:
                                            st.metric("Min interactions", min(interaction_counts))
                                            st.metric("Max interactions", max(interaction_counts))
                                        with col_dist2:
                                            st.metric("Mean", f"{np.mean(interaction_counts):.2f}")
                                            st.metric("Median", f"{np.median(interaction_counts):.2f}")
                                        with col_dist3:
                                            st.metric("Std", f"{np.std(interaction_counts):.2f}")
                                    
                                    with tab3:
                                        st.markdown("### 📈 Phân bố Trọng số per User")
                                        
                                        total_weights = [stats['total_weight'] for stats in result['user_stats'].values()]
                                        avg_weights = [stats['avg_weight'] for stats in result['user_stats'].values()]
                                        
                                        weights_df = pd.DataFrame({
                                            'User': range(len(total_weights)),
                                            'Total Weight': total_weights,
                                            'Average Weight': avg_weights
                                        })
                                        
                                        fig = go.Figure()
                                        fig.add_trace(go.Histogram(
                                            x=weights_df['Total Weight'],
                                            name='Total Weight',
                                            nbinsx=20,
                                            opacity=0.7
                                        ))
                                        fig.add_trace(go.Histogram(
                                            x=weights_df['Average Weight'],
                                            name='Average Weight',
                                            nbinsx=20,
                                            opacity=0.7
                                        ))
                                        fig.update_layout(
                                            title="Phân bố trọng số của mỗi user",
                                            xaxis_title="Trọng số",
                                            yaxis_title="Số lượng Users",
                                            barmode='overlay'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Thống kê
                                        col_weight1, col_weight2 = st.columns(2)
                                        with col_weight1:
                                            st.markdown("#### Total Weight")
                                            st.metric("Min", f"{min(total_weights):.2f}")
                                            st.metric("Max", f"{max(total_weights):.2f}")
                                            st.metric("Mean", f"{np.mean(total_weights):.2f}")
                                        with col_weight2:
                                            st.markdown("#### Average Weight")
                                            st.metric("Min", f"{min(avg_weights):.2f}")
                                            st.metric("Max", f"{max(avg_weights):.2f}")
                                            st.metric("Mean", f"{np.mean(avg_weights):.2f}")
                                    
                                    # Ví dụ tính toán
                                    st.markdown("### 🧮 Ví dụ tính toán")
                                    st.markdown("""
                                    **Ví dụ:** User $u$ tương tác ba sản phẩm:
                                    - Product 1: purchase (weight=5.0), vector $\\mathbf{v}_1 = [1, 1, 1]$
                                    - Product 2: like (weight=3.0), vector $\\mathbf{v}_2 = [0, 1, 0]$
                                    - Product 3: view (weight=1.0), vector $\\mathbf{v}_3 = [1, 0, 1]$
                                    
                                    **Tính toán:**
                                    - $\\sum w_{ui} \\mathbf{v}_i = 5[1, 1, 1] + 3[0, 1, 0] + 1[1, 0, 1] = [5, 5, 5] + [0, 3, 0] + [1, 0, 1] = [6, 8, 6]$
                                    - $\\sum w_{ui} = 5 + 3 + 1 = 9$
                                    - $\\mathbf{P}_u = [6/9, 8/9, 6/9] = [0.67, 0.89, 0.67]$
                                    """)
                                    
                                    st.markdown("""
                                    **✅ Kết quả đạt được:**
                                    - ✅ Vector $\\mathbf{P}_u$ (Hồ sơ Người dùng) được tính toán cho mỗi người dùng
                                    - ✅ Đại diện cho sở thích trung bình có trọng số của họ trong không gian thuộc tính sản phẩm
                                    - ✅ Vector này là cơ sở để tính toán điểm tương đồng cho CBF (Content-Based Filtering)
                                    """)
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi xây dựng user profiles: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
            else:
                st.info("💡 Vui lòng hoàn thành Bước 1.2 (Pruning) và Bước 1.3 (Feature Encoding) trước khi tiếp tục.")

        with st.expander("Bước 2.2: Tính Điểm Dự đoán và Xếp hạng", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:** Tính độ tương đồng Cosine giữa Hồ sơ Người dùng $\\mathbf{P}_u$ và Item Profile $\\mathbf{v}_i$ để dự đoán điểm tương tác $\\hat{r}_{ui}^{\\text{CBF}}$.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 2.1 (User Profiles) và Bước 1.3 (Feature Encoding)")

            st.markdown("""
            **Công thức Tính điểm (Tương đồng Cosine):**
            $$\\hat{r}_{ui}^{\\text{CBF}} = \\text{cos}(\\mathbf{P}_u, \\mathbf{v}_i) = \\frac{\\mathbf{P}_u \\cdot \\mathbf{v}_i}{\\|\\mathbf{P}_u\\| \\|\\mathbf{v}_i\\|}$$
                
                Trong đó:
            - $\\hat{r}_{ui}^{\\text{CBF}}$: Điểm dự đoán CBF cho user $u$ và item $i$
            - $\\mathbf{P}_u$: Vector hồ sơ người dùng $u$
            - $\\mathbf{v}_i$: Item Profile Vector của sản phẩm $i$
            - $\\mathbf{P}_u \\cdot \\mathbf{v}_i$: Tích vô hướng của hai vectors
            - $\\|\\mathbf{P}_u\\|$, $\\|\\mathbf{v}_i\\|$: Chuẩn L2 của các vectors
            - Kết quả: Điểm số trong khoảng $[-1, 1]$ (1 = hoàn toàn tương đồng, -1 = hoàn toàn đối lập)

            **Kết quả mong đợi:**
            - Một danh sách các sản phẩm tiềm năng được gán điểm $\\hat{r}_{ui}^{\\text{CBF}} \\in [-1, 1]$
            - Điểm số này phản ánh mức độ phù hợp về mặt thuộc tính nội dung giữa sản phẩm và sở thích lịch sử của người dùng
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_user_profiles = 'user_profiles' in st.session_state
            has_feature_encoding = 'feature_encoding' in st.session_state

            if not has_user_profiles:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 2.1 (User Profiles). Vui lòng chạy Bước 2.1 trước.")
            if not has_feature_encoding:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 1.3 (Feature Encoding). Vui lòng chạy Bước 1.3 trước.")

            if has_user_profiles and has_feature_encoding:
                if compute_cbf_predictions is None:
                    st.error(f"❌ Không thể import user_profile module: {_user_profile_import_error}")
                    st.info("Vui lòng đảm bảo file apps/utils/user_profile.py tồn tại và có thể import được.")
                else:
                    # Lấy dữ liệu từ session state
                    user_profiles_result = st.session_state['user_profiles']
                    encoding_result = st.session_state['feature_encoding']
                    
                    user_profiles = user_profiles_result['user_profiles']
                    encoded_matrix = encoding_result['encoded_matrix']
                    product_ids = encoding_result['product_ids']
                    
                    # Hiển thị thông tin dữ liệu
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.info(f"👥 Users: {len(user_profiles)}")
                        st.info(f"📐 Feature Matrix: {encoded_matrix.shape[0]} products × {encoded_matrix.shape[1]} features")
                    with col_info2:
                        st.info(f"📦 Products: {len(product_ids)}")
                        st.info(f"🔢 Total Predictions: {len(user_profiles) * len(product_ids):,}")
                    
                    # Configuration
                    col_config1, col_config2 = st.columns(2)
                    with col_config1:
                        top_k = st.number_input(
                            "Số lượng sản phẩm Top-K để xếp hạng",
                            min_value=5,
                            max_value=100,
                            value=20,
                            step=5,
                            key="cbf_top_k"
                        )
                    
                    with col_config2:
                        st.write("")  # Spacing
                        process_button = st.button(
                            "🔧 Tính Điểm Dự đoán và Xếp hạng",
                            type="primary",
                            use_container_width=True,
                            key="cbf_predictions_process_button"
                        )
                    
                    if process_button:
                        with st.spinner("Đang tính điểm dự đoán và xếp hạng..."):
                            try:
                                result = compute_cbf_predictions(
                                    user_profiles,
                                    encoded_matrix,
                                    product_ids,
                                    top_k=top_k
                                )
                                
                                if result['stats']['total_predictions'] == 0:
                                    st.error("❌ Không thể tính điểm dự đoán. Vui lòng kiểm tra lại dữ liệu.")
                                else:
                                    st.success(f"✅ **Hoàn thành!** Đã tính điểm dự đoán cho {result['stats']['total_users']} users và {result['stats']['total_products']} products.")
                                    
                                    # Store in session state & lưu ra artifacts
                                    st.session_state['cbf_predictions'] = result
                                    save_predictions_artifact("cbf", result)
                                    
                                    # Display statistics
                                    st.markdown("### 📊 Thống kê kết quả Predictions")
                                    
                                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                                    with col_stat1:
                                        st.metric("Tổng số predictions", f"{result['stats']['total_predictions']:,}")
                                        st.metric("Số users", result['stats']['total_users'])
                                    with col_stat2:
                                        st.metric("Số products", result['stats']['total_products'])
                                        st.metric("Top-K", top_k)
                                    with col_stat3:
                                        st.metric("Min score", f"{result['stats']['min_score']:.4f}")
                                        st.metric("Max score", f"{result['stats']['max_score']:.4f}")
                                        st.metric("Mean score", f"{result['stats']['mean_score']:.4f}")
                                        st.metric("Std score", f"{result['stats']['std_score']:.4f}")
                                    
                                    # Create tabs for different visualizations
                                    tab1, tab2, tab3 = st.tabs([
                                        "📋 Mẫu Rankings (Top-K)",
                                        "📊 Phân bố Điểm số",
                                        "🔍 Chi tiết Predictions"
                                    ])
                                    
                                    with tab1:
                                        st.markdown(f"### 📋 Mẫu Rankings Top-{top_k} (5 users đầu tiên)")
                                        
                                        # Lấy 5 users đầu tiên
                                        sample_users = list(result['rankings'].keys())[:5]
                                        
                                        for idx, user_id in enumerate(sample_users, 1):
                                            ranking = result['rankings'][user_id]
                                            
                                            with st.expander(f"User {user_id} - Top {len(ranking)} sản phẩm", expanded=False):
                                                ranking_df = pd.DataFrame([
                                                    {
                                                        'Rank': rank + 1,
                                                        'Product ID': product_id,
                                                        'Score': f"{score:.4f}"
                                                    }
                                                    for rank, (product_id, score) in enumerate(ranking)
                                                ])
                                                st.dataframe(ranking_df, use_container_width=True)
                                    
                                    with tab2:
                                        st.markdown("### 📊 Phân bố Điểm số Predictions")
                                        
                                        # Lấy tất cả scores
                                        all_scores = []
                                        for user_preds in result['predictions'].values():
                                            all_scores.extend(user_preds.values())
                                        
                                        scores_df = pd.DataFrame({
                                            'Score': all_scores
                                        })
                                        
                                        # Histogram
                                        fig = px.histogram(
                                            scores_df,
                                            x='Score',
                                            nbins=50,
                                            title="Phân bố điểm số predictions (Cosine Similarity)",
                                            labels={'Score': 'Điểm số (Cosine Similarity)', 'count': 'Số lượng'}
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Box plot
                                        fig_box = go.Figure()
                                        fig_box.add_trace(go.Box(
                                            y=all_scores,
                                            name='Predictions Scores',
                                            boxmean='sd'
                                        ))
                                        fig_box.update_layout(
                                            title="Box Plot - Phân bố điểm số predictions",
                                            yaxis_title="Điểm số (Cosine Similarity)"
                                        )
                                        st.plotly_chart(fig_box, use_container_width=True)
                                        
                                        # Thống kê chi tiết
                                        col_dist1, col_dist2 = st.columns(2)
                                        with col_dist1:
                                            st.markdown("#### Thống kê mô tả")
                                            stats_desc = pd.DataFrame({
                                                'Metric': ['Min', 'Q1 (25%)', 'Median (50%)', 'Q3 (75%)', 'Max', 'Mean', 'Std'],
                                                'Value': [
                                                    f"{np.min(all_scores):.4f}",
                                                    f"{np.percentile(all_scores, 25):.4f}",
                                                    f"{np.percentile(all_scores, 50):.4f}",
                                                    f"{np.percentile(all_scores, 75):.4f}",
                                                    f"{np.max(all_scores):.4f}",
                                                    f"{np.mean(all_scores):.4f}",
                                                    f"{np.std(all_scores):.4f}"
                                                ]
                                            })
                                            st.dataframe(stats_desc, use_container_width=True)
                                        
                                        with col_dist2:
                                            st.markdown("#### Phân bố theo khoảng")
                                            # Phân chia thành các khoảng
                                            bins = np.linspace(-1, 1, 21)  # 20 bins từ -1 đến 1
                                            hist, bin_edges = np.histogram(all_scores, bins=bins)
                                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                            
                                            dist_df = pd.DataFrame({
                                                'Khoảng': [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(hist))],
                                                'Số lượng': hist,
                                                'Tỉ lệ (%)': (hist / len(all_scores) * 100).round(2)
                                            })
                                            st.dataframe(dist_df, use_container_width=True)
                                    
                                    with tab3:
                                        st.markdown("### 🔍 Chi tiết Predictions cho một User")
                                        
                                        # Chọn user
                                        selected_user = st.selectbox(
                                            "Chọn User để xem chi tiết",
                                            list(result['predictions'].keys()),
                                            key="cbf_user_selector"
                                        )
                                        
                                        if selected_user:
                                            user_predictions = result['predictions'][selected_user]
                                            user_ranking = result['rankings'][selected_user]
                                            
                                            col_detail1, col_detail2 = st.columns(2)
                                            with col_detail1:
                                                st.metric("Tổng số predictions", len(user_predictions))
                                                st.metric("Top score", f"{user_ranking[0][1]:.4f}" if user_ranking else "N/A")
                                                st.metric("Min score", f"{min(user_predictions.values()):.4f}")
                                            with col_detail2:
                                                st.metric("Max score", f"{max(user_predictions.values()):.4f}")
                                                st.metric("Mean score", f"{np.mean(list(user_predictions.values())):.4f}")
                                                st.metric("Median score", f"{np.median(list(user_predictions.values())):.4f}")
                                            
                                            # Hiển thị top-K
                                            st.markdown(f"#### Top-{top_k} Recommendations cho User {selected_user}")
                                            top_k_df = pd.DataFrame([
                                                {
                                                    'Rank': rank + 1,
                                                    'Product ID': product_id,
                                                    'Score': score,
                                                    'Score (Rounded)': f"{score:.4f}"
                                                }
                                                for rank, (product_id, score) in enumerate(user_ranking)
                                            ])
                                            st.dataframe(top_k_df, use_container_width=True)
                                            
                                            # Biểu đồ top-K scores
                                            fig = px.bar(
                                                top_k_df,
                                                x='Rank',
                                                y='Score',
                                                title=f"Top-{top_k} Scores cho User {selected_user}",
                                                labels={'Rank': 'Xếp hạng', 'Score': 'Điểm số (Cosine Similarity)'}
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Ví dụ tính toán
                                    st.markdown("### 🧮 Ví dụ tính toán")
                                    st.markdown("""
                                    **Ví dụ:** User $u$ có $\\mathbf{P}_u \\approx [0.89, 1.0, 0.67]$ và sản phẩm $i_{\\text{cand}}$ có $\\mathbf{v}_{\\text{cand}} = [1, 1, 0]$ (Red, Casual, Women):
                                    
                                    **Tính toán:**
                                    - Tích vô hướng: $\\mathbf{P}_u \\cdot \\mathbf{v}_{\\text{cand}} = (0.89 \\times 1) + (1.0 \\times 1) + (0.67 \\times 0) = 0.89 + 1.0 + 0 = 1.89$
                                    - Chuẩn L2: $\\|\\mathbf{P}_u\\| = \\sqrt{0.89^2 + 1.0^2 + 0.67^2} = \\sqrt{0.7921 + 1.0 + 0.4489} \\approx 1.57$
                                    - Chuẩn L2: $\\|\\mathbf{v}_{\\text{cand}}\\| = \\sqrt{1^2 + 1^2 + 0^2} = \\sqrt{2} \\approx 1.41$
                                    - Điểm Dự đoán: $\\hat{r}_{ui}^{\\text{CBF}} = \\frac{1.89}{1.57 \\times 1.41} \\approx \\frac{1.89}{2.21} \\approx 0.85$
                                    
                                    **Kết quả:** Điểm số $0.85$ cho thấy sản phẩm này có độ tương đồng cao với sở thích của user (gần 1.0 = hoàn toàn tương đồng).
                                    """)
                                    
                                    st.markdown("""
                                    **✅ Kết quả đạt được:**
                                    - ✅ Một danh sách các sản phẩm tiềm năng được gán điểm $\\hat{r}_{ui}^{\\text{CBF}} \\in [-1, 1]$
                                    - ✅ Điểm số này phản ánh mức độ phù hợp về mặt thuộc tính nội dung giữa sản phẩm và sở thích lịch sử của người dùng
                                    - ✅ Top-K rankings cho mỗi user, sẵn sàng cho recommendation
                                    """)
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi tính điểm dự đoán: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
            else:
                st.info("💡 Vui lòng hoàn thành Bước 2.1 (User Profiles) và Bước 1.3 (Feature Encoding) trước khi tiếp tục.")

        with st.expander("Bước 2.3: Tạo Danh sách gợi ý cá nhân hóa", expanded=True):
            st.write("**Nội dung thực hiện:** Quy trình tạo ra danh sách Top-K Personalized dựa trên hai cấp độ lọc cứng (strict filtering) và sau đó là ưu tiên (prioritization) bằng điểm mô hình.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 2.2 (CBF Predictions) và dữ liệu Products/Users")

            st.markdown("""
            **Quy trình lọc và xếp hạng:**
            
            1. **Lọc Cứng theo articleType (STRICT):**
               - Logic: $i_{\\text{cand}} \\in I_{\\text{valid}}$ nếu và chỉ nếu $i_{\\text{cand}}.\\text{articleType} = i_{\\text{payload}}.\\text{articleType}$
               - Kết quả: Loại bỏ tất cả các sản phẩm không cùng loại với sản phẩm đầu vào
            
            2. **Lọc và Ưu tiên theo Giới tính/Độ tuổi (Age/Gender Priority):**
               - **Logic Áp dụng (Strict Filtering):**
                 - Nếu $u.\\text{age} < 13$ và $u.\\text{gender} = \\text{'male'}$: $i_{\\text{cand}}.\\text{gender}$ phải là $\\text{'Boys'}$
                 - Nếu $u.\\text{age} \\ge 13$ và $u.\\text{gender} = \\text{'female'}$: $i_{\\text{cand}}.\\text{gender}$ phải là $\\text{'Women'}$ hoặc $\\text{'Unisex'}$
               - **Phân tích Ưu tiên/Xếp hạng:** Các sản phẩm còn lại sau khi lọc cứng được xếp hạng trực tiếp bằng điểm mô hình ($\\hat{r}_{ui}^{\\text{CBF}}$)
            
            **Kết quả mong đợi:** Danh sách ứng viên được lọc chỉ chứa các sản phẩm hợp lệ về articleType, age, và gender. Danh sách này sau đó được xếp hạng theo điểm $\\hat{r}_{ui}^{\\text{CBF}}$ để tạo ra danh sách Top-K Personalized cuối cùng.
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_cbf_predictions = 'cbf_predictions' in st.session_state
            has_feature_encoding = 'feature_encoding' in st.session_state

            if not has_cbf_predictions:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 2.2 (CBF Predictions). Vui lòng chạy Bước 2.2 trước.")
            
            if has_cbf_predictions and apply_personalized_filters is not None:
                # Load products and users data
                products_path = os.path.join(current_dir, 'apps', 'exports', 'products.csv')
                users_path = os.path.join(current_dir, 'apps', 'exports', 'users.csv')
                
                products_df = None
                users_df = None
                
                if os.path.exists(products_path):
                    products_df = pd.read_csv(products_path)
                    if 'id' in products_df.columns:
                        products_df['id'] = products_df['id'].astype(str)
                        products_df.set_index('id', inplace=True)
                else:
                    st.warning("⚠️ Không tìm thấy file products.csv. Vui lòng đảm bảo file tồn tại trong apps/exports/")
                
                if os.path.exists(users_path):
                    users_df = pd.read_csv(users_path)
                    if 'id' in users_df.columns:
                        users_df['id'] = users_df['id'].astype(str)
                else:
                    st.warning("⚠️ Không tìm thấy file users.csv. Vui lòng đảm bảo file tồn tại trong apps/exports/")
                
                if products_df is not None:
                    cbf_predictions = st.session_state['cbf_predictions']
                    
                    # Configuration
                    col_config1, col_config2 = st.columns(2)
                    with col_config1:
                        selected_user_id = st.selectbox(
                            "Chọn User ID để áp dụng lọc",
                            list(cbf_predictions['predictions'].keys()) if cbf_predictions else [],
                            key="filter_user_id"
                        )
                    
                    with col_config2:
                        payload_articletype = st.selectbox(
                            "Chọn articleType của sản phẩm đầu vào (payload)",
                            products_df['articleType'].unique().tolist() if 'articleType' in products_df.columns else [],
                            key="payload_articletype"
                        )
                    
                    # Get user info
                    user_age = None
                    user_gender = None
                    if users_df is not None and selected_user_id:
                        user_row = users_df[users_df['id'] == selected_user_id]
                        if not user_row.empty:
                            user_age = user_row.iloc[0].get('age', None)
                            user_gender = user_row.iloc[0].get('gender', None)
                    
                    if selected_user_id and payload_articletype:
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            if user_age is not None:
                                st.info(f"👤 User Age: {user_age}")
                            if user_gender is not None:
                                st.info(f"👤 User Gender: {user_gender}")
                        with col_info2:
                            st.info(f"📦 Payload articleType: {payload_articletype}")
                            if user_age is not None and user_gender is not None:
                                allowed_genders = get_allowed_genders(user_age, user_gender) if get_allowed_genders else []
                                st.info(f"✅ Allowed Genders: {', '.join(allowed_genders)}")
                        
                        # Top-K configuration
                        top_k_personalized = st.number_input(
                            "Số lượng sản phẩm Top-K Personalized",
                            min_value=5,
                            max_value=100,
                            value=20,
                            step=5,
                            key="top_k_personalized"
                        )
                        
                        process_button = st.button(
                            "🔧 Áp dụng Personalized Filters và Xếp hạng Top-K",
                            type="primary",
                            use_container_width=True,
                            key="personalized_filter_button"
                        )
                        
                        if process_button:
                            # Đo Inference Time (từ khi nhận user đến khi tạo L(u) - Bước 2.3)
                            inference_start_time = time.time()
                            
                            with st.spinner("Đang áp dụng các bộ lọc cá nhân hóa và xếp hạng..."):
                                try:
                                    # Lấy danh sách candidate products từ CBF predictions
                                    user_predictions = cbf_predictions['predictions'][selected_user_id]
                                    candidate_products = list(user_predictions.keys())
                                    
                                    # Áp dụng filters và xếp hạng Top-K
                                    result = apply_personalized_filters(
                                        candidate_products,
                                        products_df,
                                        payload_articletype=payload_articletype,
                                        user_age=user_age,
                                        user_gender=user_gender,
                                        cbf_scores=user_predictions,
                                        top_k=top_k_personalized
                                    )
                                    
                                    # Kết thúc đo Inference Time
                                    inference_end_time = time.time()
                                    inference_time_measured = inference_end_time - inference_start_time
                                    
                                    st.success(f"✅ **Hoàn thành!** Đã lọc danh sách ứng viên.")
                                    
                                    # Store in session state
                                    if 'personalized_filters' not in st.session_state:
                                        st.session_state['personalized_filters'] = {}
                                    st.session_state['personalized_filters'][selected_user_id] = result
                                    # Lưu vào artifacts để không bị mất khi chạy bước khác
                                    save_intermediate_artifact('personalized_filters', st.session_state['personalized_filters'])
                                    
                                    # Lưu Inference Time vào session state (lấy trung bình nếu có nhiều users)
                                    if 'inference_times' not in st.session_state:
                                        st.session_state['inference_times'] = []
                                    st.session_state['inference_times'].append(inference_time_measured)
                                    st.session_state['inference_time'] = np.mean(st.session_state['inference_times'])
                                    
                                    # Display statistics
                                    st.markdown("### 📊 Thống kê quá trình lọc")
                                    
                                    stats = result['stats']
                                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                    with col_stat1:
                                        st.metric("Danh sách ban đầu", f"{stats['initial_count']:,}")
                                    with col_stat2:
                                        st.metric("Sau lọc articleType", f"{stats['after_articletype']:,}")
                                    with col_stat3:
                                        st.metric("Sau lọc Age/Gender", f"{stats['after_age_gender']:,}")
                                    with col_stat4:
                                        st.metric(f"Top-K Personalized ({top_k_personalized})", f"{stats['final_count']:,}")
                                    
                                    # Hiển thị Top-K Personalized Rankings
                                    if result.get('ranked_products'):
                                        st.markdown(f"### 📋 Danh sách Top-{top_k_personalized} Personalized")
                                        ranked_df = pd.DataFrame([
                                            {
                                                'Rank': rank + 1,
                                                'Product ID': product_id,
                                                'CBF Score': f"{score:.4f}"
                                            }
                                            for rank, (product_id, score) in enumerate(result['ranked_products'])
                                        ])
                                        st.dataframe(ranked_df, use_container_width=True)
                                        
                                        # Biểu đồ Top-K scores
                                        fig_scores = px.bar(
                                            ranked_df,
                                            x='Rank',
                                            y='CBF Score',
                                            title=f"Top-{top_k_personalized} Personalized Scores",
                                            labels={'Rank': 'Xếp hạng', 'CBF Score': 'Điểm CBF'}
                                        )
                                        st.plotly_chart(fig_scores, use_container_width=True)
                                    
                                    # Reduction visualization
                                    st.markdown("### 📉 Biểu đồ giảm kích thước danh sách")
                                    reduction_df = pd.DataFrame({
                                        'Bước': ['Ban đầu', 'Sau articleType', 'Sau Age/Gender', f'Top-{top_k_personalized}'],
                                        'Số lượng': [
                                            stats['initial_count'],
                                            stats['after_articletype'],
                                            stats['after_age_gender'],
                                            stats['final_count']
                                        ]
                                    })
                                    
                                    fig = px.bar(
                                        reduction_df,
                                        x='Bước',
                                        y='Số lượng',
                                        title="Quá trình giảm kích thước danh sách ứng viên",
                                        labels={'Số lượng': 'Số lượng sản phẩm', 'Bước': 'Bước lọc'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Ví dụ tính toán
                                    st.markdown("### 🧮 Ví dụ tính toán")
                                    st.markdown(f"""
                                    **Ví dụ:** User {selected_user_id} với danh sách ứng viên ban đầu:
                                    
                                    - **Danh sách ban đầu:** {stats['initial_count']:,} sản phẩm
                                    - **Sau Lọc Cứng 1 (articleType='{payload_articletype}'):** {stats['after_articletype']:,} sản phẩm (giảm {stats['initial_count'] - stats['after_articletype']:,} sản phẩm)
                                    - **Sau Lọc Cứng 2 (User age {'< 13' if user_age and user_age < 13 else '≥ 13'}, gender='{user_gender}'):** {stats['after_age_gender']:,} sản phẩm (giảm {stats['after_articletype'] - stats['after_age_gender']:,} sản phẩm)
                                    - **Sau Xếp hạng Top-{top_k_personalized}:** {stats['final_count']:,} sản phẩm (xếp hạng theo $\\hat{{r}}_{{ui}}^{{\\text{{CBF}}}}$)
                                    - **Tổng giảm:** {stats['removed_count']:,} sản phẩm ({stats['reduction_rate']:.2f}%)
                                    
                                    **✅ Kết quả đạt được:**
                                    - ✅ Danh sách ứng viên được lọc chỉ chứa các sản phẩm hợp lệ về articleType, age, và gender
                                    - ✅ Danh sách được xếp hạng theo điểm $\\hat{{r}}_{{ui}}^{{\\text{{CBF}}}}$ để tạo ra danh sách Top-K Personalized cuối cùng
                                    - ✅ Đảm bảo tính hợp lệ cơ bản và độ ưu tiên của các đề xuất
                                    """)
                                
                                except Exception as e:
                                    st.error(f"❌ Lỗi khi áp dụng personalized filters: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    else:
                        st.info("💡 Vui lòng chọn User ID và articleType để tiếp tục.")
                else:
                    st.warning("⚠️ Không thể tải dữ liệu products. Vui lòng kiểm tra lại.")
            elif apply_personalized_filters is None:
                st.error(f"❌ Không thể import cbf_utils module: {_cbf_utils_import_error}")

        with st.expander("Bước 2.4: Tính toán Số liệu (Đánh giá Mô hình)", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:** Tính toán tất cả các chỉ số so sánh (Recall@K, NDCG@K,...) trên danh sách Top-K từ CBF Predictions (Bước 2.2).")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 2.2 (CBF Predictions) và dữ liệu Ground Truth (interactions)")
            st.info("💡 **Lưu ý:** Metrics được tính trên CBF Predictions (Bước 2.2), không phải Top-K Personalized (Bước 2.3) vì ground truth nên so sánh với toàn bộ recommendations, không chỉ phần đã lọc.")

            st.markdown("**Bảng chỉ số đánh giá:**")

            st.markdown("- **Training Time (s)**: Đo thời gian từ Bước 2.1 đến 2.2 (xây dựng $\\mathbf{P}_u$).")
            st.markdown("- **Inference Time (s)**: Đo thời gian từ khi nhận $u$ đến khi tạo $L(u)$ cuối cùng (Bước 2.3).")

            st.markdown("- **Recall@K** (K = 5, 10, 20) – Công thức:")
            st.latex(r"\text{Recall}@K = \frac{|\text{Relevant}(u) \cap L(u)|}{|\text{Relevant}(u)|}")

            st.markdown("- **Precision@K** (K = 5, 10, 20) – Công thức:")
            st.latex(r"\text{Precision}@K = \frac{|\text{Relevant}(u) \cap L(u)|}{K}")

            st.markdown("- **NDCG@K** (K = 5, 10, 20) – Công thức:")
            st.latex(r"\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}")
            st.latex(r"\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{\text{rel}(i)} - 1}{\log_2(i+1)}")

            st.markdown("- **Diversity (ILD@K)** – Công thức:")
            st.latex(r"\text{ILD}@K = \frac{2}{K(K-1)} \sum_{i \in L(u)} \sum_{j \in L(u),\, j>i} \left(1 - \text{cos}(\mathbf{v}_i, \mathbf{v}_j)\right)")

            st.markdown("- **Coverage** – Công thức:")
            st.latex(r"\text{Coverage} = \frac{|\{i \in I \mid i \in L(u) \text{ cho ít nhất một user } u\}|}{|I|}")

            st.markdown("**Kết quả mong đợi:** Một hàng dữ liệu hoàn chỉnh (cho CBF) trong Bảng Tổng hợp Chỉ số, thể hiện hiệu suất cơ sở của mô hình Content-based Filtering.")

            # Kiểm tra dữ liệu từ các bước trước
            has_cbf_predictions = 'cbf_predictions' in st.session_state
            has_feature_encoding = 'feature_encoding' in st.session_state
            has_user_profiles = 'user_profiles' in st.session_state

            if not has_cbf_predictions:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 2.2 (CBF Predictions). Vui lòng chạy Bước 2.2 trước.")
            if not has_feature_encoding:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 1.3 (Feature Encoding). Vui lòng chạy Bước 1.3 trước.")
            
            if has_cbf_predictions and has_feature_encoding and compute_cbf_metrics is not None:
                cbf_predictions = st.session_state['cbf_predictions']
                encoding_result = st.session_state.get('feature_encoding', {})
                
                encoded_matrix = encoding_result.get('encoded_matrix', None)
                product_ids = encoding_result.get('product_ids', [])
                
                # Load interactions for ground truth
                interactions_path = os.path.join(current_dir, 'apps', 'exports', 'interactions.csv')
                interactions_df = None
                if os.path.exists(interactions_path):
                    interactions_df = pd.read_csv(interactions_path)
                    if 'user_id' in interactions_df.columns:
                        interactions_df['user_id'] = interactions_df['user_id'].astype(str)
                    if 'product_id' in interactions_df.columns:
                        interactions_df['product_id'] = interactions_df['product_id'].astype(str)
                
                # Configuration
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    k_values_input = st.text_input(
                        "Các giá trị K (phân cách bằng dấu phẩy)",
                        value="10,20",
                        key="k_values_input"
                    )
                    try:
                        k_values = [int(k.strip()) for k in k_values_input.split(',')]
                    except:
                        k_values = [10, 20]
                        st.warning("⚠️ Định dạng không hợp lệ. Sử dụng mặc định: [10, 20]")
                
                with col_config2:
                    # Training Time và Inference Time được đo tự động từ các bước trước
                    # Hiển thị thông tin về thời gian đã đo
                    training_time_auto = st.session_state.get('training_time', None)
                    inference_time_auto = st.session_state.get('inference_time', None)
                    
                    if training_time_auto is not None:
                        st.info(f"⏱️ **Training Time (tự động):** {training_time_auto:.3f}s (đo từ Bước 2.1)")
                    else:
                        st.warning("⚠️ Chưa có Training Time. Vui lòng chạy Bước 2.1 trước.")
                    
                    if inference_time_auto is not None:
                        st.info(f"⏱️ **Inference Time (tự động):** {inference_time_auto:.3f}s (đo từ Bước 2.3)")
                    else:
                        st.warning("⚠️ Chưa có Inference Time. Vui lòng chạy Bước 2.3 trước.")
                    
                    # Cho phép override thủ công nếu cần
                    st.markdown("**Hoặc nhập thủ công (nếu cần):**")
                    training_time_manual = st.number_input(
                        "Training Time (giây) - Thủ công",
                        min_value=0.0,
                        value=training_time_auto if training_time_auto is not None else 0.0,
                        step=0.1,
                        key="training_time_input"
                    )
                    
                    inference_time_manual = st.number_input(
                        "Inference Time (giây) - Thủ công",
                        min_value=0.0,
                        value=inference_time_auto if inference_time_auto is not None else 0.0,
                        step=0.1,
                        key="inference_time_input"
                    )
                
                process_button = st.button(
                    "🔧 Tính toán Evaluation Metrics",
                    type="primary",
                    use_container_width=True,
                    key="evaluation_metrics_button"
                )
                
                if process_button:
                    with st.spinner("Đang tính toán các chỉ số đánh giá..."):
                        try:
                            # Lấy dữ liệu từ Bước 2.2 (CBF Predictions) - TRƯỚC KHI lọc
                            cbf_predictions = st.session_state['cbf_predictions']
                            encoding_result = st.session_state['feature_encoding']
                            
                            encoded_matrix = encoding_result['encoded_matrix']
                            product_ids = encoding_result['product_ids']
                            predictions_dict = {}
                            for user_id, user_ranking in cbf_predictions['rankings'].items():
                                user_id_str = str(user_id)
                                
                                ranked_products = [(str(pid), score) for pid, score in user_ranking]
                                predictions_dict[user_id_str] = ranked_products
                            final_training_time = training_time_manual if training_time_manual > 0 else training_time_auto
                            final_inference_time = inference_time_manual if inference_time_manual > 0 else inference_time_auto
                            ground_truth_dict = {}
                            
                            # Load products để kiểm tra articleType của relevant items
                            products_path = os.path.join(current_dir, 'apps', 'exports', 'products.csv')
                            products_df_for_gt = None
                            if os.path.exists(products_path):
                                products_df_for_gt = pd.read_csv(products_path)
                                if 'id' in products_df_for_gt.columns:
                                    products_df_for_gt['id'] = products_df_for_gt['id'].astype(str)
                                    products_df_for_gt.set_index('id', inplace=True)
                            
                            if interactions_df is not None and 'user_id' in interactions_df.columns and 'product_id' in interactions_df.columns:
                                # Chuẩn hóa user_id và product_id về string
                                interactions_df['user_id'] = interactions_df['user_id'].astype(str)
                                interactions_df['product_id'] = interactions_df['product_id'].astype(str)
                                
                                # Consider only positive interactions (purchase, like, cart)
                                positive_interactions = interactions_df[
                                    interactions_df['interaction_type'].isin(['purchase', 'like', 'cart'])
                                ] if 'interaction_type' in interactions_df.columns else interactions_df
                                
                                for user_id in predictions_dict.keys():
                                    # Đảm bảo user_id là string
                                    user_id_str = str(user_id)
                                    
                                    user_interactions = positive_interactions[
                                        positive_interactions['user_id'] == user_id_str
                                    ]
                                    if not user_interactions.empty:
                                        # Lấy tất cả relevant items từ interactions gốc
                                        relevant_items_all = set(user_interactions['product_id'].astype(str).unique())
                                        ground_truth_dict[user_id_str] = relevant_items_all
                                    else:
                                        ground_truth_dict[user_id_str] = set()
                            else:
                                st.warning("⚠️ Không có dữ liệu interactions để làm ground truth. Sử dụng empty sets.")
                                for user_id in predictions_dict.keys():
                                    ground_truth_dict[str(user_id)] = set()
                            
                            # Get all items for coverage
                            all_items = set(product_ids) if product_ids else set()
                            
                            # Compute metrics
                            result = compute_cbf_metrics(
                                predictions_dict,
                                ground_truth_dict,
                                k_values=k_values,
                                item_features=encoded_matrix,
                                item_ids=product_ids,
                                all_items=all_items,
                                training_time=final_training_time,
                                inference_time=final_inference_time,
                                use_ild=True  # Sử dụng ILD@K cho Diversity
                            )
                            
                            st.success("✅ **Hoàn thành!** Đã tính toán tất cả các chỉ số đánh giá.")
                            
                            # Store in session state
                            st.session_state['cbf_evaluation_metrics'] = result
                            # Lưu vào artifacts để không bị mất khi chạy bước khác
                            save_intermediate_artifact('cbf_evaluation_metrics', result)
                            # Lưu timing metrics
                            if 'training_time' in st.session_state:
                                save_intermediate_artifact('training_time', st.session_state['training_time'])
                            if 'inference_time' in st.session_state:
                                save_intermediate_artifact('inference_time', st.session_state['inference_time'])
                            
                            # Display results
                            st.markdown("### 📊 Kết quả Evaluation Metrics")
                            
                            # Create metrics table
                            metrics_data = []
                            for k in k_values:
                                metrics_data.append({
                                    'K': k,
                                    'Recall@K': f"{result['recall'].get(k, 0.0):.4f}",
                                    'Precision@K': f"{result['precision'].get(k, 0.0):.4f}",
                                    'NDCG@K': f"{result['ndcg'].get(k, 0.0):.4f}"
                                })
                            
                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # Other metrics
                            col_other1, col_other2, col_other3, col_other4 = st.columns(4)
                            with col_other1:
                                st.metric("Diversity (ILD@K)", f"{result['diversity']:.4f}" if result['diversity'] is not None else "N/A")
                                st.caption("Intra-List Diversity")
                            with col_other2:
                                st.metric("Coverage", f"{result['coverage']:.4f}" if result['coverage'] is not None else "N/A")
                                st.caption("Tỷ lệ items được đề xuất")
                            with col_other3:
                                st.metric("Training Time", f"{result['training_time']:.2f}s" if result['training_time'] is not None else "N/A")
                                st.caption("Bước 2.1 → 2.2")
                            with col_other4:
                                st.metric("Inference Time", f"{result['inference_time']:.2f}s" if result['inference_time'] is not None else "N/A")
                                st.caption("User → L(u) (Bước 2.3)")
                            
                            # Visualization
                            st.markdown("### 📈 Biểu đồ Metrics theo K")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=k_values,
                                y=[result['recall'].get(k, 0.0) for k in k_values],
                                mode='lines+markers',
                                name='Recall@K',
                                line=dict(color='blue', width=2)
                            ))
                            fig.add_trace(go.Scatter(
                                x=k_values,
                                y=[result['precision'].get(k, 0.0) for k in k_values],
                                mode='lines+markers',
                                name='Precision@K',
                                line=dict(color='green', width=2)
                            ))
                            fig.add_trace(go.Scatter(
                                x=k_values,
                                y=[result['ndcg'].get(k, 0.0) for k in k_values],
                                mode='lines+markers',
                                name='NDCG@K',
                                line=dict(color='red', width=2)
                            ))
                            fig.update_layout(
                                title="Metrics theo K",
                                xaxis_title="K",
                                yaxis_title="Score",
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary table for export
                            st.markdown("### 📋 Bảng Tổng hợp Chỉ số (Export)")
                            summary_data = {
                                'Model': ['CBF']
                            }
                            
                            # Thêm các metrics theo K values
                            for k in k_values:
                                summary_data[f'Recall@{k}'] = [f"{result['recall'].get(k, 0.0):.4f}"]
                                summary_data[f'Precision@{k}'] = [f"{result['precision'].get(k, 0.0):.4f}"]
                                summary_data[f'NDCG@{k}'] = [f"{result['ndcg'].get(k, 0.0):.4f}"]
                            
                            # Thêm các metrics khác
                            summary_data['Diversity (ILD@K)'] = [f"{result['diversity']:.4f}" if result['diversity'] is not None else "N/A"]
                            summary_data['Coverage'] = [f"{result['coverage']:.4f}" if result['coverage'] is not None else "N/A"]
                            summary_data['Training Time (s)'] = [f"{result['training_time']:.3f}" if result['training_time'] is not None else "N/A"]
                            summary_data['Inference Time (s)'] = [f"{result['inference_time']:.3f}" if result['inference_time'] is not None else "N/A"]
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            st.markdown("""
                            **✅ Kết quả đạt được:**
                            - ✅ Một hàng dữ liệu hoàn chỉnh trong Bảng Tổng hợp Chỉ số
                            - ✅ Thể hiện hiệu suất cơ sở của mô hình Content-based Filtering
                            - ✅ Sẵn sàng để so sánh với các mô hình khác (GNN, Hybrid)
                            """)
                        
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tính toán evaluation metrics: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            elif compute_cbf_metrics is None:
                st.error(f"❌ Không thể import evaluation_metrics module: {_evaluation_import_error}")

        # PHẦN III: MÔ HÌNH MẠNG NEURAL ĐỒ THỊ (GNN)
        st.markdown('<div class="sub-header">📚 PHẦN III: MÔ HÌNH MẠNG NEURAL ĐỒ THỊ (GNN)</div>', unsafe_allow_html=True)
        st.markdown("")

        with st.expander("Bước 3.1: Xây dựng Đồ thị và Khởi tạo Nhúng", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:** Xây dựng đồ thị hai phía $G=(U, I, \\mathcal{E})$ và khởi tạo ngẫu nhiên các vector nhúng $\\mathbf{e}_u^{(0)}$ và $\\mathbf{e}_i^{(0)}$.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 1.2 (Pruned Interactions)")

            st.markdown("""
            **Cấu trúc đồ thị:**
            - **Đồ thị hai phía (Bipartite Graph):** $G=(U, I, \\mathcal{E})$
              - $U$: Tập hợp các nodes người dùng
              - $I$: Tập hợp các nodes sản phẩm
              - $\\mathcal{E}$: Tập hợp các cạnh (edges) biểu diễn tương tác giữa users và products
            
            **Khởi tạo nhúng:**
            - **User Embeddings:** $\\mathbf{e}_u^{(0)} \\in \\mathbb{R}^d$ - Vector nhúng ban đầu cho mỗi user $u$
            - **Item Embeddings:** $\\mathbf{e}_i^{(0)} \\in \\mathbb{R}^d$ - Vector nhúng ban đầu cho mỗi item $i$
            - **Phương pháp khởi tạo:** Xavier Uniform Initialization
            - **Kích thước nhúng:** $d$ (embedding_dim, mặc định: 64)
            
            **Kết quả mong đợi:**
            - Đồ thị $G$ được xây dựng từ interactions đã làm sạch
            - Các vector nhúng ban đầu $\\mathbf{e}_u^{(0)}$ và $\\mathbf{e}_i^{(0)}$ được khởi tạo ngẫu nhiên
            - Sẵn sàng cho quá trình lan truyền thông điệp (Message Propagation)
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_pruned_interactions = 'pruned_interactions' in st.session_state

            if not has_pruned_interactions:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 1.2 (Pruning). Vui lòng chạy Bước 1.2 trước.")
            else:
                pruning_result = st.session_state['pruned_interactions']
                pruned_interactions_df = pruning_result['pruned_interactions']
                
                # Configuration
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    embedding_dim = st.number_input(
                        "Kích thước nhúng (embedding_dim)",
                        min_value=16,
                        max_value=256,
                        value=64,
                        step=16,
                        key="gnn_embedding_dim"
                    )
                
                with col_config2:
                    st.write("")  # Spacing
                    process_button = st.button(
                        "🔧 Xây dựng Đồ thị và Khởi tạo Nhúng",
                        type="primary",
                        use_container_width=True,
                        key="gnn_graph_construction_button"
                    )
                
                if process_button:
                    if build_graph is None:
                        st.error(f"❌ Không thể import gnn_utils module: {_gnn_utils_import_error}")
                        st.info("Vui lòng đảm bảo file apps/utils/gnn_utils.py tồn tại và có thể import được.")
                    else:
                        with st.spinner("Đang xây dựng đồ thị và khởi tạo nhúng..."):
                            try:
                                # Build graph
                                graph_result = build_graph(pruned_interactions_df, embedding_dim)
                                
                                # Store in session state
                                st.session_state['gnn_graph'] = graph_result
                                # Lưu vào artifacts để không bị mất khi chạy bước khác
                                save_intermediate_artifact('gnn_graph', graph_result)
                                
                                st.success(f"✅ **Hoàn thành!** Đã xây dựng đồ thị với {graph_result['num_users']} users và {graph_result['num_products']} products.")
                                
                                # Display statistics
                                st.markdown("### 📊 Thống kê Đồ thị")
                                
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1:
                                    st.metric("Số lượng Users", graph_result['num_users'])
                                    st.metric("Số lượng Products", graph_result['num_products'])
                                with col_stat2:
                                    st.metric("Số lượng Edges", graph_result['num_edges'])
                                    st.metric("Kích thước nhúng", f"{embedding_dim}D")
                                with col_stat3:
                                    density = (2 * graph_result['num_edges']) / (graph_result['num_users'] * graph_result['num_products']) if (graph_result['num_users'] * graph_result['num_products']) > 0 else 0
                                    st.metric("Mật độ đồ thị", f"{density:.6f}")
                                
                                # Display sample embeddings
                                st.markdown("### 🔢 Mẫu Vector Nhúng Ban đầu")
                                
                                if 'user_embeddings' in graph_result and 'product_embeddings' in graph_result:
                                    sample_user_emb = graph_result['user_embeddings'][:3] if len(graph_result['user_embeddings']) >= 3 else graph_result['user_embeddings']
                                    sample_product_emb = graph_result['product_embeddings'][:3] if len(graph_result['product_embeddings']) >= 3 else graph_result['product_embeddings']
                                    
                                    col_emb1, col_emb2 = st.columns(2)
                                    with col_emb1:
                                        st.write("**Sample User Embeddings (3 users đầu tiên):**")
                                        user_emb_df = pd.DataFrame(
                                            sample_user_emb,
                                            index=[f"User {i+1}" for i in range(len(sample_user_emb))],
                                            columns=[f"Dim {j+1}" for j in range(embedding_dim)]
                                        )
                                        st.dataframe(user_emb_df, use_container_width=True)
                                    
                                    with col_emb2:
                                        st.write("**Sample Product Embeddings (3 products đầu tiên):**")
                                        product_emb_df = pd.DataFrame(
                                            sample_product_emb,
                                            index=[f"Product {i+1}" for i in range(len(sample_product_emb))],
                                            columns=[f"Dim {j+1}" for j in range(embedding_dim)]
                                        )
                                        st.dataframe(product_emb_df, use_container_width=True)
                                
                                st.markdown("""
                                **✅ Kết quả đạt được:**
                                - ✅ Đồ thị $G=(U, I, \\mathcal{E})$ được xây dựng từ interactions đã làm sạch
                                - ✅ Các vector nhúng ban đầu $\\mathbf{e}_u^{(0)}$ và $\\mathbf{e}_i^{(0)}$ được khởi tạo ngẫu nhiên
                                - ✅ Sẵn sàng cho quá trình lan truyền thông điệp (Message Propagation)
                                """)
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi xây dựng đồ thị: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

        with st.expander("Bước 3.2: Cơ chế Lan truyền Thông điệp (Message Propagation)", expanded=True):
            st.write("**Nội dung thực hiện:** Lan truyền thông điệp qua $L$ lớp để cập nhật nhúng $\\mathbf{e}_u^{(l)}$ và $\\mathbf{e}_i^{(l)}$.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 3.1 (Graph Construction)")

            st.markdown("""
            **Công thức Cập nhật Nhúng:**
            $$\\mathbf{e}_u^{(l)} = \\text{LeakyReLU} \\left( \\mathbf{W}_1 \\mathbf{e}_u^{(l-1)} + \\sum_{i \\in N_u} M_{u \\leftarrow i} \\right)$$
            
            Trong đó:
            - $\\mathbf{e}_u^{(l)}$: Vector nhúng của user $u$ ở lớp $l$
            - $\\mathbf{W}_1$: Ma trận trọng số học được
            - $N_u$: Tập hợp các items mà user $u$ đã tương tác (neighbors)
            - $M_{u \\leftarrow i}$: Thông điệp từ item $i$ đến user $u$
            - $\\text{LeakyReLU}$: Hàm kích hoạt
            
            **Quá trình lan truyền:**
            1. **Lớp 0:** Sử dụng nhúng ban đầu $\\mathbf{e}_u^{(0)}$ và $\\mathbf{e}_i^{(0)}$
            2. **Lớp 1 đến L:** Cập nhật nhúng dựa trên thông điệp từ neighbors
            3. **Normalization:** Chuẩn hóa theo degree của nodes để ổn định training
            
            **Kết quả mong đợi:**
            - Các vector nhúng được cập nhật qua $L$ lớp
            - Nhúng cuối cùng $\\mathbf{e}_u^{(L)}$ và $\\mathbf{e}_i^{(L)}$ phản ánh cấu trúc đồ thị và tương tác
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_gnn_graph = 'gnn_graph' in st.session_state

            if not has_gnn_graph:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 3.1 (Graph Construction). Vui lòng chạy Bước 3.1 trước.")
            else:
                graph_result = st.session_state['gnn_graph']
                
                # Configuration
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    num_layers = st.number_input(
                        "Số lớp lan truyền (num_layers)",
                        min_value=1,
                        max_value=10,
                        value=3,
                        step=1,
                        key="gnn_num_layers"
                    )
                
                with col_config2:
                    st.write("")  # Spacing
                    process_button = st.button(
                        "🔧 Thực hiện Message Propagation",
                        type="primary",
                        use_container_width=True,
                        key="gnn_message_propagation_button"
                    )
                
                if process_button:
                    if message_propagation is None:
                        st.error(f"❌ Không thể import gnn_utils module: {_gnn_utils_import_error}")
                        st.info("Vui lòng đảm bảo file apps/utils/gnn_utils.py tồn tại và có thể import được.")
                    else:
                        with st.spinner("Đang thực hiện lan truyền thông điệp..."):
                            try:
                                # Perform message propagation
                                propagation_result = message_propagation(graph_result, num_layers)
                                
                                # Store in session state
                                st.session_state['gnn_propagation'] = propagation_result
                                # Lưu vào artifacts để không bị mất khi chạy bước khác
                                save_intermediate_artifact('gnn_propagation', propagation_result)
                                
                                st.success(f"✅ **Hoàn thành!** Đã thực hiện lan truyền qua {num_layers} lớp.")
                                
                                # Display statistics
                                st.markdown("### 📊 Thống kê Message Propagation")
                                
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1:
                                    st.metric("Số lớp", num_layers)
                                    st.metric("Kích thước nhúng", f"{graph_result['embedding_dim']}D")
                                with col_stat2:
                                    if 'final_user_embeddings' in propagation_result:
                                        st.metric("User Embeddings Shape", f"{propagation_result['final_user_embeddings'].shape}")
                                    if 'final_product_embeddings' in propagation_result:
                                        st.metric("Product Embeddings Shape", f"{propagation_result['final_product_embeddings'].shape}")
                                with col_stat3:
                                    if 'layer_stats' in propagation_result:
                                        st.metric("Lớp đã xử lý", len(propagation_result['layer_stats']))
                                
                                # Display layer-wise statistics
                                if 'layer_stats' in propagation_result:
                                    st.markdown("### 📈 Thống kê theo từng lớp")
                                    layer_stats_df = pd.DataFrame(propagation_result['layer_stats'])
                                    st.dataframe(layer_stats_df, use_container_width=True)
                                
                                st.markdown("""
                                **✅ Kết quả đạt được:**
                                - ✅ Các vector nhúng được cập nhật qua $L$ lớp
                                - ✅ Nhúng cuối cùng $\\mathbf{e}_u^{(L)}$ và $\\mathbf{e}_i^{(L)}$ phản ánh cấu trúc đồ thị và tương tác
                                - ✅ Sẵn sàng cho quá trình dự đoán và xếp hạng
                                """)
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi thực hiện message propagation: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

        with st.expander("Bước 3.3: Dự đoán và Xếp hạng", expanded=True):
            st.write("**Nội dung thực hiện:** Tổng hợp nhúng cuối cùng $\\mathbf{E}_u^*, \\mathbf{E}_i^*$. Tính điểm dự đoán $\\hat{r}_{ui}^{\\text{GNN}} = \\mathbf{E}_u^* \\cdot \\mathbf{E}_i^*$.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 3.2 (Message Propagation)")

            st.markdown("""
            **Công thức Dự đoán:**
            $$\\hat{r}_{ui}^{\\text{GNN}} = \\mathbf{E}_u^* \\cdot \\mathbf{E}_i^*$$
            
            Trong đó:
            - $\\mathbf{E}_u^*$: Vector nhúng cuối cùng của user $u$ sau $L$ lớp lan truyền
            - $\\mathbf{E}_i^*$: Vector nhúng cuối cùng của item $i$ sau $L$ lớp lan truyền
            - $\\hat{r}_{ui}^{\\text{GNN}}$: Điểm dự đoán GNN cho user $u$ và item $i$
            
            **Quá trình:**
            1. Lấy nhúng cuối cùng từ Bước 3.2
            2. Tính tích vô hướng giữa user embedding và product embedding
            3. Xếp hạng các sản phẩm theo điểm dự đoán giảm dần
            
            **Kết quả mong đợi:**
            - Điểm dự đoán $\\hat{r}_{ui}^{\\text{GNN}}$ cho tất cả user-item pairs
            - Top-K rankings cho mỗi user
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_gnn_propagation = 'gnn_propagation' in st.session_state

            if not has_gnn_propagation:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 3.2 (Message Propagation). Vui lòng chạy Bước 3.2 trước.")
            else:
                propagation_result = st.session_state['gnn_propagation']
                
                # Configuration
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    top_k = st.number_input(
                        "Số lượng sản phẩm Top-K để xếp hạng",
                        min_value=5,
                        max_value=100,
                        value=20,
                        step=5,
                        key="gnn_top_k"
                    )
                
                with col_config2:
                    st.write("")  # Spacing
                    process_button = st.button(
                        "🔧 Tính Điểm Dự đoán và Xếp hạng",
                        type="primary",
                        use_container_width=True,
                        key="gnn_predictions_button"
                    )
                
                if process_button:
                    if compute_gnn_predictions is None:
                        st.error(f"❌ Không thể import gnn_utils module: {_gnn_utils_import_error}")
                        st.info("Vui lòng đảm bảo file apps/utils/gnn_utils.py tồn tại và có thể import được.")
                    else:
                        with st.spinner("Đang tính điểm dự đoán và xếp hạng..."):
                            try:
                                # Compute predictions
                                predictions_result = compute_gnn_predictions(propagation_result, top_k)
                                
                                # Store in session state & lưu ra artifacts
                                st.session_state['gnn_predictions'] = predictions_result
                                save_predictions_artifact("gnn", predictions_result)
                                
                                st.success(f"✅ **Hoàn thành!** Đã tính điểm dự đoán cho {predictions_result['stats']['total_users']} users.")
                                
                                # Display statistics
                                st.markdown("### 📊 Thống kê Predictions")
                                
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1:
                                    st.metric("Tổng số predictions", f"{predictions_result['stats']['total_predictions']:,}")
                                    st.metric("Số users", predictions_result['stats']['total_users'])
                                with col_stat2:
                                    st.metric("Số products", predictions_result['stats']['total_products'])
                                    st.metric("Top-K", top_k)
                                with col_stat3:
                                    st.metric("Min score", f"{predictions_result['stats']['min_score']:.4f}")
                                    st.metric("Max score", f"{predictions_result['stats']['max_score']:.4f}")
                                    st.metric("Mean score", f"{predictions_result['stats']['mean_score']:.4f}")
                                
                                # Display sample rankings
                                st.markdown(f"### 📋 Mẫu Rankings Top-{top_k} (5 users đầu tiên)")
                                
                                if 'rankings' in predictions_result:
                                    sample_users = list(predictions_result['rankings'].keys())[:5]
                                    
                                    for idx, user_id in enumerate(sample_users, 1):
                                        ranking = predictions_result['rankings'][user_id]
                                        
                                        with st.expander(f"User {user_id} - Top {len(ranking)} sản phẩm", expanded=False):
                                            ranking_df = pd.DataFrame([
                                                {
                                                    'Rank': rank + 1,
                                                    'Product ID': product_id,
                                                    'Score': f"{score:.4f}"
                                                }
                                                for rank, (product_id, score) in enumerate(ranking)
                                            ])
                                            st.dataframe(ranking_df, use_container_width=True)
                                
                                st.markdown("""
                                **✅ Kết quả đạt được:**
                                - ✅ Điểm dự đoán $\\hat{r}_{ui}^{\\text{GNN}}$ cho tất cả user-item pairs
                                - ✅ Top-K rankings cho mỗi user
                                - ✅ Sẵn sàng cho quá trình huấn luyện hoặc đánh giá
                                """)
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi tính điểm dự đoán: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

        with st.expander("Bước 3.4: Huấn luyện Mô hình: Tối ưu hóa bằng BPR Loss", expanded=True):
            st.write("**Nội dung thực hiện:** Huấn luyện mô hình bằng cách tối ưu hóa trực tiếp thứ hạng.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 3.2 (Message Propagation) và Bước 1.2 (Pruned Interactions)")

            st.markdown("""
            **Công thức BPR Loss:**
            $$L_{BPR} = - \\sum_{(u, i, j) \\in D_S} \\ln \\sigma(\\hat{r}_{ui} - \\hat{r}_{uj}) + \\lambda ||\\Theta||^2$$
            
            Trong đó:
            - $D_S$: Tập hợp các triplets $(u, i, j)$ với $i$ là positive item và $j$ là negative item
            - $\\hat{r}_{ui}$: Điểm dự đoán cho positive pair $(u, i)$
            - $\\hat{r}_{uj}$: Điểm dự đoán cho negative pair $(u, j)$
            - $\\sigma$: Hàm sigmoid
            - $\\lambda$: Hệ số regularization
            - $||\\Theta||^2$: L2 regularization của các tham số mô hình
            
            **Quá trình huấn luyện:**
            1. **Sampling:** Tạo các triplets $(u, i, j)$ từ interactions
            2. **Forward Pass:** Tính $\\hat{r}_{ui}$ và $\\hat{r}_{uj}$
            3. **Loss Calculation:** Tính $L_{BPR}$
            4. **Backward Pass:** Cập nhật tham số $\\Theta$ bằng gradient descent
            5. **Lặp lại** qua các epochs cho đến khi hội tụ
            
            **Kết quả mong đợi:**
            - Giá trị $L_{BPR}$ giảm dần và hội tụ
            - Tối ưu hóa các vector nhúng ($\\Theta$)
            - Mô hình học được patterns từ đồ thị tương tác
            """)

            has_gnn_propagation = 'gnn_propagation' in st.session_state
            has_pruned_interactions = 'pruned_interactions' in st.session_state

            if not has_gnn_propagation:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 3.2 (Message Propagation). Vui lòng chạy Bước 3.2 trước.")
            if not has_pruned_interactions:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 1.2 (Pruning). Vui lòng chạy Bước 1.2 trước.")
            
            if has_gnn_propagation and has_pruned_interactions:
                propagation_result = st.session_state['gnn_propagation']
                pruning_result = st.session_state['pruned_interactions']
                pruned_interactions_df = pruning_result['pruned_interactions']
                
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    num_epochs = st.number_input(
                        "Số epochs",
                        min_value=1,
                        max_value=100,
                        value=10,
                        step=1,
                        key="gnn_num_epochs"
                    )
                    
                    learning_rate = st.number_input(
                        "Learning Rate",
                        min_value=0.0001,
                        max_value=0.1,
                        value=0.001,
                        step=0.0001,
                        format="%.4f",
                        key="gnn_learning_rate"
                    )
                
                with col_config2:
                    reg_weight = st.number_input(
                        "Regularization Weight (λ)",
                        min_value=0.0,
                        max_value=0.01,
                        value=0.0001,
                        step=0.0001,
                        format="%.4f",
                        key="gnn_reg_weight"
                    )
                    
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=32,
                        max_value=1024,
                        value=256,
                        step=32,
                        key="gnn_batch_size"
                    )
                
                process_button = st.button(
                    "🔧 Huấn luyện Mô hình (BPR Loss)",
                    type="primary",
                    use_container_width=True,
                    key="gnn_training_button"
                )

                if process_button:
                    if train_gnn_model is None:
                        st.error(f"❌ Không thể import gnn_utils module: {_gnn_utils_import_error}")
                        st.info("Vui lòng đảm bảo file apps/utils/gnn_utils.py tồn tại và có thể import được.")
                    else:
                        # Đo Training Time (Bước 3.2 đến 3.4)
                        training_start_time = time.time()
                        
                        with st.spinner(f"Đang huấn luyện mô hình qua {num_epochs} epochs (có thể mất vài phút)..."):
                            try:
                                # Train model
                                training_result = train_gnn_model(
                                    propagation_result,
                                    pruned_interactions_df,
                                    num_epochs=num_epochs,
                                    learning_rate=learning_rate,
                                    reg_weight=reg_weight,
                                    batch_size=batch_size
                                )
                                
                                # Kết thúc đo Training Time
                                training_end_time = time.time()
                                training_time_measured = training_end_time - training_start_time
                                
                                # Lưu vào session state
                                st.session_state['gnn_training'] = training_result
                                st.session_state['gnn_training_time'] = training_time_measured
                                # Lưu vào artifacts để không bị mất khi chạy bước khác
                                save_intermediate_artifact('gnn_training', training_result)
                                
                                st.success(f"✅ **Hoàn thành!** Đã huấn luyện mô hình qua {num_epochs} epochs.")
                                
                                # Debug thêm để kiểm tra nguyên nhân BPR Loss luôn là 0.0000
                                with st.expander("🔍 Debug Training Result (GNN BPR Loss)", expanded=False):
                                    st.markdown("**Raw `training_result` từ `train_gnn_model`:**")
                                    try:
                                        st.json(training_result)
                                    except Exception:
                                        st.write(training_result)
                                    
                                    if isinstance(training_result, dict):
                                        initial_loss_val = training_result.get('initial_loss', None)
                                        final_loss_val = training_result.get('final_loss', None)
                                        loss_history_val = training_result.get('loss_history', None)
                                        
                                        if (initial_loss_val in [0, 0.0, None]) and (final_loss_val in [0, 0.0, None]):
                                            st.warning(
                                                "⚠️ `initial_loss` và/hoặc `final_loss` đang là 0.\n\n"
                                                "- Nếu đồng thời `loss_history` rỗng và trong kết quả có khóa "
                                                "`warning` giống như: **\"No positive pairs found for training. "
                                                "Using embeddings from propagation only.\"** thì mô hình **không "
                                                "thực sự train**, mà chỉ dùng embeddings từ bước message propagation.\n"
                                                "- Nguyên nhân thường là **không tạo được positive pair (u, i, j)** "
                                                "từ `pruned_interactions_df` trong `train_gnn_model` "
                                                "(ví dụ do dữ liệu quá ít, hoặc logic lọc triplet quá chặt).\n"
                                                "- Khi đó các thống kê BPR Loss ở UI sẽ hiển thị 0.0000 là đúng với "
                                                "kết quả hiện tại (không có bước tối ưu hóa)."
                                            )
                                        
                                        if isinstance(loss_history_val, (list, tuple)) and loss_history_val:
                                            st.write("**Sample `loss_history` (5 giá trị đầu tiên):**", loss_history_val[:5])
                                        else:
                                            st.warning("⚠️ `loss_history` rỗng hoặc không tồn tại – đây cũng có thể là nguyên nhân các số liệu hiển thị là 0.0000.")

                                # Display statistics
                                st.markdown("### 📊 Thống kê Huấn luyện")
                                
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1:
                                    st.metric("Số epochs", num_epochs)
                                    st.metric("Training Time", f"{training_time_measured:.2f}s")
                                with col_stat2:
                                    warning_msg = training_result.get('warning') if isinstance(training_result, dict) else None
                                    if warning_msg and "No positive pairs found for training" in str(warning_msg):
                                        st.warning("⚠️ Không tìm được positive pairs để train BPR. "
                                                   "Mô hình chỉ dùng embeddings từ propagation, không có bước tối ưu hóa BPR.")
                                        st.metric("Final BPR Loss", "N/A")
                                        st.metric("Initial BPR Loss", "N/A")
                                    else:
                                        if 'final_loss' in training_result:
                                            st.metric("Final BPR Loss", f"{training_result['final_loss']:.4f}")
                                        if 'initial_loss' in training_result:
                                            st.metric("Initial BPR Loss", f"{training_result['initial_loss']:.4f}")
                                with col_stat3:
                                    if warning_msg and "No positive pairs found for training" in str(warning_msg):
                                        st.metric("Loss Reduction", "N/A")
                                    elif 'final_loss' in training_result and 'initial_loss' in training_result:
                                        loss_reduction = training_result['initial_loss'] - training_result['final_loss']
                                        st.metric("Loss Reduction", f"{loss_reduction:.4f}")
                                
                                # Display training history
                                if 'loss_history' in training_result:
                                    st.markdown("### 📈 Lịch sử BPR Loss qua các Epochs")
                                    
                                    loss_history_df = pd.DataFrame({
                                        'Epoch': range(1, len(training_result['loss_history']) + 1),
                                        'BPR Loss': training_result['loss_history']
                                    })
                                    
                                    fig = px.line(
                                        loss_history_df,
                                        x='Epoch',
                                        y='BPR Loss',
                                        title="BPR Loss qua các Epochs",
                                        labels={'BPR Loss': 'BPR Loss', 'Epoch': 'Epoch'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **✅ Kết quả đạt được:**
                                - ✅ Giá trị $L_{BPR}$ giảm dần và hội tụ
                                - ✅ Tối ưu hóa các vector nhúng ($\\Theta$)
                                - ✅ Mô hình học được patterns từ đồ thị tương tác
                                """)
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi huấn luyện mô hình: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

        with st.expander("Bước 3.5: Tạo Danh sách gợi ý cá nhân hóa và Tính toán Số liệu (Đánh giá Mô hình)", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:**")
            st.write("1. **Gợi ý Cá nhân hóa:** Áp dụng Logic Lọc và Ưu tiên (Bước 2.3) lên danh sách ứng viên được xếp hạng bởi $\\hat{r}_{ui}^{\\text{GNN}}$.")
            st.write("2. **Tính toán Số liệu:** Tính toán tất cả các chỉ số (Recall@K, NDCG@K,...) tương tự như Bước 2.4, sử dụng $L(u)$ và các tham số thời gian tương ứng của GNN.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 3.3 (GNN Predictions) hoặc Bước 3.4 (Trained Model)")

            st.markdown("""
            **Dữ liệu Đầu vào (Được lấy từ):**
            - **Training Time (s):** Đo thời gian từ Bước 3.2 đến 3.4 (quá trình lặp lại BPR Loss qua các epoch).
            - **Inference Time (s):** Đo thời gian cho quá trình tính toán $\\hat{r}_{ui}^{\\text{GNN}}$ và hậu xử lý (Bước 3.5).
            - **ILD, NDCG, Recall, Precision:** Dữ liệu tương tự Bước 2.4, nhưng sử dụng $L(u)$ được tạo từ $\\hat{r}_{ui}^{\\text{GNN}}$.
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_gnn_predictions = 'gnn_predictions' in st.session_state
            has_gnn_training = 'gnn_training' in st.session_state

            if not has_gnn_predictions and not has_gnn_training:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 3.3 (GNN Predictions) hoặc Bước 3.4 (Trained Model). Vui lòng chạy một trong hai bước trước.")
            else:
                if has_gnn_training:
                    gnn_predictions = st.session_state['gnn_training']
                elif has_gnn_predictions:
                    gnn_predictions = st.session_state['gnn_predictions']
                else:
                    gnn_predictions = None
                
                # Kiểm tra format
                if gnn_predictions is not None:
                    if not isinstance(gnn_predictions, dict):
                        st.error(f"❌ **Lỗi:** gnn_predictions không phải là dictionary. Type: {type(gnn_predictions)}")
                        st.write(f"Value: {gnn_predictions}")
                        gnn_predictions = None
                    elif len(gnn_predictions) == 0:
                        st.warning("⚠️ **Cảnh báo:** gnn_predictions là dictionary rỗng. Vui lòng chạy lại Bước 3.3 hoặc 3.4.")
                        gnn_predictions = None
                
                has_feature_encoding = 'feature_encoding' in st.session_state
                if not has_feature_encoding:
                    st.warning("⚠️ Chưa có dữ liệu từ Bước 1.3 (Feature Encoding). Cần cho tính toán Diversity.")
                
                if gnn_predictions is not None:
                    encoding_result = st.session_state.get('feature_encoding', {})
                    encoded_matrix = encoding_result.get('encoded_matrix', None)
                    product_ids = encoding_result.get('product_ids', [])
                    
                    # Load interactions for ground truth
                    interactions_path = os.path.join(current_dir, 'apps', 'exports', 'interactions.csv')
                    interactions_df = None
                    if os.path.exists(interactions_path):
                        interactions_df = pd.read_csv(interactions_path)
                        if 'user_id' in interactions_df.columns:
                            interactions_df['user_id'] = interactions_df['user_id'].astype(str)
                        if 'product_id' in interactions_df.columns:
                            interactions_df['product_id'] = interactions_df['product_id'].astype(str)
                    
                    # Configuration
                    col_config1, col_config2 = st.columns(2)
                    with col_config1:
                        k_values_input = st.text_input(
                            "Các giá trị K (phân cách bằng dấu phẩy)",
                            value="10,20",
                            key="gnn_k_values_input"
                        )
                        try:
                            k_values = [int(k.strip()) for k in k_values_input.split(',')]
                        except:
                            k_values = [10, 20]
                            st.warning("⚠️ Định dạng không hợp lệ. Sử dụng mặc định: [10, 20]")
                    
                    with col_config2:
                        # Training Time và Inference Time được đo tự động từ các bước trước
                        training_time_auto = st.session_state.get('gnn_training_time', None)
                        inference_time_auto = st.session_state.get('gnn_inference_time', None)
                        
                        if training_time_auto is not None:
                            st.info(f"⏱️ **Training Time (tự động):** {training_time_auto:.3f}s (đo từ Bước 3.2-3.4)")
                        else:
                            st.warning("⚠️ Chưa có Training Time. Vui lòng chạy Bước 3.4 trước.")
                        
                        # Cho phép override thủ công nếu cần
                        st.markdown("**Hoặc nhập thủ công (nếu cần):**")
                        training_time_manual = st.number_input(
                            "Training Time (giây) - Thủ công",
                            min_value=0.0,
                            value=training_time_auto if training_time_auto is not None else 0.0,
                            step=0.1,
                            key="gnn_training_time_input"
                        )
                        
                        inference_time_manual = st.number_input(
                            "Inference Time (giây) - Thủ công",
                            min_value=0.0,
                            value=inference_time_auto if inference_time_auto is not None else 0.0,
                            step=0.1,
                            key="gnn_inference_time_input"
                        )
                    
                    process_button = st.button(
                        "🔧 Tính toán Evaluation Metrics",
                        type="primary",
                        use_container_width=True,
                        key="gnn_evaluation_metrics_button"
                    )
                    
                    if process_button:
                        # Đo Inference Time
                        inference_start_time = time.time()
                        
                        with st.spinner("Đang tính toán các chỉ số đánh giá..."):
                            try:
                                # Prepare predictions format từ GNN Predictions
                                predictions_dict = {}
                                
                                if 'rankings' in gnn_predictions:
                                    for user_id, user_ranking in gnn_predictions['rankings'].items():
                                        user_id_str = str(user_id)
                                        # Handle both tuple and non-tuple formats
                                        if user_ranking and len(user_ranking) > 0:
                                            if isinstance(user_ranking[0], tuple):
                                                ranked_products = [(str(pid), score) for pid, score in user_ranking]
                                            else:
                                                # If it's a dict, convert to list of tuples
                                                if isinstance(user_ranking, dict):
                                                    ranked_products = [(str(pid), score) for pid, score in user_ranking.items()]
                                                else:
                                                    ranked_products = [(str(item), 0.0) for item in user_ranking]
                                            predictions_dict[user_id_str] = ranked_products
                                elif 'predictions' in gnn_predictions:
                                    # Convert predictions dict to rankings format
                                    user_predictions_dict = gnn_predictions['predictions']
                                    if isinstance(user_predictions_dict, dict) and len(user_predictions_dict) > 0:
                                        # Get top_k from k_values (use max k)
                                        max_k = max(k_values) if k_values else 20
                                        
                                        for user_id, user_preds in user_predictions_dict.items():
                                            user_id_str = str(user_id)
                                            if isinstance(user_preds, dict) and len(user_preds) > 0:
                                                ranked_products = sorted(
                                                    [(str(pid), score) for pid, score in user_preds.items()],
                                                    key=lambda x: x[1],
                                                    reverse=True
                                                )[:max_k]  # Limit to max_k
                                                predictions_dict[user_id_str] = ranked_products
                                    else:
                                        st.warning(f"⚠️ 'predictions' key tồn tại nhưng không phải dict hoặc rỗng. Type: {type(user_predictions_dict)}, Length: {len(user_predictions_dict) if isinstance(user_predictions_dict, dict) else 'N/A'}")
                                else:
                                    st.error("❌ GNN predictions không có cả 'rankings' và 'predictions' keys!")
                                    st.write(f"Available keys: {list(gnn_predictions.keys()) if isinstance(gnn_predictions, dict) else 'N/A'}")
                                
                                final_training_time = training_time_manual if training_time_manual > 0 else training_time_auto
                                
                                ground_truth_dict = {}
                                
                                if interactions_df is not None and 'user_id' in interactions_df.columns and 'product_id' in interactions_df.columns:
                                    positive_interactions = interactions_df[
                                        interactions_df['interaction_type'].isin(['purchase', 'like', 'cart'])
                                    ] if 'interaction_type' in interactions_df.columns else interactions_df
                                    
                                    for user_id in predictions_dict.keys():
                                        user_id_str = str(user_id)
                                        user_interactions = positive_interactions[
                                            positive_interactions['user_id'] == user_id_str
                                        ]
                                        if not user_interactions.empty:
                                            relevant_items = set(user_interactions['product_id'].astype(str).unique())
                                            ground_truth_dict[user_id_str] = relevant_items
                                        else:
                                            ground_truth_dict[user_id_str] = set()
                                else:
                                    st.warning("⚠️ Không có dữ liệu interactions để làm ground truth. Sử dụng empty sets.")
                                    for user_id in predictions_dict.keys():
                                        ground_truth_dict[str(user_id)] = set()
                                
                                # Get all items for coverage
                                all_items = set(product_ids) if product_ids else set()
                                
                                # Kết thúc đo Inference Time
                                inference_end_time = time.time()
                                inference_time_measured = inference_end_time - inference_start_time
                                # Sử dụng inference time đã đo hoặc thủ công
                                final_inference_time = inference_time_manual if inference_time_manual > 0 else inference_time_measured
                                
                                # Lưu vào session state
                                st.session_state['gnn_inference_time'] = inference_time_measured
                                
                                # Compute metrics
                                if compute_cbf_metrics is not None:
                                    result = compute_cbf_metrics(
                                        predictions_dict,
                                        ground_truth_dict,
                                        k_values=k_values,
                                        item_features=encoded_matrix,
                                        item_ids=product_ids,
                                        all_items=all_items,
                                        training_time=final_training_time,
                                        inference_time=final_inference_time,
                                        use_ild=True
                                    )
                                    
                                    st.success("✅ **Hoàn thành!** Đã tính toán tất cả các chỉ số đánh giá.")
                                    
                                    # Store in session state
                                    st.session_state['gnn_evaluation_metrics'] = result
                                    # Lưu vào artifacts để không bị mất khi chạy bước khác
                                    save_intermediate_artifact('gnn_evaluation_metrics', result)
                                    # Lưu timing metrics
                                    if 'gnn_training_time' in st.session_state:
                                        save_intermediate_artifact('gnn_training_time', st.session_state['gnn_training_time'])
                                    if 'gnn_inference_time' in st.session_state:
                                        save_intermediate_artifact('gnn_inference_time', st.session_state['gnn_inference_time'])
                                    
                                    # Display results (similar to Step 2.5)
                                    st.markdown("### 📊 Kết quả Evaluation Metrics")
                                    
                                    # Create metrics table
                                    metrics_data = []
                                    for k in k_values:
                                        metrics_data.append({
                                            'K': k,
                                            'Recall@K': f"{result['recall'].get(k, 0.0):.4f}",
                                            'Precision@K': f"{result['precision'].get(k, 0.0):.4f}",
                                            'NDCG@K': f"{result['ndcg'].get(k, 0.0):.4f}"
                                        })
                                    
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.dataframe(metrics_df, use_container_width=True)
                                    
                                    # Other metrics
                                    col_other1, col_other2, col_other3, col_other4 = st.columns(4)
                                    with col_other1:
                                        st.metric("Diversity (ILD@K)", f"{result['diversity']:.4f}" if result['diversity'] is not None else "N/A")
                                    with col_other2:
                                        st.metric("Coverage", f"{result['coverage']:.4f}" if result['coverage'] is not None else "N/A")
                                    with col_other3:
                                        st.metric("Training Time", f"{result['training_time']:.2f}s" if result['training_time'] is not None else "N/A")
                                    with col_other4:
                                        st.metric("Inference Time", f"{result['inference_time']:.2f}s" if result['inference_time'] is not None else "N/A")
                                    
                                    # Visualization
                                    st.markdown("### 📈 Biểu đồ Metrics theo K")
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=k_values,
                                        y=[result['recall'].get(k, 0.0) for k in k_values],
                                        mode='lines+markers',
                                        name='Recall@K',
                                        line=dict(color='blue', width=2)
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=k_values,
                                        y=[result['precision'].get(k, 0.0) for k in k_values],
                                        mode='lines+markers',
                                        name='Precision@K',
                                        line=dict(color='green', width=2)
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=k_values,
                                        y=[result['ndcg'].get(k, 0.0) for k in k_values],
                                        mode='lines+markers',
                                        name='NDCG@K',
                                        line=dict(color='red', width=2)
                                    ))
                                    fig.update_layout(
                                        title="Metrics theo K (GNN)",
                                        xaxis_title="K",
                                        yaxis_title="Score",
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Summary table for export
                                    st.markdown("### 📋 Bảng Tổng hợp Chỉ số (Export)")
                                    summary_data = {
                                        'Model': ['GNN']
                                    }
                                    
                                    # Thêm các metrics theo K values
                                    for k in k_values:
                                        summary_data[f'Recall@{k}'] = [f"{result['recall'].get(k, 0.0):.4f}"]
                                        summary_data[f'Precision@{k}'] = [f"{result['precision'].get(k, 0.0):.4f}"]
                                        summary_data[f'NDCG@{k}'] = [f"{result['ndcg'].get(k, 0.0):.4f}"]
                                    
                                    # Thêm các metrics khác
                                    summary_data['Diversity (ILD@K)'] = [f"{result['diversity']:.4f}" if result['diversity'] is not None else "N/A"]
                                    summary_data['Coverage'] = [f"{result['coverage']:.4f}" if result['coverage'] is not None else "N/A"]
                                    summary_data['Training Time (s)'] = [f"{result['training_time']:.3f}" if result['training_time'] is not None else "N/A"]
                                    summary_data['Inference Time (s)'] = [f"{result['inference_time']:.3f}" if result['inference_time'] is not None else "N/A"]
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    st.markdown("""
                                    **✅ Kết quả đạt được:**
                                    - ✅ Một hàng dữ liệu hoàn chỉnh trong Bảng Tổng hợp Chỉ số cho GNN
                                    - ✅ Thể hiện hiệu suất của mô hình GNN
                                    - ✅ Sẵn sàng để so sánh với các mô hình khác (CBF, Hybrid)
                                    """)
                                else:
                                    st.error("❌ Không thể import evaluation_metrics module.")
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi tính toán evaluation metrics: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
        st.markdown('<div class="sub-header">📚 PHẦN IV: MÔ HÌNH KẾT HỢP (HYBRID GNN + CONTENT-BASED)</div>', unsafe_allow_html=True)
        st.markdown("")

        with st.expander("Bước 4.1 & 4.2: Hợp nhất Điểm số Tuyến tính", expanded=True):
            st.write("**Nội dung thực hiện:** Kết hợp tuyến tính điểm dự đoán đã chuẩn hóa của GNN ($\\hat{r}_{ui}^{\\text{GNN}}$ từ Phần III) và CBF ($\\hat{r}_{ui}^{\\text{CBF}}$ từ Phần II).")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 2.2 (CBF Predictions) và Bước 3.3/3.4 (GNN Predictions)")

            st.markdown("""
            **Công thức Tính điểm Hybrid:**
            $$Score_{Hybrid}(u, i) = \\alpha \\cdot \\hat{r}_{ui}^{\\text{GNN}} + (1 - \\alpha) \\cdot \\hat{r}_{ui}^{\\text{CBF}}$$
            
            Trong đó:
            - $\\hat{r}_{ui}^{\\text{GNN}}$: Điểm dự đoán từ mô hình GNN (đã chuẩn hóa về [0, 1])
            - $\\hat{r}_{ui}^{\\text{CBF}}$: Điểm dự đoán từ mô hình CBF (đã chuẩn hóa về [0, 1])
            - $\\alpha$: Trọng số kết hợp (0 ≤ α ≤ 1)
              - $\\alpha = 0$: Chỉ sử dụng CBF
              - $\\alpha = 0.5$: Cân bằng giữa GNN và CBF
              - $\\alpha = 1$: Chỉ sử dụng GNN
            
            **Quá trình chuẩn hóa:**
            1. Chuẩn hóa điểm GNN về [0, 1]: $\\hat{r}_{ui}^{\\text{GNN}} = \\frac{\\hat{r}_{ui}^{\\text{GNN}} - \\min(\\hat{r}^{\\text{GNN}})}{\\max(\\hat{r}^{\\text{GNN}}) - \\min(\\hat{r}^{\\text{GNN}})}$
            2. Chuẩn hóa điểm CBF về [0, 1]: $\\hat{r}_{ui}^{\\text{CBF}} = \\frac{\\hat{r}_{ui}^{\\text{CBF}} - \\min(\\hat{r}^{\\text{CBF}})}{\\max(\\hat{r}^{\\text{CBF}}) - \\min(\\hat{r}^{\\text{CBF}})}$
            3. Kết hợp tuyến tính với trọng số $\\alpha$
            
            **Kết quả mong đợi:** Điểm $Score_{Hybrid}(u, i)$ có độ chính xác dự đoán cao nhất, kết hợp ưu điểm của cả GNN (collaborative filtering) và CBF (content-based filtering).
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_cbf_predictions = 'cbf_predictions' in st.session_state
            has_gnn_predictions = 'gnn_predictions' in st.session_state
            has_gnn_training = 'gnn_training' in st.session_state

            if not has_cbf_predictions:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 2.2 (CBF Predictions). Vui lòng chạy Bước 2.2 trước.")
            if not has_gnn_predictions and not has_gnn_training:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 3.3 (GNN Predictions) hoặc Bước 3.4 (Trained Model). Vui lòng chạy một trong hai bước trước.")
            
            if has_cbf_predictions and (has_gnn_predictions or has_gnn_training):
                # Get GNN predictions
                if has_gnn_training:
                    gnn_predictions = st.session_state['gnn_training']
                elif has_gnn_predictions:
                    gnn_predictions = st.session_state['gnn_predictions']
                
                cbf_predictions = st.session_state['cbf_predictions']
                
                # Configuration
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    alpha = st.slider(
                        "Trọng số kết hợp (α)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        key="hybrid_alpha"
                    )
                    st.info(f"**α = {alpha}:** {alpha*100:.0f}% GNN + {(1-alpha)*100:.0f}% CBF")
                
                with col_config2:
                    top_k = st.number_input(
                        "Số lượng sản phẩm Top-K để xếp hạng",
                        min_value=5,
                        max_value=100,
                        value=20,
                        step=5,
                        key="hybrid_top_k"
                    )
                
                process_button = st.button(
                    "🔧 Hợp nhất Điểm số Hybrid",
                    type="primary",
                    use_container_width=True,
                    key="hybrid_combine_button"
                )
                
                if process_button:
                    if combine_hybrid_scores is None:
                        st.error(f"❌ Không thể import hybrid_utils module: {_hybrid_utils_import_error}")
                        st.info("Vui lòng đảm bảo file apps/utils/hybrid_utils.py tồn tại và có thể import được.")
                    else:
                        with st.spinner("Đang hợp nhất điểm số GNN và CBF..."):
                            try:
                                # Combine scores
                                hybrid_result = combine_hybrid_scores(cbf_predictions, gnn_predictions, alpha, top_k)
                                
                                # Store in session state & lưu ra artifacts
                                st.session_state['hybrid_predictions'] = hybrid_result
                                save_predictions_artifact("hybrid", hybrid_result)
                                
                                st.success(f"✅ **Hoàn thành!** Đã hợp nhất điểm số cho {hybrid_result['stats']['total_users']} users.")
                                
                                # Display statistics
                                st.markdown("### 📊 Thống kê Hybrid Predictions")
                                
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1:
                                    st.metric("Tổng số users", hybrid_result['stats']['total_users'])
                                    st.metric("Trọng số α", f"{alpha:.2f}")
                                with col_stat2:
                                    st.metric("CBF Score Range", f"[{hybrid_result['stats']['cbf_min']:.4f}, {hybrid_result['stats']['cbf_max']:.4f}]")
                                with col_stat3:
                                    st.metric("GNN Score Range", f"[{hybrid_result['stats']['gnn_min']:.4f}, {hybrid_result['stats']['gnn_max']:.4f}]")
                                
                                # Display sample rankings
                                st.markdown(f"### 📋 Mẫu Rankings Top-{top_k} (5 users đầu tiên)")
                                
                                if 'rankings' in hybrid_result:
                                    sample_users = list(hybrid_result['rankings'].keys())[:5]
                                    
                                    for idx, user_id in enumerate(sample_users, 1):
                                        ranking = hybrid_result['rankings'][user_id]
                                        
                                        with st.expander(f"User {user_id} - Top {len(ranking)} sản phẩm", expanded=False):
                                            ranking_df = pd.DataFrame([
                                                {
                                                    'Rank': rank + 1,
                                                    'Product ID': product_id,
                                                    'Hybrid Score': f"{score:.4f}"
                                                }
                                                for rank, (product_id, score) in enumerate(ranking)
                                            ])
                                            st.dataframe(ranking_df, use_container_width=True)
                                
                                st.markdown("""
                                **✅ Kết quả đạt được:**
                                - ✅ Điểm $Score_{Hybrid}(u, i)$ kết hợp ưu điểm của cả GNN và CBF
                                - ✅ Top-K rankings cho mỗi user
                                - ✅ Sẵn sàng cho quá trình gợi ý cá nhân hóa và đánh giá
                                """)
                            
                            except Exception as e:
                                st.error(f"❌ Lỗi khi hợp nhất điểm số: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

        with st.expander("Bước 4.3: Tạo Danh sách gợi ý cá nhân hóa với Hybrid", expanded=True):
            st.write("**Nội dung thực hiện:**")
            st.write("1. **Gợi ý Cá nhân hóa:** Áp dụng Logic Lọc và Ưu tiên (Bước 2.3) lên danh sách ứng viên được xếp hạng bởi $Score_{Hybrid}(u, i)$.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 4.1 & 4.2 (Hybrid Predictions)")

            st.markdown("""
            **Kết quả mong đợi:**
            - Danh sách Personalized có chất lượng và tính đa dạng cao nhất
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_hybrid_predictions = 'hybrid_predictions' in st.session_state

            if not has_hybrid_predictions:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 4.1 & 4.2 (Hybrid Predictions). Vui lòng chạy Bước 4.1 & 4.2 trước.")
            else:
                hybrid_predictions = st.session_state['hybrid_predictions']
                
                st.markdown("""
                **Quy trình lọc và xếp hạng với Hybrid Scores:**
                
                Bước 4.3 áp dụng cùng logic lọc cá nhân hóa như Bước 2.3, nhưng sử dụng điểm số Hybrid ($Score_{Hybrid}(u, i)$) thay vì điểm CBF ($\\hat{r}_{ui}^{\\text{CBF}}$). 
                Điểm Hybrid kết hợp ưu điểm của cả GNN và CBF, mang lại độ chính xác và tính đa dạng cao hơn.
                
                **1. Lọc Cứng theo articleType (STRICT):**
                   - **Logic:** $i_{\\text{cand}} \\in I_{\\text{valid}}$ nếu và chỉ nếu $i_{\\text{cand}}.\\text{articleType} = i_{\\text{payload}}.\\text{articleType}$
                   - **Mục đích:** Đảm bảo các sản phẩm gợi ý cùng loại với sản phẩm đầu vào (payload)
                   - **Kết quả:** Loại bỏ tất cả các sản phẩm không cùng loại với sản phẩm đầu vào
                   - **Ví dụ:** Nếu payload là "Trousers", chỉ các sản phẩm "Trousers" mới được giữ lại
                
                **2. Lọc và Ưu tiên theo Giới tính/Độ tuổi (Age/Gender Priority):**
                   - **Logic Áp dụng (Strict Filtering):**
                     - Nếu $u.\\text{age} < 13$ và $u.\\text{gender} = \\text{'male'}$: $i_{\\text{cand}}.\\text{gender}$ phải là $\\text{'Boys'}$
                     - Nếu $u.\\text{age} \\ge 13$ và $u.\\text{gender} = \\text{'female'}$: $i_{\\text{cand}}.\\text{gender}$ phải là $\\text{'Women'}$ hoặc $\\text{'Unisex'}$
                   - **Mục đích:** Đảm bảo các sản phẩm phù hợp với đặc điểm nhân khẩu học của người dùng
                   - **Phân tích Ưu tiên/Xếp hạng:** Các sản phẩm còn lại sau khi lọc cứng được xếp hạng trực tiếp bằng điểm Hybrid ($Score_{Hybrid}(u, i)$)
                
                **3. Xếp hạng theo Hybrid Score:**
                   - **Công thức:** $Score_{Hybrid}(u, i) = \\alpha \\cdot \\hat{r}_{ui}^{\\text{GNN}} + (1 - \\alpha) \\cdot \\hat{r}_{ui}^{\\text{CBF}}$
                   - **Ưu điểm:** Kết hợp sức mạnh của Graph Neural Network (học từ cấu trúc đồ thị tương tác) và Content-Based Filtering (dựa trên đặc trưng sản phẩm)
                   - **Kết quả:** Danh sách Top-K được sắp xếp theo điểm Hybrid giảm dần
                
                **Kết quả mong đợi:**
                - ✅ Danh sách ứng viên được lọc chỉ chứa các sản phẩm hợp lệ về articleType, age, và gender
                - ✅ Danh sách được xếp hạng theo điểm $Score_{Hybrid}(u, i)$ để tạo ra danh sách Top-K Personalized cuối cùng
                - ✅ Đảm bảo tính hợp lệ cơ bản và độ ưu tiên của các đề xuất
                - ✅ Chất lượng gợi ý cao hơn nhờ kết hợp ưu điểm của cả GNN và CBF
                
                **So sánh với Bước 2.3:**
                - **Bước 2.3:** Sử dụng $\\hat{r}_{ui}^{\\text{CBF}}$ (chỉ dựa trên đặc trưng nội dung)
                - **Bước 4.3:** Sử dụng $Score_{Hybrid}(u, i)$ (kết hợp GNN + CBF)
                - **Lợi ích:** Hybrid score mang lại độ chính xác cao hơn và khả năng phát hiện các mẫu phức tạp từ đồ thị tương tác
                """)
                
                # Kiểm tra xem có hàm apply_personalized_filters không
                if apply_personalized_filters is not None:
                    # Load products and users data
                    products_path = os.path.join(current_dir, 'apps', 'exports', 'products.csv')
                    users_path = os.path.join(current_dir, 'apps', 'exports', 'users.csv')
                    
                    products_df = None
                    users_df = None
                    
                    if os.path.exists(products_path):
                        products_df = pd.read_csv(products_path)
                        if 'id' in products_df.columns:
                            products_df['id'] = products_df['id'].astype(str)
                            products_df.set_index('id', inplace=True)
                    else:
                        st.warning("⚠️ Không tìm thấy file products.csv. Vui lòng đảm bảo file tồn tại trong apps/exports/")
                    
                    if os.path.exists(users_path):
                        users_df = pd.read_csv(users_path)
                        if 'id' in users_df.columns:
                            users_df['id'] = users_df['id'].astype(str)
                    else:
                        st.warning("⚠️ Không tìm thấy file users.csv. Vui lòng đảm bảo file tồn tại trong apps/exports/")
                    
                    if products_df is not None:
                        # Kiểm tra format của hybrid_predictions
                        if 'predictions' in hybrid_predictions:
                            predictions_dict = hybrid_predictions['predictions']
                        elif 'rankings' in hybrid_predictions:
                            # Convert rankings to predictions format
                            predictions_dict = {}
                            for user_id, ranking in hybrid_predictions['rankings'].items():
                                user_id_str = str(user_id)
                                predictions_dict[user_id_str] = {str(pid): score for pid, score in ranking}
                        else:
                            st.error("❌ Không tìm thấy 'predictions' hoặc 'rankings' trong hybrid_predictions")
                            predictions_dict = {}
                        
                        if predictions_dict:
                            # Configuration
                            col_config1, col_config2 = st.columns(2)
                            with col_config1:
                                selected_user_id = st.selectbox(
                                    "Chọn User ID để áp dụng lọc",
                                    list(predictions_dict.keys()),
                                    key="hybrid_filter_user_id"
                                )
                            
                            with col_config2:
                                payload_articletype = st.selectbox(
                                    "Chọn articleType của sản phẩm đầu vào (payload)",
                                    products_df['articleType'].unique().tolist() if 'articleType' in products_df.columns else [],
                                    key="hybrid_payload_articletype"
                                )
                            
                            # Get user info
                            user_age = None
                            user_gender = None
                            if users_df is not None and selected_user_id:
                                user_row = users_df[users_df['id'] == selected_user_id]
                                if not user_row.empty:
                                    user_age = user_row.iloc[0].get('age', None)
                                    user_gender = user_row.iloc[0].get('gender', None)
                            
                            if selected_user_id and payload_articletype:
                                col_info1, col_info2 = st.columns(2)
                                with col_info1:
                                    if user_age is not None:
                                        st.info(f"👤 User Age: {user_age}")
                                    if user_gender is not None:
                                        st.info(f"👤 User Gender: {user_gender}")
                                with col_info2:
                                    st.info(f"📦 Payload articleType: {payload_articletype}")
                                    if user_age is not None and user_gender is not None:
                                        allowed_genders = get_allowed_genders(user_age, user_gender) if get_allowed_genders else []
                                        st.info(f"✅ Allowed Genders: {', '.join(allowed_genders)}")
                                
                                # Top-K configuration
                                top_k_personalized = st.number_input(
                                    "Số lượng sản phẩm Top-K Personalized",
                                    min_value=5,
                                    max_value=100,
                                    value=20,
                                    step=5,
                                    key="hybrid_top_k_personalized"
                                )
                                
                                process_button = st.button(
                                    "🔧 Áp dụng Personalized Filters và Xếp hạng Top-K với Hybrid",
                                    type="primary",
                                    use_container_width=True,
                                    key="hybrid_personalized_filter_button"
                                )
                                
                                if process_button:
                                    # Đo Inference Time (từ khi nhận user đến khi tạo L(u) - Bước 4.3)
                                    inference_start_time = time.time()
                                    
                                    with st.spinner("Đang áp dụng các bộ lọc cá nhân hóa và xếp hạng với Hybrid scores..."):
                                        try:
                                            # Lấy danh sách candidate products từ Hybrid predictions
                                            user_predictions = predictions_dict[selected_user_id]
                                            candidate_products = list(user_predictions.keys())
                                            
                                            # Áp dụng filters và xếp hạng Top-K với Hybrid scores
                                            result = apply_personalized_filters(
                                                candidate_products,
                                                products_df,
                                                payload_articletype=payload_articletype,
                                                user_age=user_age,
                                                user_gender=user_gender,
                                                cbf_scores=user_predictions,  # Sử dụng hybrid scores như cbf_scores
                                                top_k=top_k_personalized
                                            )
                                            
                                            # Kết thúc đo Inference Time
                                            inference_end_time = time.time()
                                            inference_time_measured = inference_end_time - inference_start_time
                                            
                                            st.success(f"✅ **Hoàn thành!** Đã lọc danh sách ứng viên với Hybrid scores.")
                                            
                                            # Store in session state
                                            if 'hybrid_personalized_filters' not in st.session_state:
                                                st.session_state['hybrid_personalized_filters'] = {}
                                            st.session_state['hybrid_personalized_filters'][selected_user_id] = result
                                            # Lưu vào artifacts để không bị mất khi chạy bước khác
                                            save_intermediate_artifact('hybrid_personalized_filters', st.session_state['hybrid_personalized_filters'])
                                            
                                            # Lưu Inference Time vào session state (lấy trung bình nếu có nhiều users)
                                            if 'hybrid_inference_times' not in st.session_state:
                                                st.session_state['hybrid_inference_times'] = []
                                            st.session_state['hybrid_inference_times'].append(inference_time_measured)
                                            st.session_state['hybrid_inference_time'] = np.mean(st.session_state['hybrid_inference_times'])
                                            
                                            # Display statistics
                                            st.markdown("### 📊 Thống kê quá trình lọc")
                                            
                                            stats = result['stats']
                                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                            with col_stat1:
                                                st.metric("Danh sách ban đầu", f"{stats['initial_count']:,}")
                                            with col_stat2:
                                                st.metric("Sau lọc articleType", f"{stats['after_articletype']:,}")
                                            with col_stat3:
                                                st.metric("Sau lọc Age/Gender", f"{stats['after_age_gender']:,}")
                                            with col_stat4:
                                                st.metric(f"Top-K Personalized ({top_k_personalized})", f"{stats['final_count']:,}")
                                            
                                            # Hiển thị Top-K Personalized Rankings
                                            if result.get('ranked_products'):
                                                st.markdown(f"### 📋 Danh sách Top-{top_k_personalized} Personalized (Hybrid)")
                                                ranked_df = pd.DataFrame([
                                                    {
                                                        'Rank': rank + 1,
                                                        'Product ID': product_id,
                                                        'Hybrid Score': f"{score:.4f}"
                                                    }
                                                    for rank, (product_id, score) in enumerate(result['ranked_products'])
                                                ])
                                                st.dataframe(ranked_df, use_container_width=True)
                                                
                                                # Biểu đồ Top-K scores
                                                fig_scores = px.bar(
                                                    ranked_df,
                                                    x='Rank',
                                                    y='Hybrid Score',
                                                    title=f"Top-{top_k_personalized} Personalized Hybrid Scores",
                                                    labels={'Rank': 'Xếp hạng', 'Hybrid Score': 'Điểm Hybrid'}
                                                )
                                                st.plotly_chart(fig_scores, use_container_width=True)
                                            
                                            # Reduction visualization
                                            st.markdown("### 📉 Biểu đồ giảm kích thước danh sách")
                                            reduction_df = pd.DataFrame({
                                                'Bước': ['Ban đầu', 'Sau articleType', 'Sau Age/Gender', f'Top-{top_k_personalized}'],
                                                'Số lượng': [
                                                    stats['initial_count'],
                                                    stats['after_articletype'],
                                                    stats['after_age_gender'],
                                                    stats['final_count']
                                                ]
                                            })
                                            
                                            fig = px.bar(
                                                reduction_df,
                                                x='Bước',
                                                y='Số lượng',
                                                title="Quá trình giảm kích thước danh sách ứng viên (Hybrid)",
                                                labels={'Số lượng': 'Số lượng sản phẩm', 'Bước': 'Bước lọc'}
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Ví dụ tính toán
                                            st.markdown("### 🧮 Ví dụ tính toán")
                                            st.markdown(f"""
                                            **Ví dụ:** User {selected_user_id} với danh sách ứng viên ban đầu:
                                            
                                            - **Danh sách ban đầu:** {stats['initial_count']:,} sản phẩm
                                            - **Sau Lọc Cứng 1 (articleType='{payload_articletype}'):** {stats['after_articletype']:,} sản phẩm (giảm {stats['initial_count'] - stats['after_articletype']:,} sản phẩm)
                                            - **Sau Lọc Cứng 2 (User age {'< 13' if user_age and user_age < 13 else '≥ 13'}, gender='{user_gender}'):** {stats['after_age_gender']:,} sản phẩm (giảm {stats['after_articletype'] - stats['after_age_gender']:,} sản phẩm)
                                            - **Sau Xếp hạng Top-{top_k_personalized}:** {stats['final_count']:,} sản phẩm (xếp hạng theo $Score_{{Hybrid}}(u, i)$)
                                            - **Tổng giảm:** {stats['removed_count']:,} sản phẩm ({stats['reduction_rate']:.2f}%)
                                            
                                            **✅ Kết quả đạt được:**
                                            - ✅ Danh sách ứng viên được lọc chỉ chứa các sản phẩm hợp lệ về articleType, age, và gender
                                            - ✅ Danh sách được xếp hạng theo điểm $Score_{{Hybrid}}(u, i)$ để tạo ra danh sách Top-K Personalized cuối cùng
                                            - ✅ Đảm bảo tính hợp lệ cơ bản và độ ưu tiên của các đề xuất
                                            - ✅ Chất lượng gợi ý cao hơn nhờ kết hợp ưu điểm của cả GNN và CBF
                                            """)
                                        
                                        except Exception as e:
                                            st.error(f"❌ Lỗi khi áp dụng personalized filters với Hybrid: {str(e)}")
                                            import traceback
                                            st.code(traceback.format_exc())
                            else:
                                st.info("💡 Vui lòng chọn User ID và articleType để tiếp tục.")
                        else:
                            st.warning("⚠️ Không có predictions trong hybrid_predictions. Vui lòng kiểm tra lại dữ liệu.")
                    else:
                        st.warning("⚠️ Không thể tải dữ liệu products. Vui lòng kiểm tra lại.")
                elif apply_personalized_filters is None:
                    st.error(f"❌ Không thể import cbf_utils module: {_cbf_utils_import_error}")

        with st.expander("Bước 4.4: Tính toán Số liệu (Đánh giá Mô hình)", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:** Tính toán tất cả các chỉ số (Recall@K, NDCG@K,...) tương tự như Bước 2.4, sử dụng $L(u)$ và các tham số thời gian tương ứng của Hybrid.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 4.1 & 4.2 (Hybrid Predictions)")

            st.markdown("""
            **Dữ liệu Đầu vào (Được lấy từ):**
            - **Training Time (s):** Tổng thời gian huấn luyện của GNN và CBF ($\\text{Time}_{\\text{GNN}} + \\text{Time}_{\\text{CBF}}$).
            - **Inference Time (s):** Tổng thời gian tính toán $\\hat{r}_{ui}^{\\text{GNN}}$, $\\hat{r}_{ui}^{\\text{CBF}}$ và bước hợp nhất điểm số.
            - **ILD, NDCG, Recall, Precision:** Dữ liệu tương tự Bước 2.4, nhưng sử dụng $L(u)$ được tạo từ $Score_{Hybrid}(u, i)$.
            """)

            # Kiểm tra dữ liệu từ các bước trước
            has_hybrid_predictions = 'hybrid_predictions' in st.session_state
            has_feature_encoding = 'feature_encoding' in st.session_state

            if not has_hybrid_predictions:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 4.1 & 4.2 (Hybrid Predictions). Vui lòng chạy Bước 4.1 & 4.2 trước.")
            if not has_feature_encoding:
                st.warning("⚠️ Chưa có dữ liệu từ Bước 1.3 (Feature Encoding). Cần cho tính toán Diversity.")
            
            if has_hybrid_predictions and has_feature_encoding:
                hybrid_predictions = st.session_state['hybrid_predictions']
                encoding_result = st.session_state.get('feature_encoding', {})
                encoded_matrix = encoding_result.get('encoded_matrix', None)
                product_ids = encoding_result.get('product_ids', [])
                
                # Load interactions for ground truth
                interactions_path = os.path.join(current_dir, 'apps', 'exports', 'interactions.csv')
                interactions_df = None
                if os.path.exists(interactions_path):
                    interactions_df = pd.read_csv(interactions_path)
                    if 'user_id' in interactions_df.columns:
                        interactions_df['user_id'] = interactions_df['user_id'].astype(str)
                    if 'product_id' in interactions_df.columns:
                        interactions_df['product_id'] = interactions_df['product_id'].astype(str)
                
                # Configuration
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    k_values_input = st.text_input(
                        "Các giá trị K (phân cách bằng dấu phẩy)",
                        value="10,20",
                        key="hybrid_k_values_input"
                    )
                    try:
                        k_values = [int(k.strip()) for k in k_values_input.split(',')]
                    except:
                        k_values = [10, 20]
                        st.warning("⚠️ Định dạng không hợp lệ. Sử dụng mặc định: [10, 20]")
                
                with col_config2:
                    # Training Time = GNN Training Time + CBF Training Time
                    gnn_training_time = st.session_state.get('gnn_training_time', None)
                    cbf_training_time = st.session_state.get('training_time', None)
                    
                    training_time_auto = None
                    if gnn_training_time is not None and cbf_training_time is not None:
                        training_time_auto = gnn_training_time + cbf_training_time
                        st.info(f"⏱️ **Training Time (tự động):** {training_time_auto:.3f}s (GNN: {gnn_training_time:.3f}s + CBF: {cbf_training_time:.3f}s)")
                    elif gnn_training_time is not None:
                        st.warning(f"⚠️ Chỉ có GNN Training Time: {gnn_training_time:.3f}s. Thiếu CBF Training Time.")
                        training_time_auto = gnn_training_time
                    elif cbf_training_time is not None:
                        st.warning(f"⚠️ Chỉ có CBF Training Time: {cbf_training_time:.3f}s. Thiếu GNN Training Time.")
                        training_time_auto = cbf_training_time
                    else:
                        st.warning("⚠️ Chưa có Training Time. Vui lòng chạy Bước 2.1 và Bước 3.4 trước.")
                    
                    # Inference Time = GNN Inference + CBF Inference + Combination Time
                    gnn_inference_time = st.session_state.get('gnn_inference_time', None)
                    cbf_inference_time = st.session_state.get('inference_time', None)
                    
                    inference_time_auto = None
                    if gnn_inference_time is not None and cbf_inference_time is not None:
                        # Estimate combination time (usually very small, ~0.001s)
                        combination_time = 0.001
                        inference_time_auto = gnn_inference_time + cbf_inference_time + combination_time
                        st.info(f"⏱️ **Inference Time (tự động):** {inference_time_auto:.3f}s (GNN: {gnn_inference_time:.3f}s + CBF: {cbf_inference_time:.3f}s + Combine: {combination_time:.3f}s)")
                    elif gnn_inference_time is not None:
                        st.warning(f"⚠️ Chỉ có GNN Inference Time: {gnn_inference_time:.3f}s. Thiếu CBF Inference Time.")
                        inference_time_auto = gnn_inference_time + 0.001
                    elif cbf_inference_time is not None:
                        st.warning(f"⚠️ Chỉ có CBF Inference Time: {cbf_inference_time:.3f}s. Thiếu GNN Inference Time.")
                        inference_time_auto = cbf_inference_time + 0.001
                    else:
                        st.warning("⚠️ Chưa có Inference Time. Vui lòng chạy Bước 2.3 và Bước 3.5 trước.")
                    
                    # Cho phép override thủ công nếu cần
                    st.markdown("**Hoặc nhập thủ công (nếu cần):**")
                    training_time_manual = st.number_input(
                        "Training Time (giây) - Thủ công",
                        min_value=0.0,
                        value=training_time_auto if training_time_auto is not None else 0.0,
                        step=0.1,
                        key="hybrid_training_time_input"
                    )
                    
                    inference_time_manual = st.number_input(
                        "Inference Time (giây) - Thủ công",
                        min_value=0.0,
                        value=inference_time_auto if inference_time_auto is not None else 0.0,
                        step=0.1,
                        key="hybrid_inference_time_input"
                    )
                
                process_button = st.button(
                    "🔧 Tính toán Evaluation Metrics",
                    type="primary",
                    use_container_width=True,
                    key="hybrid_evaluation_metrics_button"
                )
                
                if process_button:
                    # Đo Inference Time
                    inference_start_time = time.time()
                    
                    with st.spinner("Đang tính toán các chỉ số đánh giá..."):
                        try:
                            # Prepare predictions format từ Hybrid Predictions
                            predictions_dict = {}
                            
                            if 'rankings' in hybrid_predictions:
                                for user_id, user_ranking in hybrid_predictions['rankings'].items():
                                    user_id_str = str(user_id)
                                    ranked_products = [(str(pid), score) for pid, score in user_ranking]
                                    predictions_dict[user_id_str] = ranked_products
                            
                            # Sử dụng thời gian đã đo tự động hoặc thời gian nhập thủ công
                            final_training_time = training_time_manual if training_time_manual > 0 else training_time_auto
                            
                            # Prepare ground truth from interactions
                            ground_truth_dict = {}
                            
                            if interactions_df is not None and 'user_id' in interactions_df.columns and 'product_id' in interactions_df.columns:
                                # Consider only positive interactions (purchase, like, cart)
                                positive_interactions = interactions_df[
                                    interactions_df['interaction_type'].isin(['purchase', 'like', 'cart'])
                                ] if 'interaction_type' in interactions_df.columns else interactions_df
                                
                                for user_id in predictions_dict.keys():
                                    user_id_str = str(user_id)
                                    user_interactions = positive_interactions[
                                        positive_interactions['user_id'] == user_id_str
                                    ]
                                    if not user_interactions.empty:
                                        relevant_items = set(user_interactions['product_id'].astype(str).unique())
                                        ground_truth_dict[user_id_str] = relevant_items
                                    else:
                                        ground_truth_dict[user_id_str] = set()
                            
                            # Get all items for coverage
                            all_items = set(product_ids) if product_ids else set()
                            
                            # Kết thúc đo Inference Time
                            inference_end_time = time.time()
                            inference_time_measured = inference_end_time - inference_start_time
                            
                            # Sử dụng inference time đã đo hoặc thủ công
                            final_inference_time = inference_time_manual if inference_time_manual > 0 else inference_time_measured
                            
                            # Compute metrics
                            if compute_cbf_metrics is not None:
                                result = compute_cbf_metrics(
                                    predictions_dict,
                                    ground_truth_dict,
                                    k_values=k_values,
                                    item_features=encoded_matrix,
                                    item_ids=product_ids,
                                    all_items=all_items,
                                    training_time=final_training_time,
                                    inference_time=final_inference_time,
                                    use_ild=True
                                )
                                
                                st.success("✅ **Hoàn thành!** Đã tính toán tất cả các chỉ số đánh giá.")
                                
                                # Store in session state
                                st.session_state['hybrid_evaluation_metrics'] = result
                                # Lưu vào artifacts để không bị mất khi chạy bước khác
                                save_intermediate_artifact('hybrid_evaluation_metrics', result)
                                
                                # Display results (similar to Step 2.5 and 3.5)
                                st.markdown("### 📊 Kết quả Evaluation Metrics")
                                
                                # Create metrics table
                                metrics_data = []
                                for k in k_values:
                                    metrics_data.append({
                                        'K': k,
                                        'Recall@K': f"{result['recall'].get(k, 0.0):.4f}",
                                        'Precision@K': f"{result['precision'].get(k, 0.0):.4f}",
                                        'NDCG@K': f"{result['ndcg'].get(k, 0.0):.4f}"
                                    })
                                
                                metrics_df = pd.DataFrame(metrics_data)
                                st.dataframe(metrics_df, use_container_width=True)
                                
                                # Other metrics
                                col_other1, col_other2, col_other3, col_other4 = st.columns(4)
                                with col_other1:
                                    st.metric("Diversity (ILD@K)", f"{result['diversity']:.4f}" if result['diversity'] is not None else "N/A")
                                with col_other2:
                                    st.metric("Coverage", f"{result['coverage']:.4f}" if result['coverage'] is not None else "N/A")
                                with col_other3:
                                    st.metric("Training Time", f"{result['training_time']:.2f}s" if result['training_time'] is not None else "N/A")
                                with col_other4:
                                    st.metric("Inference Time", f"{result['inference_time']:.2f}s" if result['inference_time'] is not None else "N/A")
                                
                                # Visualization
                                st.markdown("### 📈 Biểu đồ Metrics theo K")
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=k_values,
                                    y=[result['recall'].get(k, 0.0) for k in k_values],
                                    mode='lines+markers',
                                    name='Recall@K',
                                    line=dict(color='blue', width=2)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=k_values,
                                    y=[result['precision'].get(k, 0.0) for k in k_values],
                                    mode='lines+markers',
                                    name='Precision@K',
                                    line=dict(color='green', width=2)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=k_values,
                                    y=[result['ndcg'].get(k, 0.0) for k in k_values],
                                    mode='lines+markers',
                                    name='NDCG@K',
                                    line=dict(color='red', width=2)
                                ))
                                fig.update_layout(
                                    title="Metrics theo K (Hybrid)",
                                    xaxis_title="K",
                                    yaxis_title="Score",
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Summary table for export
                                st.markdown("### 📋 Bảng Tổng hợp Chỉ số (Export)")
                                summary_data = {
                                    'Model': ['Hybrid']
                                }
                                
                                # Thêm các metrics theo K values
                                for k in k_values:
                                    summary_data[f'Recall@{k}'] = [f"{result['recall'].get(k, 0.0):.4f}"]
                                    summary_data[f'Precision@{k}'] = [f"{result['precision'].get(k, 0.0):.4f}"]
                                    summary_data[f'NDCG@{k}'] = [f"{result['ndcg'].get(k, 0.0):.4f}"]
                                
                                # Thêm các metrics khác
                                summary_data['Diversity (ILD@K)'] = [f"{result['diversity']:.4f}" if result['diversity'] is not None else "N/A"]
                                summary_data['Coverage'] = [f"{result['coverage']:.4f}" if result['coverage'] is not None else "N/A"]
                                summary_data['Training Time (s)'] = [f"{result['training_time']:.3f}" if result['training_time'] is not None else "N/A"]
                                summary_data['Inference Time (s)'] = [f"{result['inference_time']:.3f}" if result['inference_time'] is not None else "N/A"]
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df, use_container_width=True)
                                
                                st.markdown("""
                                **✅ Kết quả đạt được:**
                                - ✅ Một hàng dữ liệu hoàn chỉnh trong Bảng Tổng hợp Chỉ số cho Hybrid
                                - ✅ Thể hiện hiệu suất của mô hình Hybrid (kết hợp GNN + CBF)
                                - ✅ Sẵn sàng để so sánh với các mô hình khác (CBF, GNN)
                                """)
                            else:
                                st.error("❌ Không thể import evaluation_metrics module.")
                        
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tính toán evaluation metrics: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

        st.markdown('<div class="sub-header">📚 PHẦN V: BẢNG TỔNG KẾT VÀ SO SÁNH CHỈ SỐ</div>', unsafe_allow_html=True)
        st.markdown("")
        
        with st.expander("Bước 5: Bảng Tổng kết và So sánh Chỉ số", expanded=True):
            # Tự động restore artifacts trước khi kiểm tra dữ liệu
            restore_all_artifacts()
            
            st.write("**Nội dung thực hiện:** Tổng hợp và so sánh tất cả các chỉ số đánh giá từ 3 mô hình: CBF, GNN, và Hybrid.")
            st.write("**Dữ liệu sử dụng:** Kết quả từ Bước 2.4 (CBF Metrics), Bước 3.5 (GNN Metrics), và Bước 4.4 (Hybrid Metrics)")
            
            st.markdown("""
            **Mục đích:**
            - So sánh hiệu suất của 3 mô hình trên cùng một bộ metrics
            - Xác định mô hình tối ưu dựa trên các tiêu chí đánh giá
            - Phân tích điểm mạnh và điểm yếu của từng mô hình
            
            **Các chỉ số được so sánh:**
            - **Recall@K** (K=10, 20): Tỷ lệ relevant items được đề xuất
            - **Precision@K** (K=10, 20): Tỷ lệ items đề xuất là relevant
            - **NDCG@K** (K=10, 20): Chất lượng xếp hạng (chỉ số ưu tiên)
            - **Training Time (s):** Thời gian huấn luyện mô hình
            - **Inference Time (s):** Thời gian tính toán recommendations
            - **Coverage:** Tỷ lệ items được đề xuất ít nhất một lần
            - **Diversity (ILD@K):** Độ đa dạng trong danh sách đề xuất
            """)

            # Kiểm tra dữ liệu từ các bước evaluation
            has_cbf_metrics = 'cbf_evaluation_metrics' in st.session_state
            has_gnn_metrics = 'gnn_evaluation_metrics' in st.session_state
            has_hybrid_metrics = 'hybrid_evaluation_metrics' in st.session_state

            col_check1, col_check2, col_check3 = st.columns(3)
            with col_check1:
                if has_cbf_metrics:
                    st.success("✅ CBF Metrics")
                else:
                    st.warning("⚠️ Chưa có CBF Metrics")
            with col_check2:
                if has_gnn_metrics:
                    st.success("✅ GNN Metrics")
                else:
                    st.warning("⚠️ Chưa có GNN Metrics")
            with col_check3:
                if has_hybrid_metrics:
                    st.success("✅ Hybrid Metrics")
                else:
                    st.warning("⚠️ Chưa có Hybrid Metrics")

            if has_cbf_metrics or has_gnn_metrics or has_hybrid_metrics:
                # Configuration for K values
                k_values_input = st.text_input(
                    "Các giá trị K để hiển thị (phân cách bằng dấu phẩy)",
                    value="10,20",
                    key="comparison_k_values"
                )
                try:
                    k_values = [int(k.strip()) for k in k_values_input.split(',')]
                except:
                    k_values = [10, 20]
                    st.warning("⚠️ Định dạng không hợp lệ. Sử dụng mặc định: [10, 20]")

                # Collect metrics from all models
                comparison_data = []

                # CBF Metrics
                if has_cbf_metrics:
                    cbf_metrics = st.session_state['cbf_evaluation_metrics']
                    cbf_row = {'Model': 'CBF (Content-based)'}
                    
                    for k in k_values:
                        cbf_row[f'Recall@{k}'] = f"{cbf_metrics['recall'].get(k, 0.0):.4f}"
                        cbf_row[f'Precision@{k}'] = f"{cbf_metrics['precision'].get(k, 0.0):.4f}"
                        cbf_row[f'NDCG@{k}'] = f"{cbf_metrics['ndcg'].get(k, 0.0):.4f}"
                    
                    cbf_row['Training Time (s)'] = f"{cbf_metrics.get('training_time', 0.0):.3f}" if cbf_metrics.get('training_time') is not None else "N/A"
                    cbf_row['Inference Time (s)'] = f"{cbf_metrics.get('inference_time', 0.0):.3f}" if cbf_metrics.get('inference_time') is not None else "N/A"
                    cbf_row['Coverage'] = f"{cbf_metrics.get('coverage', 0.0):.4f}" if cbf_metrics.get('coverage') is not None else "N/A"
                    cbf_row['Diversity (ILD@K)'] = f"{cbf_metrics.get('diversity', 0.0):.4f}" if cbf_metrics.get('diversity') is not None else "N/A"
                    
                    comparison_data.append(cbf_row)

                # GNN Metrics
                if has_gnn_metrics:
                    gnn_metrics = st.session_state['gnn_evaluation_metrics']
                    gnn_row = {'Model': 'GNN'}
                    
                    for k in k_values:
                        gnn_row[f'Recall@{k}'] = f"{gnn_metrics['recall'].get(k, 0.0):.4f}"
                        gnn_row[f'Precision@{k}'] = f"{gnn_metrics['precision'].get(k, 0.0):.4f}"
                        gnn_row[f'NDCG@{k}'] = f"{gnn_metrics['ndcg'].get(k, 0.0):.4f}"
                    
                    gnn_row['Training Time (s)'] = f"{gnn_metrics.get('training_time', 0.0):.3f}" if gnn_metrics.get('training_time') is not None else "N/A"
                    gnn_row['Inference Time (s)'] = f"{gnn_metrics.get('inference_time', 0.0):.3f}" if gnn_metrics.get('inference_time') is not None else "N/A"
                    gnn_row['Coverage'] = f"{gnn_metrics.get('coverage', 0.0):.4f}" if gnn_metrics.get('coverage') is not None else "N/A"
                    gnn_row['Diversity (ILD@K)'] = f"{gnn_metrics.get('diversity', 0.0):.4f}" if gnn_metrics.get('diversity') is not None else "N/A"
                    
                    comparison_data.append(gnn_row)

                if has_hybrid_metrics:
                    hybrid_metrics = st.session_state['hybrid_evaluation_metrics']
                    hybrid_row = {'Model': 'Hybrid (GNN+CBF)'}
                    
                    for k in k_values:
                        hybrid_row[f'Recall@{k}'] = f"{hybrid_metrics['recall'].get(k, 0.0):.4f}"
                        hybrid_row[f'Precision@{k}'] = f"{hybrid_metrics['precision'].get(k, 0.0):.4f}"
                        hybrid_row[f'NDCG@{k}'] = f"{hybrid_metrics['ndcg'].get(k, 0.0):.4f}"
                    
                    hybrid_row['Training Time (s)'] = f"{hybrid_metrics.get('training_time', 0.0):.3f}" if hybrid_metrics.get('training_time') is not None else "N/A"
                    hybrid_row['Inference Time (s)'] = f"{hybrid_metrics.get('inference_time', 0.0):.3f}" if hybrid_metrics.get('inference_time') is not None else "N/A"
                    hybrid_row['Coverage'] = f"{hybrid_metrics.get('coverage', 0.0):.4f}" if hybrid_metrics.get('coverage') is not None else "N/A"
                    hybrid_row['Diversity (ILD@K)'] = f"{hybrid_metrics.get('diversity', 0.0):.4f}" if hybrid_metrics.get('diversity') is not None else "N/A"
                    
                    comparison_data.append(hybrid_row)

                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    st.markdown("### 📊 Bảng Tổng kết và So sánh Chỉ số")
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Tải xuống Bảng So sánh (CSV)",
                        csv,
                        file_name="model_comparison.csv",
                        mime="text/csv",
                        key="comparison_download"
                    )
                    
                    # Visualization
                    st.markdown("### 📈 Biểu đồ So sánh Metrics")
                    
                    # Select metrics to visualize
                    metric_types = st.multiselect(
                        "Chọn metrics để so sánh",
                        ['Recall', 'Precision', 'NDCG'],
                        default=['Recall', 'Precision', 'NDCG'],
                        key="comparison_metrics"
                    )
                    
                    if metric_types:
                        for metric_type in metric_types:
                            fig = go.Figure()
                            
                            for idx, row in comparison_df.iterrows():
                                model_name = row['Model']
                                metric_values = []
                                
                                for k in k_values:
                                    value_str = row.get(f'{metric_type}@{k}', '0.0000')
                                    try:
                                        value = float(value_str)
                                    except:
                                        value = 0.0
                                    metric_values.append(value)
                                
                                fig.add_trace(go.Scatter(
                                    x=k_values,
                                    y=metric_values,
                                    mode='lines+markers',
                                    name=model_name,
                                    line=dict(width=2)
                                ))
                            
                            fig.update_layout(
                                title=f"{metric_type}@K Comparison",
                                xaxis_title="K",
                                yaxis_title=f"{metric_type}@K",
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparison of other metrics
                    st.markdown("### 📊 So sánh Training Time, Inference Time, Coverage, và Diversity")
                    
                    other_metrics = ['Training Time (s)', 'Inference Time (s)', 'Coverage', 'Diversity (ILD@K)']
                    other_data = []
                    
                    for metric in other_metrics:
                        metric_row = {'Metric': metric}
                        for idx, row in comparison_df.iterrows():
                            model_name = row['Model']
                            value_str = row.get(metric, 'N/A')
                            if value_str != 'N/A':
                                try:
                                    value = float(value_str)
                                    metric_row[model_name] = value
                                except:
                                    metric_row[model_name] = 0.0
                            else:
                                metric_row[model_name] = None
                        other_data.append(metric_row)
                    
                    other_df = pd.DataFrame(other_data)
                    
                    # Create bar chart for each metric
                    for metric in other_metrics:
                        metric_data = other_df[other_df['Metric'] == metric]
                        if not metric_data.empty:
                            fig = go.Figure()
                            
                            for col in comparison_df['Model'].values:
                                value = metric_data[col].iloc[0] if col in metric_data.columns else None
                                if value is not None:
                                    fig.add_trace(go.Bar(
                                        name=col,
                                        x=[metric],
                                        y=[value],
                                        text=f"{value:.4f}" if isinstance(value, float) else str(value),
                                        textposition='auto'
                                    ))
                            
                            fig.update_layout(
                                title=f"{metric} Comparison",
                                xaxis_title="Metric",
                                yaxis_title="Value",
                                barmode='group'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Hướng dẫn So sánh và Lựa chọn Mô hình Tối ưu
                    st.markdown("### 📖 Hướng dẫn So sánh và Lựa chọn Mô hình Tối ưu")
                    
                    st.markdown("""
                    **Phân tích tập trung vào việc xác định Mô hình Hybrid có đạt được sự cân bằng tối ưu giữa các nhóm chỉ số hay không, chứng minh tính ưu việt của kiến trúc kết hợp:**
                    
                    #### 1. Chỉ số Ưu tiên (NDCG@10/20)
                    - **NDCG là thước đo chính của chất lượng xếp hạng.**
                    - **Hybrid phải đạt NDCG cao nhất** so với CBF và GNN riêng lẻ.
                    - NDCG cao cho thấy mô hình có khả năng xếp hạng các items relevant ở vị trí cao hơn.
                    
                    #### 2. Chỉ số Hỗ trợ (Diversity/Coverage)
                    - **Đây là bằng chứng cho khả năng giải quyết vấn đề Cold Start và tránh Định kiến Phổ biến.**
                    - **Hybrid phải có Diversity và Coverage cao hơn GNN.**
                    - Diversity cao: Danh sách đề xuất đa dạng, không chỉ tập trung vào popular items.
                    - Coverage cao: Nhiều items được đề xuất, giúp khám phá items mới.
                    
                    #### 3. Chỉ số Vận hành (Inference Time)
                    - **Mặc dù Hybrid có Inference Time cao nhất**, sự tăng này phải được cân bằng bởi:
                      - Sự cải thiện đáng kể về NDCG
                      - Sự cải thiện về Diversity và Coverage
                    - Inference Time của Hybrid = GNN Inference + CBF Inference + Combination Time
                    
                    #### 4. Phân tích Tổng hợp
                    - **Mô hình tối ưu:** Hybrid nên đạt được sự cân bằng tốt nhất giữa:
                      - ✅ NDCG cao nhất (chất lượng xếp hạng)
                      - ✅ Diversity và Coverage cao (khả năng khám phá)
                      - ✅ Inference Time chấp nhận được (hiệu suất vận hành)
                    
                    #### 5. Kết luận
                    - Nếu Hybrid đạt được cả 3 mục tiêu trên, nó chứng minh tính ưu việt của kiến trúc kết hợp.
                    - Nếu không, cần điều chỉnh trọng số α hoặc cải thiện từng thành phần (GNN hoặc CBF).
                    """)
                    
                    # Automatic analysis
                    st.markdown("### 🤖 Phân tích Tự động")
                    
                    if len(comparison_df) >= 3:
                        # Extract numeric values for comparison
                        def extract_value(value_str):
                            try:
                                return float(value_str)
                            except:
                                return 0.0
                        
                        # Compare NDCG@10 and NDCG@20
                        ndcg_10_values = {}
                        ndcg_20_values = {}
                        diversity_values = {}
                        coverage_values = {}
                        inference_times = {}
                        
                        for idx, row in comparison_df.iterrows():
                            model = row['Model']
                            ndcg_10_values[model] = extract_value(row.get('NDCG@10', '0.0000'))
                            ndcg_20_values[model] = extract_value(row.get('NDCG@20', '0.0000'))
                            diversity_values[model] = extract_value(row.get('Diversity (ILD@K)', '0.0000'))
                            coverage_values[model] = extract_value(row.get('Coverage', '0.0000'))
                            inference_times[model] = extract_value(row.get('Inference Time (s)', '0.0000'))
                        
                        # Find best model for each metric
                        best_ndcg_10 = max(ndcg_10_values.items(), key=lambda x: x[1])
                        best_ndcg_20 = max(ndcg_20_values.items(), key=lambda x: x[1])
                        best_diversity = max(diversity_values.items(), key=lambda x: x[1])
                        best_coverage = max(coverage_values.items(), key=lambda x: x[1])
                        fastest_inference = min(inference_times.items(), key=lambda x: x[1] if x[1] > 0 else float('inf'))
                        
                        col_analysis1, col_analysis2 = st.columns(2)
                        
                        with col_analysis1:
                            st.markdown("#### 🏆 Mô hình Tốt nhất theo từng Metric")
                            st.write(f"**NDCG@10:** {best_ndcg_10[0]} ({best_ndcg_10[1]:.4f})")
                            st.write(f"**NDCG@20:** {best_ndcg_20[0]} ({best_ndcg_20[1]:.4f})")
                            st.write(f"**Diversity:** {best_diversity[0]} ({best_diversity[1]:.4f})")
                            st.write(f"**Coverage:** {best_coverage[0]} ({best_coverage[1]:.4f})")
                            st.write(f"**Inference Time (nhanh nhất):** {fastest_inference[0]} ({fastest_inference[1]:.3f}s)")
                        
                        with col_analysis2:
                            st.markdown("#### 📊 Đánh giá Hybrid Model")
                            
                            hybrid_ndcg_10 = ndcg_10_values.get('Hybrid (GNN+CBF)', 0.0)
                            hybrid_ndcg_20 = ndcg_20_values.get('Hybrid (GNN+CBF)', 0.0)
                            hybrid_diversity = diversity_values.get('Hybrid (GNN+CBF)', 0.0)
                            hybrid_coverage = coverage_values.get('Hybrid (GNN+CBF)', 0.0)
                            gnn_diversity = diversity_values.get('GNN', 0.0)
                            gnn_coverage = coverage_values.get('GNN', 0.0)
                            
                            if hybrid_ndcg_10 >= max([v for k, v in ndcg_10_values.items() if k != 'Hybrid (GNN+CBF)']):
                                st.success("✅ Hybrid có NDCG@10 cao nhất")
                            else:
                                st.warning("⚠️ Hybrid không có NDCG@10 cao nhất")
                            
                            if hybrid_ndcg_20 >= max([v for k, v in ndcg_20_values.items() if k != 'Hybrid (GNN+CBF)']):
                                st.success("✅ Hybrid có NDCG@20 cao nhất")
                            else:
                                st.warning("⚠️ Hybrid không có NDCG@20 cao nhất")
                            
                            if hybrid_diversity > gnn_diversity:
                                st.success("✅ Hybrid có Diversity cao hơn GNN")
                            else:
                                st.warning("⚠️ Hybrid không có Diversity cao hơn GNN")
                            
                            if hybrid_coverage > gnn_coverage:
                                st.success("✅ Hybrid có Coverage cao hơn GNN")
                            else:
                                st.warning("⚠️ Hybrid không có Coverage cao hơn GNN")
                            
                            if hybrid_ndcg_10 >= max([v for k, v in ndcg_10_values.items() if k != 'Hybrid (GNN+CBF)']) and \
                               hybrid_diversity > gnn_diversity:
                                st.success("🎯 **Kết luận:** Hybrid đạt được sự cân bằng tối ưu!")
                            else:
                                st.info("💡 **Gợi ý:** Có thể cần điều chỉnh trọng số α hoặc cải thiện từng thành phần.")
                else:
                    st.info("💡 Vui lòng chạy các bước evaluation (2.5, 3.5, 4.4) để có dữ liệu so sánh.")
            else:
                st.warning("⚠️ Chưa có dữ liệu metrics từ bất kỳ mô hình nào. Vui lòng chạy các bước evaluation trước.")
    else:
        st.markdown("## 👗 Recommendations")
        st.write("Tạo danh sách gợi ý cá nhân hóa và outfit dựa trên Hybrid (GNN + CBF).")

        products_df = load_products_data()
        users_df = load_users_data()
        interactions_df = load_interactions_data()

        if products_df is None or users_df is None:
            st.warning("⚠️ Không tìm thấy dữ liệu `products.csv` hoặc `users.csv`. Vui lòng chạy bước xuất dữ liệu (1.1).")
            st.stop()

        user_index = users_df.index.astype(str)
        product_index = products_df.index.astype(str)

        # Chỉ hiển thị các user đã có predictions (đủ điều kiện) nếu có
        eligible_user_ids = None
        try:
            # Ưu tiên Hybrid → GNN → CBF
            pred_sources = [
                st.session_state.get("hybrid_predictions"),
                st.session_state.get("gnn_predictions") or st.session_state.get("gnn_training"),
                st.session_state.get("cbf_predictions"),
            ]
            for src in pred_sources:
                if not src or not isinstance(src, dict):
                    continue
                preds = src.get("predictions")
                if preds:
                    eligible_user_ids = {str(uid) for uid in preds.keys()}
                    break
        except Exception:
            eligible_user_ids = None

        if eligible_user_ids:
            # Chỉ giữ lại những user nằm trong tập có predictions
            user_index_filtered = user_index[user_index.isin(eligible_user_ids)]
            user_options = user_index_filtered.tolist()
        else:
            # Fallback: hiển thị toàn bộ user nếu chưa có predictions nào
            user_options = user_index.tolist()

        product_options = product_index.tolist()

        def format_user_option(uid: str) -> str:
            row = get_user_record(uid, users_df)
            if row is None:
                return uid
            name = row.get('name') or row.get('email') or 'Unknown'
            return f"{name} ({uid})"

        def format_product_option(pid: str) -> str:
            row = get_product_record(pid, products_df)
            if row is None:
                return pid
            name = row.get('productDisplayName') or row.get('articleType') or 'Product'
            return f"{name} ({pid})"

        input_cols = st.columns(2)
        with input_cols[0]:
            selected_user = st.selectbox(
                "Chọn User",
                options=user_options,
                format_func=format_user_option,
                key="rec_user_select"
            )
            active_user_id = selected_user

        with input_cols[1]:
            selected_product = st.selectbox(
                "Chọn Product",
                options=product_options,
                format_func=format_product_option,
                key="rec_product_select"
            )
            active_product_id = selected_product

        config_cols = st.columns(3)
        with config_cols[0]:
            alpha = st.slider("Trọng số Hybrid α (GNN ↔ CBF)", 0.0, 1.0, 0.5, 0.05)
        with config_cols[1]:
            top_k_personalized = st.number_input(
                "Số lượng sản phẩm Personalized",
                min_value=3,
                max_value=50,
                value=6,
                step=1
            )
        with config_cols[2]:
            top_outfits = st.number_input(
                "Số lượng outfit muốn xem",
                min_value=1,
                max_value=5,
                value=3,
                step=1
            )

        if active_product_id:
            st.markdown("### 📌 Sản phẩm đầu vào (payload)")
            payload_row = get_product_record(active_product_id, products_df)
            display_product_info(payload_row.to_dict() if payload_row is not None else {}, score=None)
            if payload_row is not None:
                st.caption(
                    f"ArticleType: {payload_row.get('articleType', 'N/A')} • "
                    f"Usage: {payload_row.get('usage', 'N/A')} • "
                    f"Gender: {payload_row.get('gender', 'N/A')}"
                )

        run_button = st.button("✨ Tạo gợi ý", type="primary", use_container_width=True)

        if run_button:
            if not active_user_id or not active_product_id:
                st.warning("Vui lòng chọn đầy đủ User và Product để tiếp tục.")
                st.stop()

            candidate_pool = max(int(top_k_personalized * 3), 100)
            hybrid_data = ensure_hybrid_predictions(alpha, candidate_pool)
            if hybrid_data is None:
                st.error("Không tìm thấy dữ liệu hybrid predictions. Vui lòng chạy các bước Training trước.")
                st.stop()

            user_record = get_user_record(active_user_id, users_df)
            user_age = None
            if user_record is not None and pd.notna(user_record.get('age')):
                try:
                    user_age = int(user_record.get('age'))
                except (ValueError, TypeError):
                    user_age = None
            user_gender = user_record.get('gender') if user_record is not None else None

            personalized_items = build_personalized_candidates(
                user_id=active_user_id,
                payload_product_id=active_product_id,
                hybrid_predictions=hybrid_data,
                products_df=products_df,
                users_df=users_df,
                interactions_df=interactions_df,
                top_k=int(top_k_personalized)
            )

            if not personalized_items:
                preds = hybrid_data.get("predictions", {}) or {}
                has_hybrid_for_user = any(str(k) == str(active_user_id) for k in preds.keys())
                if not has_hybrid_for_user:
                    st.warning(
                        "Không có bất kỳ điểm Hybrid nào cho user này (chưa được train hoặc đã bị lọc ở bước trước). "
                        "Vui lòng kiểm tra lại dữ liệu train hoặc chọn user khác."
                    )
                else:
                    st.warning(
                        "Không tìm thấy sản phẩm nào thỏa **articleType = articleType của sản phẩm đầu vào** "
                        "trong Top candidate Hybrid. Vui lòng thử sản phẩm khác hoặc nới lỏng điều kiện."
                    )
            else:
                st.subheader("🎯 Personalized Products")
                allowed_genders = get_allowed_genders(user_age, user_gender)
                st.caption(f"Ưu tiên giới tính theo luật: {', '.join(allowed_genders)}")

                personal_table = []
                for idx, item in enumerate(personalized_items, start=1):
                    row = item['product_row']
                    personal_table.append({
                        "Rank": idx,
                        "Product ID": item['product_id'],
                        "Name": row.get('productDisplayName', 'N/A'),
                        "ArticleType": row.get('articleType', 'N/A'),
                        "Usage": row.get('usage', 'N/A'),
                        "Gender": row.get('gender', 'N/A'),
                        "Hybrid Score": round(item['base_score'], 4),
                        "Priority Score": round(item['score'], 4),
                        "Highlights": " • ".join(item['reasons']) or "-"
                    })

                st.dataframe(pd.DataFrame(personal_table), use_container_width=True)

                for idx, item in enumerate(personalized_items, start=1):
                    with st.expander(f"#{idx} - {item['product_row'].get('productDisplayName', 'Product')}"):
                        display_product_info(item['product_row'].to_dict(), score=item['score'])
                        st.write(f"- ArticleType: {item['product_row'].get('articleType', 'N/A')}")
                        st.write(f"- Usage: {item['product_row'].get('usage', 'N/A')}")
                        st.write(f"- Gender: {item['product_row'].get('gender', 'N/A')}")
                        if item['reasons']:
                            st.write(f"- Ưu tiên: {', '.join(item['reasons'])}")

                st.subheader("🧥 Outfit Suggestions")
                outfits = build_outfit_suggestions(
                    user_id=active_user_id,
                    payload_product_id=active_product_id,
                    personalized_items=personalized_items,
                    products_df=products_df,
                    hybrid_predictions=hybrid_data,
                    user_age=user_age,
                    user_gender=user_gender,
                    max_outfits=int(top_outfits)
                )

                if not outfits:
                    # Kiểm tra payload có phải Unisex không để hiển thị message phù hợp
                    payload_row = get_product_record(current_product_id, products_df)
                    is_unisex = False
                    if payload_row is not None:
                        payload_gender = str(payload_row.get('gender', '')).strip().lower()
                        is_unisex = payload_gender == 'unisex'
                    
                    if is_unisex:
                        st.info("Chưa đủ thành phần để tạo outfit thoả điều kiện (Accessories / Topwear / Bottomwear / Footwear cùng usage).")
                    else:
                        st.info("Chưa đủ thành phần để tạo outfit thoả điều kiện (Accessories / Topwear / Bottomwear / Footwear cùng gender và cùng usage).")
                else:
                    for idx, outfit in enumerate(outfits, start=1):
                        st.markdown(f"#### 👗 Outfit #{idx}")
                        for pid in outfit['products']:
                            product_row = get_product_record(pid, products_df)
                            if product_row is not None:
                                display_product_info(product_row.to_dict(), score=None)
                        st.divider()

if __name__ == "__main__":
    main()
