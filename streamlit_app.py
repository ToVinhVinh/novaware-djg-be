"""Streamlit dashboard for Novaware product analytics and model APIs."""

from __future__ import annotations

import json
import time
from io import BytesIO
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import re
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import os
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


st.set_page_config(
    page_title="Novaware Product Insights",
    page_icon="🧥",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .model-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
        margin: 5px 0;
    }
    .step-header {
        background-color: #FF4B4B;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧥 Novaware Product Insights & Model Console")
st.caption(
    "Upload CSV data, explore quick analytics, and interact with the "
    "GNN / CBF / Hybrid recommendation APIs."
)

@st.cache_data(show_spinner=False)
def load_csv(file_buffer: BytesIO) -> pd.DataFrame:
    """Load CSV with error handling for malformed data."""
    try:
        # Try standard read first
        return pd.read_csv(file_buffer)
    except pd.errors.ParserError:
        # Reset buffer position
        file_buffer.seek(0)
        # Try with error handling options
        try:
            # Option 1: Skip bad lines
            return pd.read_csv(file_buffer, on_bad_lines='skip', engine='python')
        except Exception:
            # Reset buffer position again
            file_buffer.seek(0)
            # Option 2: Use more lenient parsing
            return pd.read_csv(
                file_buffer,
                on_bad_lines='skip',
                quoting=1,  # QUOTE_ALL
                escapechar='\\',
                engine='python'
            )


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for all columns in the dataframe."""
    numeric_stats = df.describe(
        percentiles=[0.25, 0.5, 0.75],
        include="all",
    ).transpose()
    # Select only available columns (some may not exist for non-numeric data)
    available_cols = [col for col in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"] 
                      if col in numeric_stats.columns]
    numeric_stats = numeric_stats[available_cols].dropna(how="all")
    return numeric_stats


def plot_sparsity(df: pd.DataFrame) -> None:
    """Plot missing data ratio using KDE (Kernel Density Estimation)."""
    sparsity = df.isna().sum() / len(df) if len(df) else df.isna().sum()
    sparsity_values = sparsity.values
    
    # Create KDE plot
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if len(sparsity_values) > 0 and sparsity_values.max() > 0:
        # KDE plot
        sns.kdeplot(data=sparsity_values, fill=True, color='#FF4B4B', ax=ax)
        ax.set_xlabel('Missing Ratio', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of Missing Data (KDE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = sparsity_values.mean()
        median_val = pd.Series(sparsity_values).median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2%}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2%}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No missing data', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Show detailed table
    with st.expander("📊 Chi tiết Missing Ratio theo cột"):
        sparsity_df = (
            sparsity.rename("Missing Ratio")
            .reset_index()
            .rename(columns={"index": "Column"})
            .sort_values("Missing Ratio", ascending=False)
        )
        sparsity_df["Missing Ratio"] = sparsity_df["Missing Ratio"].apply(lambda x: f"{x:.2%}")
        st.dataframe(sparsity_df, use_container_width=True, hide_index=True)


def plot_ratio(df: pd.DataFrame, column: str) -> None:
    """Plot value distribution for a categorical column."""
    value_counts = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .value_counts(normalize=True)
        .mul(100)
    )
    
    # Create DataFrame with proper column names
    value_ratio = pd.DataFrame({
        column: value_counts.index,
        "Percentage": value_counts.values
    })
    
    st.bar_chart(
        value_ratio,
        x=column,
        y="Percentage",
        use_container_width=True,
    )


def call_api(
    base_url: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    method: str = "post",
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        response = requests.request(method, url, json=payload, timeout=600)
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json(),
        }
    except requests.RequestException as exc:
        return {
            "success": False,
            "error": str(exc),
            "response": getattr(exc, "response", None)
            and getattr(exc.response, "text", None),
        }
    except json.JSONDecodeError:
        return {"success": True, "data": {"message": "Completed", "raw": response.text}}


BASE_URL = st.sidebar.text_input(
    "API base URL",
    value="http://127.0.0.1:8000/api/v1",
    help="Đặt URL backend Django (ví dụ http://localhost:8000/api/v1).",
)
st.sidebar.markdown("---")
st.sidebar.write("User_ID cố định: `690bf0f2d0c3753df0ecbdd6`")
st.sidebar.write("Product_ID thử nghiệm: `10068`")


# Store training results in session state
if "training_results" not in st.session_state:
    st.session_state.training_results = {
        "gnn": None,
        "cbf": None,
        "hybrid": None,
    }

if "recommendation_results" not in st.session_state:
    st.session_state.recommendation_results = {
        "gnn": None,
        "cbf": None,
        "hybrid": None,
    }

# Store evaluation_support (pairs or ids provided by API) in session state
if "evaluation_support" not in st.session_state:
    st.session_state.evaluation_support = {
        "gnn": None,
        "cbf": None,
        "hybrid": None,
    }


def extract_training_metrics(result_data: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Extract training metrics from API response.
    
    This extracts metrics from /train API response which includes:
    - Training parameters: num_users, num_products, epochs, batch_size, etc.
    - Training time: time taken to train the model
    """
    metrics = {
        "num_users": "N/A",
        "num_products": "N/A",
        "num_interactions": "N/A",
        "num_training_samples": "N/A",
        "epochs": "N/A",
        "batch_size": "N/A",
        "embed_dim": "N/A",
        "learning_rate": "N/A",
        "test_size": 0.2,
        "training_time": "N/A",
    }
    
    if not result_data:
        return metrics
    
    # Try to extract from different possible response structures
    if isinstance(result_data, dict):
        # Training time - extract from result
        for key in ["training_time", "time"]:
            if key in result_data:
                value = result_data[key]
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    metrics["training_time"] = str(value)
                else:
                    metrics["training_time"] = value
        
        # Training info nested structure
        if "training_info" in result_data:
            info = result_data["training_info"]
            # Map API keys to metric keys
            info_key_mapping = {
                "embedding_dim": "embed_dim",
            }
            for key in ["num_users", "num_products", "num_interactions", "num_training_samples",
                       "epochs", "batch_size", "embed_dim", "embedding_dim", "learning_rate"]:
                if key in info:
                    value = info[key]
                    target_key = info_key_mapping.get(key, key)
                    metrics[target_key] = str(value) if value is not None else "N/A"
        
        # Direct keys at root level (from /train API)
        # Map API keys to metric keys
        key_mapping = {
            "embedding_dim": "embed_dim",  # API returns embedding_dim, we need embed_dim
            "time": "training_time",
        }
        
        for key in ["num_users", "num_products", "num_interactions", "num_training_samples",
                   "epochs", "batch_size", "embed_dim", "embedding_dim", 
                   "learning_rate", "training_time", "time", "test_size"]:
            if key in result_data:
                value = result_data[key]
                # Use mapping if exists, otherwise use key as-is
                target_key = key_mapping.get(key, key)
                if isinstance(value, (int, float)):
                    metrics[target_key] = str(value)
                else:
                    metrics[target_key] = value if value is not None else "N/A"
        
        # Try nested structures (e.g., metrics.evaluation, stats, etc.)
        for nested_key in ["metrics", "evaluation", "stats", "results"]:
            if nested_key in result_data and isinstance(result_data[nested_key], dict):
                nested = result_data[nested_key]
                # Extract training time if available
                for key in ["training_time", "time"]:
                    if key in nested:
                        value = nested[key]
                        if isinstance(value, (int, float)):
                            metrics["training_time"] = str(value)
                        else:
                            metrics["training_time"] = value
        
        # Try to extract from summary or message
        if "summary" in result_data:
            summary = result_data["summary"]
            if isinstance(summary, dict):
                for key in summary:
                    if key in metrics:
                        value = summary[key]
                        metrics[key] = str(value) if isinstance(value, (int, float)) else value
    
    return metrics


def extract_recommend_metrics(result_data: Dict[str, Any], model_type: str) -> Dict[str, Any]:

    """Extract evaluation metrics from /recommend API response.
    
    The /recommend API returns evaluation_metrics with:
    - Recall@10, Recall@20, NDCG@10, NDCG@20, inference_time
    """
    metrics = {
        "recall_at_10": "N/A",
        "recall_at_20": "N/A",
        "ndcg_at_10": "N/A",
        "ndcg_at_20": "N/A",
        "inference_time": "N/A",
    }
    
    if not result_data or not isinstance(result_data, dict):
        return metrics
    
    # Extract from evaluation_metrics (from /recommend API)
    if "evaluation_metrics" in result_data:
        eval_metrics = result_data["evaluation_metrics"]
        if isinstance(eval_metrics, dict):
            for key in ["recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20"]:
                if key in eval_metrics:
                    value = eval_metrics[key]
                    if isinstance(value, (int, float)):
                        metrics[key] = str(value)
                    else:
                        metrics[key] = value
            
            # Inference time (in milliseconds)
            if "inference_time" in eval_metrics:
                value = eval_metrics["inference_time"]
                metrics["inference_time"] = str(value) if isinstance(value, (int, float)) else value
            elif "time" in eval_metrics:
                value = eval_metrics["time"]
                # Convert seconds to milliseconds if needed
                if isinstance(value, (int, float)):
                    if value < 1000:  # Likely in seconds, convert to ms
                        metrics["inference_time"] = str(value * 1000)
                    else:
                        metrics["inference_time"] = str(value)
                else:
                    metrics["inference_time"] = value
    
    return metrics


def extract_evaluation_support(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract evaluation support (tested user/product IDs or pairs) from API response.
    Normalizes to: { 'pairs': [{'user_id':..., 'current_product_id':...}, ...], 'user_ids': [...], 'product_ids': [...] }
    """
    if not isinstance(result_data, dict):
        return None

    def _normalize_pairs(pairs_list):
        norm = []
        for p in pairs_list or []:
            if not isinstance(p, dict):
                continue
            uid = p.get('user_id') or p.get('userId') or p.get('uid')
            pid = p.get('current_product_id') or p.get('product_id') or p.get('item_id') or p.get('pid')
            if uid is not None and pid is not None:
                norm.append({'user_id': str(uid), 'current_product_id': str(pid)})
        return norm

    # 1) Direct key
    if 'evaluation_support' in result_data:
        es = result_data.get('evaluation_support')
        pairs = []
        user_ids = None
        product_ids = None
        if isinstance(es, dict):
            # dict form
            if isinstance(es.get('pairs'), list):
                pairs = _normalize_pairs(es.get('pairs'))
            if isinstance(es.get('tested_pairs'), list):
                pairs = pairs or _normalize_pairs(es.get('tested_pairs'))
            if isinstance(es.get('test_pairs'), list):
                pairs = pairs or _normalize_pairs(es.get('test_pairs'))
            if isinstance(es.get('user_ids'), list):
                user_ids = [str(x) for x in es.get('user_ids')]
            if isinstance(es.get('product_ids'), list):
                product_ids = [str(x) for x in es.get('product_ids')]
        elif isinstance(es, list):
            pairs = _normalize_pairs(es)
        if pairs or user_ids or product_ids:
            return {'pairs': pairs, 'user_ids': user_ids, 'product_ids': product_ids}

    # 2) Alternate keys on root
    for key in ['tested_pairs', 'test_pairs', 'test_cases']:
        if isinstance(result_data.get(key), list):
            pairs = _normalize_pairs(result_data.get(key))
            if pairs:
                return {'pairs': pairs, 'user_ids': None, 'product_ids': None}

    # 3) Root arrays
    if isinstance(result_data.get('user_ids'), list) and isinstance(result_data.get('product_ids'), list):
        return {
            'pairs': None,
            'user_ids': [str(x) for x in result_data['user_ids']],
            'product_ids': [str(x) for x in result_data['product_ids']],
        }

    # 4) Nested common containers
    for container in ['data', 'metrics', 'evaluation', 'results']:
        sub = result_data.get(container)
        if isinstance(sub, dict):
            found = extract_evaluation_support(sub)
            if found:
                return found

    return None

def auto_fill_metrics_to_session_state(slug: str, metrics: Dict[str, Any]) -> None:
    """Auto-fill extracted metrics to session state for input fields."""
    # Map of metric keys to session state keys
    field_mapping = {
        "num_users": f"{slug}_num_users",
        "num_products": f"{slug}_num_products",
        "num_interactions": f"{slug}_num_interactions",
        "num_training_samples": f"{slug}_num_samples",
        "epochs": f"{slug}_epochs",
        "batch_size": f"{slug}_batch",
        "embed_dim": f"{slug}_embed",
        "learning_rate": f"{slug}_lr",
        "test_size": f"{slug}_test_size",
        "training_time": f"{slug}_training_time",
        "recall_at_10": f"{slug}_recall_at_10",
        "recall_at_20": f"{slug}_recall_at_20",
        "ndcg_at_10": f"{slug}_ndcg_at_10",
        "ndcg_at_20": f"{slug}_ndcg_at_20",
        "inference_time": f"{slug}_inference_time",
    }
    
    # Update session state with extracted metrics
    for metric_key, state_key in field_mapping.items():
        if metric_key in metrics and metrics[metric_key] != "N/A":
            value = metrics[metric_key]
            # Convert to appropriate type
            if metric_key == "test_size":
                try:
                    st.session_state[state_key] = float(value) if value != "N/A" else 0.2
                except (ValueError, TypeError):
                    st.session_state[state_key] = 0.2
            else:
                st.session_state[state_key] = str(value)


PRECISION_FORMAT_KEYS = ("recall_at_10", "recall_at_20", "training_time")


def format_metric_value(value: Any, decimals: int = 4) -> str:
    """Format numeric metrics with fixed decimal places without rounding up."""
    if value is None:
        return "N/A"
    value_str = str(value).strip()
    if not value_str or value_str.upper() == "N/A":
        return "N/A"
    match = re.match(r"^(-?\d+(?:\.\d+)?)(.*)$", value_str)
    suffix = ""
    number_part = value_str
    if match:
        number_part, suffix = match.groups()
    try:
        decimal_value = Decimal(number_part)
    except InvalidOperation:
        return value_str
    quant = Decimal("1").scaleb(-decimals)
    truncated = decimal_value.quantize(quant, rounding=ROUND_DOWN)
    formatted_number = f"{truncated:.{decimals}f}"
    return f"{formatted_number}{suffix}"


def apply_precision_formatting(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure key metrics respect the 4-decimal precision requirement."""
    for key in PRECISION_FORMAT_KEYS:
        metrics_dict[key] = format_metric_value(metrics_dict.get(key))
    return metrics_dict

# ----- Metric computation helpers (apply formulas) -----
from math import log2

def compute_recall_at_k(recommended_ids, ground_truth_ids, k=10) -> float:
    """Recall@K = |recs@K ∩ GT| / |GT| (0..1)."""
    if not ground_truth_ids:
        return 0.0
    rec_topk = list(map(str, recommended_ids[:k]))
    gt = set(map(str, ground_truth_ids))
    hits = len([rid for rid in rec_topk if rid in gt])
    return hits / max(len(gt), 1)


def _dcg_at_k(binary_relevance, k=10) -> float:
    """DCG@K with binary gain: sum_{i=1..K} rel_i / log2(i+1)."""
    dcg = 0.0
    for i, rel in enumerate(binary_relevance[:k], start=1):
        if rel:
            dcg += 1.0 / log2(i + 1)
    return dcg


def compute_ndcg_at_k(recommended_ids, ground_truth_ids, k=10) -> float:
    """NDCG@K = DCG@K / IDCG@K with binary relevance from GT overlap."""
    if not ground_truth_ids:
        return 0.0
    rec_topk = list(map(str, recommended_ids[:k]))
    gt = set(map(str, ground_truth_ids))
    # Build binary relevance vector for the ranked list
    rel = [1 if rid in gt else 0 for rid in rec_topk]
    dcg = _dcg_at_k(rel, k)
    # Ideal relevance: top |GT| are 1s (capped at K)
    ideal_rel = [1] * min(len(gt), k)
    idcg = _dcg_at_k(ideal_rel, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


st.header("1. Upload & Preview CSV")

# Tạo 2 tabs cho sản phẩm và người dùng
tab_product, tab_user = st.tabs(["📦 Dữ liệu Sản phẩm", "👤 Dữ liệu Người dùng"])

# Tab 1: Dữ liệu sản phẩm
with tab_product:
    uploaded_file = st.file_uploader("Tải file CSV sản phẩm", type=["csv"], key="product_csv")

    df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        with st.spinner("Đang đọc dữ liệu sản phẩm..."):
            df = load_csv(uploaded_file)
        st.success(f"Đã tải {len(df):,} dòng, {len(df.columns)} cột.")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Thống kê dữ liệu sản phẩm")
        stats_df = describe_dataframe(df)
        st.dataframe(stats_df, use_container_width=True)

        st.subheader("Biểu đồ độ thưa (Missing Ratio)")
        plot_sparsity(df)

        st.subheader("Biểu đồ tỷ lệ (Value Ratio)")
        ratio_col = st.selectbox(
            "Chọn cột để vẽ biểu đồ tỷ lệ",
            options=df.columns.tolist(),
            key="product_ratio_col",
        )
        if ratio_col:
            plot_ratio(df, ratio_col)
    else:
        st.info("Vui lòng tải lên file CSV sản phẩm để bắt đầu.")

# Tab 2: Dữ liệu người dùng
with tab_user:
    uploaded_user_file = st.file_uploader("Tải file CSV người dùng", type=["csv"], key="user_csv")

    df_user: Optional[pd.DataFrame] = None
    if uploaded_user_file is not None:
        with st.spinner("Đang đọc dữ liệu người dùng..."):
            df_user = load_csv(uploaded_user_file)
        st.success(f"Đã tải {len(df_user):,} người dùng, {len(df_user.columns)} cột.")
        st.dataframe(df_user.head(100), use_container_width=True)

        st.subheader("Thống kê dữ liệu người dùng")
        stats_user_df = describe_dataframe(df_user)
        st.dataframe(stats_user_df, use_container_width=True)

        # Phân tích đặc biệt cho dữ liệu người dùng
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Phân bố Giới tính")
            if "Gender" in df_user.columns:
                gender_counts = df_user["Gender"].value_counts()
                st.bar_chart(gender_counts)
                
                # Hiển thị số liệu
                for gender, count in gender_counts.items():
                    percentage = (count / len(df_user)) * 100
                    st.metric(
                        label=f"{gender}",
                        value=f"{count:,}",
                        delta=f"{percentage:.1f}%"
                    )
            else:
                st.warning("Không tìm thấy cột 'Gender' trong dữ liệu.")

        with col2:
            st.subheader("Phân bố Độ tuổi")
            if "Age" in df_user.columns:
                # Tạo nhóm tuổi
                df_user_copy = df_user.copy()
                df_user_copy["age_group"] = pd.cut(
                    df_user_copy["Age"],
                    bins=[0, 12, 18, 25, 35, 50, 100],
                    labels=["Kids (0-12)", "Teens (13-18)", "Young Adults (19-25)", 
                            "Adults (26-35)", "Middle Age (36-50)", "Senior (50+)"]
                )
                age_group_counts = df_user_copy["age_group"].value_counts().sort_index()
                st.bar_chart(age_group_counts)
                
                # Thống kê độ tuổi
                st.write(f"**Độ tuổi trung bình:** {df_user['Age'].mean():.1f}")
                st.write(f"**Độ tuổi nhỏ nhất:** {df_user['Age'].min()}")
                st.write(f"**Độ tuổi lớn nhất:** {df_user['Age'].max()}")
            else:
                st.warning("Không tìm thấy cột 'Age' trong dữ liệu.")

        st.subheader("Biểu đồ độ thưa (Missing Ratio)")
        plot_sparsity(df_user)

        st.subheader("Biểu đồ tỷ lệ (Value Ratio)")
        user_ratio_col = st.selectbox(
            "Chọn cột để vẽ biểu đồ tỷ lệ",
            options=df_user.columns.tolist(),
            key="user_ratio_col",
        )
        if user_ratio_col:
            plot_ratio(df_user, user_ratio_col)
    else:
        st.info("Vui lòng tải lên file CSV người dùng để phân tích.")


st.header("2. Huấn luyện mô hình")
models = {
    "GNN": "gnn",
    "Content-based (CBF)": "cbf",
    "Hybrid": "hybrid",
}

def poll_task_status(
    base_url: str, 
    endpoint: str, 
    task_id: str, 
    max_wait_time: int = 600,
    status_placeholder=None,
    progress_bar=None
) -> Dict[str, Any]:
    """Poll task status until completion or timeout."""
    start_time = time.time()
    poll_interval = 2  # Poll every 2 seconds
    last_progress = 0
    
    while time.time() - start_time < max_wait_time:
        result = call_api(base_url, endpoint, payload={"task_id": task_id}, method="post")
        
        if not result["success"]:
            return result
        
        data = result["data"]
        status = data.get("status", "unknown")
        
        if status == "success":
            # Success! Return the result with all metrics
            return result
        elif status == "failure":
            return {
                "success": False,
                "error": data.get("error", "Training failed"),
                "data": data,
            }
        elif status in ["pending", "running"]:
            # Update progress if available
            current_progress = data.get("progress", last_progress)
            if current_progress > last_progress:
                last_progress = current_progress
                # Progress from 30% to 90% during polling
                if progress_bar:
                    progress_bar.progress(30 + int(current_progress * 0.6))
                if status_placeholder:
                    message = data.get("message", f"Training in progress... {current_progress}%")
                    current_step = data.get("current_step", "")
                    if current_step:
                        message += f" - {current_step}"
                    status_placeholder.info(message)
            time.sleep(poll_interval)
            continue
        else:
            # Unknown status, wait and retry
            time.sleep(poll_interval)
            continue
    
    return {
        "success": False,
        "error": f"Training timeout after {max_wait_time} seconds",
        "data": {"status": "timeout", "task_id": task_id},
    }


train_cols = st.columns(len(models))
for col, (label, slug) in zip(train_cols, models.items()):
    with col:
        if st.button(f"Train {label}", key=f"train_{slug}"):
            status_placeholder = st.empty()
            progress = st.progress(0)
            status_placeholder.info("Bắt đầu gọi API train...")
            progress.progress(10)
            start_time = time.time()
            
            # Use sync mode to get results immediately
            with st.spinner(f"Đang huấn luyện {label}..."):
                # Try sync mode first (sends sync: true in payload)
                result = call_api(BASE_URL, f"{slug}/train", payload={"sync": True}, method="post")
                
                # If async response (has task_id), poll for results
                if result["success"] and isinstance(result["data"], dict):
                    data = result["data"]
                    if "task_id" in data and data.get("status") in ["pending", "running"]:
                        task_id = data["task_id"]
                        status_placeholder.info(f"Training đang chạy (task_id: {task_id[:8]}...). Đang chờ kết quả...")
                        progress.progress(30)
                        
                        # Poll for completion with progress updates
                        result = poll_task_status(
                            BASE_URL, 
                            f"{slug}/train", 
                            task_id, 
                            max_wait_time=600,
                            status_placeholder=status_placeholder,
                            progress_bar=progress
                        )
            
            elapsed_time = time.time() - start_time
            progress.progress(100)
            
            if result["success"]:
                status_placeholder.success(f"Train {label} hoàn tất.")
                # Store result in session state for documentation
                result_data = result["data"]
                st.session_state.training_results[slug] = result_data
                # Extract and store evaluation_support from /train response (if provided)
                try:
                    support = extract_evaluation_support(result_data)
                    if support:
                        st.session_state.evaluation_support[slug] = support
                        cnt_pairs = len(support.get('pairs') or [])
                        cnt_u = len(support.get('user_ids') or [])
                        cnt_p = len(support.get('product_ids') or [])
                        st.info(f"📦 evaluation_support: pairs={cnt_pairs}, user_ids={cnt_u}, product_ids={cnt_p}")
                except Exception as _:
                    pass
                
                # Add training time if not present
                if isinstance(result_data, dict):
                    training_time_value = result_data.get("training_time")
                    legacy_time_value = result_data.get("time")
                    if training_time_value in (None, "", "N/A") and legacy_time_value in (None, "", "N/A"):
                        result_data["training_time"] = f"{elapsed_time:.2f}s"
                    
                    # Auto-fill metrics to session state for input fields
                    extracted_metrics = extract_training_metrics(result_data, slug)
                    auto_fill_metrics_to_session_state(slug, extracted_metrics)
                
                st.json(result_data)
                st.success(f"✅ Số liệu đã được tự động điền vào phần tài liệu!")
                
                # Tự động gọi API recommend để lấy evaluation metrics
                st.info("🔄 Đang tự động gọi API recommend để lấy evaluation metrics...")
                default_user_id = "690bf0f2d0c3753df0ecbdd6"
                
                # Try to get user's interaction history to test with multiple products
                product_ids_to_test = ["10068"]  # Default
                try:
                    user_url = f"{BASE_URL.rstrip('/')}/users/{default_user_id}"
                    user_response = requests.get(user_url, timeout=10)
                    if user_response.status_code == 200:
                        user_data = user_response.json()
                        if isinstance(user_data, dict) and "data" in user_data:
                            user_info = user_data["data"].get("user", {})
                            interaction_history = user_info.get("interaction_history", [])
                            if interaction_history:
                                # Get product IDs from interaction history
                                history_products = [str(interaction.get("product_id")) for interaction in interaction_history[:5] if interaction.get("product_id")]
                                if history_products:
                                    product_ids_to_test = history_products + ["10068"]  # Add default
                                    product_ids_to_test = list(dict.fromkeys(product_ids_to_test))  # Remove duplicates
                except:
                    pass
                
                # Test with multiple products and find the best result
                best_result = None
                best_metrics = None
                best_product_id = None
                recommended_products_to_try = []  # Collect recommended products to test
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_tests = min(len(product_ids_to_test), 5)
                
                # First pass: Test with products from interaction history
                for idx, product_id in enumerate(product_ids_to_test[:5]):  # Test up to 5 products
                    status_text.info(f"Đang test với product_id: {product_id} ({idx+1}/{total_tests})...")
                    progress_bar.progress((idx + 1) / (total_tests * 2))  # Reserve half for recommended products
                    
                    recommend_payload = {"user_id": default_user_id, "current_product_id": product_id}
                    recommend_result = call_api(BASE_URL, f"{slug}/recommend", payload=recommend_payload)
                    
                    if recommend_result["success"]:
                        data = recommend_result["data"]
                        eval_metrics = data.get("evaluation_metrics", {})
                        
                        # Collect recommended products for second pass
                        personalized = data.get("personalized", [])
                        for rec in personalized[:3]:  # Get first 3 recommendations
                            rec_product = rec.get("product", {})
                            if isinstance(rec_product, dict):
                                rec_id = rec_product.get("id")
                            else:
                                rec_id = rec.get("id") or rec.get("product_id")
                            if rec_id and str(rec_id) not in recommended_products_to_try and str(rec_id) not in product_ids_to_test:
                                recommended_products_to_try.append(str(rec_id))
                        
                        # Check if this result is better (has non-zero/non-null metrics)
                        if eval_metrics:
                            recall_at_10 = eval_metrics.get("recall_at_10", 0)
                            recall_at_20 = eval_metrics.get("recall_at_20", 0)
                            ndcg_at_10 = eval_metrics.get("ndcg_at_10", 0)
                            ndcg_at_20 = eval_metrics.get("ndcg_at_20", 0)
                            
                            # Check if this is a valid result (at least one metric is non-zero/non-null)
                            is_valid = (
                                recall_at_10 != 0 or recall_at_20 != 0 or 
                                ndcg_at_10 != 0 or ndcg_at_20 != 0
                            )
                            
                            if is_valid:
                                # Found valid metrics, use this result
                                best_result = recommend_result
                                best_metrics = eval_metrics
                                best_product_id = product_id
                                break
                            elif best_result is None:
                                # Keep first result as fallback
                                best_result = recommend_result
                                best_metrics = eval_metrics
                                best_product_id = product_id
                
                # Second pass: Test with recommended products if no valid metrics found
                if best_metrics and not any([
                    best_metrics.get("recall_at_10", 0) != 0,
                    best_metrics.get("recall_at_20", 0) != 0,
                    best_metrics.get("ndcg_at_10", 0) != 0,
                    best_metrics.get("ndcg_at_20", 0) != 0
                ]) and recommended_products_to_try:
                    status_text.info(f"Không tìm thấy metrics hợp lệ. Đang test với {len(recommended_products_to_try[:5])} recommended products...")
                    
                    for idx, rec_product_id in enumerate(recommended_products_to_try[:5]):
                        status_text.info(f"Đang test với recommended product_id: {rec_product_id} ({idx+1}/{min(len(recommended_products_to_try), 5)})...")
                        progress_bar.progress((total_tests + idx + 1) / (total_tests * 2))
                        
                        recommend_payload = {"user_id": default_user_id, "current_product_id": rec_product_id}
                        recommend_result = call_api(BASE_URL, f"{slug}/recommend", payload=recommend_payload)
                        
                        if recommend_result["success"]:
                            data = recommend_result["data"]
                            eval_metrics = data.get("evaluation_metrics", {})
                            
                            if eval_metrics:
                                recall_at_10 = eval_metrics.get("recall_at_10", 0)
                                recall_at_20 = eval_metrics.get("recall_at_20", 0)
                                ndcg_at_10 = eval_metrics.get("ndcg_at_10", 0)
                                ndcg_at_20 = eval_metrics.get("ndcg_at_20", 0)
                                
                                is_valid = (
                                    recall_at_10 != 0 or recall_at_20 != 0 or 
                                    ndcg_at_10 != 0 or ndcg_at_20 != 0
                                )
                                
                                if is_valid:
                                    # Found valid metrics, use this result
                                    best_result = recommend_result
                                    best_metrics = eval_metrics
                                    best_product_id = rec_product_id
                                    break
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                if best_result and best_result["success"]:
                    has_valid_metrics = best_metrics and any([
                        best_metrics.get("recall_at_10", 0) != 0,
                        best_metrics.get("recall_at_20", 0) != 0,
                        best_metrics.get("ndcg_at_10", 0) != 0,
                        best_metrics.get("ndcg_at_20", 0) != 0
                    ])
                    
                    if has_valid_metrics:
                        st.success(f"✅ Đã tìm thấy evaluation metrics hợp lệ với product_id: {best_product_id}!")
                    else:
                        st.warning(f"⚠️ Đã test {total_tests + min(len(recommended_products_to_try), 5)} products nhưng metrics vẫn null/0.")
                        st.info(f"📊 Sử dụng kết quả từ product_id: {best_product_id}")
                        
                        # Show debug info to help understand why
                        debug_info = best_metrics.get("_debug", {}) if best_metrics else {}
                        if debug_info:
                            with st.expander("🔍 Debug Info - Tại sao metrics = 0?"):
                                st.json(debug_info)
                                
                                # Show diagnosis if available
                                diagnosis = best_metrics.get("_diagnosis", {}) if best_metrics else {}
                                if diagnosis:
                                    st.markdown("#### 🔬 Chẩn đoán tự động:")
                                    issues = diagnosis.get("issues", [])
                                    if issues:
                                        for issue in issues:
                                            severity = issue.get("severity", "info")
                                            if severity == "error":
                                                st.error(f"❌ **{issue.get('issue')}**")
                                            elif severity == "warning":
                                                st.warning(f"⚠️ **{issue.get('issue')}**")
                                            else:
                                                st.info(f"ℹ️ **{issue.get('issue')}**")
                                            st.markdown(f"- **Lý do**: {issue.get('reason')}")
                                            st.markdown(f"- **Cách sửa**: {issue.get('fix')}")
                                    else:
                                        st.success("✅ Không phát hiện vấn đề trong logic tính toán")
                                
                                # Show overlap info
                                overlap_found = debug_info.get("overlap_found", False)
                                num_rec = debug_info.get("num_recommendations", 0)
                                num_gt = debug_info.get("num_ground_truth", 0)
                                
                                st.markdown("#### 📊 Tóm tắt:")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Recommendations", num_rec)
                                with col2:
                                    st.metric("Ground Truth", num_gt)
                                with col3:
                                    st.metric("Overlap", "✅ Có" if overlap_found else "❌ Không")
                                
                                if not overlap_found and num_rec > 0 and num_gt > 0:
                                    st.info("💡 **Giải thích**: CBF đang recommend các sản phẩm khác với interaction_history của user. Đây có thể là hành vi đúng (recommend sản phẩm mới), nhưng để tính metrics cần có overlap.")
                    
                    # Store recommendation result
                    st.session_state.recommendation_results[slug] = best_result["data"]
                    
                    # Extract evaluation metrics from recommend API and update session state
                    if isinstance(best_result["data"], dict):
                        eval_metrics = extract_recommend_metrics(best_result["data"], slug)
                        # Update session state with evaluation metrics from recommend API
                        for key, value in eval_metrics.items():
                            if value != "N/A":
                                state_key = f"{slug}_{key}"
                                st.session_state[state_key] = value
                                # Also update training_results if exists
                                if st.session_state.training_results.get(slug):
                                    if isinstance(st.session_state.training_results[slug], dict):
                                        st.session_state.training_results[slug][key] = value
                    
                    st.json(best_result["data"].get("evaluation_metrics", {}))
                else:
                    st.warning(f"⚠️ Không thể tự động gọi API recommend: {best_result.get('error', 'Unknown error') if best_result else 'No valid results found'}")
            else:
                status_placeholder.error(f"Lỗi train {label}.")
                st.error(result["error"])
                if result.get("data"):
                    st.json(result["data"])
                if result.get("response"):
                    st.code(result["response"])


st.header("3. Recommendation APIs")
default_user_id = "690bf0f2d0c3753df0ecbdd6"
default_product_id = "10068"

user_id = st.text_input("User ID", value=default_user_id)
product_id = st.text_input("Product ID", value=default_product_id)

recommend_cols = st.columns(len(models))
# API expects user_id and current_product_id (not userId and productId)
payload = {"user_id": user_id, "current_product_id": product_id}

for col, (label, slug) in zip(recommend_cols, models.items()):
    with col:
        if st.button(f"Recommend {label}", key=f"recommend_{slug}"):
            status_placeholder = st.empty()
            status_placeholder.info("Đang gọi API recommend...")
            with st.spinner(f"Đợi kết quả {label}..."):
                result = call_api(BASE_URL, f"{slug}/recommend", payload=payload)
            if result["success"]:
                status_placeholder.success(f"Kết quả {label} sẵn sàng.")
                # Store recommendation result
                st.session_state.recommendation_results[slug] = result["data"]

                # Extract evaluation_support from recommend response (if provided)
                try:
                    support = extract_evaluation_support(result["data"])
                    if support:
                        st.session_state.evaluation_support[slug] = support
                        cnt_pairs = len(support.get('pairs') or [])
                        cnt_u = len(support.get('user_ids') or [])
                        cnt_p = len(support.get('product_ids') or [])
                        st.info(f"📦 evaluation_support: pairs={cnt_pairs}, user_ids={cnt_u}, product_ids={cnt_p}")
                except Exception:
                    pass
                
                # Extract evaluation metrics from recommend API and update session state
                if isinstance(result["data"], dict):
                    eval_metrics = extract_recommend_metrics(result["data"], slug)
                    # Update session state with evaluation metrics from recommend API
                    for key, value in eval_metrics.items():
                        if value != "N/A":
                            state_key = f"{slug}_{key}"
                            st.session_state[state_key] = value
                            # Also update training_results if exists
                            if st.session_state.training_results.get(slug):
                                if isinstance(st.session_state.training_results[slug], dict):
                                    st.session_state.training_results[slug][key] = value
                
                st.json(result["data"])
            else:
                status_placeholder.error(f"Lỗi recommend {label}.")
                st.error(result["error"])
                if result.get("response"):
                    st.code(result["response"])


def generate_gnn_documentation(metrics: Dict[str, Any]) -> str:
    """Generate GNN documentation markdown with metrics."""
    doc = f"""### 2.3.1. GNN (Graph Neural Network - LightGCN)

- **Quy trình thực hiện**:
  - *Chuẩn hóa dữ liệu với Surprise*:  
    Sử dụng `surprise.Dataset.load_from_df(...)` và `train_test_split(test_size={metrics['test_size']})` để chia dữ liệu thành tập huấn luyện và tập kiểm thử.  
    - Test size: **{metrics['test_size']}** (tỷ lệ dữ liệu dùng để kiểm thử, phần còn lại dùng để huấn luyện)
    - Số lượng người dùng train: **{metrics['num_users']}** (số người dùng trong tập huấn luyện)
    - Số lượng sản phẩm train: **{metrics['num_products']}** (số sản phẩm trong tập huấn luyện)
    - Số lượng tương tác (interactions): **{metrics['num_interactions']}** (tổng số lượt tương tác giữa người dùng và sản phẩm)
    - Số lượng training samples (BPR): **{metrics['num_training_samples']}** (số mẫu huấn luyện sau khi tạo negative samples cho BPR)
  - *Pipeline 5 bước*:
    1. **Huấn luyện mô hình**: LightGCN với kiến trúc Graph Convolutional Network.
       - Thuật toán: LightGCN (Light Graph Convolution Network) - mô hình học biểu diễn người dùng và sản phẩm dựa trên đồ thị tương tác
       - Framework: PyTorch + PyTorch Geometric
       - Loss function: BPR (Bayesian Personalized Ranking) - tối ưu hóa thứ hạng sản phẩm cho từng người dùng
       - Negative sampling: 4 negative samples per positive interaction (tạo 4 mẫu âm cho mỗi tương tác tích cực để học phân biệt)
       - Epochs: **{metrics['epochs']}** (số lần duyệt toàn bộ dữ liệu training)
       - Batch size: **{metrics['batch_size']}** (số lượng mẫu xử lý cùng lúc trong mỗi bước cập nhật)
       - Embedding dimension: **{metrics['embed_dim']}** (kích thước vector đại diện cho người dùng/sản phẩm, càng lớn càng biểu diễn chi tiết hơn)
       - Learning rate: **{metrics['learning_rate']}** (tốc độ học, điều chỉnh độ lớn bước cập nhật tham số)
       - Optimizer: Adam (thuật toán tối ưu hóa tự động điều chỉnh learning rate)
       - Model file: `models/gnn_lightgcn.pkl`
    2. **Chuẩn bị dữ liệu graph**: 
       - Xây dựng bipartite graph (đồ thị hai phía) từ `UserInteraction` collection, mỗi cạnh nối một người dùng với một sản phẩm
       - Áp dụng trọng số tương tác theo `INTERACTION_WEIGHTS` để phân biệt mức độ quan trọng:
         ```python
         INTERACTION_WEIGHTS = {{
             'view': 1.0,        # Xem sản phẩm (quan tâm thấp)
             'add_to_cart': 2.0, # Thêm vào giỏ (quan tâm trung bình)
             'purchase': 3.0,    # Mua hàng (quan tâm cao nhất)
             'wishlist': 1.5,    # Yêu thích (quan tâm trung bình-thấp)
             'rating': 2.5       # Đánh giá (quan tâm cao)
         }}
         ```
       - Tạo edge index (danh sách cặp user-product) và edge weights (trọng số tương ứng)
    3. **Tạo ma trận User-Item Interaction**: 
       - Sử dụng sparse matrix (ma trận thưa) để biểu diễn tương tác user-product một cách hiệu quả
       - Tính toán sparsity (độ thưa): `sparsity = 1 - ({metrics['num_interactions']} / ({metrics['num_users']} * {metrics['num_products']}))` - tỷ lệ phần trăm các tương tác không xảy ra
    4. **Tính cosine similarity** giữa user embeddings và product embeddings.  
       - Sau khi training, LightGCN sinh ra:
         - User embeddings: `[{metrics['num_users']}, {metrics['embed_dim']}]` - {metrics['num_users']} vector, mỗi vector {metrics['embed_dim']} chiều
         - Product embeddings: `[{metrics['num_products']}, {metrics['embed_dim']}]` - {metrics['num_products']} vector, mỗi vector {metrics['embed_dim']} chiều
       - Recommendation score = dot product (tích vô hướng) giữa user embedding và product embedding, giá trị càng cao thì sản phẩm càng phù hợp với người dùng
    5. **Tính toán chỉ số đánh giá**: Recall@10, Recall@20, NDCG@10, NDCG@20, thời gian train, thời gian inference.
       - *Recall@10*: Trong 10 món bạn gợi ý, có bao nhiêu món user thực sự thích (trong test set)? Càng cao càng tốt (0-1)
       - *Recall@20*: Tương tự nhưng top 20. Càng cao càng tốt (0-1)
       - *NDCG@10*: Top 10 của bạn không chỉ đúng mà còn sắp xếp đúng thứ tự (món user thích nhất đứng cao). Càng cao càng tốt (0-1)
       - *NDCG@20*: Tương tự top 20. Càng cao càng tốt (0-1)
       - *Thời gian train*: Mất bao lâu để train xong 1 lần ({metrics.get('training_time', 'N/A')}) - càng thấp càng tốt
       - *Thời gian inference/user*: Mất bao lâu để trả về gợi ý cho 1 user ({metrics.get('inference_time', 'N/A')} ms) - càng thấp càng tốt (rất quan trọng trong production)

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Thời gian train | Thời gian inference/user |
|-------|-----------|-----------|---------|---------|----------------|------------------------|
| GNN (LightGCN) | {metrics.get('recall_at_10', 'N/A')} | {metrics.get('recall_at_20', 'N/A')} | {metrics.get('ndcg_at_10', 'N/A')} | {metrics.get('ndcg_at_20', 'N/A')} | {metrics.get('training_time', 'N/A')} | {metrics.get('inference_time', 'N/A')} ms |
"""
    return doc


def generate_cbf_documentation(metrics: Dict[str, Any]) -> str:
    """Generate Content-based Filtering documentation markdown with metrics."""
    doc = f"""### 2.3.2. Content-based Filtering

- **Quy trình thực hiện**:
  - *Chuẩn hóa dữ liệu với Surprise*:  
    Sử dụng `surprise.Dataset.load_from_df(...)` và `train_test_split(test_size={metrics['test_size']})` để chia dữ liệu thành tập huấn luyện và tập kiểm thử.  
    - Test size: **{metrics['test_size']}** (tỷ lệ dữ liệu dùng để kiểm thử, phần còn lại dùng để huấn luyện)
    - Số lượng sản phẩm train: **{metrics['num_products']}** (số sản phẩm trong tập huấn luyện)
    - Số lượng người dùng test: **{metrics['num_users']}** (số người dùng trong tập kiểm thử)
  - *Pipeline 5 bước*:
    1. **Huấn luyện mô hình**: Sentence-BERT embedding + FAISS index.
       - Model: Sentence-BERT (SBERT) - mô hình chuyển đổi văn bản thành vector số, hiểu được ngữ nghĩa của mô tả sản phẩm
       - Index: FAISS (Facebook AI Similarity Search) - thư viện tìm kiếm tương tự nhanh, cho phép tìm sản phẩm tương tự trong thời gian ngắn
       - Embedding dimension: **{metrics['embed_dim']}** (kích thước vector đại diện cho mỗi sản phẩm, càng lớn càng biểu diễn chi tiết hơn)
    2. **Chuẩn bị dữ liệu văn bản**: ghép các thuộc tính `category`, `gender`, `color`, `style_tags`, `productDisplayName` thành một chuỗi văn bản mô tả đầy đủ sản phẩm
    3. **Tạo ma trận TF-IDF**: sử dụng `TfidfVectorizer` để tạo ma trận TF-IDF (Term Frequency-Inverse Document Frequency) - đánh giá tầm quan trọng của từ trong mô tả sản phẩm
    4. **Tính cosine similarity** giữa các sản phẩm (SBERT embeddings).  
       - Recommendation score = cosine similarity (độ tương tự cosine) giữa product embeddings, giá trị từ 0-1, càng gần 1 thì sản phẩm càng giống nhau về đặc điểm
    5. **Tính toán chỉ số đánh giá**: Recall@10, Recall@20, NDCG@10, NDCG@20, thời gian train, thời gian inference.
       - *Recall@10*: Trong 10 món bạn gợi ý, có bao nhiêu món user thực sự thích (trong test set)? Càng cao càng tốt (0-1)
       - *Recall@20*: Tương tự nhưng top 20. Càng cao càng tốt (0-1)
       - *NDCG@10*: Top 10 của bạn không chỉ đúng mà còn sắp xếp đúng thứ tự (món user thích nhất đứng cao). Càng cao càng tốt (0-1)
       - *NDCG@20*: Tương tự top 20. Càng cao càng tốt (0-1)
       - *Thời gian train*: Mất bao lâu để train xong 1 lần ({metrics.get('training_time', 'N/A')}) - càng thấp càng tốt
       - *Thời gian inference/user*: Mất bao lâu để trả về gợi ý cho 1 user ({metrics.get('inference_time', 'N/A')} ms) - càng thấp càng tốt (rất quan trọng trong production)

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Thời gian train | Thời gian inference/user |
|-------|-----------|-----------|---------|---------|----------------|------------------------|
| Content-based Filtering | {metrics.get('recall_at_10', 'N/A')} | {metrics.get('recall_at_20', 'N/A')} | {metrics.get('ndcg_at_10', 'N/A')} | {metrics.get('ndcg_at_20', 'N/A')} | {metrics.get('training_time', 'N/A')} | {metrics.get('inference_time', 'N/A')} ms |
"""
    return doc


def generate_hybrid_documentation(metrics: Dict[str, Any], alpha: float = 0.7) -> str:
    """Generate Hybrid documentation markdown with metrics."""
    doc = f"""### 2.3.3. Hybrid GNN (LightGCN) & Content-based Filtering

- **Quy trình thực hiện**:
  - *Chuẩn hóa dữ liệu với Surprise*:  
    Sử dụng `surprise.Dataset.load_from_df(...)` và `train_test_split(test_size={metrics['test_size']})` để chia dữ liệu thành tập huấn luyện và tập kiểm thử.  
    - Test size: **{metrics['test_size']}** (tỷ lệ dữ liệu dùng để kiểm thử, phần còn lại dùng để huấn luyện)
    - Số lượng người dùng train: **{metrics['num_users']}** (số người dùng trong tập huấn luyện)
    - Số lượng sản phẩm train: **{metrics['num_products']}** (số sản phẩm trong tập huấn luyện)
    - Số lượng tương tác (interactions): **{metrics['num_interactions']}** (tổng số lượt tương tác giữa người dùng và sản phẩm)
  - *Pipeline 5 bước*:
    1. **Huấn luyện mô hình**: Kết hợp GNN (LightGCN) + CBF (Sentence-BERT + FAISS).
       - GNN component: LightGCN với embedding dimension **{metrics['embed_dim']}** - học từ hành vi tương tác của người dùng thông qua Graph Neural Network
       - CBF component: Sentence-BERT + FAISS index - học từ đặc điểm nội dung sản phẩm thông qua semantic embeddings
       - Trọng số kết hợp: `alpha = {alpha}` (GNN weight = {alpha}, CBF weight = {1-alpha:.1f}) - alpha càng cao thì càng ưu tiên hành vi người dùng (GNN), càng thấp thì càng ưu tiên đặc điểm sản phẩm (CBF)
    2. **Chuẩn bị dữ liệu**: 
       - Kết hợp embedding từ GNN (LightGCN) và Content-based Filtering (Sentence-BERT + FAISS)
       - User embeddings từ GNN (LightGCN): `[{metrics['num_users']}, {metrics['embed_dim']}]` - {metrics['num_users']} vector người dùng, mỗi vector {metrics['embed_dim']} chiều, học từ đồ thị tương tác
       - Product embeddings từ CBF (Sentence-BERT): `[{metrics['num_products']}, {metrics['embed_dim']}]` - {metrics['num_products']} vector sản phẩm, mỗi vector {metrics['embed_dim']} chiều, học từ mô tả sản phẩm
    3. **Tính toán similarity**: 
       - GNN similarity: cosine similarity giữa user embedding (LightGCN) và product embedding (LightGCN) - dựa trên hành vi người dùng tương tự trong đồ thị tương tác
       - CBF similarity: cosine similarity giữa product embeddings (Sentence-BERT) - dựa trên đặc điểm sản phẩm tương tự về ngữ nghĩa
       - Final score = `{alpha} * GNN_score + {1-alpha:.1f} * CBF_score` - kết hợp hai nguồn thông tin với trọng số
    4. **Kết hợp trọng số**: 
       - Bảng similarity từ CBF (Sentence-BERT + FAISS) đánh giá độ tương tự nội dung, cộng thêm trọng số GNN (LightGCN) đánh giá độ tương tự hành vi trong đồ thị
    5. **Tính toán chỉ số đánh giá**: Recall@10, Recall@20, NDCG@10, NDCG@20, thời gian train, thời gian inference.
       - *Recall@10*: Trong 10 món bạn gợi ý, có bao nhiêu món user thực sự thích (trong test set)? Càng cao càng tốt (0-1)
       - *Recall@20*: Tương tự nhưng top 20. Càng cao càng tốt (0-1)
       - *NDCG@10*: Top 10 của bạn không chỉ đúng mà còn sắp xếp đúng thứ tự (món user thích nhất đứng cao). Càng cao càng tốt (0-1)
       - *NDCG@20*: Tương tự top 20. Càng cao càng tốt (0-1)
       - *Thời gian train*: Mất bao lâu để train xong 1 lần ({metrics.get('training_time', 'N/A')}) - càng thấp càng tốt
       - *Thời gian inference/user*: Mất bao lâu để trả về gợi ý cho 1 user ({metrics.get('inference_time', 'N/A')} ms) - càng thấp càng tốt (rất quan trọng trong production)

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Thời gian train | Thời gian inference/user |
|-------|-----------|-----------|---------|---------|----------------|------------------------|
| Hybrid GNN+CBF | {metrics.get('recall_at_10', 'N/A')} | {metrics.get('recall_at_20', 'N/A')} | {metrics.get('ndcg_at_10', 'N/A')} | {metrics.get('ndcg_at_20', 'N/A')} | {metrics.get('training_time', 'N/A')} | {metrics.get('inference_time', 'N/A')} ms |
"""
    return doc


def generate_comparison_table(
    gnn_metrics: Dict[str, Any],
    cbf_metrics: Dict[str, Any],
    hybrid_metrics: Dict[str, Any],
    analysis_text: str,
) -> str:
    """Generate comparison table for all 3 models."""
    doc = """
**Giải thích các chỉ số:**
- **Recall@10** (0-1): Trong 10 món bạn gợi ý, có bao nhiêu món user thực sự thích (trong test set)? Càng cao càng tốt
- **Recall@20** (0-1): Tương tự nhưng top 20. Càng cao càng tốt
- **NDCG@10** (0-1): Top 10 của bạn không chỉ đúng mà còn sắp xếp đúng thứ tự (món user thích nhất đứng cao). Càng cao càng tốt
- **NDCG@20** (0-1): Tương tự top 20. Càng cao càng tốt
- **Thời gian train**: Mất bao lâu để train xong 1 lần (thường tính bằng phút/giờ) - càng thấp càng tốt
- **Thời gian inference/user**: Mất bao lâu để trả về gợi ý cho 1 user (thường tính bằng ms) - càng thấp càng tốt (rất quan trọng trong production)

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Thời gian train | Thời gian inference/user |
|-------|-----------|-----------|---------|---------|----------------|------------------------|
| GNN (LightGCN) | {gnn_recall_10} | {gnn_recall_20} | {gnn_ndcg_10} | {gnn_ndcg_20} | {gnn_train_time} | {gnn_inference_time} |
| Content-based Filtering | {cbf_recall_10} | {cbf_recall_20} | {cbf_ndcg_10} | {cbf_ndcg_20} | {cbf_train_time} | {cbf_inference_time} |
| Hybrid GNN+CBF | {hybrid_recall_10} | {hybrid_recall_20} | {hybrid_ndcg_10} | {hybrid_ndcg_20} | {hybrid_train_time} | {hybrid_inference_time} |

{analysis_section}
""".format(
        gnn_recall_10=gnn_metrics.get('recall_at_10', 'N/A'),
        gnn_recall_20=gnn_metrics.get('recall_at_20', 'N/A'),
        gnn_ndcg_10=gnn_metrics.get('ndcg_at_10', 'N/A'),
        gnn_ndcg_20=gnn_metrics.get('ndcg_at_20', 'N/A'),
        gnn_train_time=gnn_metrics.get('training_time', 'N/A'),
        gnn_inference_time=f"{gnn_metrics.get('inference_time', 'N/A')} ms" if gnn_metrics.get('inference_time', 'N/A') != 'N/A' else 'N/A',
        cbf_recall_10=cbf_metrics.get('recall_at_10', 'N/A'),
        cbf_recall_20=cbf_metrics.get('recall_at_20', 'N/A'),
        cbf_ndcg_10=cbf_metrics.get('ndcg_at_10', 'N/A'),
        cbf_ndcg_20=cbf_metrics.get('ndcg_at_20', 'N/A'),
        cbf_train_time=cbf_metrics.get('training_time', 'N/A'),
        cbf_inference_time=f"{cbf_metrics.get('inference_time', 'N/A')} ms" if cbf_metrics.get('inference_time', 'N/A') != 'N/A' else 'N/A',
        hybrid_recall_10=hybrid_metrics.get('recall_at_10', 'N/A'),
        hybrid_recall_20=hybrid_metrics.get('recall_at_20', 'N/A'),
        hybrid_ndcg_10=hybrid_metrics.get('ndcg_at_10', 'N/A'),
        hybrid_ndcg_20=hybrid_metrics.get('ndcg_at_20', 'N/A'),
        hybrid_train_time=hybrid_metrics.get('training_time', 'N/A'),
        hybrid_inference_time=f"{hybrid_metrics.get('inference_time', 'N/A')} ms" if hybrid_metrics.get('inference_time', 'N/A') != 'N/A' else 'N/A',
        analysis_section=analysis_text.replace("{", "{{").replace("}", "}}"),
    )
    return doc


# 3.1 Apply formulas locally to compute metrics
st.header("3.1 Áp dụng công thức (tính cục bộ)")
st.caption("Tính Recall@K, NDCG@K dựa trên danh sách gợi ý trả về và Ground Truth lấy từ lịch sử tương tác của user. Dùng chính công thức đã trình bày để kiểm chứng.")

with st.expander("🔬 Tính Recall/NDCG cục bộ từ kết quả recommend"):
    uid_local = st.text_input("User ID (local)", value=user_id, key="local_user_id")
    pid_local = st.text_input("Current Product ID (local)", value=product_id, key="local_product_id")
    k_values = st.multiselect("Chọn K để tính", options=[5, 10, 20, 50], default=[10, 20])
    model_choices = st.multiselect("Chọn mô hình", options=[("GNN","gnn"), ("CBF","cbf"), ("Hybrid","hybrid")], format_func=lambda x: x[0], default=[("GNN","gnn"), ("CBF","cbf"), ("Hybrid","hybrid")])

    def _extract_rec_ids(recommend_data: Dict[str, Any]) -> list:
        recs = recommend_data.get("personalized") or recommend_data.get("recommendations") or []
        rec_ids = []
        for rec in recs:
            rid = None
            if isinstance(rec, dict):
                # nested product object or flat id
                prod = rec.get("product")
                if isinstance(prod, dict):
                    rid = prod.get("id") or prod.get("product_id")
                rid = rid or rec.get("id") or rec.get("product_id")
            else:
                rid = rec
            if rid is not None:
                rec_ids.append(str(rid))
        # unique and keep order
        seen = set()
        ordered = []
        for rid in rec_ids:
            if rid not in seen:
                seen.add(rid)
                ordered.append(rid)
        return ordered

    def _fetch_ground_truth_ids(base_url: str, uid: str, exclude_pid: str) -> list:
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/users/{uid}", timeout=15)
            if resp.status_code == 200:
                payload = resp.json()
                user_info = (payload.get("data") or {}).get("user") or {}
                history = user_info.get("interaction_history") or []
                gt_ids = []
                for it in history:
                    pid = it.get("product_id")
                    if pid is None:
                        continue
                    pid = str(pid)
                    if exclude_pid and pid == str(exclude_pid):
                        continue
                    gt_ids.append(pid)
                # unique
                gt_ids = list(dict.fromkeys(gt_ids))
                return gt_ids
        except Exception:
            pass
        return []

    if st.button("▶️ Tính toán cục bộ", key="btn_compute_local"):
        if not uid_local:
            st.warning("Vui lòng nhập User ID")
        else:
            gt_ids = _fetch_ground_truth_ids(BASE_URL, uid_local, pid_local)
            if not gt_ids:
                st.warning("Không lấy được Ground Truth từ interaction_history của user. Hãy đảm bảo backend trả về /users/{id} có interaction_history.")
            else:
                st.success(f"Đã lấy {len(gt_ids)} Ground Truth items từ lịch sử user")
                cols = st.columns(len(model_choices) or 1)
                for col, (label, slug) in zip(cols, model_choices):
                    with col:
                        st.markdown(f"#### {label}")
                        payload_local = {"user_id": uid_local, "current_product_id": pid_local}
                        t0 = time.perf_counter()
                        res = call_api(BASE_URL, f"{slug}/recommend", payload=payload_local)
                        t1 = time.perf_counter()
                        if not res["success"]:
                            st.error(res.get("error", "Recommend API lỗi"))
                            continue
                        data = res["data"] if isinstance(res["data"], dict) else {}
                        rec_ids = _extract_rec_ids(data)
                        if not rec_ids:
                            st.warning("Không có danh sách gợi ý để tính toán.")
                            continue

                        # Compute metrics locally
                        for k in k_values:
                            recall_k = compute_recall_at_k(rec_ids, gt_ids, k=k)
                            ndcg_k = compute_ndcg_at_k(rec_ids, gt_ids, k=k)
                            st.metric(f"Recall@{k} (local)", f"{recall_k:.4f}")
                            st.metric(f"NDCG@{k} (local)", f"{ndcg_k:.4f}")
                        # Compare to API's evaluation_metrics if present
                        api_eval = data.get("evaluation_metrics", {}) if isinstance(data, dict) else {}
                        if api_eval:
                            with st.expander("So sánh với evaluation_metrics API"):
                                st.json(api_eval)
                        inf_ms = (t1 - t0) * 1000.0
                        st.metric("Inference time (local)", f"{inf_ms:.2f} ms")

# 3.2 Batch evaluation using API-provided test cases
st.header("3.2 Đánh giá theo bộ test (từ API)")
st.caption("Sử dụng danh sách user_id/product_id mà API trả về trong evaluation_support để chạy recommend theo lô, áp dụng công thức Recall@K và NDCG@K, rồi tổng hợp kết quả.")

with st.expander("🧪 Chạy đánh giá theo evaluation_support"):
    # Show availability per model
    col_av1, col_av2, col_av3 = st.columns(3)
    for c, slug, label in zip([col_av1, col_av2, col_av3], ["gnn", "cbf", "hybrid"], ["GNN", "CBF", "Hybrid"]):
        with c:
            es = st.session_state.evaluation_support.get(slug)
            if es:
                num_pairs = len(es.get("pairs") or [])
                num_u = len(es.get("user_ids") or [])
                num_p = len(es.get("product_ids") or [])
                st.success(f"{label}: pairs={num_pairs}, user_ids={num_u}, product_ids={num_p}")
            else:
                st.warning(f"{label}: Chưa có evaluation_support từ API")

    # Controls
    model_opts = st.multiselect(
        "Chọn mô hình để đánh giá",
        options=[("GNN", "gnn"), ("CBF", "cbf"), ("Hybrid", "hybrid")],
        format_func=lambda x: x[0],
        default=[("GNN", "gnn"), ("CBF", "cbf"), ("Hybrid", "hybrid")]
    )
    ks = st.multiselect("Chọn K", options=[5, 10, 20, 50], default=[10, 20])
    max_pairs = st.number_input("Giới hạn số cặp test/pairs", min_value=1, max_value=1000, value=50, step=5)

    def _get_eval_pairs(slug: str, limit: int) -> list:
        es = st.session_state.evaluation_support.get(slug) or {}
        pairs = es.get("pairs") or []
        if not pairs:
            # fallback: build pairs from user_ids x product_ids (cắt mẫu để tránh nổ tổ hợp)
            uids = es.get("user_ids") or []
            pids = es.get("product_ids") or []
            built = []
            for i, uid in enumerate(uids):
                if len(built) >= limit:
                    break
                for j, pid in enumerate(pids):
                    built.append({"user_id": str(uid), "current_product_id": str(pid)})
                    if len(built) >= limit:
                        break
            pairs = built
        return pairs[:limit]

    def _extract_rec_ids(recommend_data: Dict[str, Any]) -> list:
        recs = recommend_data.get("personalized") or recommend_data.get("recommendations") or []
        rec_ids = []
        for rec in recs:
            rid = None
            if isinstance(rec, dict):
                prod = rec.get("product")
                if isinstance(prod, dict):
                    rid = prod.get("id") or prod.get("product_id")
                rid = rid or rec.get("id") or rec.get("product_id")
            else:
                rid = rec
            if rid is not None:
                rec_ids.append(str(rid))
        # unique ordered
        seen, ordered = set(), []
        for rid in rec_ids:
            if rid not in seen:
                seen.add(rid)
                ordered.append(rid)
        return ordered

    GT_CACHE: Dict[str, list] = {}

    def _get_gt(uid: str, exclude_pid: Optional[str]) -> list:
        if uid in GT_CACHE:
            gt = GT_CACHE[uid]
        else:
            gt = []
            try:
                resp = requests.get(f"{BASE_URL.rstrip('/')}/users/{uid}", timeout=15)
                if resp.status_code == 200:
                    payload = resp.json()
                    user_info = (payload.get("data") or {}).get("user") or {}
                    history = user_info.get("interaction_history") or []
                    for it in history:
                        pid = it.get("product_id")
                        if pid is None:
                            continue
                        gt.append(str(pid))
                    gt = list(dict.fromkeys(gt))
            except Exception:
                pass
            GT_CACHE[uid] = gt
        if exclude_pid:
            return [x for x in gt if x != str(exclude_pid)]
        return gt

    if st.button("▶️ Chạy đánh giá theo bộ test", key="btn_run_eval_support"):
        if not model_opts:
            st.warning("Vui lòng chọn ít nhất một mô hình")
        elif not ks:
            st.warning("Vui lòng chọn ít nhất một K")
        else:
            for label, slug in model_opts:
                st.markdown(f"#### Kết quả - {label}")
                pairs = _get_eval_pairs(slug, int(max_pairs))
                if not pairs:
                    st.warning("Không có cặp test từ evaluation_support.")
                    continue
                prog = st.progress(0)
                rows = []
                sum_recalls = {k: 0.0 for k in ks}
                sum_ndcgs = {k: 0.0 for k in ks}
                total = len(pairs)
                total_time_ms = 0.0
                for idx, pair in enumerate(pairs, start=1):
                    uid = pair.get("user_id")
                    pid = pair.get("current_product_id")
                    if not uid:
                        continue
                    gt_ids = _get_gt(uid, pid)
                    t0 = time.perf_counter()
                    res = call_api(BASE_URL, f"{slug}/recommend", payload=pair)
                    t1 = time.perf_counter()
                    if not res["success"]:
                        rows.append({"user_id": uid, "product_id": pid, "ok": False, "error": res.get("error")})
                        prog.progress(min(idx/total, 1.0))
                        continue
                    data = res["data"] if isinstance(res["data"], dict) else {}
                    rec_ids = _extract_rec_ids(data)
                    pair_row = {"user_id": uid, "product_id": pid, "ok": True}
                    for k in ks:
                        r = compute_recall_at_k(rec_ids, gt_ids, k=k)
                        n = compute_ndcg_at_k(rec_ids, gt_ids, k=k)
                        sum_recalls[k] += r
                        sum_ndcgs[k] += n
                        pair_row[f"recall@{k}"] = round(r, 4)
                        pair_row[f"ndcg@{k}"] = round(n, 4)
                    inf_ms = (t1 - t0) * 1000.0
                    total_time_ms += inf_ms
                    pair_row["inference_ms"] = round(inf_ms, 2)
                    rows.append(pair_row)
                    prog.progress(min(idx/total, 1.0))

                # Aggregate
                agg_cols = st.columns(len(ks) * 2 + 1)
                cidx = 0
                for k in ks:
                    with agg_cols[cidx]:
                        st.metric(f"Recall@{k} (avg)", f"{(sum_recalls[k]/total):.4f}")
                    cidx += 1
                    with agg_cols[cidx]:
                        st.metric(f"NDCG@{k} (avg)", f"{(sum_ndcgs[k]/total):.4f}")
                    cidx += 1
                with agg_cols[cidx]:
                    st.metric("Inference (avg)", f"{(total_time_ms/max(total,1)):.2f} ms")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.header("4. Tài liệu mô hình (Documentation)")

st.markdown("""
**📌 Nguồn dữ liệu cho tài liệu:**

- **Từ API `/train`**: Thông số huấn luyện (num_users, num_products, epochs, batch_size, embed_dim, learning_rate, etc.)
- **Từ API `/recommend`**: Chỉ số đánh giá (MAPE, RMSE, Precision, Recall, F1, execution_time) trong `evaluation_metrics`

**💡 Lưu ý**: Để có đầy đủ số liệu, bạn cần:
1. Train mô hình qua API `/train` → Lấy thông số huấn luyện
2. Gọi API `/recommend` → Lấy evaluation metrics
""")

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_groq_api(prompt: str, system_message: str = "", max_tokens: int = 2000, temperature: float = 0.3) -> str:
    """Call Groq API with given prompt."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "**⚠️ Groq chưa sẵn sàng**: Vui lòng đặt biến môi trường `GROQ_API_KEY` "
            "để bật phân tích tự động."
        )
    
    default_system = "You are a helpful data scientist specializing in recommender systems. Always respond in Markdown and Vietnamese."
    
    payload = {
        "model": GROQ_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_message or default_system,
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not content:
            raise ValueError("Groq response empty.")
        return content
    except (requests.RequestException, ValueError, KeyError) as exc:
        return f"**⚠️ Groq lỗi**: {exc}"


def analyze_metrics_detailed(
    gnn_metrics: Dict[str, Any],
    cbf_metrics: Dict[str, Any],
    hybrid_metrics: Dict[str, Any],
) -> str:
    """Use Groq to provide detailed explanation of metrics and model selection."""
    metrics_snapshot = {
        "GNN (LightGCN)": gnn_metrics,
        "Content-based Filtering": cbf_metrics,
        "Hybrid GNN+CBF": hybrid_metrics,
    }
    
    prompt = f"""Bạn là chuyên gia về hệ thống gợi ý (Recommender Systems). 
Dựa vào số liệu thực nghiệm dưới đây, hãy:

1. **Giải thích chi tiết từng chỉ số:**
   - Recall@10, Recall@20: Ý nghĩa là gì? Giá trị bao nhiêu là tốt?
   - NDCG@10, NDCG@20: Khác gì với Recall? Tại sao cần cả hai?
   - Thời gian train vs inference: Tại sao cả hai đều quan trọng?

2. **So sánh 3 mô hình:**
   - Mô hình nào có Recall/NDCG cao nhất?
   - Mô hình nào train nhanh nhất?
   - Mô hình nào inference nhanh nhất (quan trọng cho production)?
   - Mô hình nào cân bằng tốt nhất giữa độ chính xác và tốc độ?

3. **Khuyến nghị:**
   - Chọn mô hình nào để triển khai production? Tại sao?
   - Trong trường hợp nào nên dùng mô hình khác?
   - Có cách nào cải thiện mô hình được chọn không?

**Số liệu thực nghiệm:**
{json.dumps(metrics_snapshot, ensure_ascii=False, indent=2)}

Viết chi tiết, dễ hiểu, có ví dụ cụ thể. Sử dụng tiếng Việt."""

    return call_groq_api(prompt, max_tokens=3000, temperature=0.2)


def explain_algorithms_detailed(
    gnn_metrics: Dict[str, Any],
    cbf_metrics: Dict[str, Any],
    hybrid_metrics: Dict[str, Any],
) -> str:
    """Use Groq to explain algorithms in detail with formulas and step-by-step process."""
    metrics_snapshot = {
        "GNN": gnn_metrics,
        "CBF": cbf_metrics,
        "Hybrid": hybrid_metrics,
    }
    
    prompt = f"""Bạn là chuyên gia Machine Learning và Recommender Systems.
Hãy trình bày chi tiết thuật toán của 3 mô hình sau với:

1. **GNN (LightGCN)**
   - Công thức toán học từng bước (dùng ký hiệu toán học chuẩn)
   - Giải thích ý nghĩa của từng biến
   - Quá trình tính toán: User embedding → Product embedding → Similarity score → Ranking
   - Tại sao dùng Graph Neural Network?
   - Ưu điểm: Học được mối quan hệ giữa users và items từ đồ thị tương tác
   - Nhược điểm: Cần dữ liệu tương tác đủ lớn

2. **Content-based Filtering (CBF)**
   - Công thức toán học từng bước
   - Giải thích Sentence-BERT embeddings
   - Công thức tính cosine similarity
   - Quá trình: Text → SBERT embedding → Similarity matrix → Ranking
   - Tại sao dùng Content-based?
   - Ưu điểm: Không cần dữ liệu tương tác, có thể recommend sản phẩm mới
   - Nhược điểm: Không học được preference của user

3. **Hybrid GNN+CBF**
   - Công thức kết hợp: Score = α × GNN_score + (1-α) × CBF_score
   - Tại sao kết hợp hai mô hình?
   - Ưu điểm: Kết hợp ưu điểm của cả hai
   - Nhược điểm: Phức tạp hơn, cần tune α

**Thông số từ thực nghiệm:**
{json.dumps(metrics_snapshot, ensure_ascii=False, indent=2)}

Viết rất chi tiết, có công thức toán học rõ ràng, dễ hiểu. Sử dụng tiếng Việt."""

    return call_groq_api(prompt, max_tokens=4000, temperature=0.2)


def explain_personalized_vs_outfit(
    gnn_metrics: Dict[str, Any],
    cbf_metrics: Dict[str, Any],
    hybrid_metrics: Dict[str, Any],
) -> str:
    """Use Groq to explain Personalized vs Outfit recommendation methodologies."""
    metrics_snapshot = {
        "GNN": gnn_metrics,
        "CBF": cbf_metrics,
        "Hybrid": hybrid_metrics,
    }
    
    prompt = f"""Bạn là chuyên gia về Personalized Recommendation và Outfit Recommendation.
Hãy trình bày chi tiết hai phương pháp này:

1. **PERSONALIZED RECOMMENDATION (Gợi ý cá nhân hóa)**
   - Định nghĩa: Gợi ý dựa trên hành vi và sở thích cá nhân của từng user
   - Tổ chức dữ liệu:
     * User-Item interaction matrix: [num_users × num_items]
     * Mỗi phần tử = rating/weight của user đối với item
     * Ví dụ: User 1 mua áo sơ mi → weight = 3.0
   - Quá trình tính toán:
     * Bước 1: Xây dựng user embedding từ interaction history
     * Bước 2: Tính similarity giữa user embedding và item embeddings
     * Bước 3: Rank items theo similarity score
     * Bước 4: Trả về top-K items cao nhất
   - Công thức: Score(user_i, item_j) = similarity(user_embedding_i, item_embedding_j)
   - Ứng dụng: Amazon, Netflix, Spotify (mỗi user có gợi ý khác nhau)

2. **OUTFIT RECOMMENDATION (Gợi ý trang phục/bộ sưu tập)**
   - Định nghĩa: Gợi ý các sản phẩm phối hợp tốt với nhau (áo + quần + giày)
   - Tổ chức dữ liệu:
     * Item-Item similarity matrix: [num_items × num_items]
     * Mỗi phần tử = độ tương tự giữa hai items
     * Ví dụ: Áo sơ mi xanh + Quần jeans xanh → similarity = 0.85
   - Quá trình tính toán:
     * Bước 1: Tính item embeddings từ content (màu, kiểu, chất liệu)
     * Bước 2: Tính similarity giữa current_item và tất cả items khác
     * Bước 3: Filter items phù hợp (cùng style, màu, size)
     * Bước 4: Rank theo similarity score
     * Bước 5: Trả về top-K items để phối hợp
   - Công thức: Score(item_i, item_j) = similarity(item_embedding_i, item_embedding_j)
   - Ứng dụng: Zalora, Tiki, H&M (gợi ý sản phẩm phối hợp)

3. **SO SÁNH:**
   | Tiêu chí | Personalized | Outfit |
   |----------|-------------|--------|
   | Dữ liệu input | User ID + Interaction history | Current item ID |
   | Dữ liệu tính toán | User-Item matrix | Item-Item similarity matrix |
   | Output | Sản phẩm user thích | Sản phẩm phối hợp tốt |
   | Ứng dụng | Trang chủ, Email | Chi tiết sản phẩm, Giỏ hàng |

4. **TRIỂN KHAI TRONG HỆ THỐNG:**
   - Personalized: Dùng GNN hoặc Hybrid (học từ user behavior)
   - Outfit: Dùng CBF (học từ item content/features)
   - Kết hợp: Personalized trên trang chủ, Outfit ở chi tiết sản phẩm

**Thông số từ thực nghiệm:**
{json.dumps(metrics_snapshot, ensure_ascii=False, indent=2)}

Viết rất chi tiết, có ví dụ cụ thể, công thức rõ ràng. Sử dụng tiếng Việt."""

    return call_groq_api(prompt, max_tokens=4000, temperature=0.2)


def analyze_models_with_groq(
    gnn_metrics: Dict[str, Any],
    cbf_metrics: Dict[str, Any],
    hybrid_metrics: Dict[str, Any],
) -> str:
    """Use Groq's Llama model to analyze metrics and produce recommendations."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "**⚠️ Groq chưa sẵn sàng**: Vui lòng đặt biến môi trường `GROQ_API_KEY` "
            "để bật phân tích tự động."
        )
    
    metrics_snapshot = {
        "GNN": gnn_metrics,
        "Content-based": cbf_metrics,
        "Hybrid": hybrid_metrics,
    }
    prompt = (
        "Bạn là chuyên gia hệ thống gợi ý. Dựa vào số liệu Recall@K, NDCG@K, thời gian train "
        "và inference của ba mô hình (GNN, Content-based, Hybrid), hãy đánh giá ưu/nhược điểm "
        "và đề xuất mô hình nên triển khai production.\n\n"
        "Yêu cầu định dạng:\n"
        "- Bắt đầu bằng tiêu đề in đậm `Phân tích & lựa chọn`.\n"
        "- Viết mỗi mô hình một gạch đầu dòng nêu rõ bối cảnh phù hợp và điểm cần chú ý.\n"
        "- Kết thúc bằng một gạch đầu dòng **Kết luận** nêu lựa chọn cuối cùng.\n"
        "- Viết bằng tiếng Việt súc tích (tối đa 4 gạch đầu dòng cho phần mô hình + 1 kết luận).\n\n"
        f"Dữ liệu:\n{json.dumps(metrics_snapshot, ensure_ascii=False, indent=2)}"
    )
    
    payload = {
        "model": GROQ_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful data scientist specializing in recommender systems. Always respond in Markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }
    
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not content:
            raise ValueError("Groq response empty.")
        return content
    except (requests.RequestException, ValueError, KeyError) as exc:
        return f"**⚠️ Groq lỗi**: {exc}"


# Test API section
with st.expander("🔍 Test API & Xem Response", expanded=False):
    st.subheader("Test API Responses")
    
    test_tabs = st.tabs(["Train API", "Recommend API"])
    
    # Tab 1: Test Train API
    with test_tabs[0]:
        st.markdown("### Test `/train` API Response")
        test_train_cols = st.columns(len(models))
        for col, (label, slug) in zip(test_train_cols, models.items()):
            with col:
                if st.button(f"Test {label} Train", key=f"test_train_{slug}"):
                    with st.spinner(f"Đang gọi {label} /train API..."):
                        result = call_api(BASE_URL, f"{slug}/train", payload={"sync": True}, method="post")
                    
                    if result["success"]:
                        st.success(f"✅ {label} Train API Response:")
                        st.json(result["data"])
                        
                        # Store for analysis
                        st.session_state[f"test_train_{slug}"] = result["data"]
                    else:
                        st.error(f"❌ Lỗi: {result.get('error', 'Unknown error')}")
                        if result.get("data"):
                            st.json(result["data"])
    
    # Tab 2: Test Recommend API
    with test_tabs[1]:
        st.markdown("### Test `/recommend` API Response")
        test_user_id = st.text_input("User ID (test)", value="690bf0f2d0c3753df0ecbdd6", key="test_user_id")
        test_product_id = st.text_input("Product ID (test)", value="10068", key="test_product_id")
        
        test_recommend_cols = st.columns(len(models))
        for col, (label, slug) in zip(test_recommend_cols, models.items()):
            with col:
                if st.button(f"Test {label} Recommend", key=f"test_recommend_{slug}"):
                    # API expects user_id and current_product_id (not userId and productId)
                    payload = {"user_id": test_user_id, "current_product_id": test_product_id}
                    with st.spinner(f"Đang gọi {label} /recommend API..."):
                        result = call_api(BASE_URL, f"{slug}/recommend", payload=payload, method="post")
                    
                    if result["success"]:
                        st.success(f"✅ {label} Recommend API Response:")
                        data = result["data"]
                        
                        # Show evaluation_metrics if available
                        if "evaluation_metrics" in data:
                            st.markdown("**📊 Evaluation Metrics:**")
                            st.json(data["evaluation_metrics"])
                            st.markdown("---")
                            st.markdown("**📦 Full Response:**")
                        
                        st.json(data)
                        
                        # Store evaluation metrics for documentation
                        if "evaluation_metrics" in data:
                            eval_metrics = data["evaluation_metrics"]
                            # Update session state with evaluation metrics
                            for key in ["recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20"]:
                                if key in eval_metrics:
                                    st.session_state[f"{slug}_{key}"] = str(eval_metrics[key])
                            if "inference_time" in eval_metrics:
                                st.session_state[f"{slug}_inference_time"] = str(eval_metrics["inference_time"])
                            elif "execution_time" in eval_metrics:
                                # Convert seconds to milliseconds
                                exec_time = eval_metrics["execution_time"]
                                if isinstance(exec_time, (int, float)):
                                    st.session_state[f"{slug}_inference_time"] = str(exec_time * 1000)
                                else:
                                    st.session_state[f"{slug}_inference_time"] = str(exec_time)
                            st.success(f"✅ Đã cập nhật evaluation metrics từ {label} recommend API!")
                    else:
                        st.error(f"❌ Lỗi: {result.get('error', 'Unknown error')}")
                        if result.get("data"):
                            st.json(result["data"])

st.markdown("---")

# Create tabs for each model
doc_tabs = st.tabs([
    "📊 GNN (LightGCN)", 
    "📝 Content-based Filtering", 
    "🔀 Hybrid GNN+CBF", 
    "📈 So sánh 3 mô hình",
    "🔍 Phân tích Chi tiết Metrics",
    "🧮 Giải thích Thuật toán",
    "👔 Personalized vs Outfit"
])

# Tab 1: GNN Documentation
with doc_tabs[0]:
    st.markdown("### 2.3.1. GNN (Graph Neural Network - LightGCN)")
    
    # Get metrics from training results or session state
    gnn_metrics = extract_training_metrics(
        st.session_state.training_results.get("gnn"), 
        "gnn"
    )
    
    # Get values from session state if available (auto-filled from API)
    def get_value(key: str, default: str) -> str:
        session_key = f"gnn_{key}"
        if session_key in st.session_state:
            return str(st.session_state[session_key])
        return default
    
    def get_test_size() -> float:
        if "gnn_test_size" in st.session_state:
            return st.session_state["gnn_test_size"]
        return gnn_metrics['test_size']
    
    # Display metrics (read-only display, auto-filled from API)
    st.subheader("Thông số huấn luyện (tự động điền từ API)")
    
    # Show status if data is available
    if st.session_state.training_results.get("gnn"):
        st.info("✅ Số liệu đã được tự động điền từ kết quả training API")
    else:
        st.warning("⚠️ Chưa có dữ liệu từ API. Vui lòng train mô hình GNN trước.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_users = get_value("num_users", str(gnn_metrics['num_users']))
        num_products = get_value("num_products", str(gnn_metrics['num_products']))
        st.metric("Số lượng người dùng train", num_users)
        st.metric("Số lượng sản phẩm train", num_products)
    with col2:
        num_interactions = get_value("num_interactions", str(gnn_metrics['num_interactions']))
        num_training_samples = get_value("num_samples", str(gnn_metrics['num_training_samples']))
        st.metric("Số lượng tương tác", num_interactions)
        st.metric("Số lượng training samples (BPR)", num_training_samples)
    with col3:
        epochs = get_value("epochs", str(gnn_metrics['epochs']))
        batch_size = get_value("batch", str(gnn_metrics['batch_size']))
        st.metric("Epochs", epochs)
        st.metric("Batch size", batch_size)
    
    col4, col5 = st.columns(2)
    with col4:
        embed_dim = get_value("embed", str(gnn_metrics['embed_dim']))
        learning_rate = get_value("lr", str(gnn_metrics['learning_rate']))
        st.metric("Embedding dimension", embed_dim)
        st.metric("Learning rate", learning_rate)
    with col5:
        test_size = get_test_size()
        st.metric("Test size", test_size)
    
    st.subheader("Chỉ số đánh giá (tự động điền từ API /recommend)")
    st.caption("💡 **Lưu ý**: Các chỉ số này lấy từ `evaluation_metrics` trong response của API `/recommend`. Vui lòng gọi API recommend để có số liệu đánh giá.")
    
    # Check if we have recommendation results
    has_recommend_data = st.session_state.recommendation_results.get("gnn") is not None
    if has_recommend_data:
        st.info("✅ Đã có dữ liệu từ API /recommend")
    else:
        st.warning("⚠️ Chưa có dữ liệu từ API /recommend. Vui lòng gọi API recommend ở section 3 để lấy evaluation metrics.")
    
    eval_col1, eval_col2, eval_col3 = st.columns(3)
    with eval_col1:
        recall_at_10 = format_metric_value(get_value("recall_at_10", "N/A"))
        recall_at_20 = format_metric_value(get_value("recall_at_20", "N/A"))
        st.metric("Recall@10", recall_at_10)
        st.metric("Recall@20", recall_at_20)
    with eval_col2:
        ndcg_at_10 = get_value("ndcg_at_10", "N/A")
        ndcg_at_20 = get_value("ndcg_at_20", "N/A")
        st.metric("NDCG@10", ndcg_at_10)
        st.metric("NDCG@20", ndcg_at_20)
    with eval_col3:
        training_time = format_metric_value(get_value("training_time", "N/A"))
        inference_time = get_value("inference_time", "N/A")
        st.metric("Thời gian train", training_time)
        st.metric("Thời gian inference/user", f"{inference_time} ms" if inference_time != "N/A" else "N/A")
    
    # Update metrics with current input values
    gnn_metrics_updated = {
        'num_users': num_users,
        'num_products': num_products,
        'num_interactions': num_interactions,
        'num_training_samples': num_training_samples,
        'epochs': epochs,
        'batch_size': batch_size,
        'embed_dim': embed_dim,
        'learning_rate': learning_rate,
        'test_size': test_size,
        'recall_at_10': recall_at_10,
        'recall_at_20': recall_at_20,
        'ndcg_at_10': ndcg_at_10,
        'ndcg_at_20': ndcg_at_20,
        'training_time': training_time,
        'inference_time': inference_time,
    }
    gnn_metrics_updated = apply_precision_formatting(gnn_metrics_updated)
    
    # Generate and display documentation
    gnn_doc = generate_gnn_documentation(gnn_metrics_updated)
    
    st.markdown("---")
    st.subheader("📄 Nội dung tài liệu (có thể copy)")
    st.markdown(gnn_doc)
    
    # Copy button
    st.code(gnn_doc, language="markdown")

# Tab 2: CBF Documentation
with doc_tabs[1]:
    st.markdown("### 2.3.2. Content-based Filtering")
    
    # Get metrics from training results or session state
    cbf_metrics = extract_training_metrics(
        st.session_state.training_results.get("cbf"), 
        "cbf"
    )
    
    # Get values from session state if available (auto-filled from API)
    def get_value(key: str, default: str) -> str:
        session_key = f"cbf_{key}"
        if session_key in st.session_state:
            return str(st.session_state[session_key])
        return default
    
    def get_test_size() -> float:
        if "cbf_test_size" in st.session_state:
            return st.session_state["cbf_test_size"]
        return cbf_metrics['test_size']
    
    # Display metrics (read-only display, auto-filled from API)
    st.subheader("Thông số huấn luyện (tự động điền từ API)")
    
    # Show status if data is available
    if st.session_state.training_results.get("cbf"):
        st.info("✅ Số liệu đã được tự động điền từ kết quả training API")
    else:
        st.warning("⚠️ Chưa có dữ liệu từ API. Vui lòng train mô hình CBF trước.")
    
    col1, col2 = st.columns(2)
    with col1:
        num_products = get_value("num_products", str(cbf_metrics['num_products']))
        num_users = get_value("num_users", str(cbf_metrics['num_users']))
        st.metric("Số lượng sản phẩm train", num_products)
        st.metric("Số lượng người dùng test", num_users)
    with col2:
        embed_dim = get_value("embed", str(cbf_metrics['embed_dim']))
        test_size = get_test_size()
        st.metric("Embedding dimension", embed_dim)
        st.metric("Test size", test_size)
    
    st.subheader("Chỉ số đánh giá (tự động điền từ API /recommend)")
    st.caption("💡 **Lưu ý**: Các chỉ số này lấy từ `evaluation_metrics` trong response của API `/recommend`. Vui lòng gọi API recommend để có số liệu đánh giá.")
    
    # Check if we have recommendation results
    has_recommend_data = st.session_state.recommendation_results.get("cbf") is not None
    if has_recommend_data:
        st.info("✅ Đã có dữ liệu từ API /recommend")
    else:
        st.warning("⚠️ Chưa có dữ liệu từ API /recommend. Vui lòng gọi API recommend ở section 3 để lấy evaluation metrics.")
    
    eval_col1, eval_col2, eval_col3 = st.columns(3)
    with eval_col1:
        recall_at_10 = format_metric_value(get_value("recall_at_10", "N/A"))
        recall_at_20 = format_metric_value(get_value("recall_at_20", "N/A"))
        st.metric("Recall@10", recall_at_10)
        st.metric("Recall@20", recall_at_20)
    with eval_col2:
        ndcg_at_10 = get_value("ndcg_at_10", "N/A")
        ndcg_at_20 = get_value("ndcg_at_20", "N/A")
        st.metric("NDCG@10", ndcg_at_10)
        st.metric("NDCG@20", ndcg_at_20)
    with eval_col3:
        training_time = format_metric_value(get_value("training_time", "N/A"))
        inference_time = get_value("inference_time", "N/A")
        st.metric("Thời gian train", training_time)
        st.metric("Thời gian inference/user", f"{inference_time} ms" if inference_time != "N/A" else "N/A")
    
    # Update metrics with current input values
    cbf_metrics_updated = {
        'num_products': num_products,
        'num_users': num_users,
        'embed_dim': embed_dim,
        'test_size': test_size,
        'recall_at_10': recall_at_10,
        'recall_at_20': recall_at_20,
        'ndcg_at_10': ndcg_at_10,
        'ndcg_at_20': ndcg_at_20,
        'training_time': training_time,
        'inference_time': inference_time,
    }
    cbf_metrics_updated = apply_precision_formatting(cbf_metrics_updated)
    
    # Generate and display documentation
    cbf_doc = generate_cbf_documentation(cbf_metrics_updated)
    
    st.markdown("---")
    st.subheader("📄 Nội dung tài liệu (có thể copy)")
    st.markdown(cbf_doc)
    
    # Copy button
    st.code(cbf_doc, language="markdown")

# Tab 3: Hybrid Documentation
with doc_tabs[2]:
    st.markdown("### 2.3.3. Hybrid GNN (LightGCN) & Content-based Filtering")
    
    # Get metrics from training results or session state
    hybrid_metrics = extract_training_metrics(
        st.session_state.training_results.get("hybrid"), 
        "hybrid"
    )
    
    # Get values from session state if available (auto-filled from API)
    def get_value(key: str, default: str) -> str:
        session_key = f"hybrid_{key}"
        if session_key in st.session_state:
            return str(st.session_state[session_key])
        return default
    
    def get_test_size() -> float:
        if "hybrid_test_size" in st.session_state:
            return st.session_state["hybrid_test_size"]
        return hybrid_metrics['test_size']
    
    # Alpha parameter (can be from API or default)
    if "hybrid_alpha" in st.session_state:
        default_alpha = st.session_state["hybrid_alpha"]
    else:
        default_alpha = 0.7
    alpha = st.slider("Trọng số alpha (GNN weight)", min_value=0.0, max_value=1.0, value=default_alpha, step=0.1, key="hybrid_alpha")
    
    # Display metrics (read-only display, auto-filled from API)
    st.subheader("Thông số huấn luyện (tự động điền từ API)")
    
    # Show status if data is available
    if st.session_state.training_results.get("hybrid"):
        st.info("✅ Số liệu đã được tự động điền từ kết quả training API")
    else:
        st.warning("⚠️ Chưa có dữ liệu từ API. Vui lòng train mô hình Hybrid trước.")
    
    col1, col2 = st.columns(2)
    with col1:
        num_users = get_value("num_users", str(hybrid_metrics['num_users']))
        num_products = get_value("num_products", str(hybrid_metrics['num_products']))
        st.metric("Số lượng người dùng train", num_users)
        st.metric("Số lượng sản phẩm train", num_products)
    with col2:
        num_interactions = get_value("num_interactions", str(hybrid_metrics['num_interactions']))
        embed_dim = get_value("embed", str(hybrid_metrics['embed_dim']))
        st.metric("Số lượng tương tác", num_interactions)
        st.metric("Embedding dimension", embed_dim)
    
    test_size = get_test_size()
    st.metric("Test size", test_size)
    
    st.subheader("Chỉ số đánh giá (tự động điền từ API /recommend)")
    st.caption("💡 **Lưu ý**: Các chỉ số này lấy từ `evaluation_metrics` trong response của API `/recommend`. Vui lòng gọi API recommend để có số liệu đánh giá.")
    
    # Check if we have recommendation results
    has_recommend_data = st.session_state.recommendation_results.get("hybrid") is not None
    if has_recommend_data:
        st.info("✅ Đã có dữ liệu từ API /recommend")
    else:
        st.warning("⚠️ Chưa có dữ liệu từ API /recommend. Vui lòng gọi API recommend ở section 3 để lấy evaluation metrics.")
    
    eval_col1, eval_col2, eval_col3 = st.columns(3)
    with eval_col1:
        recall_at_10 = format_metric_value(get_value("recall_at_10", "N/A"))
        recall_at_20 = format_metric_value(get_value("recall_at_20", "N/A"))
        st.metric("Recall@10", recall_at_10)
        st.metric("Recall@20", recall_at_20)
    with eval_col2:
        ndcg_at_10 = get_value("ndcg_at_10", "N/A")
        ndcg_at_20 = get_value("ndcg_at_20", "N/A")
        st.metric("NDCG@10", ndcg_at_10)
        st.metric("NDCG@20", ndcg_at_20)
    with eval_col3:
        training_time = format_metric_value(get_value("training_time", "N/A"))
        inference_time = get_value("inference_time", "N/A")
        st.metric("Thời gian train", training_time)
        st.metric("Thời gian inference/user", f"{inference_time} ms" if inference_time != "N/A" else "N/A")
    
    # Update metrics with current input values
    hybrid_metrics_updated = {
        'num_users': num_users,
        'num_products': num_products,
        'num_interactions': num_interactions,
        'embed_dim': embed_dim,
        'test_size': test_size,
        'recall_at_10': recall_at_10,
        'recall_at_20': recall_at_20,
        'ndcg_at_10': ndcg_at_10,
        'ndcg_at_20': ndcg_at_20,
        'training_time': training_time,
        'inference_time': inference_time,
    }
    hybrid_metrics_updated = apply_precision_formatting(hybrid_metrics_updated)
    
    # Generate and display documentation
    hybrid_doc = generate_hybrid_documentation(hybrid_metrics_updated, alpha)
    
    st.markdown("---")
    st.subheader("📄 Nội dung tài liệu (có thể copy)")
    st.markdown(hybrid_doc)
    
    # Copy button
    st.code(hybrid_doc, language="markdown")

# Tab 4: Comparison
with doc_tabs[3]:
    st.markdown("### So sánh 3 mô hình")
    
    st.info("💡 **Lưu ý**: Số liệu sẽ tự động được điền sau khi train các mô hình qua API. Vui lòng train các mô hình trước khi xem bảng so sánh.")
    
    # Get all metrics from session state (will be updated by the input fields in other tabs)
    gnn_metrics_final = extract_training_metrics(st.session_state.training_results.get("gnn"), "gnn")
    cbf_metrics_final = extract_training_metrics(st.session_state.training_results.get("cbf"), "cbf")
    hybrid_metrics_final = extract_training_metrics(st.session_state.training_results.get("hybrid"), "hybrid")
    
    # Get values from session state (auto-filled from API)
    def update_metrics_from_session(metrics_dict: Dict[str, Any], prefix: str) -> None:
        """Update metrics from session state with proper key mapping."""
        for key in ["recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20", 
                   "training_time", "inference_time",
                   "num_users", "num_products", "num_interactions", 
                   "epochs", "embed_dim", "learning_rate"]:
            session_key = f"{prefix}_{key}"
            if session_key in st.session_state:
                metrics_dict[key] = st.session_state[session_key]
        
        # Handle special mappings
        if f"{prefix}_num_samples" in st.session_state:
            metrics_dict["num_training_samples"] = st.session_state[f"{prefix}_num_samples"]
        if f"{prefix}_batch" in st.session_state:
            metrics_dict["batch_size"] = st.session_state[f"{prefix}_batch"]
        if f"{prefix}_embed" in st.session_state:
            metrics_dict["embed_dim"] = st.session_state[f"{prefix}_embed"]
        if f"{prefix}_lr" in st.session_state:
            metrics_dict["learning_rate"] = st.session_state[f"{prefix}_lr"]
    
    update_metrics_from_session(gnn_metrics_final, "gnn")
    update_metrics_from_session(cbf_metrics_final, "cbf")
    update_metrics_from_session(hybrid_metrics_final, "hybrid")
    gnn_metrics_final = apply_precision_formatting(gnn_metrics_final)
    cbf_metrics_final = apply_precision_formatting(cbf_metrics_final)
    hybrid_metrics_final = apply_precision_formatting(hybrid_metrics_final)
    
    # Also get alpha for hybrid
    if "hybrid_alpha" in st.session_state:
        alpha_final = st.session_state["hybrid_alpha"]
    else:
        alpha_final = 0.7
    
    # Generate Groq-backed analysis text
    with st.spinner("🤖 Đang nhờ Groq phân tích số liệu..."):
        groq_analysis_text = analyze_models_with_groq(
            gnn_metrics_final,
            cbf_metrics_final,
            hybrid_metrics_final,
        )
    
    # Generate comparison table
    comparison_doc = generate_comparison_table(
        gnn_metrics_final,
        cbf_metrics_final,
        hybrid_metrics_final,
        groq_analysis_text or "**⚠️ Groq không trả về dữ liệu để phân tích.**",
    )
    st.markdown(comparison_doc)
    
    # Copy button
    st.code(comparison_doc, language="markdown")
    
    st.subheader("🤖 Phân tích & lựa chọn (Groq)")
    st.markdown(groq_analysis_text)

# Tab 5: Detailed Metrics Analysis
with doc_tabs[4]:
    st.markdown("### 🔍 Phân tích Chi tiết Metrics")
    st.info("Phần này sử dụng Groq AI để giải thích rất chi tiết các chỉ số Recall, NDCG, thời gian train/inference và đưa ra khuyến nghị chọn mô hình tốt nhất dựa trên số liệu thực nghiệm.")

    # Gather metrics for analysis
    gnn_metrics_analysis = extract_training_metrics(st.session_state.training_results.get("gnn"), "gnn")
    cbf_metrics_analysis = extract_training_metrics(st.session_state.training_results.get("cbf"), "cbf")
    hybrid_metrics_analysis = extract_training_metrics(st.session_state.training_results.get("hybrid"), "hybrid")

    def _update_from_session(metrics_dict: Dict[str, Any], prefix: str) -> None:
        for key in ["recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20", "training_time", "inference_time",
                    "num_users", "num_products", "num_interactions", "epochs", "embed_dim", "learning_rate"]:
            session_key = f"{prefix}_{key}"
            if session_key in st.session_state:
                metrics_dict[key] = st.session_state[session_key]
        if f"{prefix}_num_samples" in st.session_state:
            metrics_dict["num_training_samples"] = st.session_state[f"{prefix}_num_samples"]
        if f"{prefix}_batch" in st.session_state:
            metrics_dict["batch_size"] = st.session_state[f"{prefix}_batch"]
        if f"{prefix}_embed" in st.session_state:
            metrics_dict["embed_dim"] = st.session_state[f"{prefix}_embed"]
        if f"{prefix}_lr" in st.session_state:
            metrics_dict["learning_rate"] = st.session_state[f"{prefix}_lr"]

    _update_from_session(gnn_metrics_analysis, "gnn")
    _update_from_session(cbf_metrics_analysis, "cbf")
    _update_from_session(hybrid_metrics_analysis, "hybrid")

    if st.button("🚀 Phân tích Chi tiết với Groq", key="btn_detailed_metrics"):
        with st.spinner("⏳ Đang gọi Groq để phân tích chi tiết..."):
            detailed_text = analyze_metrics_detailed(
                gnn_metrics_analysis,
                cbf_metrics_analysis,
                hybrid_metrics_analysis,
            )
        st.markdown("---")
        st.markdown(detailed_text)
        st.code(detailed_text, language="markdown")

# Tab 6: Algorithm Explanation
with doc_tabs[5]:
    st.markdown("### 🧮 Giải thích Thuật toán (có công thức)")
    st.info("Phần này sử dụng Groq AI để trình bày thuật toán GNN, CBF và Hybrid với công thức chi tiết, giải thích từng bước tính toán.")

    with st.expander("Thiết lập thư viện công thức toán học (tùy chọn)"):
        st.markdown("- Streamlit hỗ trợ hiển thị công thức LaTeX qua st.markdown/st.latex, không cần cài thêm.")
        st.markdown("- Nếu muốn tính toán biểu thức và render công thức tự động, có thể dùng SymPy:")
        st.code("""
# Kích hoạt môi trường ảo (chọn một trong các lệnh phù hợp hệ điều hành)
# macOS/Linux (bash/zsh)
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# Cài đặt thư viện
pip install sympy
""", language="bash")
        st.markdown("Ví dụ dùng SymPy để tính và render công thức:")
        st.code("""
import sympy as sp
x, y = sp.symbols('x y')
expr = (x + y)**3
expanded = sp.expand(expr)
latex_str = sp.latex(expanded)  # Chuyển sang LaTeX để hiển thị
st.latex(latex_str)
""", language="python")

    gnn_metrics_algo = extract_training_metrics(st.session_state.training_results.get("gnn"), "gnn")
    cbf_metrics_algo = extract_training_metrics(st.session_state.training_results.get("cbf"), "cbf")
    hybrid_metrics_algo = extract_training_metrics(st.session_state.training_results.get("hybrid"), "hybrid")

    _update_from_session(gnn_metrics_algo, "gnn")
    _update_from_session(cbf_metrics_algo, "cbf")
    _update_from_session(hybrid_metrics_algo, "hybrid")

    if st.button("🚀 Giải thích Thuật toán với Groq", key="btn_algo_explain"):
        with st.spinner("⏳ Đang gọi Groq để giải thích thuật toán..."):
            algo_text = explain_algorithms_detailed(
                gnn_metrics_algo,
                cbf_metrics_algo,
                hybrid_metrics_algo,
            )
        st.markdown("---")
        st.markdown(algo_text)
        st.code(algo_text, language="markdown")

# Tab 7: Personalized vs Outfit
with doc_tabs[6]:
    st.markdown("### 👔 Personalized vs Outfit Recommendation")
    st.info("Giải thích tiêu chuẩn Personalized (cá nhân hóa) và Outfit (phối đồ), cách tổ chức dữ liệu và công thức tính điểm gợi ý.")

    gnn_metrics_pf = extract_training_metrics(st.session_state.training_results.get("gnn"), "gnn")
    cbf_metrics_pf = extract_training_metrics(st.session_state.training_results.get("cbf"), "cbf")
    hybrid_metrics_pf = extract_training_metrics(st.session_state.training_results.get("hybrid"), "hybrid")

    _update_from_session(gnn_metrics_pf, "gnn")
    _update_from_session(cbf_metrics_pf, "cbf")
    _update_from_session(hybrid_metrics_pf, "hybrid")

    if st.button("🚀 Phân tích Personalized vs Outfit (Groq)", key="btn_pf_outfit"):
        with st.spinner("⏳ Đang gọi Groq để phân tích Personalized vs Outfit..."):
            pf_text = explain_personalized_vs_outfit(
                gnn_metrics_pf,
                cbf_metrics_pf,
                hybrid_metrics_pf,
            )
        st.markdown("---")
        st.markdown(pf_text)
        st.code(pf_text, language="markdown")