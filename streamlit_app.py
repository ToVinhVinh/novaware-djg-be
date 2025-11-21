"""Streamlit dashboard for Novaware product analytics and model APIs."""

from __future__ import annotations

import json
import time
from io import BytesIO
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st


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

# Quick start guide
with st.expander("📋 Hướng dẫn sử dụng (Quick Start Guide)", expanded=False):
    st.markdown("""
    ### 🎯 Quy trình tạo tài liệu tự động:
    
    **Bước 1: Kiểm tra kết nối API**
    - Đảm bảo Django server đang chạy tại URL trong sidebar
    - Mặc định: `http://127.0.0.1:8000/api/v1`
    
    **Bước 2: Train các mô hình (Section 2)**
    - Click nút "Train GNN" → Chờ training hoàn tất
    - Click nút "Train Content-based (CBF)" → Chờ training hoàn tất  
    - Click nút "Train Hybrid" → Chờ training hoàn tất
    - ✅ Sau khi train, thông số huấn luyện sẽ tự động điền vào tài liệu
    
    **Bước 3: Gọi API Recommend (Section 3)**
    - Nhập User ID và Product ID (hoặc dùng giá trị mặc định)
    - Click "Recommend GNN" → Lấy evaluation metrics
    - Click "Recommend Content-based (CBF)" → Lấy evaluation metrics
    - Click "Recommend Hybrid" → Lấy evaluation metrics
    - ✅ Sau khi recommend, evaluation metrics sẽ tự động điền vào tài liệu
    
    **Bước 4: Xem và copy tài liệu (Section 4)**
    - Chọn tab tương ứng (GNN, CBF, Hybrid, hoặc So sánh)
    - Xem số liệu đã được tự động điền
    - Copy markdown code để dán vào báo cáo
    
    **💡 Mẹo**: 
    - Sử dụng section "🔍 Test API & Xem Response" để kiểm tra response của API
    - Tất cả số liệu được tự động điền, không cần nhập thủ công
    """)


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
                # Convert to string if numeric
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
            "training_time": "time",
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
                
                # Add training time if not present
                if isinstance(result_data, dict):
                    if "training_time" not in result_data and "time" not in result_data:
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


def generate_comparison_table(gnn_metrics: Dict[str, Any], cbf_metrics: Dict[str, Any], 
                              hybrid_metrics: Dict[str, Any]) -> str:
    """Generate comparison table for all 3 models."""
    doc = """# 3. Đánh giá 3 mô hình

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

- **Phân tích & lựa chọn**:
  - **GNN (LightGCN)**: Phù hợp khi có nhiều dữ liệu tương tác người dùng, thường cho Recall@K và NDCG@K cao nhất nhờ học từ hành vi người dùng tương tự thông qua Graph Neural Network.
  - **Content-based Filtering**: Phù hợp khi cần xử lý cold-start (người dùng/sản phẩm mới) hoặc catalog phong phú, đảm bảo gợi ý hợp lý nhờ lọc theo đặc điểm sản phẩm (age/gender/style) sử dụng Sentence-BERT + FAISS.
  - **Hybrid GNN+CBF**: Lựa chọn production mặc định vì kết hợp ưu điểm của cả hai phương pháp (GNN LightGCN + CBF Sentence-BERT), duy trì ổn định trong nhiều tình huống, có thể tinh chỉnh trọng số `alpha` để ưu tiên hành vi người dùng (GNN) hoặc đặc điểm sản phẩm (CBF).
  - **Kết luận**: Hybrid thường đạt Recall@K và NDCG@K cao nhất và thời gian inference chấp nhận được, phù hợp cho môi trường production.
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
    )
    return doc


st.header("4. Tài liệu mô hình (Documentation)")

st.markdown("""
**📌 Nguồn dữ liệu cho tài liệu:**

- **Từ API `/train`**: Thông số huấn luyện (num_users, num_products, epochs, batch_size, embed_dim, learning_rate, etc.)
- **Từ API `/recommend`**: Chỉ số đánh giá (MAPE, RMSE, Precision, Recall, F1, execution_time) trong `evaluation_metrics`

**💡 Lưu ý**: Để có đầy đủ số liệu, bạn cần:
1. Train mô hình qua API `/train` → Lấy thông số huấn luyện
2. Gọi API `/recommend` → Lấy evaluation metrics
""")

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
doc_tabs = st.tabs(["📊 GNN (LightGCN)", "📝 Content-based Filtering", "🔀 Hybrid GNN+CBF", "📈 So sánh 3 mô hình"])

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
        recall_at_10 = get_value("recall_at_10", "N/A")
        recall_at_20 = get_value("recall_at_20", "N/A")
        st.metric("Recall@10", recall_at_10)
        st.metric("Recall@20", recall_at_20)
    with eval_col2:
        ndcg_at_10 = get_value("ndcg_at_10", "N/A")
        ndcg_at_20 = get_value("ndcg_at_20", "N/A")
        st.metric("NDCG@10", ndcg_at_10)
        st.metric("NDCG@20", ndcg_at_20)
    with eval_col3:
        training_time = get_value("training_time", "N/A")
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
        recall_at_10 = get_value("recall_at_10", "N/A")
        recall_at_20 = get_value("recall_at_20", "N/A")
        st.metric("Recall@10", recall_at_10)
        st.metric("Recall@20", recall_at_20)
    with eval_col2:
        ndcg_at_10 = get_value("ndcg_at_10", "N/A")
        ndcg_at_20 = get_value("ndcg_at_20", "N/A")
        st.metric("NDCG@10", ndcg_at_10)
        st.metric("NDCG@20", ndcg_at_20)
    with eval_col3:
        training_time = get_value("training_time", "N/A")
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
        recall_at_10 = get_value("recall_at_10", "N/A")
        recall_at_20 = get_value("recall_at_20", "N/A")
        st.metric("Recall@10", recall_at_10)
        st.metric("Recall@20", recall_at_20)
    with eval_col2:
        ndcg_at_10 = get_value("ndcg_at_10", "N/A")
        ndcg_at_20 = get_value("ndcg_at_20", "N/A")
        st.metric("NDCG@10", ndcg_at_10)
        st.metric("NDCG@20", ndcg_at_20)
    with eval_col3:
        training_time = get_value("training_time", "N/A")
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
    
    # Also get alpha for hybrid
    if "hybrid_alpha" in st.session_state:
        alpha_final = st.session_state["hybrid_alpha"]
    else:
        alpha_final = 0.7
    
    # Generate comparison table
    comparison_doc = generate_comparison_table(gnn_metrics_final, cbf_metrics_final, hybrid_metrics_final)
    
    st.markdown("---")
    st.subheader("📄 Bảng so sánh (có thể copy)")
    st.markdown(comparison_doc)
    
    # Copy button
    st.code(comparison_doc, language="markdown")
    
    # Visual comparison
    st.subheader("📊 Biểu đồ so sánh")
    comparison_data = {
        "Mô hình": ["GNN (LightGCN)", "Content-based Filtering", "Hybrid GNN+CBF"],
        "Recall@10": [gnn_metrics_final.get('recall_at_10', 'N/A'), cbf_metrics_final.get('recall_at_10', 'N/A'), hybrid_metrics_final.get('recall_at_10', 'N/A')],
        "Recall@20": [gnn_metrics_final.get('recall_at_20', 'N/A'), cbf_metrics_final.get('recall_at_20', 'N/A'), hybrid_metrics_final.get('recall_at_20', 'N/A')],
        "NDCG@10": [gnn_metrics_final.get('ndcg_at_10', 'N/A'), cbf_metrics_final.get('ndcg_at_10', 'N/A'), hybrid_metrics_final.get('ndcg_at_10', 'N/A')],
        "NDCG@20": [gnn_metrics_final.get('ndcg_at_20', 'N/A'), cbf_metrics_final.get('ndcg_at_20', 'N/A'), hybrid_metrics_final.get('ndcg_at_20', 'N/A')],
    }
    
    # Try to convert to numeric for plotting
    try:
        comparison_df = pd.DataFrame(comparison_data)
        for col in ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]:
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
        
        st.bar_chart(comparison_df.set_index("Mô hình")[["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]], use_container_width=True)
    except:
        st.info("Vui lòng nhập số liệu để hiển thị biểu đồ so sánh.")


# Update session state when training completes
st.markdown("---")
st.caption(
    "Ứng dụng Streamlit này giúp kiểm thử nhanh các API gợi ý sản phẩm của Novaware và tạo tài liệu tự động."
)

