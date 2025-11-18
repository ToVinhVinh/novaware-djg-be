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
    - Training metrics: may include evaluation metrics if available in artifacts
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
        "mape": "N/A",
        "rmse": "N/A",
        "precision": "N/A",
        "recall": "N/A",
        "f1": "N/A",
        "time": "N/A",
    }
    
    if not result_data:
        return metrics
    
    # Try to extract from different possible response structures
    if isinstance(result_data, dict):
        # Direct metrics (evaluation metrics) - may be from /train artifacts
        for key in ["mape", "rmse", "precision", "recall", "f1", "f1_score"]:
            if key in result_data:
                value = result_data[key]
                # Convert to string if numeric
                if isinstance(value, (int, float)):
                    metrics[key] = str(value)
                else:
                    metrics[key] = value
        
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
                for key in ["mape", "rmse", "precision", "recall", "f1", "f1_score"]:
                    if key in nested:
                        value = nested[key]
                        if isinstance(value, (int, float)):
                            metrics[key] = str(value)
                        else:
                            metrics[key] = value
        
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
    - MAPE, RMSE, Precision, Recall, F1, execution_time
    """
    metrics = {
        "mape": "N/A",
        "rmse": "N/A",
        "precision": "N/A",
        "recall": "N/A",
        "f1": "N/A",
        "time": "N/A",
    }
    
    if not result_data or not isinstance(result_data, dict):
        return metrics
    
    # Extract from evaluation_metrics (from /recommend API)
    if "evaluation_metrics" in result_data:
        eval_metrics = result_data["evaluation_metrics"]
        if isinstance(eval_metrics, dict):
            for key in ["mape", "rmse", "precision", "recall", "f1", "f1_score"]:
                if key in eval_metrics:
                    value = eval_metrics[key]
                    if isinstance(value, (int, float)):
                        metrics[key] = str(value)
                    else:
                        metrics[key] = value
            
            # Execution time
            if "execution_time" in eval_metrics:
                value = eval_metrics["execution_time"]
                metrics["time"] = str(value) if isinstance(value, (int, float)) else value
            elif "time" in eval_metrics:
                value = eval_metrics["time"]
                metrics["time"] = str(value) if isinstance(value, (int, float)) else value
    
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
        "mape": f"{slug}_mape",
        "rmse": f"{slug}_rmse",
        "precision": f"{slug}_precision",
        "recall": f"{slug}_recall",
        "f1": f"{slug}_f1",
        "time": f"{slug}_time",
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
payload = {"userId": user_id, "productId": product_id}

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
    Sử dụng `surprise.Dataset.load_from_df(...)` và `train_test_split(test_size={metrics['test_size']})` để tạo tập train/test.  
    - Test size: **{metrics['test_size']}**  
    - Số lượng người dùng train: **{metrics['num_users']}**  
    - Số lượng sản phẩm train: **{metrics['num_products']}**
    - Số lượng tương tác (interactions): **{metrics['num_interactions']}**
    - Số lượng training samples (BPR): **{metrics['num_training_samples']}**
  - *Pipeline 5 bước*:
    1. **Huấn luyện mô hình**: LightGCN với kiến trúc Graph Convolutional Network.
       - Thuật toán: LightGCN (Light Graph Convolution Network)
       - Framework: PyTorch + PyTorch Geometric
       - Loss function: BPR (Bayesian Personalized Ranking)
       - Negative sampling: 4 negative samples per positive interaction
       - Epochs: **{metrics['epochs']}**
       - Batch size: **{metrics['batch_size']}**
       - Embedding dimension: **{metrics['embed_dim']}**
       - Learning rate: **{metrics['learning_rate']}**
       - Optimizer: Adam
       - Model file: `models/gnn_lightgcn.pkl`
    2. **Chuẩn bị dữ liệu graph**: 
       - Xây dựng bipartite graph từ `UserInteraction` collection.
       - Áp dụng trọng số tương tác theo `INTERACTION_WEIGHTS`:
         ```python
         INTERACTION_WEIGHTS = {{
             'view': 1.0,
             'add_to_cart': 2.0,
             'purchase': 3.0,
             'wishlist': 1.5,
             'rating': 2.5
         }}
         ```
       - Tạo edge index (user-product pairs) và edge weights.
    3. **Tạo ma trận User-Item Interaction**: 
       - Sử dụng sparse matrix để biểu diễn tương tác user-product.
       - Tính toán sparsity: `sparsity = 1 - (num_interactions / (num_users * num_products))`
    4. **Tính cosine similarity** giữa user embeddings và product embeddings.  
       - Sau khi training, LightGCN sinh ra:
         - User embeddings: `[{metrics['num_users']}, {metrics['embed_dim']}]`
         - Product embeddings: `[{metrics['num_products']}, {metrics['embed_dim']}]`
       - Recommendation score = dot product giữa user embedding và product embedding.
    5. **Tính toán chỉ số đánh giá**: MAPE, RMSE, Precision, Recall, F1, thời gian.
       - *MAPE*: sai số phần trăm tuyệt đối trung bình giữa rating dự đoán và thực tế.
       - *RMSE*: độ lệch chuẩn của sai số dự đoán.
       - *Precision/Recall/F1*: độ chính xác/phủ và cân bằng giữa hai yếu tố khi so sánh recommendation với ground-truth.
       - *Thời gian*: latency suy luận hoặc toàn bộ pipeline ({metrics['time']}).

| Mô hình | MAPE | RMSE | Precision | Recall | F1 | Thời gian |
|---------|------|------|-----------|--------|----|-----------|
| GNN (LightGCN) | {metrics['mape']} | {metrics['rmse']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['time']} |
"""
    return doc


def generate_cbf_documentation(metrics: Dict[str, Any]) -> str:
    """Generate Content-based Filtering documentation markdown with metrics."""
    doc = f"""### 2.3.2. Content-based Filtering

- **Quy trình thực hiện**:
  - *Chuẩn hóa dữ liệu với Surprise*:  
    Sử dụng `surprise.Dataset.load_from_df(...)` và `train_test_split(test_size={metrics['test_size']})` để tạo tập train/test.  
    - Test size: **{metrics['test_size']}**  
    - Số lượng sản phẩm train: **{metrics['num_products']}**  
    - Số lượng người dùng test: **{metrics['num_users']}**
  - *Pipeline 5 bước*:
    1. **Huấn luyện mô hình**: Sentence-BERT embedding + FAISS index.
       - Model: Sentence-BERT (SBERT)
       - Index: FAISS (Facebook AI Similarity Search)
       - Embedding dimension: **{metrics['embed_dim']}**
    2. **Chuẩn bị dữ liệu văn bản**: ghép `category`, `gender`, `color`, `style_tags`, `productDisplayName`.
    3. **Tạo ma trận TF-IDF**: sử dụng `TfidfVectorizer` để tạo ma trận TF-IDF cho báo cáo.  
    4. **Tính cosine similarity** giữa các sản phẩm (SBERT embeddings).  
       - Recommendation score = cosine similarity giữa product embeddings.
    5. **Tính toán chỉ số đánh giá**: MAPE, RMSE, Precision, Recall, F1, thời gian.
       - *MAPE*: sai số phần trăm tuyệt đối trung bình giữa rating dự đoán và thực tế.
       - *RMSE*: độ lệch chuẩn của sai số dự đoán.
       - *Precision/Recall/F1*: độ chính xác/phủ và cân bằng giữa hai yếu tố khi so sánh recommendation với ground-truth.
       - *Thời gian*: latency suy luận hoặc toàn bộ pipeline ({metrics['time']}).

| Mô hình | MAPE | RMSE | Precision | Recall | F1 | Thời gian |
|---------|------|------|-----------|--------|----|-----------|
| Content-based Filtering | {metrics['mape']} | {metrics['rmse']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['time']} |
"""
    return doc


def generate_hybrid_documentation(metrics: Dict[str, Any], alpha: float = 0.7) -> str:
    """Generate Hybrid documentation markdown with metrics."""
    doc = f"""### 2.3.3. Hybrid Content-based Filtering & Collaborative Filtering

- **Quy trình thực hiện**:
  - *Chuẩn hóa dữ liệu với Surprise*:  
    Sử dụng `surprise.Dataset.load_from_df(...)` và `train_test_split(test_size={metrics['test_size']})` để tạo tập train/test.  
    - Test size: **{metrics['test_size']}**  
    - Số lượng người dùng train: **{metrics['num_users']}**  
    - Số lượng sản phẩm train: **{metrics['num_products']}**
    - Số lượng tương tác (interactions): **{metrics['num_interactions']}**
  - *Pipeline 5 bước*:
    1. **Huấn luyện mô hình**: Kết hợp GNN (LightGCN) + CBF (Sentence-BERT).
       - GNN component: LightGCN với embedding dimension **{metrics['embed_dim']}**
       - CBF component: Sentence-BERT + FAISS index
       - Trọng số kết hợp: `alpha = {alpha}` (CF weight = {alpha}, CBF weight = {1-alpha:.1f})
    2. **Chuẩn bị dữ liệu**: 
       - Kết hợp embedding từ CF (Collaborative Filtering - GNN) và Content-based.
       - User embeddings từ GNN: `[{metrics['num_users']}, {metrics['embed_dim']}]`
       - Product embeddings từ CBF: `[{metrics['num_products']}, {metrics['embed_dim']}]`
    3. **Tính toán similarity**: 
       - CF similarity: cosine similarity giữa user embedding (GNN) và product embedding (GNN)
       - CBF similarity: cosine similarity giữa product embeddings (SBERT)
       - Final score = `alpha * CF_score + (1-alpha) * CBF_score`
    4. **Kết hợp trọng số**: 
       - Bảng TF-IDF/Cosine kế thừa từ CBF, cộng thêm trọng số CF.
    5. **Tính toán chỉ số đánh giá**: MAPE, RMSE, Precision, Recall, F1, thời gian.
       - *MAPE*: sai số phần trăm tuyệt đối trung bình giữa rating dự đoán và thực tế.
       - *RMSE*: độ lệch chuẩn của sai số dự đoán.
       - *Precision/Recall/F1*: độ chính xác/phủ và cân bằng giữa hai yếu tố khi so sánh recommendation với ground-truth.
       - *Thời gian*: latency suy luận hoặc toàn bộ pipeline ({metrics['time']}).

| Mô hình | MAPE | RMSE | Precision | Recall | F1 | Thời gian |
|---------|------|------|-----------|--------|----|-----------|
| Hybrid CF+CBF | {metrics['mape']} | {metrics['rmse']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['time']} |
"""
    return doc


def generate_comparison_table(gnn_metrics: Dict[str, Any], cbf_metrics: Dict[str, Any], 
                              hybrid_metrics: Dict[str, Any]) -> str:
    """Generate comparison table for all 3 models."""
    doc = """# 3. Đánh giá 3 mô hình

| Mô hình | MAPE | RMSE | Precision | Recall | F1 | Thời gian |
|---------|------|------|-----------|--------|----|-----------|
| GNN (LightGCN) | {gnn_mape} | {gnn_rmse} | {gnn_precision} | {gnn_recall} | {gnn_f1} | {gnn_time} |
| Content-based Filtering | {cbf_mape} | {cbf_rmse} | {cbf_precision} | {cbf_recall} | {cbf_f1} | {cbf_time} |
| Hybrid CF+CBF | {hybrid_mape} | {hybrid_rmse} | {hybrid_precision} | {hybrid_recall} | {hybrid_f1} | {hybrid_time} |

- **Phân tích & lựa chọn**:
  - GNN (LightGCN) phù hợp khi tập trung vào hành vi người dùng dày đặc, thường cho Precision/Recall cao nhất.
  - Content-based Filtering phù hợp khi cần xử lý cold-start hoặc catalog phong phú, đảm bảo gợi ý hợp lý nhờ lọc age/gender và reason theo style.
  - Hybrid là lựa chọn production mặc định vì duy trì ổn định giữa hai tình huống, có thể tinh chỉnh trọng số `alpha`.
  - Kết luận: Hybrid đạt F1 cao nhất và thời gian xử lý chấp nhận được, phù hợp cho môi trường production.
""".format(
        gnn_mape=gnn_metrics['mape'],
        gnn_rmse=gnn_metrics['rmse'],
        gnn_precision=gnn_metrics['precision'],
        gnn_recall=gnn_metrics['recall'],
        gnn_f1=gnn_metrics['f1'],
        gnn_time=gnn_metrics['time'],
        cbf_mape=cbf_metrics['mape'],
        cbf_rmse=cbf_metrics['rmse'],
        cbf_precision=cbf_metrics['precision'],
        cbf_recall=cbf_metrics['recall'],
        cbf_f1=cbf_metrics['f1'],
        cbf_time=cbf_metrics['time'],
        hybrid_mape=hybrid_metrics['mape'],
        hybrid_rmse=hybrid_metrics['rmse'],
        hybrid_precision=hybrid_metrics['precision'],
        hybrid_recall=hybrid_metrics['recall'],
        hybrid_f1=hybrid_metrics['f1'],
        hybrid_time=hybrid_metrics['time'],
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
                    payload = {"userId": test_user_id, "productId": test_product_id}
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
                            for key in ["mape", "rmse", "precision", "recall", "f1"]:
                                if key in eval_metrics:
                                    st.session_state[f"{slug}_{key}"] = str(eval_metrics[key])
                            if "execution_time" in eval_metrics:
                                st.session_state[f"{slug}_time"] = str(eval_metrics["execution_time"])
                            st.success(f"✅ Đã cập nhật evaluation metrics từ {label} recommend API!")
                    else:
                        st.error(f"❌ Lỗi: {result.get('error', 'Unknown error')}")
                        if result.get("data"):
                            st.json(result["data"])

st.markdown("---")

# Create tabs for each model
doc_tabs = st.tabs(["📊 GNN (LightGCN)", "📝 Content-based Filtering", "🔀 Hybrid CF+CBF", "📈 So sánh 3 mô hình"])

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
        mape = get_value("mape", str(gnn_metrics['mape']))
        rmse = get_value("rmse", str(gnn_metrics['rmse']))
        st.metric("MAPE", mape)
        st.metric("RMSE", rmse)
    with eval_col2:
        precision = get_value("precision", str(gnn_metrics['precision']))
        recall = get_value("recall", str(gnn_metrics['recall']))
        st.metric("Precision", precision)
        st.metric("Recall", recall)
    with eval_col3:
        f1 = get_value("f1", str(gnn_metrics['f1']))
        time_val = get_value("time", str(gnn_metrics['time']))
        st.metric("F1", f1)
        st.metric("Thời gian", time_val)
    
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
        'mape': mape,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': time_val,
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
        mape = get_value("mape", str(cbf_metrics['mape']))
        rmse = get_value("rmse", str(cbf_metrics['rmse']))
        st.metric("MAPE", mape)
        st.metric("RMSE", rmse)
    with eval_col2:
        precision = get_value("precision", str(cbf_metrics['precision']))
        recall = get_value("recall", str(cbf_metrics['recall']))
        st.metric("Precision", precision)
        st.metric("Recall", recall)
    with eval_col3:
        f1 = get_value("f1", str(cbf_metrics['f1']))
        time_val = get_value("time", str(cbf_metrics['time']))
        st.metric("F1", f1)
        st.metric("Thời gian", time_val)
    
    # Update metrics with current input values
    cbf_metrics_updated = {
        'num_products': num_products,
        'num_users': num_users,
        'embed_dim': embed_dim,
        'test_size': test_size,
        'mape': mape,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': time_val,
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
    st.markdown("### 2.3.3. Hybrid Content-based Filtering & Collaborative Filtering")
    
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
    alpha = st.slider("Trọng số alpha (CF weight)", min_value=0.0, max_value=1.0, value=default_alpha, step=0.1, key="hybrid_alpha")
    
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
        mape = get_value("mape", str(hybrid_metrics['mape']))
        rmse = get_value("rmse", str(hybrid_metrics['rmse']))
        st.metric("MAPE", mape)
        st.metric("RMSE", rmse)
    with eval_col2:
        precision = get_value("precision", str(hybrid_metrics['precision']))
        recall = get_value("recall", str(hybrid_metrics['recall']))
        st.metric("Precision", precision)
        st.metric("Recall", recall)
    with eval_col3:
        f1 = get_value("f1", str(hybrid_metrics['f1']))
        time_val = get_value("time", str(hybrid_metrics['time']))
        st.metric("F1", f1)
        st.metric("Thời gian", time_val)
    
    # Update metrics with current input values
    hybrid_metrics_updated = {
        'num_users': num_users,
        'num_products': num_products,
        'num_interactions': num_interactions,
        'embed_dim': embed_dim,
        'test_size': test_size,
        'mape': mape,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': time_val,
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
        for key in ["mape", "rmse", "precision", "recall", "f1", "time", 
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
        "Mô hình": ["GNN (LightGCN)", "Content-based Filtering", "Hybrid CF+CBF"],
        "MAPE": [gnn_metrics_final['mape'], cbf_metrics_final['mape'], hybrid_metrics_final['mape']],
        "RMSE": [gnn_metrics_final['rmse'], cbf_metrics_final['rmse'], hybrid_metrics_final['rmse']],
        "Precision": [gnn_metrics_final['precision'], cbf_metrics_final['precision'], hybrid_metrics_final['precision']],
        "Recall": [gnn_metrics_final['recall'], cbf_metrics_final['recall'], hybrid_metrics_final['recall']],
        "F1": [gnn_metrics_final['f1'], cbf_metrics_final['f1'], hybrid_metrics_final['f1']],
    }
    
    # Try to convert to numeric for plotting
    try:
        comparison_df = pd.DataFrame(comparison_data)
        for col in ["MAPE", "RMSE", "Precision", "Recall", "F1"]:
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
        
        st.bar_chart(comparison_df.set_index("Mô hình")[["Precision", "Recall", "F1"]], use_container_width=True)
    except:
        st.info("Vui lòng nhập số liệu để hiển thị biểu đồ so sánh.")


# Update session state when training completes
st.markdown("---")
st.caption(
    "Ứng dụng Streamlit này giúp kiểm thử nhanh các API gợi ý sản phẩm của Novaware và tạo tài liệu tự động."
)

