"""
Utility functions for building weighted user profiles for Content-Based Filtering.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


# Trọng số theo loại tương tác
INTERACTION_WEIGHTS = {
    'purchase': 5.0,
    'like': 3.0,
    'cart': 2.0,
    'view': 1.0
}


def get_interaction_weight(interaction_type: str) -> float:
    """
    Lấy trọng số tương tác theo loại.
    
    Args:
        interaction_type: Loại tương tác (purchase, like, cart, view)
    
    Returns:
        Trọng số tương tác (mặc định 1.0 nếu không tìm thấy)
    """
    return INTERACTION_WEIGHTS.get(interaction_type.lower(), 1.0)


def build_weighted_user_profile(
    interactions_df: pd.DataFrame,
    encoded_matrix: np.ndarray,
    product_ids: list,
    interaction_weights: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Xây dựng hồ sơ người dùng có trọng số từ các tương tác và item profiles.
    
    Công thức:
        P_u = (sum(w_ui * v_i)) / (sum(w_ui))
    
    Trong đó:
        - P_u: Vector hồ sơ người dùng u
        - w_ui: Trọng số tương tác giữa user u và item i
        - v_i: Item profile vector của sản phẩm i
        - I_u^+: Tập các sản phẩm user u đã tương tác
    
    Args:
        interactions_df: DataFrame chứa interactions (user_id, product_id, interaction_type)
        encoded_matrix: Ma trận encoded features (n_products x n_features)
        product_ids: Danh sách product IDs tương ứng với encoded_matrix
        interaction_weights: Dictionary mapping interaction_type -> weight (optional)
    
    Returns:
        Dictionary chứa:
            - user_profiles: Dict[user_id, np.ndarray] - Vector profile cho mỗi user
            - user_stats: Dict[user_id, Dict] - Thống kê cho mỗi user
            - total_users: int - Tổng số users
            - feature_dim: int - Số chiều của feature vector
    """
    if interactions_df.empty:
        return {
            'user_profiles': {},
            'user_stats': {},
            'total_users': 0,
            'feature_dim': encoded_matrix.shape[1] if len(encoded_matrix.shape) > 1 else 0
        }
    
    if 'user_id' not in interactions_df.columns or 'product_id' not in interactions_df.columns:
        raise ValueError("interactions_df phải có columns 'user_id' và 'product_id'")
    
    # Sử dụng interaction_weights nếu được cung cấp, nếu không dùng mặc định
    weights = interaction_weights if interaction_weights is not None else INTERACTION_WEIGHTS
    
    # Tạo mapping từ product_id -> index trong encoded_matrix
    # Hỗ trợ cả trường hợp product_ids là list index hoặc list id
    product_id_to_idx = {}
    for idx, pid in enumerate(product_ids):
        # Cho phép mapping theo cả giá trị và kiểu dữ liệu
        product_id_to_idx[pid] = idx
        # Nếu pid là số, cũng thử convert sang int/str để matching
        try:
            if isinstance(pid, (int, float)):
                product_id_to_idx[int(pid)] = idx
        except:
            pass
    
    # Lấy interaction_type nếu có, nếu không mặc định là 'view'
    has_interaction_type = 'interaction_type' in interactions_df.columns
    
    user_profiles = {}
    user_stats = {}
    skipped_products = set()
    
    # Nhóm interactions theo user
    for user_id, user_interactions in interactions_df.groupby('user_id'):
        weighted_sum = np.zeros(encoded_matrix.shape[1])
        total_weight = 0.0
        interaction_count = 0
        
        for _, interaction in user_interactions.iterrows():
            product_id = interaction['product_id']
            
            # Thử convert product_id để matching
            product_idx = None
            if product_id in product_id_to_idx:
                product_idx = product_id_to_idx[product_id]
            else:
                # Thử convert sang int hoặc str
                try:
                    product_id_int = int(product_id)
                    if product_id_int in product_id_to_idx:
                        product_idx = product_id_to_idx[product_id_int]
                except:
                    pass
                
                if product_idx is None:
                    skipped_products.add(product_id)
                    continue
            
            item_vector = encoded_matrix[product_idx]
            
            # Lấy trọng số
            if has_interaction_type:
                interaction_type = str(interaction.get('interaction_type', 'view')).lower()
                weight = weights.get(interaction_type, 1.0)
            else:
                weight = 1.0
            
            # Cộng dồn: w_ui * v_i
            weighted_sum += weight * item_vector
            total_weight += weight
            interaction_count += 1
        
        # Tính vector profile: P_u = (sum(w_ui * v_i)) / (sum(w_ui))
        if total_weight > 0:
            user_profile = weighted_sum / total_weight
            user_profiles[user_id] = user_profile
            
            # Thống kê
            user_stats[user_id] = {
                'interaction_count': interaction_count,
                'total_weight': total_weight,
                'avg_weight': total_weight / interaction_count if interaction_count > 0 else 0
            }
        else:
            # User không có interaction hợp lệ
            user_stats[user_id] = {
                'interaction_count': 0,
                'total_weight': 0.0,
                'avg_weight': 0.0
            }
    
    return {
        'user_profiles': user_profiles,
        'user_stats': user_stats,
        'total_users': len(user_profiles),
        'feature_dim': encoded_matrix.shape[1] if len(encoded_matrix.shape) > 1 else 0,
        'skipped_products': len(skipped_products),
        'skipped_product_ids': list(skipped_products)[:10]  # Chỉ lấy 10 đầu tiên để hiển thị
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tính cosine similarity giữa hai vectors.
    
    Công thức:
        cos(a, b) = (a · b) / (||a|| * ||b||)
    
    Args:
        a: Vector thứ nhất
        b: Vector thứ hai
    
    Returns:
        Cosine similarity score trong khoảng [-1, 1]
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def compute_cbf_predictions(
    user_profiles: Dict,
    encoded_matrix: np.ndarray,
    product_ids: list,
    top_k: int = 20
) -> Dict:
    """
    Tính điểm dự đoán CBF cho tất cả users và sản phẩm.
    
    Công thức:
        r_hat_ui^CBF = cos(P_u, v_i) = (P_u · v_i) / (||P_u|| * ||v_i||)
    
    Args:
        user_profiles: Dictionary mapping user_id -> user profile vector
        encoded_matrix: Ma trận encoded features (n_products x n_features)
        product_ids: Danh sách product IDs tương ứng với encoded_matrix
        top_k: Số lượng sản phẩm top-K để xếp hạng
    
    Returns:
        Dictionary chứa:
            - predictions: Dict[user_id, Dict[product_id, score]] - Điểm dự đoán cho mỗi user-product pair
            - rankings: Dict[user_id, List[Tuple[product_id, score]]] - Top-K rankings cho mỗi user
            - stats: Dict - Thống kê về predictions
    """
    predictions = {}
    rankings = {}
    
    # Tính predictions cho mỗi user
    for user_id, user_profile in user_profiles.items():
        user_predictions = {}
        
        # Tính cosine similarity với tất cả sản phẩm
        for idx, product_id in enumerate(product_ids):
            item_vector = encoded_matrix[idx]
            score = cosine_similarity(user_profile, item_vector)
            user_predictions[product_id] = score
        
        predictions[user_id] = user_predictions
        
        # Xếp hạng và lấy top-K
        sorted_predictions = sorted(
            user_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        rankings[user_id] = sorted_predictions[:top_k]
    
    # Tính thống kê
    all_scores = []
    for user_preds in predictions.values():
        all_scores.extend(user_preds.values())
    
    stats = {
        'total_predictions': len(all_scores),
        'min_score': float(np.min(all_scores)) if all_scores else 0.0,
        'max_score': float(np.max(all_scores)) if all_scores else 0.0,
        'mean_score': float(np.mean(all_scores)) if all_scores else 0.0,
        'std_score': float(np.std(all_scores)) if all_scores else 0.0,
        'total_users': len(predictions),
        'total_products': len(product_ids)
    }
    
    return {
        'predictions': predictions,
        'rankings': rankings,
        'stats': stats
    }

