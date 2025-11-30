"""
Utility functions for Outfit Recommendation using compatibility scoring.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import combinations


def compute_pairwise_compatibility(
    item_vector_i: np.ndarray,
    item_vector_j: np.ndarray
) -> float:
    """
    Tính điểm tương thích cặp giữa hai sản phẩm.
    
    Công thức:
        Comp(i, j) = cos(v_i, v_j)
    
    Args:
        item_vector_i: Item profile vector của sản phẩm i
        item_vector_j: Item profile vector của sản phẩm j
    
    Returns:
        Điểm tương thích (cosine similarity) trong khoảng [-1, 1]
    """
    dot_product = np.dot(item_vector_i, item_vector_j)
    norm_i = np.linalg.norm(item_vector_i)
    norm_j = np.linalg.norm(item_vector_j)
    
    if norm_i == 0 or norm_j == 0:
        return 0.0
    
    return dot_product / (norm_i * norm_j)


def check_usage_compatibility(
    outfit_products: List,
    products_df: pd.DataFrame
) -> bool:
    """
    Kiểm tra ràng buộc tương thích usage (STRICT).
    Outfit O = {i1, i2, ..., in} là hợp lệ nếu ∀i, j ∈ O: i.usage = j.usage
    
    Args:
        outfit_products: Danh sách product IDs trong outfit
        products_df: DataFrame chứa thông tin sản phẩm
    
    Returns:
        True nếu tất cả sản phẩm có cùng usage, False nếu không
    """
    if 'usage' not in products_df.columns or len(outfit_products) == 0:
        return True
    
    # Lấy usage của các sản phẩm
    if products_df.index.name is not None or (products_df.index.name is None and not isinstance(products_df.index, pd.RangeIndex)):
        usages = products_df.loc[products_df.index.astype(str).isin([str(p) for p in outfit_products]), 'usage'].dropna().unique()
    elif 'id' in products_df.columns:
        usages = products_df[products_df['id'].astype(str).isin([str(p) for p in outfit_products])]['usage'].dropna().unique()
    else:
        return True
    
    # Tất cả phải có cùng usage
    return len(usages) == 1


def check_outfit_structure(
    outfit_products: List,
    products_df: pd.DataFrame,
    payload_product_id: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Kiểm tra ràng buộc cấu trúc danh mục của Outfit.
    
    Ràng buộc:
    1. Outfit O phải chứa i_payload (nếu được chỉ định)
    2. Outfit O phải có đầy đủ các thành phần:
       - Accessories (1/3 subs): ít nhất 1 sản phẩm từ Accessories
       - Bottomwear: ít nhất 1 sản phẩm
       - Topwear: ít nhất 1 sản phẩm
       - Footwear (1/3 subs): ít nhất 1 sản phẩm từ Footwear
    
    Args:
        outfit_products: Danh sách product IDs trong outfit
        products_df: DataFrame chứa thông tin sản phẩm
        payload_product_id: ID của sản phẩm payload (bắt buộc phải có trong outfit)
    
    Returns:
        Tuple (is_valid, reason): True nếu hợp lệ, False và lý do nếu không
    """
    if len(outfit_products) == 0:
        return False, "Outfit rỗng"
    
    # Lấy thông tin sản phẩm
    if products_df.index.name is not None or (products_df.index.name is None and not isinstance(products_df.index, pd.RangeIndex)):
        outfit_df = products_df.loc[products_df.index.astype(str).isin([str(p) for p in outfit_products])]
    elif 'id' in products_df.columns:
        outfit_df = products_df[products_df['id'].astype(str).isin([str(p) for p in outfit_products])]
    else:
        return False, "Không thể truy cập dữ liệu sản phẩm"
    
    if len(outfit_df) == 0:
        return False, "Không tìm thấy thông tin sản phẩm"
    
    # Kiểm tra 1: Phải chứa payload product
    if payload_product_id is not None:
        payload_str = str(payload_product_id)
        if payload_str not in [str(p) for p in outfit_products]:
            return False, f"Outfit không chứa payload product {payload_product_id}"
    
    # Kiểm tra 2: Phải có đầy đủ các thành phần
    if 'subCategory' not in outfit_df.columns and 'masterCategory' not in outfit_df.columns:
        return False, "Thiếu thông tin subCategory/masterCategory"
    
    # Phân loại sản phẩm theo subCategory
    has_accessories = False
    has_bottomwear = False
    has_topwear = False
    has_footwear = False
    
    for _, row in outfit_df.iterrows():
        subcat = str(row.get('subCategory', '')).lower()
        mastercat = str(row.get('masterCategory', '')).lower()
        
        # Kiểm tra Accessories (có thể là subCategory hoặc masterCategory)
        if 'accessor' in subcat or 'accessor' in mastercat:
            has_accessories = True
        # Kiểm tra Bottomwear
        elif 'bottom' in subcat or 'bottom' in mastercat:
            has_bottomwear = True
        # Kiểm tra Topwear
        elif 'top' in subcat or 'top' in mastercat:
            has_topwear = True
        # Kiểm tra Footwear
        elif 'foot' in subcat or 'foot' in mastercat or 'shoe' in subcat:
            has_footwear = True
    
    missing_components = []
    if not has_accessories:
        missing_components.append("Accessories")
    if not has_bottomwear:
        missing_components.append("Bottomwear")
    if not has_topwear:
        missing_components.append("Topwear")
    if not has_footwear:
        missing_components.append("Footwear")
    
    if missing_components:
        return False, f"Thiếu các thành phần: {', '.join(missing_components)}"
    
    return True, "Hợp lệ"


def compute_outfit_score(
    outfit_products: List,
    cbf_scores: Dict,
    encoded_matrix: np.ndarray,
    product_ids: list,
    products_df: pd.DataFrame,
    user_id: str,
    item_weight: float = 1.0,
    compatibility_weight: float = 0.5
) -> float:
    """
    Tính điểm Outfit tổng thể.
    
    Công thức:
        Score(O) = sum(w_i * r_hat_ui^CBF) + sum(w_i,j * Comp(i, j))
    
    Args:
        outfit_products: Danh sách product IDs trong outfit
        cbf_scores: Dictionary mapping (user_id, product_id) -> CBF score
        encoded_matrix: Ma trận encoded features
        product_ids: Danh sách product IDs tương ứng với encoded_matrix
        products_df: DataFrame chứa thông tin sản phẩm
        user_id: ID của user
        item_weight: Trọng số cho phần CBF score (w_i)
        compatibility_weight: Trọng số cho phần compatibility (w_i,j)
    
    Returns:
        Điểm số tổng thể của outfit
    """
    if len(outfit_products) == 0:
        return 0.0
    
    # Tạo mapping product_id -> index (hỗ trợ cả string và int)
    product_id_to_idx = {}
    for idx, pid in enumerate(product_ids):
        # Store as string
        product_id_to_idx[str(pid)] = idx
        # Also store as int if possible
        try:
            pid_int = int(pid)
            product_id_to_idx[pid_int] = idx
        except (ValueError, TypeError):
            pass
    
    # Phần 1: Tổng CBF scores
    cbf_sum = 0.0
    for product_id in outfit_products:
        # Tìm index trong encoded_matrix
        product_idx = product_id_to_idx.get(str(product_id)) or product_id_to_idx.get(product_id)
        if product_idx is None:
            try:
                product_id_int = int(product_id)
                product_idx = product_id_to_idx.get(product_id_int)
            except (ValueError, TypeError):
                pass
        
        if product_idx is None:
            continue
        
        # Lấy CBF score (thử cả string và int key)
        cbf_score = None
        if user_id in cbf_scores:
            cbf_score = cbf_scores[user_id].get(str(product_id)) or cbf_scores[user_id].get(product_id)
            if cbf_score is None:
                try:
                    product_id_int = int(product_id)
                    cbf_score = cbf_scores[user_id].get(product_id_int)
                except (ValueError, TypeError):
                    pass
        
        if cbf_score is not None:
            cbf_sum += item_weight * cbf_score
    
    # Phần 2: Tổng compatibility scores
    compatibility_sum = 0.0
    for i, j in combinations(outfit_products, 2):
        # Tìm indices
        idx_i = product_id_to_idx.get(str(i)) or product_id_to_idx.get(i)
        if idx_i is None:
            try:
                idx_i = product_id_to_idx.get(int(i))
            except (ValueError, TypeError):
                pass
        
        idx_j = product_id_to_idx.get(str(j)) or product_id_to_idx.get(j)
        if idx_j is None:
            try:
                idx_j = product_id_to_idx.get(int(j))
            except (ValueError, TypeError):
                pass
        
        if idx_i is not None and idx_j is not None:
            item_vector_i = encoded_matrix[idx_i]
            item_vector_j = encoded_matrix[idx_j]
            comp_score = compute_pairwise_compatibility(item_vector_i, item_vector_j)
            compatibility_sum += compatibility_weight * comp_score
    
    return cbf_sum + compatibility_sum


def generate_outfit_recommendations(
    candidate_products: List,
    cbf_scores: Dict,
    encoded_matrix: np.ndarray,
    product_ids: list,
    products_df: pd.DataFrame,
    user_id: str,
    payload_product_id: Optional[str] = None,
    outfit_size: int = 4,  # Mặc định 4 để đủ các thành phần
    max_outfits: int = 10,
    item_weight: float = 1.0,
    compatibility_weight: float = 0.5
) -> List[Dict]:
    """
    Tạo danh sách outfit recommendations với các ràng buộc cấu trúc và tương thích.
    
    Ràng buộc:
    1. Outfit O phải chứa i_payload (nếu được chỉ định)
    2. Outfit O phải có đầy đủ: Accessories, Bottomwear, Topwear, Footwear
    3. Tất cả sản phẩm phải có cùng usage
    
    Args:
        candidate_products: Danh sách product IDs ứng viên
        cbf_scores: Dictionary mapping user_id -> {product_id: score}
        encoded_matrix: Ma trận encoded features
        product_ids: Danh sách product IDs
        products_df: DataFrame chứa thông tin sản phẩm
        user_id: ID của user
        payload_product_id: ID của sản phẩm payload (bắt buộc phải có trong outfit)
        outfit_size: Kích thước outfit (số lượng sản phẩm, mặc định 4)
        max_outfits: Số lượng outfit tối đa để trả về
        item_weight: Trọng số cho phần CBF score
        compatibility_weight: Trọng số cho phần compatibility
    
    Returns:
        Danh sách outfits, mỗi outfit là dict chứa:
            - products: List[product_id]
            - score: float
            - cbf_component: float
            - compatibility_component: float
    """
    from itertools import combinations
    
    outfits = []
    
    # Đảm bảo payload product có trong candidate_products
    if payload_product_id is not None:
        payload_str = str(payload_product_id)
        if payload_str not in [str(p) for p in candidate_products]:
            # Thêm payload vào candidate nếu chưa có
            candidate_products = [payload_str] + [p for p in candidate_products if str(p) != payload_str]
    
    # Tạo tất cả các tổ hợp có thể
    for outfit_products in combinations(candidate_products, outfit_size):
        outfit_products_list = list(outfit_products)
        
        # Kiểm tra 1: Usage compatibility (STRICT)
        if not check_usage_compatibility(outfit_products_list, products_df):
            continue
        
        # Kiểm tra 2: Cấu trúc danh mục (phải có đầy đủ thành phần và payload)
        is_valid, reason = check_outfit_structure(outfit_products_list, products_df, payload_product_id)
        if not is_valid:
            continue
        
        # Tính điểm
        score = compute_outfit_score(
            list(outfit_products),
            cbf_scores,
            encoded_matrix,
            product_ids,
            products_df,
            user_id,
            item_weight,
            compatibility_weight
        )
        
        # Tính riêng từng component để hiển thị
        cbf_component = 0.0
        for product_id in outfit_products:
            if user_id in cbf_scores and product_id in cbf_scores[user_id]:
                cbf_component += item_weight * cbf_scores[user_id][product_id]
        
        compatibility_component = 0.0
        # Tạo mapping product_id -> index (hỗ trợ cả string và int)
        product_id_to_idx_local = {}
        for idx, pid in enumerate(product_ids):
            product_id_to_idx_local[str(pid)] = idx
            try:
                product_id_to_idx_local[int(pid)] = idx
            except (ValueError, TypeError):
                pass
        
        for i, j in combinations(outfit_products, 2):
            # Tìm indices
            idx_i = product_id_to_idx_local.get(str(i)) or product_id_to_idx_local.get(i)
            if idx_i is None:
                try:
                    idx_i = product_id_to_idx_local.get(int(i))
                except (ValueError, TypeError):
                    pass
            
            idx_j = product_id_to_idx_local.get(str(j)) or product_id_to_idx_local.get(j)
            if idx_j is None:
                try:
                    idx_j = product_id_to_idx_local.get(int(j))
                except (ValueError, TypeError):
                    pass
            
            if idx_i is not None and idx_j is not None:
                comp_score = compute_pairwise_compatibility(
                    encoded_matrix[idx_i],
                    encoded_matrix[idx_j]
                )
                compatibility_component += compatibility_weight * comp_score
        
        outfits.append({
            'products': list(outfit_products),
            'score': score,
            'cbf_component': cbf_component,
            'compatibility_component': compatibility_component
        })
    
    # Sắp xếp theo điểm số giảm dần và lấy top-K
    outfits.sort(key=lambda x: x['score'], reverse=True)
    
    return outfits[:max_outfits]

