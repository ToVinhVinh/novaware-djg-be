"""
Utility functions for Content-Based Filtering: Personalized Filtering and Outfit Recommendation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations


def apply_articletype_filter(
    candidate_products: List,
    products_df: pd.DataFrame,
    payload_articletype: str
) -> List:
    """
    Lọc cứng theo articleType (STRICT).
    Chỉ giữ lại các sản phẩm có articleType trùng khớp với articleType của sản phẩm đầu vào.
    
    Args:
        candidate_products: Danh sách product IDs ứng viên
        products_df: DataFrame chứa thông tin sản phẩm
        payload_articletype: articleType của sản phẩm đầu vào
    
    Returns:
        Danh sách product IDs sau khi lọc
    """
    if 'articleType' not in products_df.columns:
        return candidate_products
    
    # Convert candidate_products to strings for comparison
    candidate_products_str = [str(p) for p in candidate_products]
    
    # Check if index is set (not default RangeIndex)
    if products_df.index.name is not None or (products_df.index.name is None and not isinstance(products_df.index, pd.RangeIndex)):
        # Index is set, use index
        filtered = products_df[
            products_df.index.astype(str).isin(candidate_products_str) &
            (products_df['articleType'] == payload_articletype)
        ]
        return filtered.index.astype(str).tolist()
    elif 'id' in products_df.columns:
        # Use 'id' column
        filtered = products_df[
            products_df['id'].astype(str).isin(candidate_products_str) &
            (products_df['articleType'] == payload_articletype)
        ]
        return filtered['id'].astype(str).tolist()
    else:
        # Fallback: return original if structure is unexpected
        return candidate_products


def get_allowed_genders(user_age: Optional[int], user_gender: Optional[str]) -> List[str]:
    """
    Xác định các gender được phép dựa trên age và gender của user (STRICT FILTERING).
    
    Logic Áp dụng (Strict Filtering):
    - Nếu u.age < 13 và u.gender = 'male': i_cand.gender phải là 'Boys'
    - Nếu u.age < 13 và u.gender = 'female': i_cand.gender phải là 'Girls'
    - Nếu u.age >= 13 và u.gender = 'male': i_cand.gender phải là 'Men' hoặc 'Unisex'
    - Nếu u.age >= 13 và u.gender = 'female': i_cand.gender phải là 'Women' hoặc 'Unisex'
    
    Args:
        user_age: Tuổi của user (có thể None)
        user_gender: Giới tính của user ('male' hoặc 'female', có thể None)
    
    Returns:
        Danh sách các gender được phép
    """
    if user_age is None or user_gender is None:
        # Nếu thiếu thông tin, cho phép tất cả
        return ['Men', 'Women', 'Boys', 'Girls', 'Unisex']
    
    user_gender_lower = str(user_gender).lower()
    
    if user_age < 13:
        # Trẻ em - STRICT: chỉ giữ đúng gender
        if user_gender_lower == 'male':
            return ['Boys']  # STRICT: chỉ Boys, không có Unisex
        elif user_gender_lower == 'female':
            return ['Girls']  # STRICT: chỉ Girls, không có Unisex
        else:
            return ['Boys', 'Girls']  # Nếu gender không rõ, cho phép cả hai
    else:
        # Người lớn - cho phép Unisex
        if user_gender_lower == 'male':
            return ['Men', 'Unisex']
        elif user_gender_lower == 'female':
            return ['Women', 'Unisex']
        else:
            return ['Men', 'Women', 'Unisex']


def apply_age_gender_filter(
    candidate_products: List,
    products_df: pd.DataFrame,
    user_age: Optional[int] = None,
    user_gender: Optional[str] = None
) -> List:
    """
    Lọc cứng theo Giới tính/Độ tuổi (Age/Gender Priority).
    
    Args:
        candidate_products: Danh sách product IDs ứng viên
        products_df: DataFrame chứa thông tin sản phẩm
        user_age: Tuổi của user
        user_gender: Giới tính của user
    
    Returns:
        Danh sách product IDs sau khi lọc
    """
    if 'gender' not in products_df.columns:
        return candidate_products
    
    allowed_genders = get_allowed_genders(user_age, user_gender)
    
    # Convert candidate_products to strings for comparison
    candidate_products_str = [str(p) for p in candidate_products]
    
    # Check if index is set (not default RangeIndex)
    if products_df.index.name is not None or (products_df.index.name is None and not isinstance(products_df.index, pd.RangeIndex)):
        # Index is set, use index
        filtered = products_df[
            products_df.index.astype(str).isin(candidate_products_str) &
            (products_df['gender'].isin(allowed_genders))
        ]
        return filtered.index.astype(str).tolist()
    elif 'id' in products_df.columns:
        # Use 'id' column
        filtered = products_df[
            products_df['id'].astype(str).isin(candidate_products_str) &
            (products_df['gender'].isin(allowed_genders))
        ]
        return filtered['id'].astype(str).tolist()
    else:
        # Fallback: return original if structure is unexpected
        return candidate_products


def apply_personalized_filters(
    candidate_products: List,
    products_df: pd.DataFrame,
    payload_articletype: Optional[str] = None,
    user_age: Optional[int] = None,
    user_gender: Optional[str] = None,
    cbf_scores: Optional[Dict] = None,
    top_k: Optional[int] = None
) -> Dict:
    """
    Áp dụng tất cả các bộ lọc cá nhân hóa và xếp hạng Top-K.
    
    Quy trình:
    1. Lọc Cứng theo articleType (STRICT)
    2. Lọc Cứng theo Age/Gender (STRICT)
    3. Xếp hạng bằng điểm CBF để tạo Top-K Personalized
    
    Args:
        candidate_products: Danh sách product IDs ứng viên ban đầu
        products_df: DataFrame chứa thông tin sản phẩm
        payload_articletype: articleType của sản phẩm đầu vào (nếu có)
        user_age: Tuổi của user
        user_gender: Giới tính của user
        cbf_scores: Dictionary mapping product_id -> CBF score (để xếp hạng)
        top_k: Số lượng sản phẩm Top-K (nếu None, trả về tất cả sau khi lọc)
    
    Returns:
        Dictionary chứa:
            - filtered_products: Danh sách product IDs sau khi lọc và xếp hạng
            - ranked_products: Danh sách (product_id, score) đã sắp xếp
            - stats: Thống kê về quá trình lọc
    """
    stats = {
        'initial_count': len(candidate_products),
        'after_articletype': len(candidate_products),
        'after_age_gender': len(candidate_products),
        'after_ranking': len(candidate_products),
        'final_count': len(candidate_products)
    }
    
    filtered = candidate_products.copy()
    
    # Lọc 1: articleType (STRICT)
    if payload_articletype:
        filtered = apply_articletype_filter(filtered, products_df, payload_articletype)
        stats['after_articletype'] = len(filtered)
    
    # Lọc 2: Age/Gender Priority (STRICT)
    if user_age is not None or user_gender is not None:
        filtered = apply_age_gender_filter(filtered, products_df, user_age, user_gender)
        stats['after_age_gender'] = len(filtered)
    
    # Bước 3: Xếp hạng bằng điểm CBF (nếu có)
    ranked_products = []
    if cbf_scores is not None and filtered:
        # Tạo danh sách (product_id, score) và sắp xếp
        product_scores = []
        for product_id in filtered:
            # Thử cả string và int key
            score = cbf_scores.get(str(product_id)) or cbf_scores.get(product_id)
            if score is None:
                try:
                    score = cbf_scores.get(int(product_id))
                except (ValueError, TypeError):
                    score = 0.0  # Nếu không tìm thấy score, đặt = 0
            product_scores.append((product_id, score))
        
        # Sắp xếp theo điểm giảm dần
        product_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_products = product_scores
        
        # Lấy Top-K nếu được chỉ định
        if top_k is not None and top_k > 0:
            ranked_products = ranked_products[:top_k]
            filtered = [pid for pid, _ in ranked_products]
        else:
            filtered = [pid for pid, _ in ranked_products]
    else:
        # Nếu không có CBF scores, chỉ trả về danh sách đã lọc
        ranked_products = [(pid, 0.0) for pid in filtered]
    
    stats['after_ranking'] = len(filtered)
    stats['final_count'] = len(filtered)
    stats['removed_count'] = stats['initial_count'] - stats['final_count']
    stats['reduction_rate'] = (stats['removed_count'] / stats['initial_count'] * 100) if stats['initial_count'] > 0 else 0
    
    return {
        'filtered_products': filtered,
        'ranked_products': ranked_products,
        'stats': stats
    }

