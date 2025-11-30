"""
Utility functions for evaluating recommendation system performance.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
import time


def recall_at_k(recommended_items: List, relevant_items: Set, k: int) -> float:
    """
    Tính Recall@K.
    
    Recall@K = |relevant_items ∩ recommended_items[:k]| / |relevant_items|
    
    Args:
        recommended_items: Danh sách items được đề xuất (đã sắp xếp)
        relevant_items: Set các items thực sự relevant (ground truth)
        k: Số lượng items top-K
    
    Returns:
        Recall@K trong khoảng [0, 1]
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    hits = len(set(top_k) & relevant_items)
    
    return hits / len(relevant_items)


def precision_at_k(recommended_items: List, relevant_items: Set, k: int) -> float:
    """
    Tính Precision@K.
    
    Precision@K = |relevant_items ∩ recommended_items[:k]| / k
    
    Args:
        recommended_items: Danh sách items được đề xuất (đã sắp xếp)
        relevant_items: Set các items thực sự relevant (ground truth)
        k: Số lượng items top-K
    
    Returns:
        Precision@K trong khoảng [0, 1]
    """
    if k == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    hits = len(set(top_k) & relevant_items)
    
    return hits / k


def dcg_at_k(recommended_items: List, relevant_items: Set, k: int) -> float:
    """
    Tính Discounted Cumulative Gain (DCG)@K.
    
    Công thức:
        DCG@K = sum_{i=1}^{K} (2^rel(i) - 1) / log2(i + 1)
    
    Trong đó:
        - rel(i) = 1 nếu item thứ i là relevant, 0 nếu không
        - 2^rel(i) - 1 = 1 nếu relevant, 0 nếu không (với binary relevance)
    
    Args:
        recommended_items: Danh sách items được đề xuất (đã sắp xếp)
        relevant_items: Set các items thực sự relevant (ground truth)
        k: Số lượng items top-K
    
    Returns:
        DCG@K (giá trị >= 0)
    """
    top_k = recommended_items[:k]
    dcg = 0.0
    
    for i, item in enumerate(top_k, start=1):
        if item in relevant_items:
            # rel(i) = 1, nên 2^1 - 1 = 1
            rel_score = 1.0
            dcg += (2.0 ** rel_score - 1.0) / np.log2(i + 1)
        # Nếu không relevant, rel(i) = 0, nên 2^0 - 1 = 0, không cộng vào
    
    return dcg


def ndcg_at_k(recommended_items: List, relevant_items: Set, k: int) -> float:
    """
    Tính Normalized Discounted Cumulative Gain (NDCG)@K.
    
    NDCG@K = DCG@K / IDCG@K
    Trong đó IDCG@K là DCG@K lý tưởng (khi tất cả items top-K đều relevant)
    
    Args:
        recommended_items: Danh sách items được đề xuất (đã sắp xếp)
        relevant_items: Set các items thực sự relevant (ground truth)
        k: Số lượng items top-K
    
    Returns:
        NDCG@K trong khoảng [0, 1]
    """
    dcg = dcg_at_k(recommended_items, relevant_items, k)
    
    # Tính IDCG: giả sử k items đầu tiên đều relevant
    # IDCG@K = sum_{i=1}^{min(K, |relevant|)} (2^1 - 1) / log2(i + 1)
    idcg = 0.0
    num_relevant = min(k, len(relevant_items))
    for i in range(1, num_relevant + 1):
        idcg += (2.0 ** 1.0 - 1.0) / np.log2(i + 1)  # = 1.0 / log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def ild_at_k(recommended_items: List, item_features: np.ndarray, item_ids: List, k: int) -> float:
    """
    Tính Intra-List Diversity (ILD)@K.
    
    Công thức:
        ILD@K = (2 / (K(K-1))) * sum_{i in L(u)} sum_{j in L(u), j>i} (1 - cos(v_i, v_j))
    
    Args:
        recommended_items: Danh sách items được đề xuất (có thể là list of product_ids hoặc list of (product_id, score) tuples)
        item_features: Ma trận features của items
        item_ids: Danh sách item IDs tương ứng với item_features
        k: Số lượng items top-K
    
    Returns:
        ILD@K (giá trị >= 0, cao hơn = đa dạng hơn)
    """
    if len(recommended_items) < 2 or k < 2:
        return 0.0
    
    # Lấy top-K items (nếu là tuples, lấy product_id)
    top_k_items = []
    for item in recommended_items[:k]:
        if isinstance(item, tuple):
            top_k_items.append(item[0])  # Lấy product_id từ tuple
        else:
            top_k_items.append(item)
    
    if len(top_k_items) < 2:
        return 0.0
    
    # Tạo mapping product_id -> index
    item_id_to_idx = {}
    for idx, item_id in enumerate(item_ids):
        item_id_to_idx[str(item_id)] = idx
        try:
            item_id_to_idx[int(item_id)] = idx
        except (ValueError, TypeError):
            pass
    
    # Tính tổng (1 - cosine similarity) cho tất cả các cặp (i, j) với j > i
    total_dissimilarity = 0.0
    pair_count = 0
    
    for i, item_i in enumerate(top_k_items):
        for j, item_j in enumerate(top_k_items):
            if j <= i:  # Chỉ tính các cặp với j > i
                continue
            
            # Tìm indices
            idx_i = item_id_to_idx.get(str(item_i)) or item_id_to_idx.get(item_i)
            if idx_i is None:
                try:
                    idx_i = item_id_to_idx.get(int(item_i))
                except (ValueError, TypeError):
                    continue
            
            idx_j = item_id_to_idx.get(str(item_j)) or item_id_to_idx.get(item_j)
            if idx_j is None:
                try:
                    idx_j = item_id_to_idx.get(int(item_j))
                except (ValueError, TypeError):
                    continue
            
            if idx_i is not None and idx_j is not None:
                vec_i = item_features[idx_i]
                vec_j = item_features[idx_j]
                
                # Tính cosine similarity
                dot_product = np.dot(vec_i, vec_j)
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i > 0 and norm_j > 0:
                    cosine_sim = dot_product / (norm_i * norm_j)
                    dissimilarity = 1.0 - cosine_sim
                    total_dissimilarity += dissimilarity
                    pair_count += 1
    
    if pair_count == 0:
        return 0.0
    
    # ILD@K = (2 / (K(K-1))) * total_dissimilarity
    actual_k = len(top_k_items)
    ild = (2.0 / (actual_k * (actual_k - 1))) * total_dissimilarity if actual_k > 1 else 0.0
    
    return max(0.0, ild)


def diversity(recommended_items_list: List[List], item_features: Optional[np.ndarray] = None, 
              item_ids: Optional[List] = None) -> float:
    """
    Tính Diversity của recommendations.
    
    Diversity = 1 - (average pairwise similarity giữa các items được đề xuất)
    
    Nếu có item_features, sử dụng cosine similarity.
    Nếu không, sử dụng Jaccard similarity dựa trên sets.
    
    Args:
        recommended_items_list: List of lists, mỗi list là recommendations cho một user
        item_features: Ma trận features của items (optional)
        item_ids: Danh sách item IDs tương ứng với item_features (optional)
    
    Returns:
        Diversity trong khoảng [0, 1] (cao hơn = đa dạng hơn)
    """
    if not recommended_items_list:
        return 0.0
    
    all_recommended = set()
    for rec_list in recommended_items_list:
        all_recommended.update(rec_list)
    
    if len(all_recommended) < 2:
        return 1.0  # Nếu chỉ có 1 item, diversity = 1
    
    # Tính average pairwise similarity
    similarities = []
    
    if item_features is not None and item_ids is not None:
        # Sử dụng cosine similarity với features
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        
        for i, item_i in enumerate(all_recommended):
            for j, item_j in enumerate(all_recommended):
                if i >= j:
                    continue
                
                idx_i = item_id_to_idx.get(item_i)
                idx_j = item_id_to_idx.get(item_j)
                
                if idx_i is not None and idx_j is not None:
                    vec_i = item_features[idx_i]
                    vec_j = item_features[idx_j]
                    
                    dot_product = np.dot(vec_i, vec_j)
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = dot_product / (norm_i * norm_j)
                        similarities.append(similarity)
    else:
        # Sử dụng Jaccard similarity đơn giản
        # Với diversity, chúng ta tính average pairwise dissimilarity
        # Giả sử mỗi item là unique, similarity = 0, diversity = 1
        return 1.0
    
    if not similarities:
        return 1.0
    
    avg_similarity = np.mean(similarities)
    diversity_score = 1.0 - avg_similarity
    
    return max(0.0, min(1.0, diversity_score))


def coverage(recommended_items_list: List[List], all_items: Set) -> float:
    """
    Tính Coverage của recommendations.
    
    Coverage = |unique items được đề xuất| / |tất cả items có thể|
    
    Args:
        recommended_items_list: List of lists, mỗi list là recommendations cho một user
        all_items: Set tất cả items có thể được đề xuất
    
    Returns:
        Coverage trong khoảng [0, 1]
    """
    if len(all_items) == 0:
        return 0.0
    
    recommended_set = set()
    for rec_list in recommended_items_list:
        recommended_set.update(rec_list)
    
    return len(recommended_set) / len(all_items)


def compute_cbf_metrics(
    predictions: Dict[str, List[Tuple]],  # user_id -> [(product_id, score), ...]
    ground_truth: Dict[str, Set],  # user_id -> set of relevant product_ids
    k_values: List[int] = [5, 10, 20],
    item_features: Optional[np.ndarray] = None,
    item_ids: Optional[List] = None,
    all_items: Optional[Set] = None,
    training_time: Optional[float] = None,
    inference_time: Optional[float] = None,
    use_ild: bool = True
) -> Dict:
    """
    Tính toán tất cả các metrics cho CBF model.
    
    Args:
        predictions: Dictionary mapping user_id -> sorted list of (product_id, score) tuples
        ground_truth: Dictionary mapping user_id -> set of relevant product_ids
        k_values: List các giá trị K để tính metrics
        item_features: Ma trận features của items (cho diversity)
        item_ids: Danh sách item IDs (cho diversity)
        all_items: Set tất cả items có thể (cho coverage)
        training_time: Thời gian training (giây)
        inference_time: Thời gian inference (giây)
    
    Returns:
        Dictionary chứa tất cả metrics
    """
    metrics = {
        'k_values': k_values,
        'recall': {},
        'precision': {},
        'ndcg': {},
        'diversity': None,
        'coverage': None,
        'training_time': training_time,
        'inference_time': inference_time
    }
    
    # Tính Recall, Precision, NDCG cho từng K
    for k in k_values:
        recalls = []
        precisions = []
        ndcgs = []
        
        for user_id, pred_list in predictions.items():
            if user_id not in ground_truth:
                continue
            
            # Extract recommended items và chuẩn hóa về string
            recommended_items = []
            for item in pred_list:
                if isinstance(item, tuple):
                    item_id = item[0]
                else:
                    item_id = item
                # Chuẩn hóa về string để matching
                recommended_items.append(str(item_id))
            
            # Chuẩn hóa relevant items về string
            relevant_items = {str(item_id) for item_id in ground_truth[user_id]}
            
            if len(relevant_items) == 0:
                continue
            
            recalls.append(recall_at_k(recommended_items, relevant_items, k))
            precisions.append(precision_at_k(recommended_items, relevant_items, k))
            ndcgs.append(ndcg_at_k(recommended_items, relevant_items, k))
        
        metrics['recall'][k] = np.mean(recalls) if recalls else 0.0
        metrics['precision'][k] = np.mean(precisions) if precisions else 0.0
        metrics['ndcg'][k] = np.mean(ndcgs) if ndcgs else 0.0
    
    # Tạo recommended_items_list cho Diversity và Coverage
    recommended_items_list = []
    for pred_list in predictions.values():
        if len(pred_list) > 0 and isinstance(pred_list[0], tuple):
            recommended_items_list.append([item_id for item_id, _ in pred_list])
        else:
            recommended_items_list.append(pred_list)
    
    # Tính Diversity (ILD@K hoặc average diversity)
    if recommended_items_list and item_features is not None and item_ids is not None:
        if use_ild and len(k_values) > 0:
            # Tính ILD@K cho K lớn nhất, trung bình qua tất cả users
            max_k = max(k_values)
            ild_scores = []
            for pred_list in predictions.values():
                # pred_list có thể là list of (product_id, score) hoặc list of product_ids
                if len(pred_list) > 0 and isinstance(pred_list[0], tuple):
                    recommended_items = pred_list  # Giữ nguyên tuples để ild_at_k xử lý
                else:
                    recommended_items = pred_list
                
                if len(recommended_items) >= 2:
                    ild_score = ild_at_k(recommended_items, item_features, item_ids, min(max_k, len(recommended_items)))
                    ild_scores.append(ild_score)
            metrics['diversity'] = np.mean(ild_scores) if ild_scores else 0.0
        else:
            # Fallback to old diversity calculation
            recommended_items_list_flat = []
            for pred_list in predictions.values():
                if len(pred_list) > 0 and isinstance(pred_list[0], tuple):
                    recommended_items_list_flat.append([item_id for item_id, _ in pred_list])
                else:
                    recommended_items_list_flat.append(pred_list)
            metrics['diversity'] = diversity(recommended_items_list_flat, item_features, item_ids)
    
    # Tính Coverage
    if all_items is not None:
        metrics['coverage'] = coverage(recommended_items_list, all_items)
    
    return metrics

