"""
Utility functions for Hybrid (GNN + CBF) model operations.
Includes combining scores from GNN and CBF models.
"""

from typing import Dict, List, Tuple, Optional


def combine_hybrid_scores(
    cbf_predictions: Dict,
    gnn_predictions: Dict,
    alpha: float = 0.5,
    top_k: int = 20
) -> Dict:
    """
    Kết hợp tuyến tính điểm dự đoán của GNN và CBF.
    
    Công thức:
        Score_Hybrid(u, i) = α * r_hat_GNN(u, i) + (1 - α) * r_hat_CBF(u, i)
    
    Args:
        cbf_predictions: Dictionary từ CBF predictions (có 'predictions' key)
        gnn_predictions: Dictionary từ GNN predictions (có 'predictions' hoặc 'rankings' key)
        alpha: Trọng số kết hợp (0 ≤ α ≤ 1)
        top_k: Số lượng sản phẩm Top-K để xếp hạng
    
    Returns:
        Dictionary chứa:
            - predictions: Dict[user_id, Dict[product_id, hybrid_score]]
            - rankings: Dict[user_id, List[Tuple[product_id, hybrid_score]]]
            - alpha: Trọng số đã sử dụng
            - stats: Statistics về score ranges
    """
    hybrid_predictions = {}
    hybrid_rankings = {}
    
    # Get CBF predictions dict
    cbf_preds = cbf_predictions.get('predictions', {})
    
    # Get GNN predictions dict
    gnn_preds = {}
    if 'predictions' in gnn_predictions:
        gnn_preds = gnn_predictions['predictions']
    elif 'rankings' in gnn_predictions:
        # Convert rankings to predictions dict
        for user_id, ranking in gnn_predictions['rankings'].items():
            gnn_preds[user_id] = {pid: score for pid, score in ranking}
    
    # Get all users (union of CBF and GNN users)
    all_users = set(cbf_preds.keys()) | set(gnn_preds.keys())
    
    if not all_users:
        return {
            'predictions': {},
            'rankings': {},
            'alpha': alpha,
            'stats': {
                'total_users': 0,
                'cbf_min': 0.0,
                'cbf_max': 0.0,
                'gnn_min': 0.0,
                'gnn_max': 0.0
            }
        }
    
    # Normalize scores for each model
    all_cbf_scores = []
    all_gnn_scores = []
    
    for user_id in all_users:
        if user_id in cbf_preds:
            all_cbf_scores.extend(cbf_preds[user_id].values())
        if user_id in gnn_preds:
            all_gnn_scores.extend(gnn_preds[user_id].values())
    
    cbf_min = min(all_cbf_scores) if all_cbf_scores else 0.0
    cbf_max = max(all_cbf_scores) if all_cbf_scores else 1.0
    cbf_range = cbf_max - cbf_min if cbf_max != cbf_min else 1.0
    
    gnn_min = min(all_gnn_scores) if all_gnn_scores else 0.0
    gnn_max = max(all_gnn_scores) if all_gnn_scores else 1.0
    gnn_range = gnn_max - gnn_min if gnn_max != gnn_min else 1.0
    
    # Combine scores
    for user_id in all_users:
        user_hybrid_preds = {}
        
        # Get all products for this user
        cbf_products = set(cbf_preds.get(user_id, {}).keys())
        gnn_products = set(gnn_preds.get(user_id, {}).keys())
        all_products = cbf_products | gnn_products
        
        for product_id in all_products:
            # Get normalized scores
            cbf_score = cbf_preds.get(user_id, {}).get(product_id, 0.0)
            gnn_score = gnn_preds.get(user_id, {}).get(product_id, 0.0)
            
            # Normalize to [0, 1]
            cbf_norm = (cbf_score - cbf_min) / cbf_range if cbf_range > 0 else 0.0
            gnn_norm = (gnn_score - gnn_min) / gnn_range if gnn_range > 0 else 0.0
            
            # Combine linearly
            hybrid_score = alpha * gnn_norm + (1 - alpha) * cbf_norm
            user_hybrid_preds[product_id] = hybrid_score
        
        hybrid_predictions[user_id] = user_hybrid_preds
        
        # Rank and get top-K
        sorted_preds = sorted(
            user_hybrid_preds.items(),
            key=lambda x: x[1],
            reverse=True
        )
        hybrid_rankings[user_id] = sorted_preds[:top_k]
    
    return {
        'predictions': hybrid_predictions,
        'rankings': hybrid_rankings,
        'alpha': alpha,
        'stats': {
            'total_users': len(all_users),
            'cbf_min': cbf_min,
            'cbf_max': cbf_max,
            'gnn_min': gnn_min,
            'gnn_max': gnn_max
        }
    }

