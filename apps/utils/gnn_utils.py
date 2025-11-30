"""
Utility functions for GNN (Graph Neural Network) model operations.
Includes graph construction, message propagation, predictions, and training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Try to import PyTorch and PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import degree
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None
    MessagePassing = None
    degree = None


if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
    class SimpleGNNConv(MessagePassing):
        """Simple GNN Convolution Layer for message passing."""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__(aggr='add')
            self.lin = nn.Linear(in_channels, out_channels)
            self.reset_parameters()
        
        def reset_parameters(self):
            self.lin.reset_parameters()
        
        def forward(self, x, edge_index):
            x = self.lin(x)
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            out = self.propagate(edge_index, x=x, norm=norm)
            return F.leaky_relu(out, negative_slope=0.2)
        
        def message(self, x_j, norm):
            return norm.view(-1, 1) * x_j
else:
    SimpleGNNConv = None


def build_graph(interactions_df: pd.DataFrame, embedding_dim: int = 64) -> Dict:
    """
    Xây dựng đồ thị hai phía G=(U, I, E) và khởi tạo nhúng.
    
    Args:
        interactions_df: DataFrame chứa interactions (user_id, product_id)
        embedding_dim: Kích thước vector nhúng
    
    Returns:
        Dictionary chứa:
            - num_users: Số lượng users
            - num_products: Số lượng products
            - num_edges: Số lượng edges
            - user_ids: Mapping user_id -> index
            - product_ids: Mapping product_id -> index
            - edge_index: Tensor chứa edges (2, num_edges)
            - user_embeddings: Initial user embeddings
            - product_embeddings: Initial product embeddings
            - embedding_dim: Kích thước nhúng
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Please install it with: pip install torch")
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric is not installed. Please install it with: pip install torch-geometric")
    if interactions_df.empty:
        return {
            'num_users': 0,
            'num_products': 0,
            'num_edges': 0,
            'user_ids': {},
            'product_ids': {},
            'edge_index': None,
            'user_embeddings': None,
            'product_embeddings': None,
            'embedding_dim': embedding_dim
        }
    
    # Tạo mapping user_id -> index và product_id -> index
    unique_users = sorted(interactions_df['user_id'].unique())
    unique_products = sorted(interactions_df['product_id'].unique())
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    product_id_to_idx = {product_id: idx for idx, product_id in enumerate(unique_products)}
    
    num_users = len(unique_users)
    num_products = len(unique_products)
    
    # Tạo edge_index
    edges = []
    for _, row in interactions_df.iterrows():
        user_idx = user_id_to_idx[row['user_id']]
        product_idx = product_id_to_idx[row['product_id']] + num_users  # Offset by num_users
        edges.append([user_idx, product_idx])
        edges.append([product_idx, user_idx])  # Undirected graph
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_edges = len(edges) // 2  # Each edge is counted twice (undirected)
    
    # Khởi tạo embeddings (Xavier Uniform)
    user_embeddings = torch.empty(num_users, embedding_dim)
    product_embeddings = torch.empty(num_products, embedding_dim)
    
    nn.init.xavier_uniform_(user_embeddings)
    nn.init.xavier_uniform_(product_embeddings)
    
    return {
        'num_users': num_users,
        'num_products': num_products,
        'num_edges': num_edges,
        'user_ids': user_id_to_idx,
        'product_ids': product_id_to_idx,
        'unique_user_ids': unique_users,
        'unique_product_ids': unique_products,
        'edge_index': edge_index,
        'user_embeddings': user_embeddings,
        'product_embeddings': product_embeddings,
        'embedding_dim': embedding_dim
    }


def message_propagation(graph_result: Dict, num_layers: int = 3) -> Dict:
    """
    Thực hiện lan truyền thông điệp qua L lớp.
    
    Args:
        graph_result: Kết quả từ build_graph
        num_layers: Số lớp lan truyền
    
    Returns:
        Dictionary chứa:
            - final_user_embeddings: Final user embeddings after L layers
            - final_product_embeddings: Final product embeddings after L layers
            - layer_stats: Statistics for each layer
    """
    if graph_result['edge_index'] is None:
        return {
            'final_user_embeddings': None,
            'final_product_embeddings': None,
            'layer_stats': []
        }
    
    embedding_dim = graph_result['embedding_dim']
    edge_index = graph_result['edge_index']
    
    # Concatenate user and product embeddings
    x = torch.cat([graph_result['user_embeddings'], graph_result['product_embeddings']], dim=0)
    
    # Create GNN layers
    conv_layers = nn.ModuleList([
        SimpleGNNConv(embedding_dim, embedding_dim) for _ in range(num_layers)
    ])
    
    # Message propagation through layers
    layer_stats = []
    for layer_idx in range(num_layers):
        x_prev = x.clone()
        x = conv_layers[layer_idx](x, edge_index)
        
        # Statistics
        layer_stats.append({
            'layer': layer_idx + 1,
            'mean_norm': float(x.norm(dim=1).mean()),
            'std_norm': float(x.norm(dim=1).std()),
            'change': float((x - x_prev).norm().item())
        })
    
    # Split back to user and product embeddings
    final_user_embeddings = x[:graph_result['num_users']]
    final_product_embeddings = x[graph_result['num_users']:]
    
    return {
        'final_user_embeddings': final_user_embeddings,
        'final_product_embeddings': final_product_embeddings,
        'layer_stats': layer_stats,
        'graph_result': graph_result
    }


def compute_gnn_predictions(propagation_result: Dict, top_k: int = 20) -> Dict:
    """
    Tính điểm dự đoán GNN và xếp hạng Top-K.
    
    Args:
        propagation_result: Kết quả từ message_propagation
        top_k: Số lượng sản phẩm Top-K
    
    Returns:
        Dictionary chứa:
            - predictions: Dict[user_id, Dict[product_id, score]]
            - rankings: Dict[user_id, List[Tuple[product_id, score]]]
            - stats: Statistics
    """
    if propagation_result['final_user_embeddings'] is None:
        return {
            'predictions': {},
            'rankings': {},
            'stats': {
                'total_predictions': 0,
                'total_users': 0,
                'total_products': 0,
                'min_score': 0.0,
                'max_score': 0.0,
                'mean_score': 0.0
            }
        }
    
    user_emb = propagation_result['final_user_embeddings']
    product_emb = propagation_result['final_product_embeddings']
    graph_result = propagation_result['graph_result']
    
    # Convert to numpy for easier manipulation
    user_emb_np = user_emb.detach().numpy()
    product_emb_np = product_emb.detach().numpy()
    
    # Compute predictions for all user-product pairs
    predictions = {}
    rankings = {}
    all_scores = []
    
    unique_user_ids = graph_result['unique_user_ids']
    unique_product_ids = graph_result['unique_product_ids']
    
    for user_idx, user_id in enumerate(unique_user_ids):
        user_vec = user_emb_np[user_idx]
        user_predictions = {}
        
        # Compute scores for all products
        for product_idx, product_id in enumerate(unique_product_ids):
            product_vec = product_emb_np[product_idx]
            score = float(np.dot(user_vec, product_vec))
            user_predictions[str(product_id)] = score
            all_scores.append(score)
        
        predictions[str(user_id)] = user_predictions
        
        # Rank and get top-K
        sorted_predictions = sorted(
            user_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        rankings[str(user_id)] = sorted_predictions[:top_k]
    
    stats = {
        'total_predictions': len(all_scores),
        'total_users': len(unique_user_ids),
        'total_products': len(unique_product_ids),
        'min_score': float(np.min(all_scores)) if all_scores else 0.0,
        'max_score': float(np.max(all_scores)) if all_scores else 0.0,
        'mean_score': float(np.mean(all_scores)) if all_scores else 0.0,
        'std_score': float(np.std(all_scores)) if all_scores else 0.0
    }
    
    return {
        'predictions': predictions,
        'rankings': rankings,
        'stats': stats
    }


def train_gnn_model(
    propagation_result: Dict,
    interactions_df: pd.DataFrame,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    reg_weight: float = 0.0001,
    batch_size: int = 256
) -> Dict:
    """
    Huấn luyện mô hình GNN bằng BPR Loss.
    
    Args:
        propagation_result: Kết quả từ message_propagation
        interactions_df: DataFrame chứa interactions
        num_epochs: Số epochs
        learning_rate: Learning rate
        reg_weight: Regularization weight (λ)
        batch_size: Batch size
    
    Returns:
        Dictionary chứa:
            - final_user_embeddings: Trained user embeddings
            - final_product_embeddings: Trained product embeddings
            - loss_history: List of BPR losses per epoch
            - initial_loss: Initial BPR loss
            - final_loss: Final BPR loss
            - predictions: Predictions after training
    """
    if propagation_result['final_user_embeddings'] is None:
        return {
            'final_user_embeddings': None,
            'final_product_embeddings': None,
            'loss_history': [],
            'initial_loss': 0.0,
            'final_loss': 0.0,
            'predictions': {}
        }
    
    # Get embeddings and graph info
    # Detach và tạo mới để đảm bảo là leaf tensors
    user_emb_init = propagation_result['final_user_embeddings'].detach()
    product_emb_init = propagation_result['final_product_embeddings'].detach()
    
    # Tạo mới với requires_grad=True để có thể optimize
    user_emb = nn.Parameter(user_emb_init.clone())
    product_emb = nn.Parameter(product_emb_init.clone())
    
    graph_result = propagation_result['graph_result']
    
    # Create optimizer
    optimizer = torch.optim.Adam([user_emb, product_emb], lr=learning_rate)
    
    # Prepare training data (positive interactions)
    positive_interactions = interactions_df[
        interactions_df['interaction_type'].isin(['purchase', 'like', 'cart'])
    ] if 'interaction_type' in interactions_df.columns else interactions_df
    
    user_id_to_idx = graph_result['user_ids']
    product_id_to_idx = graph_result['product_ids']
    
    # Create positive pairs
    positive_pairs = []
    for _, row in positive_interactions.iterrows():
        user_id = str(row['user_id'])
        product_id = str(row['product_id'])
        if user_id in user_id_to_idx and product_id in product_id_to_idx:
            positive_pairs.append((
                user_id_to_idx[user_id],
                product_id_to_idx[product_id]
            ))
    
    if not positive_pairs:
        # Vẫn tạo predictions và rankings ngay cả khi không có positive pairs để training
        # (có thể dùng embeddings từ propagation)
        user_emb_np = user_emb.detach().numpy()
        product_emb_np = product_emb.detach().numpy()
        unique_user_ids = graph_result['unique_user_ids']
        unique_product_ids = graph_result['unique_product_ids']
        
        predictions = {}
        for user_idx, user_id in enumerate(unique_user_ids):
            user_vec = user_emb_np[user_idx]
            user_predictions = {}
            
            for product_idx, product_id in enumerate(unique_product_ids):
                product_vec = product_emb_np[product_idx]
                score = float(np.dot(user_vec, product_vec))
                user_predictions[str(product_id)] = score
            
            predictions[str(user_id)] = user_predictions
        
        return {
            'final_user_embeddings': user_emb.detach(),
            'final_product_embeddings': product_emb.detach(),
            'loss_history': [],
            'initial_loss': 0.0,
            'final_loss': 0.0,
            'predictions': predictions,
            'rankings': {
                str(user_id): sorted(
                    user_predictions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
                for user_id, user_predictions in predictions.items()
            },
            'warning': 'No positive pairs found for training. Using embeddings from propagation only.'
        }
    
    # Training loop
    loss_history = []
    num_products = graph_result['num_products']
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Sample batches
        for i in range(0, len(positive_pairs), batch_size):
            batch_pairs = positive_pairs[i:i+batch_size]
            
            if not batch_pairs:
                continue
            
            # Sample negative products for each positive pair
            user_ids = torch.tensor([pair[0] for pair in batch_pairs], dtype=torch.long)
            pos_product_ids = torch.tensor([pair[1] for pair in batch_pairs], dtype=torch.long)
            neg_product_ids = torch.tensor(
                [np.random.randint(0, num_products) for _ in batch_pairs],
                dtype=torch.long
            )
            
            # Get embeddings
            user_emb_batch = user_emb[user_ids]
            pos_emb_batch = product_emb[pos_product_ids]
            neg_emb_batch = product_emb[neg_product_ids]
            
            # Compute scores
            pos_scores = (user_emb_batch * pos_emb_batch).sum(dim=1)
            neg_scores = (user_emb_batch * neg_emb_batch).sum(dim=1)
            
            # BPR Loss
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            # Regularization
            reg_loss = reg_weight * (
                user_emb_batch.norm(2).pow(2) +
                pos_emb_batch.norm(2).pow(2) +
                neg_emb_batch.norm(2).pow(2)
            ) / user_emb_batch.size(0)
            
            total_loss = bpr_loss + reg_loss
            epoch_losses.append(total_loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        loss_history.append(avg_loss)
    
    # Compute final predictions
    user_emb_np = user_emb.detach().numpy()
    product_emb_np = product_emb.detach().numpy()
    
    predictions = {}
    unique_user_ids = graph_result['unique_user_ids']
    unique_product_ids = graph_result['unique_product_ids']
    
    for user_idx, user_id in enumerate(unique_user_ids):
        user_vec = user_emb_np[user_idx]
        user_predictions = {}
        
        for product_idx, product_id in enumerate(unique_product_ids):
            product_vec = product_emb_np[product_idx]
            score = float(np.dot(user_vec, product_vec))
            user_predictions[str(product_id)] = score
        
        predictions[str(user_id)] = user_predictions
    
    return {
        'final_user_embeddings': user_emb.detach(),
        'final_product_embeddings': product_emb.detach(),
        'loss_history': loss_history,
        'initial_loss': loss_history[0] if loss_history else 0.0,
        'final_loss': loss_history[-1] if loss_history else 0.0,
        'predictions': predictions,
        'rankings': {
            str(user_id): sorted(
                user_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            for user_id, user_predictions in predictions.items()
        }
    }

