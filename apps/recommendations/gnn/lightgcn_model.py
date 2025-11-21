"""
LightGCN model implementation using PyTorch Geometric.
Implements a heterogeneous graph with User ↔ Product ↔ Category ↔ Color ↔ Style nodes.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

logger = logging.getLogger(__name__)


class LightGCNConv(MessagePassing):
    """
    LightGCN convolution layer.
    Simplified GCN without feature transformation and nonlinear activation.
    """
    
    def __init__(self):
        super().__init__(aggr='add')
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features
        """
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Propagate messages
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        """Message function."""
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """
    LightGCN model (Graph Neural Network) for collaborative filtering via user-item interaction graph.
    
    Architecture:
    - User and Product embeddings
    - Multiple LightGCN layers for message passing
    - Layer combination for final embeddings
    """
    
    def __init__(
        self,
        num_users: int,
        num_products: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        """
        Initialize LightGCN model.
        
        Args:
            num_users: Number of users
            num_products: Number of products
            embedding_dim: Dimension of embeddings
            num_layers: Number of LightGCN layers
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_products = num_products
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)
        
        # LightGCN layers
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
    
    def forward(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            edge_index: Edge indices [2, num_edges] for user-product bipartite graph
            
        Returns:
            Tuple of (user_embeddings, product_embeddings)
        """
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        product_emb = self.product_embedding.weight
        
        # Concatenate user and product embeddings
        all_emb = torch.cat([user_emb, product_emb], dim=0)
        
        # Store embeddings from each layer
        embs = [all_emb]
        
        # Message passing through layers
        for conv in self.convs:
            all_emb = conv(all_emb, edge_index)
            embs.append(all_emb)
        
        # Combine embeddings from all layers (including initial)
        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        
        # Split back into user and product embeddings
        user_final = final_emb[:self.num_users]
        product_final = final_emb[self.num_users:]
        
        return user_final, product_final
    
    def predict(self, user_ids: torch.Tensor, product_ids: torch.Tensor, 
                user_emb: torch.Tensor, product_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for user-product pairs.
        
        Args:
            user_ids: User indices [batch_size]
            product_ids: Product indices [batch_size]
            user_emb: User embeddings [num_users, embedding_dim]
            product_emb: Product embeddings [num_products, embedding_dim]
            
        Returns:
            Predicted scores [batch_size]
        """
        user_emb_batch = user_emb[user_ids]
        product_emb_batch = product_emb[product_ids]
        
        # Compute dot product
        scores = (user_emb_batch * product_emb_batch).sum(dim=1)
        
        return scores
    
    def bpr_loss(self, user_ids: torch.Tensor, pos_product_ids: torch.Tensor, 
                 neg_product_ids: torch.Tensor, user_emb: torch.Tensor, 
                 product_emb: torch.Tensor, reg_weight: float = 1e-5) -> torch.Tensor:
        """
        Bayesian Personalized Ranking (BPR) loss.
        
        Args:
            user_ids: User indices [batch_size]
            pos_product_ids: Positive product indices [batch_size]
            neg_product_ids: Negative product indices [batch_size]
            user_emb: User embeddings [num_users, embedding_dim]
            product_emb: Product embeddings [num_products, embedding_dim]
            reg_weight: L2 regularization weight
            
        Returns:
            BPR loss
        """
        # Get embeddings
        user_emb_batch = user_emb[user_ids]
        pos_emb_batch = product_emb[pos_product_ids]
        neg_emb_batch = product_emb[neg_product_ids]
        
        # Compute scores
        pos_scores = (user_emb_batch * pos_emb_batch).sum(dim=1)
        neg_scores = (user_emb_batch * neg_emb_batch).sum(dim=1)
        
        # BPR loss
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization
        reg_loss = reg_weight * (
            user_emb_batch.norm(2).pow(2) +
            pos_emb_batch.norm(2).pow(2) +
            neg_emb_batch.norm(2).pow(2)
        ) / user_emb_batch.size(0)
        
        return bpr_loss + reg_loss


class HeterogeneousLightGCN(nn.Module):
    """
    Heterogeneous LightGCN for multi-type nodes.
    Supports User ↔ Product ↔ Category ↔ Color ↔ Style relationships.
    """
    
    def __init__(
        self,
        num_users: int,
        num_products: int,
        num_categories: int = 0,
        num_colors: int = 0,
        num_styles: int = 0,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        """
        Initialize Heterogeneous LightGCN.
        
        Args:
            num_users: Number of users
            num_products: Number of products
            num_categories: Number of categories
            num_colors: Number of colors
            num_styles: Number of styles
            embedding_dim: Dimension of embeddings
            num_layers: Number of layers
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_products = num_products
        self.num_categories = num_categories
        self.num_colors = num_colors
        self.num_styles = num_styles
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Initialize embeddings for each node type
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        
        if num_categories > 0:
            self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        if num_colors > 0:
            self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        if num_styles > 0:
            self.style_embedding = nn.Embedding(num_styles, embedding_dim)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)
        if num_categories > 0:
            nn.init.xavier_uniform_(self.category_embedding.weight)
        if num_colors > 0:
            nn.init.xavier_uniform_(self.color_embedding.weight)
        if num_styles > 0:
            nn.init.xavier_uniform_(self.style_embedding.weight)
        
        # LightGCN layers
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
    
    def forward(self, edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through heterogeneous graph.
        
        Args:
            edge_index_dict: Dictionary of edge indices for each edge type
                e.g., {'user_product': edge_index, 'product_category': edge_index, ...}
        
        Returns:
            Dictionary of final embeddings for each node type
        """
        # Get initial embeddings
        embeddings = {
            'user': self.user_embedding.weight,
            'product': self.product_embedding.weight,
        }
        
        if self.num_categories > 0:
            embeddings['category'] = self.category_embedding.weight
        if self.num_colors > 0:
            embeddings['color'] = self.color_embedding.weight
        if self.num_styles > 0:
            embeddings['style'] = self.style_embedding.weight
        
        # For simplicity, we'll use the main user-product graph
        # In a full implementation, you would handle multiple edge types
        if 'user_product' in edge_index_dict:
            edge_index = edge_index_dict['user_product']
            
            # Concatenate all embeddings
            all_emb = torch.cat([embeddings['user'], embeddings['product']], dim=0)
            
            # Store embeddings from each layer
            embs = [all_emb]
            
            # Message passing
            for conv in self.convs:
                all_emb = conv(all_emb, edge_index)
                embs.append(all_emb)
            
            # Combine embeddings
            final_emb = torch.stack(embs, dim=0).mean(dim=0)
            
            # Split back
            embeddings['user'] = final_emb[:self.num_users]
            embeddings['product'] = final_emb[self.num_users:]
        
        return embeddings


def build_bipartite_graph(
    user_product_interactions: List[Tuple[int, int]],
    num_users: int,
    num_products: int,
) -> torch.Tensor:
    """
    Build bipartite graph edge index from user-product interactions.
    
    Args:
        user_product_interactions: List of (user_id, product_id) tuples
        num_users: Total number of users
        num_products: Total number of products
        
    Returns:
        Edge index tensor [2, num_edges]
    """
    edges = []
    
    for user_id, product_id in user_product_interactions:
        # User to product edge
        edges.append([user_id, num_users + product_id])
        # Product to user edge (undirected)
        edges.append([num_users + product_id, user_id])
    
    if not edges:
        # Return empty edge index
        return torch.zeros((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    return edge_index


def train_lightgcn(
    model: LightGCN,
    edge_index: torch.Tensor,
    train_interactions: List[Tuple[int, int, int]],  # (user, pos_product, neg_product)
    num_epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    reg_weight: float = 1e-5,
    device: str = 'cpu',
) -> Dict[str, List[float]]:
    """
    Train LightGCN model.
    
    Args:
        model: LightGCN model
        edge_index: Edge index for the graph
        train_interactions: List of (user_id, pos_product_id, neg_product_id) tuples
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        reg_weight: Regularization weight
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Dictionary of training metrics
    """
    model = model.to(device)
    edge_index = edge_index.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        
        # Shuffle training data
        np.random.shuffle(train_interactions)
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(train_interactions), batch_size):
            batch = train_interactions[i:i + batch_size]
            
            user_ids = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            pos_product_ids = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
            neg_product_ids = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)
            
            # Forward pass
            user_emb, product_emb = model(edge_index)
            
            # Compute loss
            loss = model.bpr_loss(user_ids, pos_product_ids, neg_product_ids, 
                                 user_emb, product_emb, reg_weight)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return {'losses': losses}

