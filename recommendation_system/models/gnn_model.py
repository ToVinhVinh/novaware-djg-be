"""
GNN (Graph Neural Network) Model
Sử dụng GraphSAGE cho recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import time
from tqdm import tqdm


class GraphSAGERecommender(nn.Module):
    """GraphSAGE model for recommendation"""
    
    def __init__(
        self, 
        n_users: int, 
        n_products: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Args:
            n_users: Số lượng users
            n_products: Số lượng products
            embedding_dim: Dimension của embedding
            hidden_dim: Dimension của hidden layer
            n_layers: Số lượng GNN layers
            dropout: Dropout rate
        """
        super(GraphSAGERecommender, self).__init__()
        
        self.n_users = n_users
        self.n_products = n_products
        self.embedding_dim = embedding_dim
        
        # User và Product embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.product_embedding = nn.Embedding(n_products, embedding_dim)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(embedding_dim, hidden_dim))
        
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        
        self.dropout = dropout
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)
        
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features [n_nodes, embedding_dim]
            edge_index: Edge indices [2, n_edges]
        
        Returns:
            Node embeddings [n_nodes, embedding_dim]
        """
        # GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.fc(x)
        
        return x
    
    def predict(self, user_idx, product_idx, node_embeddings):
        """
        Predict score for user-product pair
        
        Args:
            user_idx: User index
            product_idx: Product index
            node_embeddings: Node embeddings from forward pass
        
        Returns:
            Predicted score
        """
        user_emb = node_embeddings[user_idx]
        product_emb = node_embeddings[self.n_users + product_idx]
        
        # Dot product
        score = torch.sum(user_emb * product_emb, dim=-1)
        
        return score


class GNNRecommender:
    """Wrapper class cho GNN model"""
    
    def __init__(
        self,
        users_df: pd.DataFrame,
        products_df: pd.DataFrame,
        train_interactions: pd.DataFrame,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        device: str = 'cpu'
    ):
        """
        Args:
            users_df: DataFrame chứa thông tin users
            products_df: DataFrame chứa thông tin products
            train_interactions: DataFrame chứa training interactions
            embedding_dim: Dimension của embedding
            hidden_dim: Dimension của hidden layer
            n_layers: Số lượng GNN layers
            dropout: Dropout rate
            device: Device để train ('cpu' hoặc 'cuda')
        """
        self.users_df = users_df
        self.products_df = products_df
        self.train_interactions = train_interactions
        
        self.n_users = len(users_df)
        self.n_products = len(products_df)
        
        self.device = torch.device(device)
        
        # Initialize model
        self.model = GraphSAGERecommender(
            n_users=self.n_users,
            n_products=self.n_products,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)
        
        # Graph data
        self.graph_data = None
        self.node_embeddings = None
        
        # Training metrics
        self.training_time = 0
        self.training_losses = []
        
    def build_graph(self):
        """Xây dựng bipartite graph từ interactions"""
        print("\n[GNN] Building user-product bipartite graph...")
        
        # Create edges từ interactions
        edges = []
        edge_weights = []
        
        for _, row in self.train_interactions.iterrows():
            user_idx = int(row['user_idx'])
            product_idx = int(row['product_idx'])
            weight = float(row['weight'])
            
            # User -> Product edge
            edges.append([user_idx, self.n_users + product_idx])
            edge_weights.append(weight)
            
            # Product -> User edge (undirected graph)
            edges.append([self.n_users + product_idx, user_idx])
            edge_weights.append(weight)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Create initial node features
        # Concatenate user và product embeddings
        user_features = self.model.user_embedding.weight.data
        product_features = self.model.product_embedding.weight.data
        x = torch.cat([user_features, product_features], dim=0)
        
        # Create graph data
        self.graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weights
        ).to(self.device)
        
        print(f"Graph created:")
        print(f"  Nodes: {self.graph_data.x.shape[0]} ({self.n_users} users + {self.n_products} products)")
        print(f"  Edges: {self.graph_data.edge_index.shape[1]}")
        print(f"  Features: {self.graph_data.x.shape[1]}")
        
    def train(
        self,
        n_epochs: int = 30,  # Giảm từ 50 xuống 30
        learning_rate: float = 0.001,
        batch_size: int = 2048,  # Tăng batch size để tối ưu
        negative_samples: int = 4
    ):
        """
        Train GNN model (Optimized version)
        
        Args:
            n_epochs: Số epochs (mặc định 30 thay vì 50)
            learning_rate: Learning rate
            batch_size: Batch size (tăng lên 2048 để tối ưu)
            negative_samples: Số negative samples cho mỗi positive sample
        """
        print("\n" + "="*80)
        print("TRAINING GNN MODEL (GraphSAGE) - OPTIMIZED")
        print("="*80)
        
        start_time = time.time()
        
        # Build graph
        self.build_graph()
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Prepare training data
        pos_interactions = self.train_interactions[['user_idx', 'product_idx', 'weight']].values
        
        # Training loop với progress bar
        self.model.train()
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0
            
            # Shuffle interactions
            np.random.shuffle(pos_interactions)
            
            # Progress bar cho epoch
            if use_tqdm:
                pbar = tqdm(range(0, len(pos_interactions), batch_size), 
                           desc=f"Epoch {epoch+1}/{n_epochs}")
            else:
                pbar = range(0, len(pos_interactions), batch_size)
            
            # Mini-batch training với vectorization
            for i in pbar:
                batch = pos_interactions[i:i+batch_size]
                
                # Forward pass - chỉ gọi 1 lần cho toàn bộ graph
                node_embeddings = self.model(
                    self.graph_data.x,
                    self.graph_data.edge_index
                )
                
                # Vectorize: Extract tất cả user và product embeddings cùng lúc
                batch_users = torch.tensor([int(x[0]) for x in batch], dtype=torch.long, device=self.device)
                batch_products = torch.tensor([int(x[1]) for x in batch], dtype=torch.long, device=self.device)
                batch_weights = torch.tensor([float(x[2]) for x in batch], dtype=torch.float, device=self.device)
                
                # Get user embeddings (vectorized)
                user_embs = node_embeddings[batch_users]  # [batch_size, embedding_dim]
                
                # Get positive product embeddings (vectorized)
                pos_product_embs = node_embeddings[self.n_users + batch_products]  # [batch_size, embedding_dim]
                
                # Positive scores (vectorized)
                pos_scores = torch.sum(user_embs * pos_product_embs, dim=1)  # [batch_size]
                
                # Negative sampling (vectorized)
                # Tạo negative products cho tất cả samples cùng lúc
                neg_products = torch.randint(
                    0, self.n_products, 
                    (len(batch), negative_samples), 
                    device=self.device
                )  # [batch_size, negative_samples]
                
                # Get negative product embeddings
                neg_product_embs = node_embeddings[
                    self.n_users + neg_products
                ]  # [batch_size, negative_samples, embedding_dim]
                
                # Expand user_embs để match với negative samples
                user_embs_expanded = user_embs.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                
                # Negative scores (vectorized)
                neg_scores = torch.sum(
                    user_embs_expanded * neg_product_embs, dim=2
                )  # [batch_size, negative_samples]
                
                # BPR loss (vectorized)
                # pos_scores: [batch_size]
                # neg_scores: [batch_size, negative_samples]
                # Tính diff: [batch_size, negative_samples]
                score_diff = pos_scores.unsqueeze(1) - neg_scores  # [batch_size, negative_samples]
                
                # BPR loss cho mỗi negative sample
                bpr_loss_per_neg = -torch.log(torch.sigmoid(score_diff) + 1e-10)  # [batch_size, negative_samples]
                
                # Average over negative samples
                bpr_loss = torch.mean(bpr_loss_per_neg, dim=1)  # [batch_size]
                
                # Weighted loss
                weighted_loss = torch.mean(batch_weights * bpr_loss)
                
                # Backward pass
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()
                
                epoch_loss += weighted_loss.item()
                n_batches += 1
                
                # Update progress bar
                if use_tqdm:
                    pbar.set_postfix({'loss': f'{weighted_loss.item():.4f}'})
            
            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:  # Print mỗi 5 epochs thay vì 10
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        # Save final embeddings
        self.model.eval()
        with torch.no_grad():
            self.node_embeddings = self.model(
                self.graph_data.x,
                self.graph_data.edge_index
            ).cpu().numpy()
        
        self.training_time = time.time() - start_time
        
        print(f"\n[GNN] Training completed in {self.training_time:.2f}s")
        print(f"  Average time per epoch: {self.training_time / n_epochs:.2f}s")
        print("="*80)
        
    def get_user_recommendations(
        self,
        user_idx: int,
        top_k: int = 20,
        exclude_interacted: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Lấy recommendations cho user
        
        Args:
            user_idx: User index
            top_k: Số lượng recommendations
            exclude_interacted: Có loại bỏ sản phẩm đã tương tác không
        
        Returns:
            List of (product_idx, score)
        """
        if self.node_embeddings is None:
            raise ValueError("Model chưa được train!")
        
        # Get user embedding
        user_emb = self.node_embeddings[user_idx]
        
        # Get all product embeddings
        product_embs = self.node_embeddings[self.n_users:self.n_users + self.n_products]
        
        # Compute scores
        scores = np.dot(product_embs, user_emb)
        
        # Get top K
        if exclude_interacted:
            # Loại bỏ sản phẩm đã tương tác
            interacted = set(
                self.train_interactions[
                    self.train_interactions['user_idx'] == user_idx
                ]['product_idx'].values
            )
            
            for prod_idx in interacted:
                scores[prod_idx] = -np.inf
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return list(zip(top_indices, top_scores))
    
    def recommend_personalized(
        self,
        user_info: Dict,
        user_idx: int,
        payload_product_idx: int,
        top_k: int = 10
    ) -> Tuple[List[Tuple[int, float]], float]:
        """
        Gợi ý sản phẩm theo tiêu chí Personalized
        
        Args:
            user_info: Thông tin user
            user_idx: User index
            payload_product_idx: Index của sản phẩm payload
            top_k: Số lượng recommendations
        
        Returns:
            (List of (product_idx, score), inference_time)
        """
        start_time = time.time()
        
        # Lấy recommendations từ GNN
        candidates = self.get_user_recommendations(
            user_idx,
            top_k=top_k * 3,
            exclude_interacted=True
        )
        
        # Lấy thông tin sản phẩm payload
        payload_product = self.products_df[
            self.products_df['product_idx'] == payload_product_idx
        ].iloc[0]
        
        # Filter theo articleType (STRICT)
        filtered = []
        for prod_idx, score in candidates:
            product = self.products_df[
                self.products_df['product_idx'] == prod_idx
            ].iloc[0]
            
            # Check articleType
            if product['articleType'] == payload_product['articleType']:
                # Check gender
                if product['gender'] in [user_info['product_gender'], 'Unisex']:
                    filtered.append((prod_idx, score))
        
        # Lấy top K
        results = filtered[:top_k]
        
        inference_time = time.time() - start_time
        
        return results, inference_time
    
    def get_model_info(self) -> Dict:
        """Lấy thông tin về model"""
        return {
            'model_name': 'GNN (GraphSAGE)',
            'n_users': self.n_users,
            'n_products': self.n_products,
            'embedding_dim': self.model.embedding_dim,
            'n_edges': self.graph_data.edge_index.shape[1] if self.graph_data else 0,
            'training_time': self.training_time,
            'final_loss': self.training_losses[-1] if self.training_losses else 0
        }


if __name__ == "__main__":
    print("Testing GNN Recommender...")
