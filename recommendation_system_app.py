"""
Streamlit App for Product Recommendation System
Implements 3 models: LightGCN, Content-based Filtering, Hybrid
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import ast
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Import data loader
from apps.recommendations.streamlit_utils.data_loader import (
    load_users_csv,
    load_products_csv,
    load_interactions_csv,
    prepare_data_for_models,
    filter_products_by_gender_age
)

st.set_page_config(
    page_title="Hệ thống Gợi ý Sản phẩm",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #fff;
        padding: 15px;
        border-radius: 5px;
        border: 2px solid #1f77b4;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    .step-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)


# ==================== DATA LOADING ====================

@st.cache_data
def load_all_data():
    """Load all CSV files."""
    exports_dir = Path("exports")
    
    users_df = load_users_csv(exports_dir / "users.csv")
    products_df = load_products_csv(exports_dir / "products.csv")
    interactions_df = load_interactions_csv(exports_dir / "interactions.csv")
    
    user_dict, product_dict, interactions_df = prepare_data_for_models(
        users_df, products_df, interactions_df
    )
    
    return user_dict, product_dict, interactions_df, users_df, products_df


# ==================== LIGHTGCN MODEL ====================

class LightGCNLayer(nn.Module):
    """LightGCN Graph Convolutional Layer."""
    
    def __init__(self):
        super(LightGCNLayer, self).__init__()
    
    def forward(self, embeddings, edge_index):
        """LightGCN propagation: e^(l+1) = sum(e^l / sqrt(deg(u)) / sqrt(deg(v)))"""
        if edge_index.numel() == 0:
            return embeddings
        
        # Get degrees
        num_nodes = embeddings.size(0)
        row, col = edge_index
        
        # Calculate degrees
        deg = torch.zeros(num_nodes, device=embeddings.device)
        if row.numel() > 0:
            deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg = torch.clamp(deg, min=1.0)
        
        # Normalize: 1 / sqrt(deg(u) * deg(v))
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Aggregate neighbors
        out = torch.zeros_like(embeddings)
        if row.numel() > 0:
            out.scatter_add_(0, col.unsqueeze(-1).expand(-1, embeddings.size(1)), 
                            embeddings[row] * norm.unsqueeze(-1))
        else:
            out = embeddings
        
        return out


class LightGCNModel(nn.Module):
    """LightGCN Model for Recommendation."""
    
    def __init__(self, num_users, num_products, embedding_dim=64, num_layers=3):
        super(LightGCNModel, self).__init__()
        self.num_users = num_users
        self.num_products = num_products
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.product_embedding.weight, std=0.1)
        
        # LightGCN layers
        self.layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])
    
    def forward(self, edge_index):
        """Forward pass through LightGCN."""
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        product_emb = self.product_embedding.weight
        all_emb = torch.cat([user_emb, product_emb], dim=0)
        
        # LightGCN propagation: average embeddings from all layers
        emb_list = [all_emb]
        for layer_idx, layer in enumerate(self.layers):
            all_emb = layer(all_emb, edge_index)
            emb_list.append(all_emb)
        
        # Average all layer embeddings
        final_emb = torch.mean(torch.stack(emb_list), dim=0)
        
        # Split back to users and products
        user_final = final_emb[:self.num_users]
        product_final = final_emb[self.num_users:]
        
        return user_final, product_final
    
    def predict(self, user_idx, product_idx):
        """Predict rating: r_hat = e_u^T * e_i"""
        user_emb = self.user_embedding.weight[user_idx]
        product_emb = self.product_embedding.weight[product_idx]
        return torch.sum(user_emb * product_emb, dim=-1)


class LightGCNRecommender:
    """LightGCN Recommendation System."""
    
    # Interaction type weights (không dùng rating)
    INTERACTION_WEIGHTS = {
        'view': 1.0,
        'like': 2.0,
        'cart': 3.0,
        'purchase': 4.0,
        'review': 2.5  # Review có thể có rating nhưng chúng ta chỉ dùng interaction type
    }
    
    def __init__(self):
        self.model = None
        self.user_id_map = {}
        self.product_id_map = {}
        self.reverse_user_map = {}
        self.reverse_product_map = {}
        self.edge_index = None
        self.training_time = 0.0
        self.computation_steps = []  # Lưu các bước tính toán
        self.matrices = {}  # Lưu các ma trận để hiển thị
    
    def build_graph(self, interactions_df: pd.DataFrame):
        """Build bipartite graph from interactions with interaction type weights."""
        # Create mappings
        unique_users = interactions_df['user_id'].unique()
        unique_products = interactions_df['product_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.product_id_map = {pid: idx for idx, pid in enumerate(unique_products)}
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_product_map = {v: k for k, v in self.product_id_map.items()}
        
        # Build edge list with weights based on interaction types
        edges = []
        edge_weights = []  # Store weights for visualization
        
        for _, row in interactions_df.iterrows():
            user_idx = self.user_id_map[row['user_id']]
            product_idx = self.product_id_map[row['product_id']] + len(self.user_id_map)
            
            # Get interaction type weight (không dùng rating)
            interaction_type = str(row.get('interaction_type', 'view')).lower()
            weight = self.INTERACTION_WEIGHTS.get(interaction_type, 1.0)
            
            edges.append([user_idx, product_idx])
            edges.append([product_idx, user_idx])  # Undirected graph
            edge_weights.extend([weight, weight])
        
        # Store edge weights for later use
        self.edge_weights = edge_weights if edge_weights else None
        
        # Convert to tensor
        if edges:
            edges_tensor = torch.tensor(edges, dtype=torch.long)
            if edges_tensor.numel() > 0:
                self.edge_index = edges_tensor.t().contiguous()
            else:
                self.edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            self.edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # Store graph statistics for computation steps
        self.computation_steps.append({
            'step': 'Bước 1: Xây dựng đồ thị',
            'formula': 'G = (U ∪ I, E)',
            'computation': f'Số users: {len(unique_users)}, Số products: {len(unique_products)}, Số edges: {len(edges)//2}',
            'meaning': f'Đồ thị có {len(unique_users)} nodes user và {len(unique_products)} nodes product, tạo thành {len(edges)//2} cạnh tương tác'
        })
        
        return len(unique_users), len(unique_products)
    
    def train(self, interactions_df: pd.DataFrame, epochs=50, lr=0.001):
        """Train LightGCN model."""
        start_time = time.time()
        
        # Build graph
        num_users, num_products = self.build_graph(interactions_df)
        
        if num_users == 0 or num_products == 0:
            st.error("Không có dữ liệu để train!")
            return
        
        # Initialize model
        embedding_dim = 64
        num_layers = 3
        self.model = LightGCNModel(num_users, num_products, embedding_dim=embedding_dim, num_layers=num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Store initial embeddings for computation steps
        initial_user_emb = self.model.user_embedding.weight.data.clone()
        initial_product_emb = self.model.product_embedding.weight.data.clone()
        
        self.computation_steps.append({
            'step': 'Bước 2: Khởi tạo Embeddings',
            'formula': 'e_u^(0) ~ N(0, 0.1²), e_i^(0) ~ N(0, 0.1²)',
            'computation': f'User embeddings shape: {initial_user_emb.shape}, Product embeddings shape: {initial_product_emb.shape}\n'
                          f'Ví dụ e_u[0] = {initial_user_emb[0][:3].tolist()}..., e_i[0] = {initial_product_emb[0][:3].tolist()}...',
            'meaning': f'Mỗi user và product được biểu diễn bằng vector {embedding_dim} chiều, khởi tạo ngẫu nhiên từ phân phối chuẩn'
        })
        
        # Store initial embeddings matrix
        self.matrices['initial_user_embeddings'] = initial_user_emb[:min(10, num_users), :min(10, embedding_dim)].detach().numpy()
        self.matrices['initial_product_embeddings'] = initial_product_emb[:min(10, num_products), :min(10, embedding_dim)].detach().numpy()
        
        # Add propagation computation steps
        with torch.no_grad():
            # Example propagation for first layer
            example_user_idx = 0
            example_user_emb = initial_user_emb[example_user_idx]
            
            # Calculate degree for example user
            if self.edge_index.numel() > 0:
                row, col = self.edge_index
                user_edges = (row == example_user_idx).sum().item()
                
                self.computation_steps.append({
                    'step': 'Bước 3: LightGCN Propagation',
                    'formula': 'e_u^(l+1) = Σ (e_i^(l) / √(deg(u) * deg(i)))',
                    'computation': f'Ví dụ với user 0:\n'
                                  f'  deg(user_0) = {user_edges} (số edges từ user này)\n'
                                  f'  e_user_0^(0) = {example_user_emb[:3].tolist()}...\n'
                                  f'  Sau propagation qua {num_layers} layers, embedding được cập nhật',
                    'meaning': f'Embedding của user được cập nhật bằng cách tổng hợp thông tin từ các products mà user đã tương tác, với normalization theo bậc của node'
                })
                
                self.computation_steps.append({
                    'step': 'Bước 4: Average Embeddings từ tất cả layers',
                    'formula': 'e_u = (1/(L+1)) * Σ e_u^(l)',
                    'computation': f'L = {num_layers} layers\n'
                                  f'Tổng hợp embeddings từ layer 0 đến layer {num_layers}\n'
                                  f'e_u_final = (e_u^(0) + e_u^(1) + ... + e_u^({num_layers})) / {num_layers + 1}',
                    'meaning': f'Final embedding là trung bình của embeddings từ tất cả {num_layers + 1} layers (bao gồm initial), giúp giữ lại thông tin từ mọi độ sâu của graph'
                })
        
        # Training loop
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        sample_size = min(1000, len(interactions_df))
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            user_emb, product_emb = self.model(self.edge_index)
            
            # BPR Loss with interaction type weights (không dùng rating)
            # Sample positive and negative pairs
            pos_pairs = []
            neg_pairs = []
            pos_weights = []  # Store interaction weights
            
            sample_df = interactions_df.sample(sample_size)
            for _, row in sample_df.iterrows():
                user_idx = self.user_id_map[row['user_id']]
                pos_product_idx = self.product_id_map[row['product_id']]
                
                # Get interaction type weight
                interaction_type = str(row.get('interaction_type', 'view')).lower()
                weight = self.INTERACTION_WEIGHTS.get(interaction_type, 1.0)
                
                # Sample negative product
                neg_product_idx = np.random.randint(0, num_products)
                while neg_product_idx == pos_product_idx:
                    neg_product_idx = np.random.randint(0, num_products)
                
                pos_pairs.append((user_idx, pos_product_idx))
                neg_pairs.append((user_idx, neg_product_idx))
                pos_weights.append(weight)
            
            # Calculate BPR loss with weights
            if pos_pairs:
                user_indices = [u for u, _ in pos_pairs]
                pos_product_indices = [p for _, p in pos_pairs]
                neg_product_indices = [p for _, p in neg_pairs]
                
                pos_scores = torch.sum(user_emb[user_indices] * product_emb[pos_product_indices], dim=1)
                neg_scores = torch.sum(user_emb[user_indices] * product_emb[neg_product_indices], dim=1)
                
                # Apply interaction type weights
                weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)
                weighted_diff = weights_tensor * (pos_scores - neg_scores)
                
                loss = -torch.mean(torch.log(torch.sigmoid(weighted_diff) + 1e-10))
                
                # Store computation steps for first epoch
                if epoch == 0 and len(pos_pairs) > 0:
                    example_pos_score = pos_scores[0].item()
                    example_neg_score = neg_scores[0].item()
                    example_weight = pos_weights[0]
                    
                    self.computation_steps.append({
                        'step': 'Bước 5: Dự đoán Rating (với interaction weights)',
                        'formula': 'r̂_ui = w_type * (e_u^T · e_i)',
                        'computation': f'Ví dụ: r̂_pos = {example_weight} * ({example_pos_score:.4f}) = {example_weight * example_pos_score:.4f}\n'
                                      f'r̂_neg = {example_neg_score:.4f}',
                        'meaning': f'Score được nhân với weight theo interaction type: view=1.0, like=2.0, cart=3.0, purchase=4.0'
                    })
                    
                    self.computation_steps.append({
                        'step': 'Bước 6: BPR Loss',
                        'formula': 'L_BPR = -Σ log(σ(w_type * (r̂_ui - r̂_uj)))',
                        'computation': f'Ví dụ: diff = {example_weight * example_pos_score:.4f} - {example_neg_score:.4f} = {example_weight * example_pos_score - example_neg_score:.4f}\n'
                                      f'sigmoid(diff) = {torch.sigmoid(torch.tensor(example_weight * example_pos_score - example_neg_score)).item():.4f}\n'
                                      f'Loss = {loss.item():.4f}',
                        'meaning': 'Loss càng nhỏ càng tốt, nghĩa là model phân biệt tốt giữa positive và negative pairs'
                    })
            else:
                loss = torch.tensor(0.0)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Store final embeddings for visualization
        with torch.no_grad():
            final_user_emb, final_product_emb = self.model(self.edge_index)
            self.matrices['final_user_embeddings'] = final_user_emb[:min(10, num_users), :min(10, embedding_dim)].detach().numpy()
            self.matrices['final_product_embeddings'] = final_product_emb[:min(10, num_products), :min(10, embedding_dim)].detach().numpy()
            
            # Compute similarity matrix (example: first 10 users vs first 10 products)
            similarity_matrix = torch.matmul(
                final_user_emb[:min(10, num_users)],
                final_product_emb[:min(10, num_products)].t()
            ).detach().numpy()
            self.matrices['similarity_matrix'] = similarity_matrix
        
        self.computation_steps.append({
            'step': 'Bước 7: Gradient Descent',
            'formula': 'θ ← θ - α * ∇L_BPR',
            'computation': f'Learning rate α = {lr}, Số epochs = {epochs}\n'
                          f'Final loss = {loss.item():.4f}',
            'meaning': f'Model đã được train {epochs} epochs, embeddings đã được cập nhật để tối ưu BPR loss'
        })
        
        self.training_time = time.time() - start_time
        progress_bar.empty()
        status_text.empty()
    
    def recommend(self, user_id: str, product_dict: Dict, top_k: int = 20, 
                  user_gender: str = None, user_age: int = None) -> Tuple[List[Tuple[str, float]], float]:
        """Generate recommendations for a user."""
        if self.model is None or user_id not in self.user_id_map:
            return [], 0.0
        
        start_time = time.time()
        
        self.model.eval()
        with torch.no_grad():
            user_emb, product_emb = self.model(self.edge_index)
            user_idx = self.user_id_map[user_id]
            
            # Calculate scores: r_hat = e_u^T * e_i
            scores = torch.matmul(user_emb[user_idx:user_idx+1], product_emb.t())
            scores = scores.squeeze(0)
            
            # Get top-k products
            top_indices = torch.topk(scores, min(top_k * 3, len(scores))).indices.tolist()
            
            # Filter by gender and age
            recommendations = []
            for idx in top_indices:
                product_id = self.reverse_product_map[idx]
                if product_id in product_dict:
                    product = product_dict[product_id]
                    
                    # Gender filter
                    if user_gender:
                        product_gender = product.get('gender', '').lower()
                        user_gender_lower = user_gender.lower()
                        if product_gender not in [user_gender_lower, 'unisex', '']:
                            continue
                    
                    # Age filter (simplified)
                    if user_age is not None and user_age <= 12:
                        product_gender = product.get('gender', '').lower()
                        if product_gender not in ['boys', 'girls', 'unisex', '']:
                            continue
                    
                    score = scores[idx].item()
                    recommendations.append((product_id, score))
                    
                    if len(recommendations) >= top_k:
                        break
        
        inference_time = time.time() - start_time
        return recommendations, inference_time


# ==================== CONTENT-BASED FILTERING ====================

class ContentBasedRecommender:
    """Content-Based Filtering using TF-IDF and Cosine Similarity."""
    
    # Interaction type weights (không dùng rating)
    INTERACTION_WEIGHTS = {
        'view': 1.0,
        'like': 2.0,
        'cart': 3.0,
        'purchase': 4.0,
        'review': 2.5
    }
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.product_vectors = None
        self.product_ids = []
        self.training_time = 0.0
        self.computation_steps = []
        self.matrices = {}
    
    def train(self, products_df: pd.DataFrame):
        """Train content-based model."""
        start_time = time.time()
        
        # Create text features for each product
        product_texts = []
        self.product_ids = []
        
        for _, row in products_df.iterrows():
            # Chỉ sử dụng các field được chỉ định: gender, masterCategory, subCategory, articleType, baseColour, usage
            text_parts = [
                str(row.get('gender', '')),
                str(row.get('masterCategory', '')),
                str(row.get('subCategory', '')),
                str(row.get('articleType', '')),
                str(row.get('baseColour', '')),
                str(row.get('usage', ''))
            ]
            text = ' '.join([p for p in text_parts if p and p != 'nan'])
            product_texts.append(text)
            self.product_ids.append(str(row['id']))
        
        # Example text for computation steps
        example_text = product_texts[0] if product_texts else ""
        example_words = example_text.split()[:10] if example_text else []
        
        self.computation_steps.append({
            'step': 'Bước 1: Tạo Feature Vector cho mỗi sản phẩm',
            'formula': 'v_i = TF-IDF(gender, masterCategory, subCategory, articleType, baseColour, usage)',
            'computation': f'Ví dụ sản phẩm 1: "{example_text[:50]}..."\n'
                          f'Các từ khóa: {", ".join(example_words[:5])}...\n'
                          f'Tổng số sản phẩm: {len(product_texts)}\n'
                          f'Các field sử dụng: gender, masterCategory, subCategory, articleType, baseColour, usage',
            'meaning': 'Mỗi sản phẩm được biểu diễn bằng vector TF-IDF từ 6 đặc tính: giới tính, danh mục chính, danh mục phụ, loại sản phẩm, màu sắc, mục đích sử dụng'
        })
        
        # Vectorize products
        self.product_vectors = self.vectorizer.fit_transform(product_texts)
        
        # Store TF-IDF matrix for visualization
        self.matrices['tfidf_matrix'] = self.product_vectors[:min(20, len(product_texts)), :min(50, self.product_vectors.shape[1])].toarray()
        
        # Example TF-IDF calculation
        if len(product_texts) > 0:
            vocab = list(self.vectorizer.vocabulary_.keys())[:5]
            example_vector = self.product_vectors[0].toarray()[0]
            non_zero_indices = np.nonzero(example_vector)[0][:5]
            example_values = example_vector[non_zero_indices]
            
            self.computation_steps.append({
                'step': 'Bước 1 (tiếp): Tính TF-IDF',
                'formula': 'TF-IDF(t, d) = TF(t, d) * IDF(t, D)',
                'computation': f'Vocabulary size: {len(self.vectorizer.vocabulary_)}\n'
                              f'Ví dụ vector sản phẩm 1: shape = {self.product_vectors[0].shape}\n'
                              f'Các giá trị TF-IDF khác 0 đầu tiên: {example_values.tolist()}',
                'meaning': f'Vector có {self.product_vectors.shape[1]} chiều, mỗi chiều đại diện cho một từ trong vocabulary'
            })
        
        self.training_time = time.time() - start_time
    
    def recommend(self, user_interactions: pd.DataFrame, products_df: pd.DataFrame,
                  product_dict: Dict, top_k: int = 20,
                  user_gender: str = None, user_age: int = None) -> Tuple[List[Tuple[str, float]], float]:
        """Generate recommendations based on user's interaction history with interaction type weights."""
        start_time = time.time()
        
        if len(user_interactions) == 0 or self.product_vectors is None:
            return [], 0.0
        
        # Get user's interacted products with weights (không dùng rating)
        interacted_data = []
        interacted_product_ids = set()  # Track interacted product IDs to exclude from recommendations
        for _, row in user_interactions.iterrows():
            pid = str(row['product_id'])
            interacted_product_ids.add(pid)  # Add to set of interacted products
            if pid in self.product_ids:
                interaction_type = str(row.get('interaction_type', 'view')).lower()
                weight = self.INTERACTION_WEIGHTS.get(interaction_type, 1.0)
                idx = self.product_ids.index(pid)
                interacted_data.append((idx, weight))
        
        if not interacted_data:
            return [], 0.0
        
        # Build user profile: weighted average of interacted products (không dùng rating)
        # Formula: u = (1/Σw_i) * Σ(w_i * v_i) for i in I_u
        indices = [idx for idx, _ in interacted_data]
        weights = np.array([w for _, w in interacted_data])
        weights = weights / weights.sum()  # Normalize weights
        
        # Convert sparse matrix to dense for weighted calculation
        product_vectors_dense = self.product_vectors[indices].toarray()
        weighted_vectors = product_vectors_dense * weights.reshape(-1, 1)
        user_profile = np.mean(weighted_vectors, axis=0)
        
        # Store computation steps
        self.computation_steps.append({
            'step': 'Bước 2: Xây dựng User Profile (với interaction weights)',
            'formula': 'u = (1/Σw_i) * Σ(w_i * v_i)',
            'computation': f'Số sản phẩm user đã tương tác: {len(interacted_data)}\n'
                          f'Weights: {dict(zip([self.product_ids[idx] for idx in indices[:3]], weights[:3]))}...\n'
                          f'User profile shape: {user_profile.shape}\n'
                          f'Ví dụ user profile (5 giá trị đầu): {user_profile[:5].tolist()}',
            'meaning': f'User profile là trung bình có trọng số của {len(interacted_data)} sản phẩm đã tương tác, với weight theo type: view=1.0, like=2.0, cart=3.0, purchase=4.0'
        })
        
        # Calculate cosine similarity: sim(u, i) = (u · i) / (||u|| * ||i||)
        similarities = cosine_similarity(user_profile.reshape(1, -1), self.product_vectors).flatten()
        
        # Store similarity matrix (top 20 products)
        top_similar_indices = np.argsort(similarities)[::-1][:20]
        self.matrices['similarity_matrix'] = similarities[top_similar_indices].reshape(-1, 1)
        
        # Example similarity calculation
        if len(similarities) > 0:
            max_sim_idx = np.argmax(similarities)
            max_sim_value = similarities[max_sim_idx]
            example_product_vector = self.product_vectors[max_sim_idx].toarray().flatten()
            
            # Calculate dot product and norms
            dot_product = np.dot(user_profile, example_product_vector)
            user_norm = np.linalg.norm(user_profile)
            product_norm = np.linalg.norm(example_product_vector)
            
            self.computation_steps.append({
                'step': 'Bước 3: Tính Cosine Similarity',
                'formula': 'sim(u, i) = (u · v_i) / (||u|| * ||v_i||)',
                'computation': f'Ví dụ với sản phẩm {max_sim_idx}:\n'
                              f'  u · v_i = {dot_product:.4f}\n'
                              f'  ||u|| = {user_norm:.4f}\n'
                              f'  ||v_i|| = {product_norm:.4f}\n'
                              f'  sim(u, i) = {dot_product:.4f} / ({user_norm:.4f} * {product_norm:.4f}) = {max_sim_value:.4f}',
                'meaning': f'Similarity = {max_sim_value:.4f} nghĩa là sản phẩm này giống {max_sim_value*100:.1f}% với sở thích của user (giá trị từ -1 đến 1, 1 = giống nhất)'
            })
        
        # Add ranking step
        if len(similarities) > 0:
            top_5_indices = np.argsort(similarities)[::-1][:5]
            top_5_scores = similarities[top_5_indices]
            
            self.computation_steps.append({
                'step': 'Bước 4: Ranking và Recommendation',
                'formula': 'Rank products by sim(u, i) descending',
                'computation': f'Top 5 sản phẩm:\n'
                              f'  Product {top_5_indices[0]}: similarity = {top_5_scores[0]:.4f}\n'
                              f'  Product {top_5_indices[1]}: similarity = {top_5_scores[1]:.4f}\n'
                              f'  Product {top_5_indices[2]}: similarity = {top_5_scores[2]:.4f}\n'
                              f'  Product {top_5_indices[3]}: similarity = {top_5_scores[3]:.4f}\n'
                              f'  Product {top_5_indices[4]}: similarity = {top_5_scores[4]:.4f}',
                'meaning': 'Sắp xếp sản phẩm theo similarity giảm dần, chọn top-K sản phẩm có similarity cao nhất để recommend'
            })
        
        # Get top-k products (excluding already interacted)
        top_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in top_indices:
            product_id = self.product_ids[idx]
            
            # Skip already interacted products
            if product_id in interacted_product_ids:
                continue
            
            # Filter by gender and age
            if product_id in product_dict:
                product = product_dict[product_id]
                
                if user_gender:
                    product_gender = product.get('gender', '').lower()
                    user_gender_lower = user_gender.lower()
                    if product_gender not in [user_gender_lower, 'unisex', '']:
                        continue
                
                if user_age is not None and user_age <= 12:
                    product_gender = product.get('gender', '').lower()
                    if product_gender not in ['boys', 'girls', 'unisex', '']:
                        continue
                
                score = float(similarities[idx])
                recommendations.append((product_id, score))
                
                if len(recommendations) >= top_k:
                    break
        
        inference_time = time.time() - start_time
        return recommendations, inference_time


# ==================== HYBRID MODEL ====================

class HybridRecommender:
    """Hybrid Model combining LightGCN and Content-Based Filtering."""
    
    def __init__(self, lightgcn: LightGCNRecommender, cbf: ContentBasedRecommender):
        self.lightgcn = lightgcn
        self.cbf = cbf
        self.alpha = 0.6  # Weight for LightGCN
        self.training_time = 0.0
    
    def train(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame):
        """Train both models."""
        start_time = time.time()
        
        # Train LightGCN
        self.lightgcn.train(interactions_df, epochs=30, lr=0.001)
        
        # Train CBF
        self.cbf.train(products_df)
        
        self.training_time = time.time() - start_time
    
    def recommend(self, user_id: str, user_interactions: pd.DataFrame,
                  products_df: pd.DataFrame, product_dict: Dict,
                  top_k: int = 20, user_gender: str = None, user_age: int = None) -> Tuple[List[Tuple[str, float]], float]:
        """Generate hybrid recommendations."""
        start_time = time.time()
        
        # Get recommendations from both models
        lightgcn_recs, _ = self.lightgcn.recommend(
            user_id, product_dict, top_k * 2, user_gender, user_age
        )
        cbf_recs, _ = self.cbf.recommend(
            user_interactions, products_df, product_dict, top_k * 2, user_gender, user_age
        )
        
        # Combine scores: score_hybrid = α * score_gnn + (1-α) * score_cbf
        # Formula: r_hybrid = α * r_gnn + (1-α) * r_cbf
        combined_scores = defaultdict(float)
        
        # Normalize LightGCN scores
        if lightgcn_recs:
            scores_gnn = [score for _, score in lightgcn_recs]
            if scores_gnn:
                max_gnn = max(scores_gnn)
                min_gnn = min(scores_gnn)
                gnn_range = max_gnn - min_gnn if max_gnn != min_gnn else 1.0
                
                for pid, score in lightgcn_recs:
                    normalized_score = (score - min_gnn) / gnn_range if gnn_range > 0 else 0.5
                    combined_scores[pid] += self.alpha * normalized_score
        
        # Normalize CBF scores
        if cbf_recs:
            scores_cbf = [score for _, score in cbf_recs]
            if scores_cbf:
                max_cbf = max(scores_cbf)
                min_cbf = min(scores_cbf)
                cbf_range = max_cbf - min_cbf if max_cbf != min_cbf else 1.0
                
                for pid, score in cbf_recs:
                    normalized_score = (score - min_cbf) / cbf_range if cbf_range > 0 else 0.5
                    combined_scores[pid] += (1 - self.alpha) * normalized_score
        
        # Sort by combined score
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        inference_time = time.time() - start_time
        return recommendations, inference_time


# ==================== EVALUATION METRICS ====================

def calculate_recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """Calculate Recall@K: |R ∩ T| / |T|"""
    if len(relevant) == 0:
        return 0.0
    
    top_k_rec = set(recommended[:k])
    relevant_set = set(relevant)
    
    intersection = len(top_k_rec & relevant_set)
    return intersection / len(relevant_set) if len(relevant_set) > 0 else 0.0


def calculate_ndcg_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """Calculate NDCG@K: DCG@K / IDCG@K"""
    def dcg_at_k(relevance_list: List[float], k: int) -> float:
        """DCG@K = sum(rel_i / log2(i+1))"""
        dcg = 0.0
        for i in range(min(len(relevance_list), k)):
            dcg += relevance_list[i] / math.log2(i + 2)
        return dcg
    
    # Build relevance list
    relevance_list = [1.0 if pid in relevant else 0.0 for pid in recommended[:k]]
    
    # Calculate DCG@K
    dcg = dcg_at_k(relevance_list, k)
    
    # Calculate IDCG@K (ideal: all relevant items ranked first)
    ideal_relevance = [1.0] * min(len(relevant), k)
    idcg = dcg_at_k(ideal_relevance, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(recommendations: List[Tuple[str, float]], 
                  test_interactions: pd.DataFrame,
                  k_values: List[int] = [10, 20]) -> Dict:
    """Evaluate model performance."""
    recommended_ids = [pid for pid, _ in recommendations]
    
    # Get ground truth (test interactions)
    relevant_ids = test_interactions['product_id'].unique().tolist()
    
    metrics = {}
    for k in k_values:
        metrics[f'recall_at_{k}'] = calculate_recall_at_k(recommended_ids, relevant_ids, k)
        metrics[f'ndcg_at_{k}'] = calculate_ndcg_at_k(recommended_ids, relevant_ids, k)
    
    return metrics


# ==================== OUTFIT RECOMMENDATION ====================

def get_outfit_categories():
    """Define outfit categories."""
    return {
        'topwear': ['Tshirts', 'Shirts', 'Tops', 'Sweaters', 'Sweatshirts', 'Jackets'],
        'bottomwear': ['Trousers', 'Jeans', 'Shorts', 'Skirts', 'Track Pants'],
        'footwear': ['Shoes', 'Sandals', 'Flip Flops'],
        'accessories': ['Bags', 'Watches', 'Belts', 'Caps']
    }


def recommend_outfit(current_product: Dict, product_dict: Dict, 
                    user_gender: str = None, user_age: int = None) -> Dict:
    """Recommend complementary items for an outfit."""
    outfit_categories = get_outfit_categories()
    
    # Determine current product category
    current_category = None
    article_type = current_product.get('articleType', '').lower()
    sub_category = current_product.get('subCategory', '').lower()
    
    if 'topwear' in sub_category or article_type in ['tshirts', 'shirts', 'tops', 'sweaters']:
        current_category = 'topwear'
    elif 'bottomwear' in sub_category or article_type in ['trousers', 'jeans', 'shorts']:
        current_category = 'bottomwear'
    elif 'footwear' in sub_category or article_type in ['shoes', 'sandals']:
        current_category = 'footwear'
    elif 'accessories' in sub_category or article_type in ['bags', 'watches']:
        current_category = 'accessories'
    
    # Recommend items from other categories
    outfit_recommendations = {}
    
    for category, article_types in outfit_categories.items():
        if category == current_category:
            continue
        
        # Find products in this category
        candidates = []
        for pid, product in product_dict.items():
            product_article = product.get('articleType', '').lower()
            product_sub = product.get('subCategory', '').lower()
            
            # Check if matches category
            matches = False
            if category == 'topwear' and ('topwear' in product_sub or product_article in article_types):
                matches = True
            elif category == 'bottomwear' and ('bottomwear' in product_sub or product_article in article_types):
                matches = True
            elif category == 'footwear' and ('footwear' in product_sub or product_article in article_types):
                matches = True
            elif category == 'accessories' and ('accessories' in product_sub or product_article in article_types):
                matches = True
            
            if matches:
                # Gender filter
                if user_gender:
                    product_gender = product.get('gender', '').lower()
                    user_gender_lower = user_gender.lower()
                    if product_gender not in [user_gender_lower, 'unisex']:
                        continue
                
                candidates.append((pid, product))
        
        # Select top 3-5 items
        outfit_recommendations[category] = candidates[:5]
    
    return outfit_recommendations


# ==================== STREAMLIT UI ====================

def main():
    st.title("🛍️ Hệ thống Gợi ý Sản phẩm")
    st.markdown("---")
    
    # Information about fields used
    with st.expander("ℹ️ Thông tin về Fields được sử dụng", expanded=False):
        st.markdown("""
        ### Fields từ Users (users.csv):
        - **age**: Tuổi của user (dùng để filter products phù hợp)
        - **gender**: Giới tính của user (male/female, dùng để filter products)
        - **interaction_history**: Lịch sử tương tác của user với các sản phẩm
          - Không dùng **rating**, chỉ dùng **interaction types** với weights:
            - view = 1.0
            - like = 2.0
            - cart = 3.0
            - purchase = 4.0
            - review = 2.5
        
        ### Fields từ Products (products.csv):
        - **gender**: Giới tính sản phẩm (Men/Women/Boys/Girls/Unisex)
        - **masterCategory**: Danh mục chính (Apparel, Footwear, Accessories, ...)
        - **subCategory**: Danh mục phụ (Topwear, Bottomwear, Shoes, Bags, ...)
        - **articleType**: Loại sản phẩm (Tshirts, Jeans, Shoes, Handbags, ...)
        - **baseColour**: Màu sắc cơ bản (Red, Blue, Black, White, ...)
        - **usage**: Mục đích sử dụng (Casual, Formal, Sports, ...)
        
        ### Fields KHÔNG sử dụng:
        - ❌ rating (không có trong hệ thống)
        - ❌ season, year (không dùng trong tính toán)
        - ❌ productDisplayName (chỉ dùng để hiển thị, không dùng trong tính toán)
        """)
    
    # Load data
    with st.spinner("Đang tải dữ liệu..."):
        user_dict, product_dict, interactions_df, users_df, products_df = load_all_data()
    
    st.sidebar.header("⚙️ Cấu hình")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Chọn mô hình",
        ["LightGCN (GNN)", "Content-Based Filtering", "Hybrid (LightGCN + CBF)"]
    )
    
    # User selection
    user_ids = list(user_dict.keys())
    selected_user_id = st.sidebar.selectbox("Chọn người dùng", user_ids)
    
    if selected_user_id:
        user = user_dict[selected_user_id]
        interaction_count = len(user.get('interactions', []))
        st.sidebar.info(f"**Người dùng:** {user.get('name', 'N/A')}\n\n"
                       f"**Tuổi:** {user.get('age', 'N/A')}\n\n"
                       f"**Giới tính:** {user.get('gender', 'N/A')}\n\n"
                       f"**Số interactions:** {interaction_count}")
    
    # Product selection for outfit recommendation
    product_ids = list(product_dict.keys())
    selected_product_id = st.sidebar.selectbox(
        "Chọn sản phẩm (cho Outfit recommendation)",
        [""] + product_ids[:100]  # Limit for performance
    )
    
    # Training section
    if st.sidebar.button("🚀 Train Models"):
        st.header("📊 Training Models")
        
        # Split data (80% train, 20% test)
        train_size = int(len(interactions_df) * 0.8)
        train_interactions = interactions_df.iloc[:train_size]
        test_interactions = interactions_df.iloc[train_size:]
        
        # Initialize models
        if model_type == "LightGCN (GNN)":
            model = LightGCNRecommender()
            
            model.train(train_interactions, epochs=30, lr=0.001)
            st.success(f"✅ LightGCN trained in {model.training_time:.2f}s")
            
            # Display Algorithm (A-Z) với computation steps gộp chung
            with st.expander("📖 LightGCN Algorithm (A-Z)", expanded=True):
                st.markdown("""
                """)
                
                # Hiển thị từng bước với công thức, áp dụng công thức, giải thích và ma trận - gộp hoàn toàn
                if model.computation_steps:
                    for step_info in model.computation_steps:
                        # Xác định số bước từ step_info
                        step_num = step_info['step'].split(':')[0] if ':' in step_info['step'] else step_info['step']
                        
                        with st.expander(f"{step_info['step']}", expanded=False):
                            st.markdown(f"**Công thức:** `{step_info['formula']}`")
                            st.markdown(f"**Áp dụng công thức:**")
                            st.code(step_info['computation'], language='text')
                            st.markdown(f"**Giải thích ý nghĩa:** {step_info['meaning']}")
                            
                            # Hiển thị ma trận tương ứng với từng bước
                            if 'Bước 2: Khởi tạo Embeddings' in step_info['step'] and 'initial_user_embeddings' in model.matrices:
                                st.markdown("**📈 Ma trận User Embeddings ban đầu (10x10):**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(model.matrices['initial_user_embeddings'], 
                                           annot=True, fmt='.3f', cmap='viridis', ax=ax,
                                           xticklabels=False, yticklabels=False)
                                ax.set_title('Initial User Embeddings Matrix')
                                st.pyplot(fig)
                                st.caption("Ma trận embeddings ban đầu của 10 users đầu tiên, mỗi user có vector 10 chiều")
                            
                            elif 'Bước 7: Gradient Descent' in step_info['step'] and 'final_user_embeddings' in model.matrices:
                                st.markdown("**📈 Ma trận User Embeddings sau training (10x10):**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(model.matrices['final_user_embeddings'], 
                                           annot=True, fmt='.3f', cmap='viridis', ax=ax,
                                           xticklabels=False, yticklabels=False)
                                ax.set_title('Final User Embeddings Matrix')
                                st.pyplot(fig)
                                st.caption("Ma trận embeddings sau training, đã được tối ưu để phân biệt sở thích users")
                            
                            elif 'Bước 5: Dự đoán' in step_info['step'] and 'similarity_matrix' in model.matrices:
                                st.markdown("**📈 Ma trận Similarity (User x Product):**")
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(model.matrices['similarity_matrix'], 
                                           annot=True, fmt='.3f', cmap='coolwarm', ax=ax,
                                           xticklabels=False, yticklabels=False)
                                ax.set_title('User-Product Similarity Matrix (10x10)')
                                ax.set_xlabel('Products')
                                ax.set_ylabel('Users')
                                st.pyplot(fig)
                                st.caption("Ma trận similarity giữa 10 users và 10 products đầu tiên. Giá trị càng cao = user càng thích product")
            
        elif model_type == "Content-Based Filtering":
            model = ContentBasedRecommender()
            
            model.train(products_df)
            st.success(f"✅ Content-Based Filtering trained in {model.training_time:.2f}s")
            
            # Display Algorithm (A-Z) với computation steps gộp chung
            with st.expander("📖 Content-Based Filtering Algorithm (A-Z)", expanded=True):
                st.markdown("""
                """)
                
                # Hiển thị từng bước với công thức, áp dụng công thức, giải thích và ma trận - gộp hoàn toàn
                if model.computation_steps:
                    for step_info in model.computation_steps:
                        with st.expander(f"{step_info['step']}", expanded=False):
                            st.markdown(f"**Công thức:** `{step_info['formula']}`")
                            st.markdown(f"**Áp dụng công thức:**")
                            st.code(step_info['computation'], language='text')
                            st.markdown(f"**Giải thích ý nghĩa:** {step_info['meaning']}")
                            
                            # Hiển thị ma trận tương ứng với từng bước
                            if 'Bước 1' in step_info['step'] and 'TF-IDF' in step_info['step'] and 'tfidf_matrix' in model.matrices:
                                st.markdown("**📈 Ma trận TF-IDF (20 sản phẩm đầu x 50 features):**")
                                fig, ax = plt.subplots(figsize=(12, 8))
                                sns.heatmap(model.matrices['tfidf_matrix'], 
                                           annot=False, fmt='.2f', cmap='YlOrRd', ax=ax,
                                           xticklabels=False, yticklabels=False)
                                ax.set_title('TF-IDF Matrix (Products x Features)')
                                ax.set_xlabel('Features (words)')
                                ax.set_ylabel('Products')
                                st.pyplot(fig)
                                st.caption("Ma trận TF-IDF: mỗi hàng là một sản phẩm, mỗi cột là một từ trong vocabulary. Giá trị càng cao = từ đó quan trọng với sản phẩm")
                            
                            elif 'Bước 3: Tính Cosine Similarity' in step_info['step'] and 'similarity_matrix' in model.matrices:
                                st.markdown("**📈 Ma trận Similarity (Top 20 sản phẩm):**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(model.matrices['similarity_matrix'], 
                                           annot=True, fmt='.3f', cmap='coolwarm', ax=ax,
                                           xticklabels=False, yticklabels=False)
                                ax.set_title('User-Product Similarity (Top 20)')
                                ax.set_xlabel('Products')
                                ax.set_ylabel('Similarity Score')
                                st.pyplot(fig)
                                st.caption("Similarity scores của top 20 sản phẩm. Giá trị càng cao = sản phẩm càng phù hợp với user")
            
        elif model_type == "Hybrid (LightGCN + CBF)":
            lightgcn = LightGCNRecommender()
            cbf = ContentBasedRecommender()
            model = HybridRecommender(lightgcn, cbf)
            
            model.train(train_interactions, products_df)
            st.success(f"✅ Hybrid model trained in {model.training_time:.2f}s")
            
            # Display Algorithm (A-Z) với computation steps gộp chung
            with st.expander("📖 Hybrid Algorithm (A-Z)", expanded=True):
                st.markdown("""
                ### Bước 1: Train LightGCN Model
                - Áp dụng toàn bộ thuật toán LightGCN (xem phần LightGCN)
                - Input: users (age, gender, interaction_history), products (gender, masterCategory, subCategory, articleType, baseColour, usage)
                - Kết quả: r_gnn = w_type * (e_u^T · e_i) (với interaction weights, không dùng rating)
                
                ### Bước 2: Train Content-Based Model
                - Áp dụng toàn bộ thuật toán Content-Based (xem phần CBF)
                - Input: products (gender, masterCategory, subCategory, articleType, baseColour, usage)
                - Kết quả: r_cbf = sim(u, i) = (u · v_i) / (||u|| * ||v_i||)
                - User profile dựa trên interaction_history (không dùng rating)
                
                ### Bước 3: Normalize Scores
                - **Công thức:** r_norm = (r - r_min) / (r_max - r_min)
                  - Chuẩn hóa scores về khoảng [0, 1]
                
                ### Bước 4: Weighted Combination
                - **Công thức:** r_hybrid = α * r_gnn_norm + (1-α) * r_cbf_norm
                  - α: trọng số cho LightGCN (thường α = 0.6)
                  - (1-α): trọng số cho Content-Based
                  - Kết hợp ưu điểm của cả 2 mô hình
                
                ### Bước 5: Ranking
                - Sắp xếp sản phẩm theo r_hybrid giảm dần
                - Chọn top-K sản phẩm
                """)
                
                # Hiển thị từng bước với công thức, áp dụng công thức, giải thích và ma trận - gộp hoàn toàn
                if lightgcn.computation_steps or cbf.computation_steps:
                    st.markdown("**LightGCN Computation Steps:**")
                    for step_info in lightgcn.computation_steps:
                        with st.expander(f"LightGCN - {step_info['step']}", expanded=False):
                            st.markdown(f"**Công thức:** `{step_info['formula']}`")
                            st.markdown(f"**Áp dụng công thức:**")
                            st.code(step_info['computation'], language='text')
                            st.markdown(f"**Giải thích ý nghĩa:** {step_info['meaning']}")
                            
                            # Ma trận cho LightGCN trong các bước tương ứng
                            if 'Bước 2: Khởi tạo Embeddings' in step_info['step'] and 'initial_user_embeddings' in lightgcn.matrices:
                                st.markdown("**📈 Ma trận User Embeddings ban đầu:**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(lightgcn.matrices['initial_user_embeddings'], 
                                           annot=True, fmt='.3f', cmap='viridis', ax=ax)
                                ax.set_title('Initial User Embeddings Matrix')
                                st.pyplot(fig)
                                st.caption("Ma trận embeddings ban đầu của 10 users đầu tiên")
                            
                            elif 'Bước 7: Gradient Descent' in step_info['step'] and 'final_user_embeddings' in lightgcn.matrices:
                                st.markdown("**📈 Ma trận User Embeddings sau training:**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(lightgcn.matrices['final_user_embeddings'], 
                                           annot=True, fmt='.3f', cmap='viridis', ax=ax)
                                ax.set_title('Final User Embeddings Matrix')
                                st.pyplot(fig)
                                st.caption("Ma trận embeddings sau training")
                            
                            elif 'Bước 5: Dự đoán' in step_info['step'] and 'similarity_matrix' in lightgcn.matrices:
                                st.markdown("**📈 Ma trận Similarity (LightGCN):**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(lightgcn.matrices['similarity_matrix'], 
                                           annot=True, fmt='.3f', cmap='coolwarm', ax=ax)
                                ax.set_title('LightGCN Similarity Matrix')
                                st.pyplot(fig)
                                st.caption("Ma trận similarity giữa users và products")
                    
                    st.markdown("**Content-Based Computation Steps:**")
                    for step_info in cbf.computation_steps:
                        with st.expander(f"CBF - {step_info['step']}", expanded=False):
                            st.markdown(f"**Công thức:** `{step_info['formula']}`")
                            st.markdown(f"**Áp dụng công thức:**")
                            st.code(step_info['computation'], language='text')
                            st.markdown(f"**Giải thích ý nghĩa:** {step_info['meaning']}")
                            
                            # Ma trận cho CBF trong các bước tương ứng
                            if 'Bước 1' in step_info['step'] and 'TF-IDF' in step_info['step'] and 'tfidf_matrix' in cbf.matrices:
                                st.markdown("**📈 Ma trận TF-IDF:**")
                                fig, ax = plt.subplots(figsize=(12, 8))
                                sns.heatmap(cbf.matrices['tfidf_matrix'], 
                                           annot=False, fmt='.2f', cmap='YlOrRd', ax=ax)
                                ax.set_title('TF-IDF Matrix (Products x Features)')
                                ax.set_xlabel('Features (words)')
                                ax.set_ylabel('Products')
                                st.pyplot(fig)
                                st.caption("Ma trận TF-IDF: mỗi hàng là một sản phẩm, mỗi cột là một từ trong vocabulary")
                            
                            elif 'Bước 3: Tính Cosine Similarity' in step_info['step'] and 'similarity_matrix' in cbf.matrices:
                                st.markdown("**📈 Ma trận Similarity (CBF):**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cbf.matrices['similarity_matrix'], 
                                           annot=True, fmt='.3f', cmap='coolwarm', ax=ax)
                                ax.set_title('CBF Similarity Matrix (Top 20)')
                                st.pyplot(fig)
                                st.caption("Similarity scores của top 20 sản phẩm")
                    
                    # Display hybrid combination computation
                    with st.expander("Bước 3 & 4: Normalize Scores & Weighted Combination", expanded=False):
                        st.markdown("""
                        **Bước 3: Normalize Scores**
                        - **Công thức:** r_norm = (r - r_min) / (r_max - r_min)
                        - **Ví dụ:** Nếu r_gnn có range [0.2, 0.8] và r_cbf có range [0.1, 0.9]
                          - r_gnn_norm = (r_gnn - 0.2) / (0.8 - 0.2) = (r_gnn - 0.2) / 0.6
                          - r_cbf_norm = (r_cbf - 0.1) / (0.9 - 0.1) = (r_cbf - 0.1) / 0.8
                        - **Ý nghĩa:** Chuẩn hóa về cùng scale [0, 1] để có thể kết hợp công bằng
                        
                        **Bước 4: Weighted Combination**
                        - **Công thức:** r_hybrid = α * r_gnn_norm + (1-α) * r_cbf_norm
                        - **Ví dụ:** Với α = 0.6, nếu r_gnn_norm = 0.7 và r_cbf_norm = 0.8
                          - r_hybrid = 0.6 * 0.7 + 0.4 * 0.8 = 0.42 + 0.32 = 0.74
                        - **Ý nghĩa:** Kết hợp 60% từ LightGCN (collaborative) và 40% từ CBF (content-based)
                        """)
                    
                    # Hiển thị ma trận trong các bước tương ứng của LightGCN và CBF
                    st.markdown("**LightGCN Computation Steps với Ma trận:**")
                    for step_info in lightgcn.computation_steps:
                        with st.expander(f"LightGCN - {step_info['step']}", expanded=False):
                            st.markdown(f"**Công thức:** `{step_info['formula']}`")
                            st.markdown(f"**Áp dụng công thức:**")
                            st.code(step_info['computation'], language='text')
                            st.markdown(f"**Giải thích ý nghĩa:** {step_info['meaning']}")
                            
                            # Ma trận cho LightGCN trong các bước tương ứng
                            if 'Bước 2: Khởi tạo Embeddings' in step_info['step'] and 'initial_user_embeddings' in lightgcn.matrices:
                                st.markdown("**📈 Ma trận User Embeddings ban đầu:**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(lightgcn.matrices['initial_user_embeddings'], 
                                           annot=True, fmt='.3f', cmap='viridis', ax=ax)
                                ax.set_title('Initial User Embeddings Matrix')
                                st.pyplot(fig)
                                st.caption("Ma trận embeddings ban đầu của 10 users đầu tiên")
                            
                            elif 'Bước 7: Gradient Descent' in step_info['step'] and 'final_user_embeddings' in lightgcn.matrices:
                                st.markdown("**📈 Ma trận User Embeddings sau training:**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(lightgcn.matrices['final_user_embeddings'], 
                                           annot=True, fmt='.3f', cmap='viridis', ax=ax)
                                ax.set_title('Final User Embeddings Matrix')
                                st.pyplot(fig)
                                st.caption("Ma trận embeddings sau training")
                            
                            elif 'Bước 5: Dự đoán' in step_info['step'] and 'similarity_matrix' in lightgcn.matrices:
                                st.markdown("**📈 Ma trận Similarity (LightGCN):**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(lightgcn.matrices['similarity_matrix'], 
                                           annot=True, fmt='.3f', cmap='coolwarm', ax=ax)
                                ax.set_title('LightGCN Similarity Matrix')
                                st.pyplot(fig)
                                st.caption("Ma trận similarity giữa users và products")
                    
                    st.markdown("**Content-Based Computation Steps với Ma trận:**")
                    for step_info in cbf.computation_steps:
                        with st.expander(f"CBF - {step_info['step']}", expanded=False):
                            st.markdown(f"**Công thức:** `{step_info['formula']}`")
                            st.markdown(f"**Áp dụng công thức:**")
                            st.code(step_info['computation'], language='text')
                            st.markdown(f"**Giải thích ý nghĩa:** {step_info['meaning']}")
                            
                            # Ma trận cho CBF trong các bước tương ứng
                            if 'Bước 1' in step_info['step'] and 'TF-IDF' in step_info['step'] and 'tfidf_matrix' in cbf.matrices:
                                st.markdown("**📈 Ma trận TF-IDF:**")
                                fig, ax = plt.subplots(figsize=(12, 8))
                                sns.heatmap(cbf.matrices['tfidf_matrix'], 
                                           annot=False, fmt='.2f', cmap='YlOrRd', ax=ax)
                                ax.set_title('TF-IDF Matrix (Products x Features)')
                                ax.set_xlabel('Features (words)')
                                ax.set_ylabel('Products')
                                st.pyplot(fig)
                                st.caption("Ma trận TF-IDF: mỗi hàng là một sản phẩm, mỗi cột là một từ trong vocabulary")
                            
                            elif 'Bước 3: Tính Cosine Similarity' in step_info['step'] and 'similarity_matrix' in cbf.matrices:
                                st.markdown("**📈 Ma trận Similarity (CBF):**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cbf.matrices['similarity_matrix'], 
                                           annot=True, fmt='.3f', cmap='coolwarm', ax=ax)
                                ax.set_title('CBF Similarity Matrix (Top 20)')
                                st.pyplot(fig)
                                st.caption("Similarity scores của top 20 sản phẩm")
        
        # Store in session state
        st.session_state['model'] = model
        st.session_state['model_type'] = model_type
        st.session_state['train_interactions'] = train_interactions
        st.session_state['test_interactions'] = test_interactions
    
    # Recommendation section
    if 'model' in st.session_state:
        st.header("🎯 Recommendations")
        
        user = user_dict[selected_user_id]
        user_gender = user.get('gender')
        user_age = user.get('age')
        
        # Get user interactions for training
        user_train_interactions = st.session_state.get('train_interactions', interactions_df)
        user_train_interactions = user_train_interactions[
            user_train_interactions['user_id'] == selected_user_id
        ]
        
        # Generate recommendations
        model = st.session_state['model']
        model_type = st.session_state['model_type']
        
        if model_type == "LightGCN (GNN)":
            recommendations, inference_time = model.recommend(
                selected_user_id, product_dict, top_k=20,
                user_gender=user_gender, user_age=user_age
            )
        elif model_type == "Content-Based Filtering":
            recommendations, inference_time = model.recommend(
                user_train_interactions, products_df, product_dict, top_k=20,
                user_gender=user_gender, user_age=user_age
            )
        else:  # Hybrid
            recommendations, inference_time = model.recommend(
                selected_user_id, user_train_interactions, products_df, product_dict, top_k=20,
                user_gender=user_gender, user_age=user_age
            )
        
        # Personalize recommendations
        st.subheader("👤 Personalized Recommendations")
        st.markdown("**Dựa trên lịch sử tương tác (interaction_history) của bạn:**")
        st.info(f"**Thông tin user:** Tuổi: {user_age}, Giới tính: {user_gender}\n"
                f"**Fields sử dụng:** age, gender, interaction_history (không dùng rating)")
        
        # Filter recommendations by articleType if a product is selected
        filtered_recommendations = recommendations
        if selected_product_id and selected_product_id in product_dict:
            current_product = product_dict[selected_product_id]
            target_article_type = current_product.get('articleType')
            
            if target_article_type:
                st.info(f"**🔍 Lọc theo articleType:** {target_article_type} (từ sản phẩm đã chọn)")
                filtered_recommendations = [
                    (product_id, score) for product_id, score in recommendations
                    if product_id in product_dict and 
                       product_dict[product_id].get('articleType') == target_article_type
                ]
                
                if not filtered_recommendations:
                    st.warning(f"Không tìm thấy sản phẩm nào cùng articleType '{target_article_type}'. Hiển thị tất cả recommendations.")
                    filtered_recommendations = recommendations
        
        cols = st.columns(4)
        for idx, (product_id, score) in enumerate(filtered_recommendations[:12]):
            if product_id in product_dict:
                product = product_dict[product_id]
                with cols[idx % 4]:
                    st.markdown(f"**{product.get('productDisplayName', 'N/A')[:30]}...**")
                    st.caption(f"Score: {score:.4f}")
                    st.caption(f"Category: {product.get('subCategory', 'N/A')}")
                    st.caption(f"ArticleType: {product.get('articleType', 'N/A')}")
                    st.caption(f"Color: {product.get('baseColour', 'N/A')}")
        
        # Outfit recommendations
        if selected_product_id and selected_product_id in product_dict:
            st.subheader("👔 Outfit Recommendations")
            st.markdown("**Các sản phẩm đi kèm để tạo bộ trang phục hoàn chỉnh:**")
            
            current_product = product_dict[selected_product_id]
            outfit_recs = recommend_outfit(
                current_product, product_dict, user_gender=user_gender, user_age=user_age
            )
            
            for category, items in outfit_recs.items():
                if items:
                    st.markdown(f"**{category.upper()}:**")
                    cols = st.columns(min(5, len(items)))
                    for idx, (pid, product) in enumerate(items):
                        with cols[idx]:
                            st.markdown(f"• {product.get('productDisplayName', 'N/A')[:25]}...")
                            st.caption(f"{product.get('articleType', 'N/A')}")
        
        # Evaluation metrics
        st.subheader("📊 Evaluation Metrics")
        
        # Get test interactions for this user
        test_interactions = st.session_state.get('test_interactions', pd.DataFrame())
        user_test_interactions = test_interactions[
            test_interactions['user_id'] == selected_user_id
        ]
        
        if len(user_test_interactions) > 0:
            metrics = evaluate_model(recommendations, user_test_interactions, k_values=[10, 20])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Recall@10", f"{metrics['recall_at_10']:.4f}")
            with col2:
                st.metric("Recall@20", f"{metrics['recall_at_20']:.4f}")
            with col3:
                st.metric("NDCG@10", f"{metrics['ndcg_at_10']:.4f}")
            with col4:
                st.metric("NDCG@20", f"{metrics['ndcg_at_20']:.4f}")
            
            # Explanation
            with st.expander("📖 Giải thích Metrics"):
                st.markdown("""
                **Recall@K:**
                - **Công thức:** Recall@K = |R ∩ T| / |T|
                  - R: tập sản phẩm được recommend trong top-K
                  - T: tập sản phẩm thực tế user đã tương tác (ground truth)
                - **Ý nghĩa:** Tỷ lệ sản phẩm relevant được tìm thấy trong top-K
                - **Ví dụ:** Nếu user đã mua 10 sản phẩm và hệ thống recommend đúng 7 trong top-10 → Recall@10 = 0.7
                
                **NDCG@K (Normalized Discounted Cumulative Gain):**
                - **Công thức:** NDCG@K = DCG@K / IDCG@K
                  - DCG@K = Σ (rel_i / log₂(i+1)) với i từ 1 đến K
                  - IDCG@K: DCG lý tưởng (tất cả relevant items xếp đầu)
                - **Ý nghĩa:** Đánh giá chất lượng ranking, ưu tiên items relevant ở vị trí cao
                - **Ví dụ:** NDCG@10 = 0.8 nghĩa là ranking tốt 80% so với ranking lý tưởng
                """)
        
        st.info(f"⏱️ Inference time: {inference_time*1000:.2f}ms")
    
    # Model comparison table
    if st.sidebar.button("📈 Compare All Models"):
        st.header("📊 Model Comparison")
        
        # Train all models
        train_size = int(len(interactions_df) * 0.8)
        train_interactions = interactions_df.iloc[:train_size]
        test_interactions = interactions_df.iloc[train_size:]
        
        models = {}
        results = []
        
        # LightGCN
        st.write("Training LightGCN...")
        lightgcn = LightGCNRecommender()
        lightgcn.train(train_interactions, epochs=20, lr=0.001)
        models['LightGCN'] = lightgcn
        
        # CBF
        st.write("Training Content-Based...")
        cbf = ContentBasedRecommender()
        cbf.train(products_df)
        models['Content-Based'] = cbf
        
        # Hybrid
        st.write("Training Hybrid...")
        hybrid = HybridRecommender(lightgcn, cbf)
        hybrid.train(train_interactions, products_df)
        models['Hybrid'] = hybrid
        
        # Evaluate on sample users
        sample_users = user_ids[:10]  # Evaluate on 10 users
        
        for model_name, model in models.items():
            recalls_10 = []
            recalls_20 = []
            ndcgs_10 = []
            ndcgs_20 = []
            inference_times = []
            
            for user_id in sample_users:
                user = user_dict[user_id]
                user_test = test_interactions[test_interactions['user_id'] == user_id]
                user_train = train_interactions[train_interactions['user_id'] == user_id]
                
                if model_name == "LightGCN":
                    recs, inf_time = model.recommend(
                        user_id, product_dict, top_k=20,
                        user_gender=user.get('gender'), user_age=user.get('age')
                    )
                elif model_name == "Content-Based":
                    recs, inf_time = model.recommend(
                        user_train, products_df, product_dict, top_k=20,
                        user_gender=user.get('gender'), user_age=user.get('age')
                    )
                else:  # Hybrid
                    recs, inf_time = model.recommend(
                        user_id, user_train, products_df, product_dict, top_k=20,
                        user_gender=user.get('gender'), user_age=user.get('age')
                    )
                
                if len(user_test) > 0:
                    metrics = evaluate_model(recs, user_test, k_values=[10, 20])
                    recalls_10.append(metrics['recall_at_10'])
                    recalls_20.append(metrics['recall_at_20'])
                    ndcgs_10.append(metrics['ndcg_at_10'])
                    ndcgs_20.append(metrics['ndcg_at_20'])
                    inference_times.append(inf_time)
            
            # Calculate averages
            training_time = lightgcn.training_time if model_name == "LightGCN" else (
                cbf.training_time if model_name == "Content-Based" else hybrid.training_time
            )
            
            results.append({
                'Model': model_name,
                'Recall@10': f"{np.mean(recalls_10):.4f}" if recalls_10 else "N/A",
                'Recall@20': f"{np.mean(recalls_20):.4f}" if recalls_20 else "N/A",
                'NDCG@10': f"{np.mean(ndcgs_10):.4f}" if ndcgs_10 else "N/A",
                'NDCG@20': f"{np.mean(ndcgs_20):.4f}" if ndcgs_20 else "N/A",
                'Thời gian train': f"{training_time:.2f}s",
                'Thời gian inference/user': f"{np.mean(inference_times)*1000:.2f}ms" if inference_times else "N/A"
            })
        
        # Display comparison table
        comparison_df = pd.DataFrame(results)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Store results
        st.session_state['comparison_results'] = comparison_df


if __name__ == "__main__":
    main()

