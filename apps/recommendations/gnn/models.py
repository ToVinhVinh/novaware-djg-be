"""GNN recommendation engine using PyTorch Geometric + LightGCN."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from celery import shared_task

from apps.recommendations.common import BaseRecommendationEngine, CandidateFilter
from apps.recommendations.common.constants import INTERACTION_WEIGHTS
from apps.recommendations.common.context import RecommendationContext
from apps.recommendations.common.gender_utils import normalize_gender
from apps.users.models import UserInteraction

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Data = None
    MessagePassing = None
    add_self_loops = None
    degree = None


if TORCH_AVAILABLE:
    class LightGCNConv(MessagePassing):
        """LightGCN convolution layer."""
        
        def __init__(self, **kwargs):
            super().__init__(aggr='add', **kwargs)
        
        def forward(self, x, edge_index):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            
            # Propagate
            return self.propagate(edge_index, x=x, norm=norm)
        
        def message(self, x_j, norm):
            return norm.view(-1, 1) * x_j


    class LightGCNModel(nn.Module):
        """LightGCN model for collaborative filtering."""
        
        def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_layers: int = 3):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            self.num_layers = num_layers
            
            # User and item embeddings
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            # Initialize embeddings
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
            
            # LightGCN layers
            self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        
        def forward(self, edge_index, user_idx=None, item_idx=None):
            """Forward pass."""
            user_emb = self.user_embedding.weight
            item_emb = self.item_embedding.weight
            
            # Stack user and item embeddings
            all_emb = torch.cat([user_emb, item_emb], dim=0)
            
            # Apply LightGCN layers and collect embeddings
            embs = [all_emb]
            for conv in self.convs:
                all_emb = conv(all_emb, edge_index)
                embs.append(all_emb)
            
            # Average embeddings from all layers (LightGCN approach)
            final_emb = torch.mean(torch.stack(embs), dim=0)
            
            # Split back into user and item embeddings
            user_final = final_emb[:self.num_users]
            item_final = final_emb[self.num_users:]
            
            if user_idx is not None and item_idx is not None:
                return user_final[user_idx], item_final[item_idx]
            return user_final, item_final
else:
    # Dummy classes when torch is not available
    LightGCNConv = None
    LightGCNModel = None


class GNNRecommendationEngine(BaseRecommendationEngine):
    model_name = "gnn"

    def _train_impl(self) -> dict[str, Any]:
        if not TORCH_AVAILABLE:
            logger.warning(f"[{self.model_name}] PyTorch not available, falling back to co-occurrence method")
            return self._fallback_training()
        
        logger.info(f"[{self.model_name}] Loading interactions for LightGCN training...")
        
        # Load interactions from MongoDB (using mongo_models)
        try:
            from apps.users.mongo_models import UserInteraction as MongoInteraction
            from apps.products.mongo_models import Product as MongoProduct
            from apps.users.mongo_models import User as MongoUser
            from bson import ObjectId
            
            # Get all interactions
            interactions = list(MongoInteraction.objects.all())
            logger.info(f"[{self.model_name}] Loaded {len(interactions)} interactions")
            
            if not interactions:
                logger.warning(f"[{self.model_name}] No interactions found in MongoDB, using fallback method")
                return self._fallback_training()
            
            # Build user and item mappings
            user_ids_set = set()
            product_ids_set = set()
            for it in interactions:
                if it.user_id:
                    user_ids_set.add(str(it.user_id))
                if it.product_id:
                    product_ids_set.add(str(it.product_id))
            
            user_ids = sorted(list(user_ids_set))
            product_ids = sorted(list(product_ids_set))
            
            if not user_ids or not product_ids:
                logger.warning(f"[{self.model_name}] No valid users or products found (users: {len(user_ids)}, products: {len(product_ids)}), using fallback method")
                return self._fallback_training()
            
            user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
            product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
            
            num_users = len(user_ids)
            num_items = len(product_ids)
            
            logger.info(f"[{self.model_name}] Building graph: {num_users} users, {num_items} items")
            
            # Build edge list for PyTorch Geometric
            edge_list = []
            product_frequency: dict[str, float] = defaultdict(float)
            
            for it in interactions:
                if not it.user_id or not it.product_id:
                    continue
                user_str = str(it.user_id)
                product_str = str(it.product_id)
                
                if user_str not in user_to_idx or product_str not in product_to_idx:
                    continue
                
                user_idx = user_to_idx[user_str]
                item_idx = product_to_idx[product_str]
                weight = INTERACTION_WEIGHTS.get(it.interaction_type, 1.0)
                
                # Add user-item edge (bipartite graph)
                # User nodes: 0 to num_users-1
                # Item nodes: num_users to num_users+num_items-1
                edge_list.append([user_idx, num_users + item_idx])
                edge_list.append([num_users + item_idx, user_idx])  # Undirected
                
                product_frequency[product_str] += weight
            
            if not edge_list:
                logger.warning(f"[{self.model_name}] No edges found, using fallback co-occurrence")
                return self._fallback_training()
            
            # Convert to PyTorch Geometric format
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # Create PyTorch Geometric data object
            data = Data(edge_index=edge_index, num_nodes=num_users + num_items)
            
            # Train LightGCN model
            logger.info(f"[{self.model_name}] Training LightGCN model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LightGCNModel(num_users, num_items, embedding_dim=64, num_layers=3).to(device)
            data = data.to(device)
            
            # Simple training loop (can be extended with proper loss function)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            
            # Train for a few epochs
            num_epochs = 10
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                user_emb, item_emb = model(data.edge_index)
                
                # Simple reconstruction loss (can be improved with BPR loss)
                # For now, just ensure embeddings are learned
                loss = -torch.mean(user_emb) - torch.mean(item_emb)  # Placeholder loss
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"[{self.model_name}] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            
            # Get final embeddings
            model.eval()
            with torch.no_grad():
                user_emb, item_emb = model(data.edge_index)
                user_embeddings = user_emb.cpu().numpy()
                item_embeddings = item_emb.cpu().numpy()
            
            logger.info(f"[{self.model_name}] Training completed. Embeddings shape: users={user_embeddings.shape}, items={item_embeddings.shape}")
            
            # Calculate sparsity: ratio of non-zero interactions to total possible interactions
            total_possible = num_users * num_items
            actual_interactions = len(edge_list) // 2  # Divide by 2 because edges are undirected
            sparsity = 1.0 - (actual_interactions / total_possible) if total_possible > 0 else 1.0
            
            # Store training parameters
            embedding_dim = 64  # From model initialization
            num_training_samples = actual_interactions  # Number of unique user-item interactions
            
            # Create matrix data for display
            max_display = min(5, num_users, num_items) if num_users > 0 and num_items > 0 else 0
            similarity_matrix = np.dot(user_embeddings[:max_display], item_embeddings[:max_display].T) if max_display > 0 else np.array([])
            
            matrix_data = {
                "shape": [num_users, num_items],
                "display_shape": [max_display, max_display],
                "data": similarity_matrix.tolist() if max_display > 0 else [],
                "user_ids": user_ids[:max_display] if max_display > 0 else [],
                "product_ids": product_ids[:max_display] if max_display > 0 else [],
                "description": "User-Item Embedding Similarity Matrix (LightGCN)",
                "row_label": "User ID",
                "col_label": "Product ID",
                "value_description": "Similarity score from LightGCN embeddings",
                "sparsity": float(sparsity),
            }
            
            return {
                "user_ids": user_ids,
                "product_ids": product_ids,
                "user_to_idx": user_to_idx,
                "product_to_idx": product_to_idx,
                "user_embeddings": user_embeddings.tolist(),
                "item_embeddings": item_embeddings.tolist(),
                "product_frequency": dict(product_frequency),
                "matrix_data": matrix_data,
                # Training parameters for API response
                "num_users": num_users,
                "num_products": num_items,
                "num_interactions": actual_interactions,
                "num_training_samples": num_training_samples,
                "embedding_dim": embedding_dim,
            }
            
        except Exception as e:
            logger.error(f"[{self.model_name}] Error in LightGCN training: {e}", exc_info=True)
            logger.info(f"[{self.model_name}] Falling back to co-occurrence method")
            return self._fallback_training()
    
    def _fallback_training(self) -> dict[str, Any]:
        """Fallback to simple co-occurrence if LightGCN fails."""
        logger.info(f"[{self.model_name}] Using fallback co-occurrence training")
        co_occurrence: dict[int, dict[int, float]] = defaultdict(dict)
        product_frequency: dict[int, float] = defaultdict(float)

        interactions = (
            UserInteraction.objects.select_related("product")
            .order_by("user_id", "timestamp")
        )
        
        interactions_list = list(interactions)
        logger.info(f"[{self.model_name}] Fallback: Loaded {len(interactions_list)} interactions from SQL")

        if not interactions_list:
            logger.warning(f"[{self.model_name}] No interactions found in SQL database either")
            # Return empty structure with proper shape
            return {
                "co_occurrence": {},
                "product_frequency": {},
                "matrix_data": {
                    "shape": [0, 0],
                    "display_shape": [0, 0],
                    "data": [],
                    "product_ids": [],
                    "description": "Product Co-occurrence Matrix (Fallback - No Data)",
                    "row_label": "Product ID",
                    "col_label": "Product ID",
                    "value_description": "Co-occurrence weight",
                    "sparsity": 1.0,
                },
            }

        user_histories: dict[int, list[tuple[int, float]]] = defaultdict(list)

        for interaction in interactions_list:
            if not interaction.product_id:
                continue
            weight = INTERACTION_WEIGHTS.get(interaction.interaction_type, 1.0)
            user_history = user_histories[interaction.user_id]
            for other_product_id, other_weight in user_history:
                total = weight + other_weight
                co_occurrence[interaction.product_id][other_product_id] = (
                    co_occurrence[interaction.product_id].get(other_product_id, 0.0) + total
                )
                co_occurrence[other_product_id][interaction.product_id] = (
                    co_occurrence[other_product_id].get(interaction.product_id, 0.0) + total
                )
            user_history.append((interaction.product_id, weight))
            product_frequency[interaction.product_id] += weight

        serialized_graph = {
            product_id: dict(neighbours) for product_id, neighbours in co_occurrence.items()
        }
        
        all_product_ids = sorted(set(list(serialized_graph.keys()) + list(product_frequency.keys())))
        
        # Calculate sparsity: ratio of non-zero co-occurrences to total possible pairs
        num_products = len(all_product_ids)
        total_possible_pairs = num_products * (num_products - 1) // 2  # Exclude diagonal
        actual_pairs = sum(1 for pid in serialized_graph for other_pid in serialized_graph[pid] if other_pid != pid)
        sparsity = 1.0 - (actual_pairs / total_possible_pairs) if total_possible_pairs > 0 else 1.0
        
        max_display = min(5, len(all_product_ids)) if all_product_ids else 0
        display_product_ids = all_product_ids[:max_display].copy() if max_display > 0 else []
        
        matrix_data_list = []
        for i, product_id_i in enumerate(display_product_ids):
            row = []
            for j, product_id_j in enumerate(display_product_ids):
                if product_id_i == product_id_j:
                    row.append(0.0)
                else:
                    value = serialized_graph.get(product_id_i, {}).get(product_id_j, 0.0)
                    row.append(float(value))
            matrix_data_list.append(row)
        
        display_rows = 5
        if len(all_product_ids) < display_rows and all_product_ids:
            while len(matrix_data_list) < display_rows:
                matrix_data_list.append([0.0] * len(matrix_data_list[0]) if matrix_data_list else [0.0] * display_rows)
                for row in matrix_data_list[:-1]:
                    row.append(0.0)
                display_product_ids.append(-(len(matrix_data_list)))
        
        matrix_data = {
            "shape": [num_products, num_products],
            "display_shape": [len(matrix_data_list), len(matrix_data_list[0]) if matrix_data_list else 0],
            "data": matrix_data_list[:display_rows] if matrix_data_list else [],
            "product_ids": display_product_ids[:display_rows] if display_product_ids else [],
            "description": "Product Co-occurrence Matrix (Fallback)",
            "row_label": "Product ID",
            "col_label": "Product ID",
            "value_description": "Co-occurrence weight",
            "sparsity": float(sparsity),
        }
        
        # Calculate training parameters for fallback method
        num_interactions = len(interactions_list)
        num_training_samples = num_interactions  # For fallback, use total interactions
        embedding_dim = None  # Fallback doesn't use embeddings
        
        return {
            "co_occurrence": serialized_graph,
            "product_frequency": dict(product_frequency),
            "matrix_data": matrix_data,
            # Training parameters for API response
            "num_users": len(user_histories),
            "num_products": num_products,
            "num_interactions": num_interactions,
            "num_training_samples": num_training_samples,
            "embedding_dim": embedding_dim,
        }

    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        # Try to use LightGCN embeddings if available
        if "user_embeddings" in artifacts and "item_embeddings" in artifacts:
            return self._score_with_embeddings(context, artifacts)
        
        # Fallback to co-occurrence
        return self._score_with_cooccurrence(context, artifacts)
    
    def _score_with_embeddings(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        """Score candidates using LightGCN embeddings."""
        user_ids = artifacts.get("user_ids", [])
        product_ids = artifacts.get("product_ids", [])
        user_to_idx = artifacts.get("user_to_idx", {})
        product_to_idx = artifacts.get("product_to_idx", {})
        user_embeddings = np.array(artifacts.get("user_embeddings", []))
        item_embeddings = np.array(artifacts.get("item_embeddings", []))
        product_frequency = artifacts.get("product_frequency", {})
        
        # Get user index
        user_id_str = str(context.user.id) if hasattr(context.user, 'id') else None
        if not user_id_str or user_id_str not in user_to_idx:
            # Fallback to co-occurrence
            return self._score_with_cooccurrence(context, artifacts)
        
        user_idx = user_to_idx[user_id_str]
        user_emb = user_embeddings[user_idx]
        
        candidate_scores: dict[int, float] = {}
        
        for candidate in context.candidate_products:
            candidate_id = candidate.id
            if candidate_id is None:
                continue
            
            # Try to find product in embeddings
            candidate_id_str = str(candidate_id)
            score = 0.0
            
            if candidate_id_str in product_to_idx:
                item_idx = product_to_idx[candidate_id_str]
                item_emb = item_embeddings[item_idx]
                # Cosine similarity
                similarity = np.dot(user_emb, item_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(item_emb) + 1e-9)
                score = float(similarity)
            
            # Add style and brand bonuses
            score += 0.1 * sum(context.style_weight(token) for token in _style_tokens(candidate))
            # Brand field removed from Product model
            score += 0.01 * product_frequency.get(candidate_id_str, 0.0)
            
            candidate_scores[candidate_id] = score
        
        return candidate_scores
    
    def _score_with_cooccurrence(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        """Score candidates using co-occurrence graph (fallback)."""
        graph: dict[int, dict[int, float]] = artifacts.get("co_occurrence", {})
        frequency: dict[int, float] = artifacts.get("product_frequency", {})
        current_neighbors = graph.get(context.current_product.id, {}) if context.current_product.id else {}

        history_ids = list(context.iter_history_ids())
        candidate_scores: dict[int, float] = {}

        for candidate in context.candidate_products:
            candidate_id = candidate.id
            if candidate_id is None:
                continue
            score = 0.0
            candidate_neighbors = graph.get(candidate_id, {})
            for history_product_id in history_ids:
                score += candidate_neighbors.get(history_product_id, 0.0)
            score += current_neighbors.get(candidate_id, 0.0) * 1.2
            score += sum(context.style_weight(token) for token in _style_tokens(candidate))
            score += 0.1 * frequency.get(candidate_id, 1.0)
            # Brand field removed from Product model
            candidate_scores[candidate_id] = score

        if not candidate_scores:
            for candidate in context.candidate_products:
                if candidate.id is None:
                    continue
                candidate_scores[candidate.id] = sum(context.style_weight(token) for token in _style_tokens(candidate))
        return candidate_scores
    
    def _build_reason(self, product, context: RecommendationContext) -> str:
        """Build detailed reason based on user age, gender, interaction history, style, and color."""
        from apps.recommendations.utils.english_reasons import build_english_reason_from_context
        return build_english_reason_from_context(product, context, "gnn")


def _style_tokens(product) -> Iterable[str]:
    tokens = []
    if isinstance(getattr(product, "style_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.style_tags if tag)
    if isinstance(getattr(product, "outfit_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.outfit_tags if tag)
    if getattr(product, "category_type", None):
        tokens.append(product.category_type.lower())
    if getattr(product, "baseColour", None):
        tokens.append(str(product.baseColour).lower())
    return tokens


engine = GNNRecommendationEngine()


@shared_task
def train_gnn_model(force_retrain: bool = False) -> dict[str, Any]:
    return engine.train(force_retrain=force_retrain)


def recommend_gnn(
    *,
    user_id: str | int,
    current_product_id: str | int,
    top_k_personal: int,
    top_k_outfit: int,
    request_params: dict | None = None,
) -> dict[str, Any]:
    context = CandidateFilter.build_context(
        user_id=user_id,
        current_product_id=current_product_id,
        top_k_personal=top_k_personal,
        top_k_outfit=top_k_outfit,
        request_params=request_params,
    )
    payload = engine.recommend(context)
    return payload.as_dict()
