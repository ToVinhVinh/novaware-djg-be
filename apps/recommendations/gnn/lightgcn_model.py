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

    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_products: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):

        super().__init__()

        self.num_users = num_users
        self.num_products = num_products
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)

        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

    def forward(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        user_emb = self.user_embedding.weight
        product_emb = self.product_embedding.weight

        all_emb = torch.cat([user_emb, product_emb], dim=0)

        embs = [all_emb]

        for conv in self.convs:
            all_emb = conv(all_emb, edge_index)
            embs.append(all_emb)

        final_emb = torch.stack(embs, dim=0).mean(dim=0)

        user_final = final_emb[:self.num_users]
        product_final = final_emb[self.num_users:]

        return user_final, product_final

    def predict(self, user_ids: torch.Tensor, product_ids: torch.Tensor,
                user_emb: torch.Tensor, product_emb: torch.Tensor) -> torch.Tensor:

        user_emb_batch = user_emb[user_ids]
        product_emb_batch = product_emb[product_ids]

        scores = (user_emb_batch * product_emb_batch).sum(dim=1)

        return scores

    def bpr_loss(self, user_ids: torch.Tensor, pos_product_ids: torch.Tensor,
                 neg_product_ids: torch.Tensor, user_emb: torch.Tensor,
                 product_emb: torch.Tensor, reg_weight: float = 1e-5) -> torch.Tensor:

        user_emb_batch = user_emb[user_ids]
        pos_emb_batch = product_emb[pos_product_ids]
        neg_emb_batch = product_emb[neg_product_ids]

        pos_scores = (user_emb_batch * pos_emb_batch).sum(dim=1)
        neg_scores = (user_emb_batch * neg_emb_batch).sum(dim=1)

        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = reg_weight * (
            user_emb_batch.norm(2).pow(2) +
            pos_emb_batch.norm(2).pow(2) +
            neg_emb_batch.norm(2).pow(2)
        ) / user_emb_batch.size(0)

        return bpr_loss + reg_loss

class HeterogeneousLightGCN(nn.Module):

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

        super().__init__()

        self.num_users = num_users
        self.num_products = num_products
        self.num_categories = num_categories
        self.num_colors = num_colors
        self.num_styles = num_styles
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)

        if num_categories > 0:
            self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        if num_colors > 0:
            self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        if num_styles > 0:
            self.style_embedding = nn.Embedding(num_styles, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)
        if num_categories > 0:
            nn.init.xavier_uniform_(self.category_embedding.weight)
        if num_colors > 0:
            nn.init.xavier_uniform_(self.color_embedding.weight)
        if num_styles > 0:
            nn.init.xavier_uniform_(self.style_embedding.weight)

        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

    def forward(self, edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

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

        if 'user_product' in edge_index_dict:
            edge_index = edge_index_dict['user_product']

            all_emb = torch.cat([embeddings['user'], embeddings['product']], dim=0)

            embs = [all_emb]

            for conv in self.convs:
                all_emb = conv(all_emb, edge_index)
                embs.append(all_emb)

            final_emb = torch.stack(embs, dim=0).mean(dim=0)

            embeddings['user'] = final_emb[:self.num_users]
            embeddings['product'] = final_emb[self.num_users:]

        return embeddings

def build_bipartite_graph(
    user_product_interactions: List[Tuple[int, int]],
    num_users: int,
    num_products: int,
) -> torch.Tensor:

    edges = []

    for user_id, product_id in user_product_interactions:
        edges.append([user_id, num_users + product_id])
        edges.append([num_users + product_id, user_id])

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    return edge_index

def train_lightgcn(
    model: LightGCN,
    edge_index: torch.Tensor,
    train_interactions: List[Tuple[int, int, int]],
    num_epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    reg_weight: float = 1e-5,
    device: str = 'cpu',
) -> Dict[str, List[float]]:

    model = model.to(device)
    edge_index = edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        model.train()

        np.random.shuffle(train_interactions)

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_interactions), batch_size):
            batch = train_interactions[i:i + batch_size]

            user_ids = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            pos_product_ids = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
            neg_product_ids = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)

            user_emb, product_emb = model(edge_index)

            loss = model.bpr_loss(user_ids, pos_product_ids, neg_product_ids,
                                 user_emb, product_emb, reg_weight)

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

