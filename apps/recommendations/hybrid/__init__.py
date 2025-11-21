"""Hybrid recommendation module combining GNN (LightGCN) and CBF (Sentence-BERT + FAISS)."""

from .models import HybridRecommendationEngine, recommend_hybrid, train_hybrid_model  # noqa: F401

