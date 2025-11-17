"""
Enhanced GNN recommendation engine using PyTorch Geometric + LightGCN.
Implements proper filtering, outfit logic, and Vietnamese reasons.
"""

import logging
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from bson import ObjectId

from apps.products.mongo_models import Product as MongoProduct
from apps.users.mongo_models import User as MongoUser, UserInteraction as MongoInteraction
from apps.recommendations.utils import (
    filter_by_age_gender,
    get_outfit_categories,
    generate_vietnamese_reason,
    map_subcategory_to_tag,
)

from .lightgcn_model import LightGCN, build_bipartite_graph, train_lightgcn

logger = logging.getLogger(__name__)


class LightGCNRecommendationEngine:
    """
    GNN-based recommendation engine using LightGCN.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the engine.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[LightGCN] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.user_id_map: Dict[str, int] = {}
        self.product_id_map: Dict[str, int] = {}
        self.reverse_user_map: Dict[int, str] = {}
        self.reverse_product_map: Dict[int, str] = {}
        
        self.user_embeddings: Optional[torch.Tensor] = None
        self.product_embeddings: Optional[torch.Tensor] = None
        
        self.is_trained = False
    
    def train(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train the LightGCN model.
        
        Args:
            force_retrain: Force retraining even if model exists
            
        Returns:
            Training metrics and info
        """
        model_path = self.model_dir / "gnn_lightgcn.pkl"
        
        # Check if model exists
        if not force_retrain and model_path.exists():
            logger.info("Loading existing GNN model...")
            self.load_model()
            return {
                "status": "loaded",
                "message": "Model loaded from disk",
                "trained_at": datetime.now().isoformat(),
            }
        
        logger.info("Training GNN model with LightGCN...")
        
        # Step 1: Load all users and products
        users = list(MongoUser.objects.only('id'))
        products = list(MongoProduct.objects.only('id'))
        
        logger.info(f"Loaded {len(users)} users and {len(products)} products")
        
        # Step 2: Create ID mappings
        self.user_id_map = {str(user.id): idx for idx, user in enumerate(users)}
        self.product_id_map = {str(product.id): idx for idx, product in enumerate(products)}
        self.reverse_user_map = {idx: str(user.id) for idx, user in enumerate(users)}
        self.reverse_product_map = {idx: str(product.id) for idx, product in enumerate(products)}
        
        num_users = len(users)
        num_products = len(products)
        
        logger.info(f"Created mappings: {num_users} users, {num_products} products")
        
        # Step 3: Load interactions
        interactions = list(MongoInteraction.objects.only('user_id', 'product_id', 'interaction_type'))
        logger.info(f"Loaded {len(interactions)} interactions")
        
        # Step 4: Build bipartite graph
        user_product_pairs = []
        interaction_weights = {
            "view": 0.5,
            "like": 1.0,
            "cart": 1.5,
            "review": 1.2,
            "purchase": 3.0,
        }
        
        # Track positive interactions for training
        user_positive_products: Dict[int, Set[int]] = defaultdict(set)
        
        for interaction in interactions:
            user_id_str = str(interaction.user_id)
            product_id_str = str(interaction.product_id)
            
            if user_id_str not in self.user_id_map or product_id_str not in self.product_id_map:
                continue
            
            user_idx = self.user_id_map[user_id_str]
            product_idx = self.product_id_map[product_id_str]
            
            user_product_pairs.append((user_idx, product_idx))
            user_positive_products[user_idx].add(product_idx)
        
        logger.info(f"Built {len(user_product_pairs)} user-product edges")
        
        # Build edge index
        self.edge_index = build_bipartite_graph(user_product_pairs, num_users, num_products)
        logger.info(f"Edge index shape: {self.edge_index.shape}")
        
        # Step 5: Generate training data (BPR sampling)
        train_interactions = []
        num_negative_samples = 4  # Number of negative samples per positive
        
        for user_idx, pos_products in user_positive_products.items():
            pos_products_list = list(pos_products)
            
            for pos_product in pos_products_list:
                # Sample negative products
                for _ in range(num_negative_samples):
                    neg_product = np.random.randint(0, num_products)
                    # Ensure negative product is not in positive set
                    while neg_product in pos_products:
                        neg_product = np.random.randint(0, num_products)
                    
                    train_interactions.append((user_idx, pos_product, neg_product))
        
        logger.info(f"Generated {len(train_interactions)} training samples")
        
        # Step 6: Initialize and train model
        embedding_dim = 64
        num_layers = 3
        
        self.model = LightGCN(
            num_users=num_users,
            num_products=num_products,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")
        
        training_metrics = train_lightgcn(
            model=self.model,
            edge_index=self.edge_index,
            train_interactions=train_interactions,
            num_epochs=50,
            batch_size=2048,
            learning_rate=0.001,
            reg_weight=1e-5,
            device=device,
        )
        
        # Step 7: Generate final embeddings
        self.model.eval()
        with torch.no_grad():
            self.user_embeddings, self.product_embeddings = self.model(self.edge_index.to(device))
            self.user_embeddings = self.user_embeddings.cpu()
            self.product_embeddings = self.product_embeddings.cpu()
        
        logger.info("Generated final embeddings")
        
        # Step 8: Save model
        self.save_model()
        self.is_trained = True
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "num_users": num_users,
            "num_products": num_products,
            "num_interactions": len(interactions),
            "num_training_samples": len(train_interactions),
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "training_metrics": training_metrics,
            "trained_at": datetime.now().isoformat(),
        }
    
    def save_model(self):
        """Save model and mappings to disk."""
        model_path = self.model_dir / "gnn_lightgcn.pkl"
        
        data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'edge_index': self.edge_index,
            'user_id_map': self.user_id_map,
            'product_id_map': self.product_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_product_map': self.reverse_product_map,
            'user_embeddings': self.user_embeddings,
            'product_embeddings': self.product_embeddings,
            'num_users': len(self.user_id_map),
            'num_products': len(self.product_id_map),
            'embedding_dim': self.model.embedding_dim if self.model else 64,
            'num_layers': self.model.num_layers if self.model else 3,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load model and mappings from disk."""
        model_path = self.model_dir / "gnn_lightgcn.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.user_id_map = data['user_id_map']
        self.product_id_map = data['product_id_map']
        self.reverse_user_map = data['reverse_user_map']
        self.reverse_product_map = data['reverse_product_map']
        self.edge_index = data['edge_index']
        self.user_embeddings = data['user_embeddings']
        self.product_embeddings = data['product_embeddings']
        
        # Recreate model
        self.model = LightGCN(
            num_users=data['num_users'],
            num_products=data['num_products'],
            embedding_dim=data['embedding_dim'],
            num_layers=data['num_layers'],
        )
        
        if data['model_state_dict']:
            self.model.load_state_dict(data['model_state_dict'])
        
        self.model.eval()
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
    
    def recommend(
        self,
        user_id: str,
        current_product_id: str,
        top_k_personal: int = 5,
        top_k_outfit: int = 4,
    ) -> Dict[str, Any]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID (MongoDB ObjectId as string)
            current_product_id: Current product ID
            top_k_personal: Number of personalized recommendations
            top_k_outfit: Number of outfit recommendations per category
            
        Returns:
            Recommendation results with personalized and outfit sections
        """
        if not self.is_trained:
            try:
                self.load_model()
            except FileNotFoundError:
                raise ValueError("Model not trained. Please train the model first.")
        
        # Load user and current product
        user = MongoUser.objects(id=ObjectId(user_id)).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        current_product = MongoProduct.objects(id=ObjectId(current_product_id)).first()
        if not current_product:
            raise ValueError(f"Product {current_product_id} not found")
        
        # Get user index
        user_idx = self.user_id_map.get(user_id)
        if user_idx is None:
            logger.warning(f"User {user_id} not in training data, using cold start")
            return self._cold_start_recommend(user, current_product, top_k_personal, top_k_outfit)
        
        # Get user embedding
        user_emb = self.user_embeddings[user_idx]
        
        # Compute scores for all products
        scores = torch.matmul(self.product_embeddings, user_emb)
        scores = scores.numpy()
        
        # Get top products
        top_indices = np.argsort(scores)[::-1]
        
        # Load candidate products
        candidate_product_ids = [self.reverse_product_map[idx] for idx in top_indices[:500]]
        candidate_products = list(MongoProduct.objects(id__in=[ObjectId(pid) for pid in candidate_product_ids]))
        
        # Create product map for quick lookup
        product_map = {str(p.id): p for p in candidate_products}
        
        # Filter by age and gender
        exclude_ids = {current_product_id}
        filtered_products = filter_by_age_gender(candidate_products, user, exclude_ids)
        
        # Generate personalized recommendations
        personalized = []
        for product in filtered_products[:top_k_personal]:
            reason = generate_vietnamese_reason(
                product=product,
                user=user,
                reason_type="personalized",
                interaction_history=getattr(user, 'interaction_history', []),
            )
            
            personalized.append({
                "product": self._serialize_product(product),
                "score": float(scores[self.product_id_map[str(product.id)]]),
                "reason": reason,
            })
        
        # Generate outfit recommendations
        current_tag = map_subcategory_to_tag(
            current_product.subCategory,
            current_product.articleType
        )
        
        outfit_categories = get_outfit_categories(current_tag or "tops", user.gender)
        outfit = {}
        
        for category in outfit_categories:
            # Find products in this category
            category_products = [
                p for p in filtered_products
                if map_subcategory_to_tag(p.subCategory, p.articleType) == category
            ]
            
            if category_products:
                product = category_products[0]
                reason = generate_vietnamese_reason(
                    product=product,
                    user=user,
                    reason_type="outfit",
                    current_product=current_product,
                )
                
                outfit[category] = {
                    "product": self._serialize_product(product),
                    "score": float(scores[self.product_id_map[str(product.id)]]),
                    "reason": reason,
                }
        
        # Generate overall reasons
        reasons = {
            "personalized": [item["reason"] for item in personalized],
            "outfit": [f"Phối hợp hoàn hảo với {current_product.articleType or 'sản phẩm hiện tại'}"]
        }
        
        return {
            "personalized": personalized,
            "outfit": outfit,
            "reasons": reasons,
        }
    
    def _cold_start_recommend(
        self,
        user: MongoUser,
        current_product: MongoProduct,
        top_k_personal: int,
        top_k_outfit: int,
    ) -> Dict[str, Any]:
        """Handle cold start users (not in training data)."""
        # Use popularity-based recommendations
        products = list(MongoProduct.objects.limit(100))
        
        filtered_products = filter_by_age_gender(products, user, {str(current_product.id)})
        
        personalized = []
        for product in filtered_products[:top_k_personal]:
            reason = generate_vietnamese_reason(
                product=product,
                user=user,
                reason_type="personalized",
            )
            
            personalized.append({
                "product": self._serialize_product(product),
                "score": 0.5,
                "reason": reason,
            })
        
        # Generate outfit
        current_tag = map_subcategory_to_tag(
            current_product.subCategory,
            current_product.articleType
        )
        
        outfit_categories = get_outfit_categories(current_tag or "tops", user.gender)
        outfit = {}
        
        for category in outfit_categories:
            category_products = [
                p for p in filtered_products
                if map_subcategory_to_tag(p.subCategory, p.articleType) == category
            ]
            
            if category_products:
                product = category_products[0]
                reason = generate_vietnamese_reason(
                    product=product,
                    user=user,
                    reason_type="outfit",
                    current_product=current_product,
                )
                
                outfit[category] = {
                    "product": self._serialize_product(product),
                    "score": 0.5,
                    "reason": reason,
                }
        
        reasons = {
            "personalized": [item["reason"] for item in personalized],
            "outfit": ["Outfit được gợi ý dựa trên sản phẩm hiện tại"]
        }
        
        return {
            "personalized": personalized,
            "outfit": outfit,
            "reasons": reasons,
        }
    
    def _serialize_product(self, product: MongoProduct) -> Dict[str, Any]:
        """Serialize product to dictionary."""
        return {
            "id": str(product.id),
            "name": product.productDisplayName or "",
            "gender": product.gender or "",
            "masterCategory": product.masterCategory or "",
            "subCategory": product.subCategory or "",
            "articleType": product.articleType or "",
            "baseColour": product.baseColour or "",
            "season": product.season or "",
            "usage": product.usage or "",
            "images": product.images or [],
        }


# Global engine instance
_engine: Optional[LightGCNRecommendationEngine] = None


def get_engine() -> LightGCNRecommendationEngine:
    """Get or create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = LightGCNRecommendationEngine()
    return _engine

