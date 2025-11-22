"""
Enhanced GNN recommendation engine using PyTorch Geometric + GNN.
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

from apps.products.mongo_models import Product as MongoProduct, ProductVariant
from apps.users.mongo_models import User as MongoUser, UserInteraction as MongoInteraction
from apps.recommendations.utils import (
    filter_by_age_gender,
    get_outfit_categories,
    generate_english_reason,
    map_subcategory_to_tag,
)

from .gnn_model import GNN, build_bipartite_graph, train_GNN

logger = logging.getLogger(__name__)


class GNNRecommendationEngine:
    """
    GNN-based recommendation engine using GNN.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the engine.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[GNN] = None
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
        Train the GNN model.
        
        Args:
            force_retrain: Force retraining even if model exists
            
        Returns:
            Training metrics and info
        """
        model_path = self.model_dir / "gnn_gnn.pkl"
        
        # Check if model exists
        if not force_retrain and model_path.exists():
            logger.info("Loading existing GNN model...")
            self.load_model()
            return {
                "status": "loaded",
                "message": "Model loaded from disk",
                "trained_at": datetime.now().isoformat(),
            }
        
        logger.info("Training GNN model with GNN...")
        
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
        interactions = list(MongoInteraction.objects.only('user_id', 'product_id', 'interaction_type', 'timestamp'))
        logger.info(f"Loaded {len(interactions)} interactions")
        
        # Step 3.5: Split interactions into train/test (80/20)
        # Sort by timestamp to ensure temporal split
        interactions_sorted = sorted(interactions, key=lambda x: x.timestamp if hasattr(x, 'timestamp') and x.timestamp else datetime.min)
        
        # Use 80% for training, 20% for testing
        split_idx = int(len(interactions_sorted) * 0.8)
        train_interactions_list = interactions_sorted[:split_idx]
        test_interactions_list = interactions_sorted[split_idx:]
        
        logger.info(f"Split interactions: {len(train_interactions_list)} train, {len(test_interactions_list)} test")
        
        # Store test interactions for evaluation (map to indices)
        self.test_interactions = {}  # {user_idx: set(product_idx)}
        for interaction in test_interactions_list:
            user_id_str = str(interaction.user_id)
            product_id_str = str(interaction.product_id)
            
            if user_id_str not in self.user_id_map or product_id_str not in self.product_id_map:
                continue
            
            user_idx = self.user_id_map[user_id_str]
            product_idx = self.product_id_map[product_id_str]
            
            if user_idx not in self.test_interactions:
                self.test_interactions[user_idx] = set()
            self.test_interactions[user_idx].add(product_idx)
        
        logger.info(f"Stored {len(self.test_interactions)} users with test interactions")
        
        # Step 4: Build bipartite graph (ONLY from train set)
        user_product_pairs = []
        interaction_weights = {
            "view": 0.5,
            "like": 1.0,
            "cart": 1.5,
            "review": 1.2,
            "purchase": 3.0,
        }
        
        # Track positive interactions for training (ONLY from train set)
        user_positive_products: Dict[int, Set[int]] = defaultdict(set)
        
        for interaction in train_interactions_list:
            user_id_str = str(interaction.user_id)
            product_id_str = str(interaction.product_id)
            
            if user_id_str not in self.user_id_map or product_id_str not in self.product_id_map:
                continue
            
            user_idx = self.user_id_map[user_id_str]
            product_idx = self.product_id_map[product_id_str]
            
            user_product_pairs.append((user_idx, product_idx))
            user_positive_products[user_idx].add(product_idx)
        
        logger.info(f"Built {len(user_product_pairs)} user-product edges from train set")
        
        # Build edge index (ONLY from train set)
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
        
        self.model = GNN(
            num_users=num_users,
            num_products=num_products,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")
        
        training_metrics = train_GNN(
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
            "num_train_interactions": len(train_interactions_list),
            "num_test_interactions": len(test_interactions_list),
            "num_test_users": len(self.test_interactions),
            "num_training_samples": len(train_interactions),
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "training_metrics": training_metrics,
            "trained_at": datetime.now().isoformat(),
        }
    
    def save_model(self):
        """Save model and mappings to disk."""
        model_path = self.model_dir / "gnn_gnn.pkl"
        
        # Convert test_interactions sets to lists for pickle compatibility
        test_interactions_serializable = {}
        if hasattr(self, 'test_interactions') and self.test_interactions:
            for user_idx, product_set in self.test_interactions.items():
                test_interactions_serializable[user_idx] = list(product_set)
        
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
            'test_interactions': test_interactions_serializable,  # Save test set for evaluation
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {model_path} (with {len(test_interactions_serializable)} test users)")
    
    def load_model(self):
        """Load model and mappings from disk."""
        model_path = self.model_dir / "gnn_gnn.pkl"
        
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
        
        # Load test_interactions (convert lists back to sets)
        if 'test_interactions' in data and data['test_interactions']:
            self.test_interactions = {
                user_idx: set(product_list) 
                for user_idx, product_list in data['test_interactions'].items()
            }
            logger.info(f"Loaded {len(self.test_interactions)} test users from saved model")
        else:
            self.test_interactions = {}
            logger.warning("No test_interactions found in saved model (old model format)")
        
        # Recreate model
        self.model = GNN(
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
        
        user = MongoUser.objects(id=ObjectId(user_id)).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        current_product = MongoProduct.objects(id=ObjectId(current_product_id)).first()
        if not current_product:
            raise ValueError(f"Product {current_product_id} not found")
        
        user_idx = self.user_id_map.get(user_id)
        if user_idx is None:
            logger.warning(f"User {user_id} not in training data, using cold start")
            return self._cold_start_recommend(user, current_product, top_k_personal, top_k_outfit)
        
        user_emb = self.user_embeddings[user_idx]
        
        scores = torch.matmul(self.product_embeddings, user_emb)
        scores = scores.numpy()
        
        top_indices = np.argsort(scores)[::-1]
        logger.debug(f"Top 10 raw recommendation indices: {top_indices[:10]}")
        logger.debug(f"Top 10 raw scores: {scores[top_indices[:10]]}")

        personalized = []
        exclude_ids = {current_product_id}
        logger.debug(f"Starting personalized recommendation generation for user {user_id}. Excluding initial product: {current_product_id}")

        for idx in top_indices:
            if len(personalized) >= top_k_personal:
                logger.debug("Reached top_k_personal limit. Stopping personalized generation.")
                break

            product_id = self.reverse_product_map.get(idx)
            if not product_id or product_id in exclude_ids:
                continue
            
            logger.debug(f"Checking candidate product ID: {product_id} with score {scores[idx]}")
            product = MongoProduct.objects(id=ObjectId(product_id)).first()
            if not product:
                logger.warning(f"Product ID {product_id} from recommendations not found in DB.")
                continue

            if not filter_by_age_gender([product], user):
                continue

            logger.info(f"Product {product_id} passed all filters. Adding to personalized list.")
            reason = generate_english_reason(
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
            exclude_ids.add(product_id) # Ensure it's not reused in outfit

        if not personalized:
            logger.warning("Personalized recommendation list is empty after filtering.")
        
        outfit = {}
        current_tag = map_subcategory_to_tag(
            current_product.subCategory,
            current_product.articleType
        )
        outfit_categories = get_outfit_categories(current_tag or "tops", user.gender)
        
        if current_tag and current_tag in outfit_categories:
            current_product_reason = generate_english_reason(
                product=current_product,
                user=user,
                reason_type="outfit",
                current_product=current_product,
            )
            current_score = 1.0
            if str(current_product.id) in self.product_id_map:
                current_score = float(scores[self.product_id_map[str(current_product.id)]])
            else:
                current_score = min(1.0, float(getattr(current_product, 'rating', 5.0) / 5.0))
            
            outfit[current_tag] = {
                "product": self._serialize_product(current_product),
                "score": current_score,
                "reason": f"Selected product: {current_product_reason}",
            }
        
        used_outfit_product_ids = {str(current_product.id)}
        used_outfit_product_ids.update(item['product']['id'] for item in personalized)

        for category in outfit_categories:
            # Skip if we already added the current product to this category
            if category == current_tag:
                continue
            # Directly query for the best-rated product in the category
            outfit_candidate = MongoProduct.objects(
                subCategory__iexact=category,
                gender=user.gender,
                id__nin=[ObjectId(pid) for pid in used_outfit_product_ids if ObjectId.is_valid(pid)]
            ).order_by('-rating').first()

            if outfit_candidate:
                reason = generate_english_reason(
                    product=outfit_candidate,
                    user=user,
                    reason_type="outfit",
                    current_product=current_product,
                )
                
                # Calculate score if possible, otherwise use rating
                score = 0.5 # Default score
                if str(outfit_candidate.id) in self.product_id_map:
                    score = float(scores[self.product_id_map[str(outfit_candidate.id)]])
                else:
                    score = float(getattr(outfit_candidate, 'rating', 0.0) / 5.0)

                outfit[category] = {
                    "product": self._serialize_product(outfit_candidate),
                    "score": score,
                    "reason": reason,
                }
                used_outfit_product_ids.add(str(outfit_candidate.id))
            else:
                logger.warning(f"Could not find product for outfit category: {category}")
        
        # Generate overall reasons
        reasons = {
            "personalized": [item["reason"] for item in personalized],
            "outfit": [f"Perfect combination with {current_product.articleType or 'current product'}"]
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
        """Handle cold start users with a more robust popularity-based fallback."""
        logger.info(f"Executing improved cold-start logic for user {user.id}")

        # 1. Personalized recommendations: Find popular products in the same category and gender
        personalized_candidates = list(MongoProduct.objects(
            subCategory=current_product.subCategory,
            gender=current_product.gender,
            id__ne=current_product.id  # Exclude the current product
        ).order_by('-rating').limit(top_k_personal * 2)) # Fetch more to ensure we have enough after filtering

        filtered_personalized = filter_by_age_gender(personalized_candidates, user, {str(current_product.id)})

        personalized = []
        for product in filtered_personalized[:top_k_personal]:
            reason = generate_english_reason(
                product=product,
                user=user,
                reason_type="personalized",
            )
            reason = f"[Recommendation for new user] {reason}"
            
            personalized.append({
                "product": self._serialize_product(product),
                "score": float(getattr(product, 'rating', 0.0) / 5.0), # Normalize score
                "reason": reason,
            })

        # 2. Outfit recommendations: Directly query popular items for each needed category
        outfit = {}
        current_tag = map_subcategory_to_tag(
            current_product.subCategory,
            current_product.articleType
        )
        outfit_categories = get_outfit_categories(current_tag or "tops", user.gender)
        
        # Add current product to outfit in its category
        if current_tag and current_tag in outfit_categories:
            current_product_reason = generate_english_reason(
                product=current_product,
                user=user,
                reason_type="outfit",
                current_product=current_product,
            )
            current_score = min(1.0, float(getattr(current_product, 'rating', 5.0) / 5.0))
            
            outfit[current_tag] = {
                "product": self._serialize_product(current_product),
                "score": current_score,
                "reason": f"Selected product: {current_product_reason}",
            }
        
        used_outfit_product_ids = {str(current_product.id)}
        if personalized:
            used_outfit_product_ids.update(item['product']['id'] for item in personalized)

        for category in outfit_categories:
            # Skip if we already added the current product to this category
            if category == current_tag:
                continue
            # Find the best product for this category
            outfit_candidate = MongoProduct.objects(
                subCategory__iexact=category,  # Case-insensitive match for category
                gender=user.gender,
                id__nin=[ObjectId(pid) for pid in used_outfit_product_ids if ObjectId.is_valid(pid)]
            ).order_by('-rating').first()

            if outfit_candidate:
                reason = generate_english_reason(
                    product=outfit_candidate,
                    user=user,
                    reason_type="outfit",
                    current_product=current_product,
                )
                
                outfit[category] = {
                    "product": self._serialize_product(outfit_candidate),
                    "score": float(getattr(outfit_candidate, 'rating', 0.0) / 5.0),
                    "reason": reason,
                }
                used_outfit_product_ids.add(str(outfit_candidate.id))
            else:
                logger.warning(f"Could not find product for outfit category: {category}")

        reasons = {
            "personalized": [item["reason"] for item in personalized],
            "outfit": [item["reason"] for item in outfit.values() if item]
        }
        
        return {
            "personalized": personalized,
            "outfit": outfit,
            "reasons": reasons,
        }
    
    def _serialize_product(self, product: MongoProduct) -> Dict[str, Any]:
        """Serialize product to dictionary."""
        # Load variants for this product
        variants = []
        try:
            product_variants = ProductVariant.objects(product_id=product.id)
            for variant in product_variants:
                variants.append({
                    "id": str(variant.id),
                    "color": variant.color,
                    "size": variant.size,
                    "price": float(variant.price) if variant.price is not None else None,
                    "stock": variant.stock if variant.stock is not None else 0,
                })
        except Exception as e:
            logger.debug(f"Could not load variants for product {product.id}: {e}")
            variants = []
        
        return {
            "id": int(product.id) if product.id is not None else None,
            "gender": getattr(product, "gender", None),
            "masterCategory": getattr(product, "masterCategory", None),
            "subCategory": getattr(product, "subCategory", None),
            "articleType": getattr(product, "articleType", None),
            "baseColour": getattr(product, "baseColour", None),
            "season": getattr(product, "season", None),
            "year": getattr(product, "year", None),
            "usage": getattr(product, "usage", None),
            "productDisplayName": getattr(product, "productDisplayName", getattr(product, "name", None)),
            "images": list(getattr(product, "images", [])) or [],
            "rating": float(getattr(product, "rating", 0.0)) if getattr(product, "rating", None) is not None else None,
            "sale": float(getattr(product, "sale", 0.0)) if getattr(product, "sale", None) is not None else None,
            "reviews": [],
            "variants": variants,
            "created_at": getattr(product, "created_at", None).isoformat() if getattr(product, "created_at", None) else None,
            "updated_at": getattr(product, "updated_at", None).isoformat() if getattr(product, "updated_at", None) else None,
        }


# Global engine instance
_engine: Optional[GNNRecommendationEngine] = None


def get_engine() -> GNNRecommendationEngine:
    """Get or create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = GNNRecommendationEngine()
    return _engine

