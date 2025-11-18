"""
Content-Based Filtering engine using Sentence-BERT + FAISS.
Implements semantic similarity search with FAISS for efficient retrieval.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import faiss
import numpy as np
from bson import ObjectId

from apps.products.mongo_models import Product as MongoProduct
from apps.users.mongo_models import User as MongoUser, UserInteraction as MongoInteraction
from apps.recommendations.utils import (
    EmbeddingGenerator,
    filter_by_age_gender,
    get_outfit_categories,
    generate_vietnamese_reason,
    map_subcategory_to_tag,
)

logger = logging.getLogger(__name__)


class ContentBasedRecommendationEngine:
    """
    Content-based recommendation engine using Sentence-BERT embeddings and FAISS index.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the engine.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.faiss_index: Optional[faiss.Index] = None
        self.product_id_map: Dict[int, str] = {}  # FAISS index -> product ID
        self.product_embeddings: Dict[str, np.ndarray] = {}
        self.user_profiles: Dict[str, np.ndarray] = {}
        
        self.is_trained = False
    
    def train(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train the content-based model.
        
        Args:
            force_retrain: Force retraining even if model exists
            
        Returns:
            Training metrics and info
        """
        model_path = self.model_dir / "cbf_sbert_faiss.pkl"
        
        # Check if model exists
        if not force_retrain and model_path.exists():
            logger.info("Loading existing CBF model...")
            self.load_model()
            return {
                "status": "loaded",
                "message": "Model loaded from disk",
                "trained_at": datetime.now().isoformat(),
            }
        
        logger.info("Training CBF model with Sentence-BERT + FAISS...")
        
        # Step 1: Load all products
        products = list(MongoProduct.objects.all())
        logger.info(f"Loaded {len(products)} products")
        
        if not products:
            raise ValueError("No products found in database")
        
        # Step 2: Generate embeddings for all products
        logger.info("Generating product embeddings...")
        embeddings = EmbeddingGenerator.generate_embeddings_batch(products)
        
        # Store embeddings
        self.product_embeddings = {
            str(product.id): embeddings[i]
            for i, product in enumerate(products)
        }
        
        logger.info(f"Generated {len(self.product_embeddings)} product embeddings")
        
        # Step 3: Build FAISS index
        logger.info("Building FAISS index...")
        embedding_dim = embeddings.shape[1]
        
        # Use IVF Flat index for better performance on large datasets
        # For smaller datasets, we can use a simple flat index
        if len(products) > 10000:
            # IVF index with 100 clusters
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
            self.faiss_index.train(embeddings.astype('float32'))
        else:
            # Simple flat index for exact search
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Create mapping from FAISS index to product ID
        self.product_id_map = {i: str(products[i].id) for i in range(len(products))}
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
        
        # Step 4: Build user profiles from interactions
        logger.info("Building user profiles...")
        users = list(MongoUser.objects.only('id'))
        interactions = list(MongoInteraction.objects.all())
        
        # Group interactions by user
        user_interactions_map: Dict[str, List[Dict]] = {}
        for interaction in interactions:
            user_id_str = str(interaction.user_id)
            if user_id_str not in user_interactions_map:
                user_interactions_map[user_id_str] = []
            
            user_interactions_map[user_id_str].append({
                'product_id': str(interaction.product_id),
                'interaction_type': interaction.interaction_type,
            })
        
        # Generate user profiles
        for user_id_str, user_interactions in user_interactions_map.items():
            user_embedding = EmbeddingGenerator.generate_user_embedding(
                user_interactions,
                self.product_embeddings
            )
            self.user_profiles[user_id_str] = user_embedding
        
        logger.info(f"Built {len(self.user_profiles)} user profiles")
        
        # Step 5: Save model
        self.save_model()
        self.is_trained = True
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "num_products": len(products),
            "num_users": len(users),
            "num_interactions": len(interactions),
            "num_user_profiles": len(self.user_profiles),
            "embedding_dim": embedding_dim,
            "index_type": "IVF Flat" if len(products) > 10000 else "Flat L2",
            "trained_at": datetime.now().isoformat(),
        }
    
    def save_model(self):
        """Save model and mappings to disk."""
        model_path = self.model_dir / "cbf_sbert_faiss.pkl"
        faiss_index_path = self.model_dir / "cbf_faiss.index"
        
        # Save FAISS index separately
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(faiss_index_path))
        
        # Save other data
        data = {
            'product_id_map': self.product_id_map,
            'product_embeddings': self.product_embeddings,
            'user_profiles': self.user_profiles,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load model and mappings from disk."""
        model_path = self.model_dir / "cbf_sbert_faiss.pkl"
        faiss_index_path = self.model_dir / "cbf_faiss.index"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load FAISS index
        if faiss_index_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_index_path))
        
        # Load other data
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.product_id_map = data['product_id_map']
        self.product_embeddings = data['product_embeddings']
        self.user_profiles = data['user_profiles']
        
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
        
        # Get user profile or create on-the-fly
        if user_id in self.user_profiles:
            user_embedding = self.user_profiles[user_id]
        else:
            # Cold start: use current product embedding
            if current_product_id in self.product_embeddings:
                user_embedding = self.product_embeddings[current_product_id]
            else:
                # Generate embedding on-the-fly
                user_embedding = EmbeddingGenerator.generate_embedding(current_product)
        
        # Search FAISS index
        k = 100  # Retrieve more candidates for filtering
        distances, indices = self.faiss_index.search(
            user_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # Get candidate products
        candidate_product_ids = [
            self.product_id_map[idx]
            for idx in indices[0]
            if idx in self.product_id_map
        ]
        
        candidate_products = list(MongoProduct.objects(
            id__in=[ObjectId(pid) for pid in candidate_product_ids]
        ))
        
        # Create product map for quick lookup
        product_map = {str(p.id): p for p in candidate_products}
        
        # Reorder candidates based on FAISS results
        ordered_candidates = [
            product_map[pid]
            for pid in candidate_product_ids
            if pid in product_map
        ]
        
        # Filter by age and gender
        exclude_ids = {current_product_id}
        filtered_products = filter_by_age_gender(ordered_candidates, user, exclude_ids)
        
        # Generate personalized recommendations
        personalized = []
        for product in filtered_products[:top_k_personal]:
            # Compute similarity score
            if str(product.id) in self.product_embeddings:
                similarity = EmbeddingGenerator.compute_similarity(
                    user_embedding,
                    self.product_embeddings[str(product.id)]
                )
            else:
                similarity = 0.5
            
            reason = generate_vietnamese_reason(
                product=product,
                user=user,
                reason_type="personalized",
                interaction_history=getattr(user, 'interaction_history', []),
            )
            
            personalized.append({
                "product": self._serialize_product(product),
                "score": float(similarity),
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
                
                # Compute similarity
                if str(product.id) in self.product_embeddings:
                    similarity = EmbeddingGenerator.compute_similarity(
                        self.product_embeddings[current_product_id],
                        self.product_embeddings[str(product.id)]
                    )
                else:
                    similarity = 0.5
                
                reason = generate_vietnamese_reason(
                    product=product,
                    user=user,
                    reason_type="outfit",
                    current_product=current_product,
                )
                
                outfit[category] = {
                    "product": self._serialize_product(product),
                    "score": float(similarity),
                    "reason": reason,
                }
        
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
_engine: Optional[ContentBasedRecommendationEngine] = None


def get_engine() -> ContentBasedRecommendationEngine:
    """Get or create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = ContentBasedRecommendationEngine()
    return _engine

