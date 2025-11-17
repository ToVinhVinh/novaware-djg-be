"""
Embedding generation utilities using Sentence-BERT.
Generates embeddings for products on-the-fly using text and image features.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from apps.products.mongo_models import Product as MongoProduct


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for products using Sentence-BERT.
    Uses all-MiniLM-L6-v2 model for text embeddings.
    """
    
    _model: Optional[SentenceTransformer] = None
    _model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Get or initialize the Sentence-BERT model."""
        if cls._model is None:
            logger.info(f"Loading Sentence-BERT model: {cls._model_name}")
            cls._model = SentenceTransformer(cls._model_name)
            logger.info("Model loaded successfully")
        return cls._model
    
    @classmethod
    def generate_product_text(cls, product: MongoProduct) -> str:
        """
        Generate text description for a product.
        
        Args:
            product: Product object
            
        Returns:
            Text description combining all relevant fields
        """
        parts = []
        
        # Product name
        name = getattr(product, "productDisplayName", None)
        if name:
            parts.append(name)
        
        # Gender
        gender = getattr(product, "gender", None)
        if gender:
            parts.append(gender)
        
        # Category info
        master_category = getattr(product, "masterCategory", None)
        if master_category:
            parts.append(master_category)
        
        sub_category = getattr(product, "subCategory", None)
        if sub_category:
            parts.append(sub_category)
        
        article_type = getattr(product, "articleType", None)
        if article_type:
            parts.append(article_type)
        
        # Color
        color = getattr(product, "baseColour", None)
        if color:
            parts.append(color)
        
        # Usage/Style
        usage = getattr(product, "usage", None)
        if usage:
            parts.append(usage)
        
        # Season
        season = getattr(product, "season", None)
        if season:
            parts.append(season)
        
        return " ".join(parts)
    
    @classmethod
    def generate_embedding(cls, product: MongoProduct) -> np.ndarray:
        """
        Generate embedding for a single product.
        
        Args:
            product: Product object
            
        Returns:
            Embedding vector as numpy array
        """
        model = cls.get_model()
        text = cls.generate_product_text(product)
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    
    @classmethod
    def generate_embeddings_batch(cls, products: List[MongoProduct]) -> np.ndarray:
        """
        Generate embeddings for multiple products in batch.
        
        Args:
            products: List of products
            
        Returns:
            Matrix of embeddings (n_products x embedding_dim)
        """
        model = cls.get_model()
        texts = [cls.generate_product_text(p) for p in products]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
    @classmethod
    def generate_user_embedding(cls, user_interactions: List[Dict], product_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate user embedding from interaction history.
        
        Args:
            user_interactions: List of user interactions with products
            product_embeddings: Dictionary mapping product_id to embedding
            
        Returns:
            User embedding as weighted average of interacted products
        """
        if not user_interactions:
            # Return zero vector if no interactions
            model = cls.get_model()
            return np.zeros(model.get_sentence_embedding_dimension())
        
        # Weight interactions by type
        interaction_weights = {
            "view": 0.5,
            "like": 1.0,
            "cart": 1.5,
            "review": 1.2,
            "purchase": 3.0,
        }
        
        weighted_embeddings = []
        total_weight = 0.0
        
        for interaction in user_interactions:
            product_id = str(interaction.get("product_id", ""))
            interaction_type = interaction.get("interaction_type", "view")
            
            if product_id in product_embeddings:
                weight = interaction_weights.get(interaction_type, 1.0)
                weighted_embeddings.append(product_embeddings[product_id] * weight)
                total_weight += weight
        
        if not weighted_embeddings:
            # Return zero vector if no valid embeddings found
            model = cls.get_model()
            return np.zeros(model.get_sentence_embedding_dimension())
        
        # Compute weighted average
        user_embedding = np.sum(weighted_embeddings, axis=0) / total_weight
        return user_embedding
    
    @classmethod
    def compute_similarity(cls, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Clip to [0, 1] range
        return float(np.clip(similarity, 0.0, 1.0))

