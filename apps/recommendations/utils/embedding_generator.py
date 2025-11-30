import logging
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from apps.products.mongo_models import Product as MongoProduct

logger = logging.getLogger(__name__)

class EmbeddingGenerator:

    _model: Optional[SentenceTransformer] = None
    _model_name = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            logger.info(f"Loading Sentence-BERT model: {cls._model_name}")
            cls._model = SentenceTransformer(cls._model_name)
            logger.info("Model loaded successfully")
        return cls._model

    @classmethod
    def generate_product_text(cls, product: MongoProduct) -> str:

        parts = []

        name = getattr(product, "productDisplayName", None)
        if name:
            parts.append(name)

        gender = getattr(product, "gender", None)
        if gender:
            parts.append(gender)

        master_category = getattr(product, "masterCategory", None)
        if master_category:
            parts.append(master_category)

        sub_category = getattr(product, "subCategory", None)
        if sub_category:
            parts.append(sub_category)

        article_type = getattr(product, "articleType", None)
        if article_type:
            parts.append(article_type)

        color = getattr(product, "baseColour", None)
        if color:
            parts.append(color)

        usage = getattr(product, "usage", None)
        if usage:
            parts.append(usage)

        season = getattr(product, "season", None)
        if season:
            parts.append(season)

        return " ".join(parts)

    @classmethod
    def generate_embedding(cls, product: MongoProduct) -> np.ndarray:

        model = cls.get_model()
        text = cls.generate_product_text(product)
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding

    @classmethod
    def generate_embeddings_batch(cls, products: List[MongoProduct]) -> np.ndarray:

        model = cls.get_model()
        texts = [cls.generate_product_text(p) for p in products]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

    @classmethod
    def generate_user_embedding(cls, user_interactions: List[Dict], product_embeddings: Dict[str, np.ndarray]) -> np.ndarray:

        if not user_interactions:
            model = cls.get_model()
            return np.zeros(model.get_sentence_embedding_dimension())

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
            model = cls.get_model()
            return np.zeros(model.get_sentence_embedding_dimension())

        user_embedding = np.sum(weighted_embeddings, axis=0) / total_weight
        return user_embedding

    @classmethod
    def compute_similarity(cls, embedding1: np.ndarray, embedding2: np.ndarray) -> float:

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(np.clip(similarity, 0.0, 1.0))

