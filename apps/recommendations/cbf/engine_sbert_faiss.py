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
    generate_english_reason,
    map_subcategory_to_tag,
)

logger = logging.getLogger(__name__)

class ContentBasedRecommendationEngine:

    def __init__(self, model_dir: str = "models"):

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_index: Optional[faiss.Index] = None
        self.product_id_map: Dict[int, str] = {}
        self.product_embeddings: Dict[str, np.ndarray] = {}
        self.user_profiles: Dict[str, np.ndarray] = {}

        self.is_trained = False

    def train(self, force_retrain: bool = False) -> Dict[str, Any]:

        model_path = self.model_dir / "cbf_sbert_faiss.pkl"

        if not force_retrain and model_path.exists():
            logger.info("Loading existing CBF model...")
            self.load_model()
            return {
                "status": "loaded",
                "message": "Model loaded from disk",
                "trained_at": datetime.now().isoformat(),
            }

        logger.info("Training CBF model with Sentence-BERT + FAISS...")

        products = list(MongoProduct.objects.all())
        logger.info(f"Loaded {len(products)} products")

        if not products:
            raise ValueError("No products found in database")

        logger.info("Generating product embeddings...")
        embeddings = EmbeddingGenerator.generate_embeddings_batch(products)

        self.product_embeddings = {
            str(product.id): embeddings[i]
            for i, product in enumerate(products)
        }

        logger.info(f"Generated {len(self.product_embeddings)} product embeddings")

        logger.info("Building FAISS index...")
        embedding_dim = embeddings.shape[1]

        if len(products) > 10000:
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
            self.faiss_index.train(embeddings.astype('float32'))
        else:
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)

        self.faiss_index.add(embeddings.astype('float32'))

        self.product_id_map = {i: str(products[i].id) for i in range(len(products))}

        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")

        logger.info("Building user profiles...")
        users = list(MongoUser.objects.only('id'))
        interactions = list(MongoInteraction.objects.all())

        user_interactions_map: Dict[str, List[Dict]] = {}
        for interaction in interactions:
            user_id_str = str(interaction.user_id)
            if user_id_str not in user_interactions_map:
                user_interactions_map[user_id_str] = []

            user_interactions_map[user_id_str].append({
                'product_id': str(interaction.product_id),
                'interaction_type': interaction.interaction_type,
            })

        for user_id_str, user_interactions in user_interactions_map.items():
            user_embedding = EmbeddingGenerator.generate_user_embedding(
                user_interactions,
                self.product_embeddings
            )
            self.user_profiles[user_id_str] = user_embedding

        logger.info(f"Built {len(self.user_profiles)} user profiles")

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
        model_path = self.model_dir / "cbf_sbert_faiss.pkl"
        faiss_index_path = self.model_dir / "cbf_faiss.index"

        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(faiss_index_path))

        data = {
            'product_id_map': self.product_id_map,
            'product_embeddings': self.product_embeddings,
            'user_profiles': self.user_profiles,
        }

        with open(model_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {model_path}")

    def load_model(self):
        model_path = self.model_dir / "cbf_sbert_faiss.pkl"
        faiss_index_path = self.model_dir / "cbf_faiss.index"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        if faiss_index_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_index_path))

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

        if user_id in self.user_profiles:
            user_embedding = self.user_profiles[user_id]
        else:
            if current_product_id in self.product_embeddings:
                user_embedding = self.product_embeddings[current_product_id]
            else:
                user_embedding = EmbeddingGenerator.generate_embedding(current_product)

        k = 100
        distances, indices = self.faiss_index.search(
            user_embedding.reshape(1, -1).astype('float32'),
            k
        )

        candidate_product_ids = [
            self.product_id_map[idx]
            for idx in indices[0]
            if idx in self.product_id_map
        ]

        candidate_products = list(MongoProduct.objects(
            id__in=[ObjectId(pid) for pid in candidate_product_ids]
        ))

        product_map = {str(p.id): p for p in candidate_products}

        ordered_candidates = [
            product_map[pid]
            for pid in candidate_product_ids
            if pid in product_map
        ]

        exclude_ids = {current_product_id}
        filtered_products = filter_by_age_gender(ordered_candidates, user, exclude_ids)

        personalized = []
        for product in filtered_products[:top_k_personal]:
            if str(product.id) in self.product_embeddings:
                similarity = EmbeddingGenerator.compute_similarity(
                    user_embedding,
                    self.product_embeddings[str(product.id)]
                )
            else:
                similarity = 0.5

            reason = generate_english_reason(
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

        current_tag = map_subcategory_to_tag(
            current_product.subCategory,
            current_product.articleType
        )

        outfit_categories = get_outfit_categories(current_tag or "tops", user.gender)
        outfit = {}

        if current_tag and current_tag in outfit_categories:
            current_product_reason = generate_english_reason(
                product=current_product,
                user=user,
                reason_type="outfit",
                current_product=current_product,
            )
            outfit[current_tag] = {
                "product": self._serialize_product(current_product),
                "score": 1.0,
                "reason": f"Selected product: {current_product_reason}",
            }

        for category in outfit_categories:
            if category == current_tag:
                continue
            category_products = [
                p for p in filtered_products
                if map_subcategory_to_tag(p.subCategory, p.articleType) == category
            ]

            if category_products:
                product = category_products[0]

                if str(product.id) in self.product_embeddings:
                    similarity = EmbeddingGenerator.compute_similarity(
                        self.product_embeddings[current_product_id],
                        self.product_embeddings[str(product.id)]
                    )
                else:
                    similarity = 0.5

                reason = generate_english_reason(
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

_engine: Optional[ContentBasedRecommendationEngine] = None

def get_engine() -> ContentBasedRecommendationEngine:
    global _engine
    if _engine is None:
        _engine = ContentBasedRecommendationEngine()
    return _engine

