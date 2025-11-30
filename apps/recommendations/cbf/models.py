from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
from celery import shared_task

from apps.products.mongo_models import Product
from apps.recommendations.common import BaseRecommendationEngine, CandidateFilter
from apps.recommendations.common.context import RecommendationContext
from apps.recommendations.common.gender_utils import normalize_gender

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    SentenceTransformer = None

@lru_cache(maxsize=1)
def _get_sbert_model():
    if not SBERT_AVAILABLE:
        raise ImportError("sentence-transformers is not installed. Please install it with: pip install sentence-transformers")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("[cbf] Loaded Sentence-BERT model: all-MiniLM-L6-v2")
        return model
    except Exception as e:
        logger.error(f"[cbf] Failed to load Sentence-BERT model: {e}")
        raise

class ContentBasedRecommendationEngine(BaseRecommendationEngine):
    model_name = "cbf"

    def _train_impl(self) -> dict[str, Any]:
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers is not installed. Please install it with: pip install sentence-transformers")
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is not installed. Please install it with: pip install faiss-cpu")

        logger.info(f"[{self.model_name}] Loading products from database...")
        products = list(Product.objects())
        logger.info(f"[{self.model_name}] Loaded {len(products)} products")

        product_ids = [product.id for product in products if product.id is not None]
        logger.info(f"[{self.model_name}] Building documents for {len(product_ids)} products...")
        documents = [_build_document(product) for product in products if product.id is not None]
        logger.info(f"[{self.model_name}] Documents built, creating Sentence-BERT embeddings...")

        model = _get_sbert_model()

        logger.info(f"[{self.model_name}] Generating embeddings for {len(documents)} documents...")
        embeddings = model.encode(documents, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype('float32')

        logger.info(f"[{self.model_name}] Embeddings created: shape {embeddings.shape}")

        dimension = embeddings.shape[1]
        logger.info(f"[{self.model_name}] Building FAISS index with dimension {dimension}...")

        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        logger.info(f"[{self.model_name}] FAISS index built with {index.ntotal} vectors")

        max_display = min(5, len(product_ids))
        similarity_matrix = np.dot(embeddings[:max_display], embeddings[:max_display].T)

        display_rows = 5
        matrix_data_list = similarity_matrix.tolist()
        product_ids_display = product_ids[:max_display].copy()

        if len(product_ids) < display_rows:
            while len(matrix_data_list) < display_rows:
                matrix_data_list.append([0.0] * len(matrix_data_list[0]) if matrix_data_list else [0.0] * display_rows)
                for row in matrix_data_list[:-1]:
                    row.append(0.0)
                product_ids_display.append(-(len(matrix_data_list)))

        matrix_data = {
            "shape": [len(product_ids), len(product_ids)],
            "display_shape": [len(matrix_data_list), len(matrix_data_list[0]) if matrix_data_list else 0],
            "data": matrix_data_list[:display_rows],
            "product_ids": product_ids_display[:display_rows],
            "description": "Product Similarity Matrix (Sentence-BERT Cosine Similarity)",
            "row_label": "Product ID",
            "col_label": "Product ID",
            "value_description": "Similarity score (0-1, higher = more similar)",
        }

        return {
            "product_ids": product_ids,
            "embeddings": embeddings.tolist(),
            "documents": documents,
            "faiss_index": index,
            "matrix_data": matrix_data,
        }

    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is not installed. Please install it with: pip install faiss-cpu")

        product_ids: list[int] = artifacts["product_ids"]
        embeddings = np.array(artifacts["embeddings"]).astype('float32')

        faiss_index = artifacts.get("faiss_index")
        if faiss_index is None:
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            faiss_index.add(embeddings_normalized)
        else:
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)

        id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}

        user_profile = self._build_user_profile(context, embeddings, id_to_index)
        if user_profile is None:
            user_profile = self._vector_for_product(
                context.current_product,
                embeddings,
                id_to_index,
            )
            if user_profile is None:
                return {}

        user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-9)
        user_profile = user_profile.reshape(1, -1).astype('float32')

        candidate_scores: dict[int, float] = {}
        for candidate in context.candidate_products:
            candidate_id = candidate.id
            if candidate_id is None:
                continue

            candidate_vector = self._vector_for_product(candidate, embeddings_normalized, id_to_index)
            if candidate_vector is None:
                continue

            candidate_vector = candidate_vector / (np.linalg.norm(candidate_vector) + 1e-9)
            candidate_vector = candidate_vector.reshape(1, -1).astype('float32')

            similarity = float(np.dot(user_profile, candidate_vector.T)[0, 0])

            style_bonus = 0.05 * sum(context.style_weight(token) for token in _style_tokens(candidate))
            brand_bonus = 0.0

            candidate_scores[candidate_id] = float(similarity + style_bonus + brand_bonus)

        return candidate_scores

    def _build_user_profile(
        self,
        context: RecommendationContext,
        embeddings: np.ndarray,
        id_to_index: dict[int, int],
    ) -> np.ndarray | None:
        model = _get_sbert_model()
        accum_vector: np.ndarray | None = None
        total_weight = 0.0

        for product in context.history_products:
            if product.id is None:
                continue

            product_vector = self._vector_for_product(product, embeddings, id_to_index)
            if product_vector is None:
                continue

            weight = context.interaction_weight(product.id) or 1.0
            weighted_vector = product_vector * weight
            accum_vector = weighted_vector if accum_vector is None else accum_vector + weighted_vector
            total_weight += weight

        if accum_vector is None:
            return None

        if total_weight > 0:
            accum_vector = accum_vector / total_weight

        return accum_vector

    def _vector_for_product(
        self,
        product: Product,
        embeddings: np.ndarray,
        id_to_index: dict[int, int],
    ) -> np.ndarray | None:
        if product.id in id_to_index:
            idx = id_to_index[product.id]
            return embeddings[idx].copy()

        model = _get_sbert_model()
        document = _build_document(product)
        if not document.strip():
            return None

        embedding = model.encode([document], show_progress_bar=False)[0]
        embedding = embedding.astype('float32')
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        return embedding

    def _build_reason(self, product: Product, context: RecommendationContext) -> str:
        from apps.recommendations.utils.english_reasons import build_english_reason_from_context
        return build_english_reason_from_context(product, context, "cbf")

def _style_tokens(product) -> list[str]:
    tokens: list[str] = []
    if isinstance(getattr(product, "style_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.style_tags if tag)
    if isinstance(getattr(product, "outfit_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.outfit_tags if tag)
    if getattr(product, "category_type", None):
        tokens.append(product.category_type.lower())
    if getattr(product, "baseColour", None):
        tokens.append(str(product.baseColour).lower())
    return tokens

def _build_document(product: Product) -> str:
    tokens = []
    if getattr(product, "category_type", None):
        tokens.append(product.category_type.lower())
    if getattr(product, "gender", None):
        tokens.append(product.gender.lower())
    if getattr(product, "age_group", None):
        tokens.append(product.age_group.lower())
    if getattr(product, "subCategory", None):
        tokens.append(product.subCategory.lower())
    if getattr(product, "masterCategory", None):
        tokens.append(product.masterCategory.lower())
    if getattr(product, "subCategory", None):
        tokens.append(product.subCategory.lower())
    if getattr(product, "articleType", None):
        tokens.append(product.articleType.lower())
    if getattr(product, "baseColour", None):
        tokens.append(product.baseColour.lower())
    if getattr(product, "season", None):
        tokens.append(product.season.lower())
    if getattr(product, "usage", None):
        tokens.append(product.usage.lower())
    if getattr(product, "productDisplayName", None):
        tokens.append(product.productDisplayName.lower())
    for tag in getattr(product, "style_tags", []) or []:
        tokens.append(str(tag).lower())
    for tag in getattr(product, "outfit_tags", []) or []:
        tokens.append(str(tag).lower())

    colors = getattr(product, "colors", None)
    if colors is not None:
        color_names = getattr(colors, "values_list", None)
        if callable(color_names):
            for color_name in color_names("name", flat=True):
                tokens.append(str(color_name).lower())
    return " ".join(tokens)

engine = ContentBasedRecommendationEngine()

@shared_task
def train_cbf_model(force_retrain: bool = False) -> dict[str, Any]:
    return engine.train(force_retrain=force_retrain)

def recommend_cbf(
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
