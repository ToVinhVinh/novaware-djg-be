"""Content-based filtering engine using TF-IDF embeddings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import scipy.sparse as sp
from celery import shared_task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from apps.products.models import Product
from apps.recommendations.common import BaseRecommendationEngine, CandidateFilter
from apps.recommendations.common.context import RecommendationContext


class ContentBasedRecommendationEngine(BaseRecommendationEngine):
    model_name = "cbf"

    def _train_impl(self) -> dict[str, Any]:
        products = list(
            Product.objects.all().select_related("brand", "category")
        )
        product_ids = [product.id for product in products if product.id is not None]
        documents = [_build_document(product) for product in products if product.id is not None]

        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        product_matrix = vectorizer.fit_transform(documents)

        return {
            "product_ids": product_ids,
            "product_matrix": product_matrix,
            "vectorizer": vectorizer,
        }

    def _score_candidates(
        self,
        context: RecommendationContext,
        artifacts: dict[str, Any],
    ) -> dict[int, float]:
        vectorizer: TfidfVectorizer = artifacts["vectorizer"]
        product_ids: list[int] = artifacts["product_ids"]
        product_matrix: sp.csr_matrix = artifacts["product_matrix"]
        id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}

        user_profile = self._build_user_profile(context, vectorizer, product_matrix, id_to_index)
        if user_profile is None or user_profile.nnz == 0:
            user_profile = self._vector_for_product(
                context.current_product,
                vectorizer,
                product_matrix,
                id_to_index,
            )

        candidate_scores: dict[int, float] = {}
        for candidate in context.candidate_products:
            candidate_id = candidate.id
            if candidate_id is None:
                continue
            candidate_vector = self._vector_for_product(candidate, vectorizer, product_matrix, id_to_index)
            if candidate_vector is None or candidate_vector.nnz == 0:
                continue
            similarity = cosine_similarity(user_profile, candidate_vector)[0][0]
            style_bonus = 0.05 * sum(context.style_weight(token) for token in _style_tokens(candidate))
            brand_bonus = 0.15 * context.brand_weight(candidate.brand_id)
            candidate_scores[candidate_id] = float(similarity + style_bonus + brand_bonus)

        return candidate_scores

    def _build_user_profile(
        self,
        context: RecommendationContext,
        vectorizer: TfidfVectorizer,
        product_matrix: sp.csr_matrix,
        id_to_index: dict[int, int],
    ) -> sp.csr_matrix | None:
        accum_vector: sp.csr_matrix | None = None
        total_weight = 0.0
        for product in context.history_products:
            if product.id is None:
                continue
            product_vector = self._vector_for_product(product, vectorizer, product_matrix, id_to_index)
            if product_vector is None or product_vector.nnz == 0:
                continue
            weight = context.interaction_weight(product.id) or 1.0
            weighted_vector = product_vector.multiply(weight)
            accum_vector = weighted_vector if accum_vector is None else accum_vector + weighted_vector
            total_weight += weight
        if accum_vector is None:
            return None
        if total_weight > 0:
            accum_vector = accum_vector.multiply(1 / total_weight)
        return accum_vector

    def _vector_for_product(
        self,
        product,
        vectorizer: TfidfVectorizer,
        product_matrix: sp.csr_matrix,
        id_to_index: dict[int, int],
    ) -> sp.csr_matrix | None:
        if product.id in id_to_index:
            return product_matrix[id_to_index[product.id]]
        document = _build_document(product)
        if not document.strip():
            return None
        return vectorizer.transform([document])


def _style_tokens(product) -> list[str]:
    tokens: list[str] = []
    if isinstance(getattr(product, "style_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.style_tags if tag)
    if isinstance(getattr(product, "outfit_tags", None), list):
        tokens.extend(str(tag).lower() for tag in product.outfit_tags if tag)
    if getattr(product, "category_type", None):
        tokens.append(product.category_type.lower())
    return tokens


def _build_document(product) -> str:
    tokens = []
    if getattr(product, "category_type", None):
        tokens.append(product.category_type.lower())
    if getattr(product, "gender", None):
        tokens.append(product.gender.lower())
    if getattr(product, "age_group", None):
        tokens.append(product.age_group.lower())
    if getattr(product, "category", None) and getattr(product.category, "name", None):
        tokens.append(product.category.name.lower())
    for tag in getattr(product, "style_tags", []) or []:
        tokens.append(str(tag).lower())
    for tag in getattr(product, "outfit_tags", []) or []:
        tokens.append(str(tag).lower())
    if getattr(product, "brand", None) and getattr(product.brand, "name", None):
        tokens.append(product.brand.name.lower())
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

