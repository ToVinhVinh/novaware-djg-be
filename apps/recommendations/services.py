from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import tensorflow as tf
from celery import shared_task
from django.contrib.auth import get_user_model
from django.db import transaction

from apps.products.models import Product

from .models import RecommendationLog, RecommendationRequest, RecommendationResult

logger = logging.getLogger(__name__)
User = get_user_model()

@dataclass
class RecommendationContext:
    request: RecommendationRequest
    user_vector: np.ndarray
    product_matrix: np.ndarray

class BaseRecommender:
    def train(self, context: RecommendationContext) -> None:
        raise NotImplementedError

    def recommend(self, context: RecommendationContext, top_k: int = 10) -> list[int]:
        raise NotImplementedError

class CFRecommender(BaseRecommender):
    def __init__(self) -> None:
        self.model = None

    def train(self, context: RecommendationContext) -> None:
        user_input = tf.keras.Input(shape=(context.product_matrix.shape[1],))
        dense = tf.keras.layers.Dense(64, activation="relu")(user_input)
        output = tf.keras.layers.Dense(context.product_matrix.shape[1], activation="sigmoid")(dense)
        self.model = tf.keras.Model(inputs=user_input, outputs=output)
        self.model.compile(optimizer="adam", loss="mse")
        dummy_target = np.random.rand(1, context.product_matrix.shape[1])
        self.model.fit(context.product_matrix[:1], dummy_target, epochs=1, verbose=0)

    def recommend(self, context: RecommendationContext, top_k: int = 10) -> list[int]:
        if not self.model:
            raise RuntimeError("Model has not been trained")
        predictions = self.model.predict(context.product_matrix[:1], verbose=0)[0]
        top_indices = predictions.argsort()[-top_k:][::-1]
        return top_indices.tolist()

class RecommendationService:
    recommender_map = {
        "cf": CFRecommender,
        "hybrid": CFRecommender,
        "cb": CFRecommender,
        "gnn": CFRecommender,
    }

    @classmethod
    def enqueue_recommendation(cls, request_obj: RecommendationRequest) -> None:
        run_recommendation_task.delay(request_obj.id)

    @classmethod
    def build_context(cls, request_obj: RecommendationRequest) -> RecommendationContext:
        user = request_obj.user
        products = Product.objects.all()
        product_matrix = np.random.rand(len(products) or 1, 32)
        user_vector = np.random.rand(1, 32)
        return RecommendationContext(request=request_obj, user_vector=user_vector, product_matrix=product_matrix)

    @classmethod
    def run(cls, request_obj: RecommendationRequest) -> Iterable[Product]:
        context = cls.build_context(request_obj)
        recommender_cls = cls.recommender_map.get(request_obj.algorithm, CFRecommender)
        recommender = recommender_cls()
        recommender.train(context)
        params = request_obj.parameters or {}
        indices = recommender.recommend(context, top_k=params.get("top_k", 10))
        product_ids = list(
            Product.objects.values_list("id", flat=True)
        )
        selected_ids = [product_ids[i] for i in indices if i < len(product_ids)]
        return Product.objects.filter(id__in=selected_ids)

@shared_task
def run_recommendation_task(request_id: int) -> None:
    try:
        request_obj = RecommendationRequest.objects.get(pk=request_id)
    except RecommendationRequest.DoesNotExist:
        logger.warning("RecommendationRequest %s does not exist", request_id)
        return

    RecommendationLog.objects.create(request=request_obj, message="Starting recommender processing")

    products = RecommendationService.run(request_obj)

    with transaction.atomic():
        result, _ = RecommendationResult.objects.update_or_create(
            request=request_obj,
            defaults={"metadata": request_obj.parameters or {}},
        )
        result.products.set(products)

    RecommendationLog.objects.create(request=request_obj, message="Completed recommender processing")

