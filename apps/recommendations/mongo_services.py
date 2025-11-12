"""Service layer cho hệ thống gợi ý sử dụng MongoEngine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import tensorflow as tf
from bson import ObjectId
from celery import shared_task

from apps.products.mongo_models import Product
from apps.users.mongo_models import User

from .mongo_models import (
    RecommendationLog,
    RecommendationRequest,
    RecommendationResult,
)

logger = logging.getLogger(__name__)


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
        # Placeholder training step - trong thực tế cần dữ liệu lịch sử tương tác
        dummy_target = np.random.rand(1, context.product_matrix.shape[1])
        self.model.fit(context.product_matrix[:1], dummy_target, epochs=1, verbose=0)

    def recommend(self, context: RecommendationContext, top_k: int = 10) -> list[int]:
        if not self.model:
            raise RuntimeError("Model chưa được train")
        predictions = self.model.predict(context.product_matrix[:1], verbose=0)[0]
        top_indices = predictions.argsort()[-top_k:][::-1]
        return top_indices.tolist()


class RecommendationService:
    recommender_map = {
        "cf": CFRecommender,
        "hybrid": CFRecommender,  # Placeholder, có thể thay bằng HybridRecommender
        "cb": CFRecommender,
        "gnn": CFRecommender,
    }

    @classmethod
    def enqueue_recommendation(cls, request_obj: RecommendationRequest) -> None:
        """Enqueue recommendation task."""
        run_recommendation_task.delay(str(request_obj.id))

    @classmethod
    def build_context(cls, request_obj: RecommendationRequest) -> RecommendationContext:
        """Build recommendation context from request."""
        # Get user from MongoEngine
        try:
            user = User.objects.get(id=request_obj.user_id)
        except User.DoesNotExist:
            logger.warning("User %s không tồn tại", request_obj.user_id)
            user = None
        
        # Get all products
        products = list(Product.objects.all())
        product_matrix = np.random.rand(len(products) or 1, 32)
        user_vector = np.random.rand(1, 32)
        
        # Use user embedding if available
        if user and user.user_embedding:
            try:
                user_vector = np.array([user.user_embedding])
            except Exception:
                pass
        
        return RecommendationContext(
            request=request_obj,
            user_vector=user_vector,
            product_matrix=product_matrix
        )

    @classmethod
    def run(cls, request_obj: RecommendationRequest) -> Iterable[Product]:
        """Run recommendation algorithm."""
        context = cls.build_context(request_obj)
        recommender_cls = cls.recommender_map.get(request_obj.algorithm, CFRecommender)
        recommender = recommender_cls()
        recommender.train(context)
        params = request_obj.parameters or {}
        indices = recommender.recommend(context, top_k=params.get("top_k", 10))
        
        # Get all products
        products = list(Product.objects.all())
        if not products:
            return []
        
        # Select products by indices
        selected_products = []
        for idx in indices:
            if idx < len(products):
                selected_products.append(products[idx])
        
        return selected_products


@shared_task
def run_recommendation_task(request_id: str) -> None:
    """Celery task to run recommendation."""
    try:
        request_obj = RecommendationRequest.objects.get(id=ObjectId(request_id))
    except (RecommendationRequest.DoesNotExist, Exception) as e:
        logger.warning("RecommendationRequest %s không tồn tại: %s", request_id, str(e))
        return

    # Create log
    RecommendationLog.objects.create(
        request_id=request_obj.id,
        message="Bắt đầu xử lý recommender"
    )

    # Run recommendation
    products = RecommendationService.run(request_obj)

    # Create or update result
    product_ids = [str(p.id) for p in products]
    
    try:
        result = RecommendationResult.objects.get(request_id=request_obj.id)
        result.product_ids = [ObjectId(pid) for pid in product_ids]
        result.metadata = request_obj.parameters or {}
        result.save()
    except RecommendationResult.DoesNotExist:
        result = RecommendationResult(
            request_id=request_obj.id,
            product_ids=[ObjectId(pid) for pid in product_ids],
            metadata=request_obj.parameters or {}
        )
        result.save()

    # Create log
    RecommendationLog.objects.create(
        request_id=request_obj.id,
        message="Hoàn tất xử lý recommender"
    )

