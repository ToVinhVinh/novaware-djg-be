"""Custom exceptions for recommendation services."""

from __future__ import annotations


class RecommendationError(RuntimeError):
    """Base error for recommendation pipeline."""


class ModelNotTrainedError(RecommendationError):
    """Raised when inference is requested before training artifacts are available."""

    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model '{model_name}' has not been trained yet.")
        self.model_name = model_name

