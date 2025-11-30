from __future__ import annotations

class RecommendationError(RuntimeError):

class ModelNotTrainedError(RecommendationError):

    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model '{model_name}' has not been trained yet.")
        self.model_name = model_name

