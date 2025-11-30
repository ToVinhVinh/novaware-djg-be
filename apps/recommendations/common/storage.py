from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from django.conf import settings

from .exceptions import ModelNotTrainedError

class ArtifactStorage:

    def __init__(self, model_name: str, filename: str = "artifacts.pkl") -> None:
        self.model_name = model_name
        self.filename = filename

    @property
    def base_dir(self) -> Path:
        return Path(settings.BASE_DIR) / "artifacts" / self.model_name

    @property
    def file_path(self) -> Path:
        return self.base_dir / self.filename

    def exists(self) -> bool:
        return self.file_path.exists()

    def load(self) -> dict[str, Any]:
        if not self.exists():
            raise ModelNotTrainedError(self.model_name)
        with self.file_path.open("rb") as handle:
            return pickle.load(handle)

    def save(self, payload: dict[str, Any]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("wb") as handle:
            pickle.dump(payload, handle)

