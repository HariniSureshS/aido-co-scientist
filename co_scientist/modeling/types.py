"""Types for the modeling layer."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# Model types that need raw sequences for predict/fit
# "stacking" included because ensemble may contain sequence models internally
_SEQUENCE_MODELS = {"bio_cnn", "stacking", "aido_finetune"}

# Model types that use foundation model embeddings
_EMBEDDING_MODELS = {"embed_xgboost", "embed_mlp"}

# Model types that concatenate handcrafted features + embeddings
_CONCAT_MODELS = {"concat_xgboost", "concat_mlp"}


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    name: str
    tier: str  # "trivial", "simple", "standard", "advanced", "foundation"
    model_type: str  # e.g. "majority_class", "logistic_regression", "xgboost"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    task_type: str = ""  # "classification" or "regression"


class TrainedModel:
    """A trained model with its metadata."""

    def __init__(
        self,
        config: ModelConfig,
        model: Any,  # the sklearn/xgboost model object
        train_time_seconds: float = 0.0,
    ):
        self.config = config
        self.model = model
        self.train_time_seconds = train_time_seconds

    @property
    def needs_sequences(self) -> bool:
        return self.config.model_type in _SEQUENCE_MODELS

    @property
    def needs_embeddings(self) -> bool:
        return self.config.model_type in _EMBEDDING_MODELS

    def predict(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        if self.needs_sequences:
            preds = self.model.predict(X, sequences=sequences)
        else:
            preds = self.model.predict(X)
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
        return preds

    def predict_proba(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            if self.needs_sequences:
                return self.model.predict_proba(X, sequences=sequences)
            return self.model.predict_proba(X)
        return None
