"""Types for the evaluation layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    """Evaluation configuration for a task."""

    task_type: str  # "binary_classification", "multiclass_classification", "regression"
    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)


class ModelResult(BaseModel):
    """Evaluation results for a single model."""

    model_config = {"protected_namespaces": ()}

    model_name: str
    tier: str
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric_name: str = ""
    primary_metric_value: float = 0.0
    train_time_seconds: float = 0.0
