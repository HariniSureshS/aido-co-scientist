"""Shared fixtures for the co-scientist test suite."""

from __future__ import annotations

import numpy as np
import pytest

from co_scientist.data.types import DatasetProfile, Modality, SplitData, TaskType
from co_scientist.evaluation.types import EvalConfig, ModelResult


@pytest.fixture
def sample_profile() -> DatasetProfile:
    """A realistic DatasetProfile for binary classification on tabular data."""
    return DatasetProfile(
        dataset_name="test_dataset",
        dataset_path="test/binary_clf",
        task_hint="binary_clf",
        modality=Modality.TABULAR,
        task_type=TaskType.BINARY_CLASSIFICATION,
        input_columns=[f"feat_{i}" for i in range(5)],
        target_column="label",
        input_type="tabular",
        num_samples=100,
        num_features=5,
        num_classes=2,
        class_distribution={"0": 60, "1": 40},
    )


@pytest.fixture
def sample_eval_config() -> EvalConfig:
    """EvalConfig with accuracy as the primary metric."""
    return EvalConfig(
        task_type="binary_classification",
        primary_metric="accuracy",
        secondary_metrics=["f1", "roc_auc"],
    )


@pytest.fixture
def sample_split() -> SplitData:
    """A small SplitData with 10 samples per split, 5 features."""
    rng = np.random.RandomState(42)
    return SplitData(
        X_train=rng.randn(10, 5).astype(np.float32),
        y_train=rng.randint(0, 2, size=10),
        X_val=rng.randn(10, 5).astype(np.float32),
        y_val=rng.randint(0, 2, size=10),
        X_test=rng.randn(10, 5).astype(np.float32),
        y_test=rng.randint(0, 2, size=10),
        split_method="random",
        feature_names=[f"feat_{i}" for i in range(5)],
    )


@pytest.fixture
def sample_model_result() -> ModelResult:
    """A ModelResult from a hypothetical logistic regression run."""
    return ModelResult(
        model_name="logistic_regression_default",
        tier="simple",
        metrics={"accuracy": 0.85, "f1": 0.82, "roc_auc": 0.90},
        primary_metric_name="accuracy",
        primary_metric_value=0.85,
        train_time_seconds=1.5,
    )
