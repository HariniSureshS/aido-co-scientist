"""Tests for the model registry — building models and retrieving baseline configs."""

from __future__ import annotations

import pytest

from co_scientist.modeling.registry import build_model, get_baseline_configs
from co_scientist.modeling.types import ModelConfig


# ---------------------------------------------------------------------------
# Sklearn models: build each type and verify fit/predict exist
# ---------------------------------------------------------------------------

_SKLEARN_MODELS = [
    ("majority_class", "classification"),
    ("mean_predictor", "regression"),
    ("logistic_regression", "classification"),
    ("ridge_regression", "regression"),
    ("random_forest", "classification"),
    ("svm", "classification"),
    ("knn", "classification"),
]


@pytest.mark.parametrize("model_type,task_type", _SKLEARN_MODELS)
def test_build_all_sklearn_models(model_type: str, task_type: str):
    config = ModelConfig(
        name=f"{model_type}_test",
        tier="test",
        model_type=model_type,
        task_type=task_type,
        hyperparameters={},
    )
    model = build_model(config)
    assert hasattr(model, "fit"), f"{model_type} missing fit()"
    assert hasattr(model, "predict"), f"{model_type} missing predict()"


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def test_build_xgboost_classification():
    config = ModelConfig(
        name="xgb_clf",
        tier="standard",
        model_type="xgboost",
        task_type="classification",
        hyperparameters={"n_estimators": 10, "random_state": 0},
    )
    model = build_model(config)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_build_xgboost_regression():
    config = ModelConfig(
        name="xgb_reg",
        tier="standard",
        model_type="xgboost",
        task_type="regression",
        hyperparameters={"n_estimators": 10, "random_state": 0},
    )
    model = build_model(config)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def test_build_lightgbm():
    config = ModelConfig(
        name="lgbm_clf",
        tier="standard",
        model_type="lightgbm",
        task_type="classification",
        hyperparameters={"n_estimators": 10, "random_state": 0},
    )
    model = build_model(config)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# Unknown model type
# ---------------------------------------------------------------------------

def test_build_unknown_model_type():
    config = ModelConfig(
        name="bad_model",
        tier="test",
        model_type="nonexistent_model",
        task_type="classification",
    )
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model(config)


# ---------------------------------------------------------------------------
# Baseline configs
# ---------------------------------------------------------------------------

def test_get_baseline_configs(sample_profile):
    configs = get_baseline_configs(sample_profile, seed=42)
    assert isinstance(configs, list)
    assert len(configs) > 0
    for cfg in configs:
        assert isinstance(cfg, ModelConfig)
        assert cfg.tier in ("trivial", "simple", "standard", "advanced")
        assert cfg.name
        assert cfg.model_type
