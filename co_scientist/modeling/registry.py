"""Model registry — defines available models per tier and task type."""

from __future__ import annotations

from typing import Any

from co_scientist.data.types import DatasetProfile, TaskType
from co_scientist.defaults import get_model_defaults

from .types import ModelConfig


def get_baseline_configs(profile: DatasetProfile, seed: int = 42) -> list[ModelConfig]:
    """Return the baseline model configs from YAML defaults: trivial → simple → standard → advanced.

    Each tier in the YAML can be a single model (dict) or a list of models.
    """
    is_clf = profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
    task_family = "classification" if is_clf else "regression"
    task_type_str = task_family

    tier_defaults = get_model_defaults(
        task_family,
        modality=profile.modality.value,
        dataset_path=profile.dataset_path,
    )

    configs = []
    for tier in ("trivial", "simple", "standard", "advanced", "foundation"):
        if tier not in tier_defaults:
            continue
        # Foundation tier is GPU-gated — skip if no GPU
        if tier == "foundation":
            from .foundation import gpu_available
            if not gpu_available():
                continue
        tier_val = tier_defaults[tier]

        # Support both single model (dict) and multiple models (list)
        model_defs = tier_val if isinstance(tier_val, list) else [tier_val]

        for td in model_defs:
            hp = dict(td.get("hyperparameters", {}))
            # Inject random_state for models that support it
            if td["model_type"] in _SEED_MODELS:
                hp.setdefault("random_state", seed)
            # Inject AIDO model name for fine-tuning models
            if td["model_type"] == "aido_finetune":
                from .foundation import get_foundation_model_name
                hp.setdefault("model_name", get_foundation_model_name(profile.modality.value))
            configs.append(ModelConfig(
                name=td["name"],
                tier=tier,
                model_type=td["model_type"],
                task_type=task_type_str,
                hyperparameters=hp,
            ))

    return configs


# Models that accept a random_state parameter
_SEED_MODELS = {"xgboost", "lightgbm", "random_forest", "mlp", "bio_cnn", "svm", "ft_transformer", "embed_xgboost", "embed_mlp", "aido_finetune", "concat_xgboost", "concat_mlp"}


def build_model(config: ModelConfig) -> Any:
    """Instantiate a model from its config. Returns an sklearn-compatible object."""
    builders = {
        "majority_class": _build_majority_class,
        "mean_predictor": _build_mean_predictor,
        "logistic_regression": _build_logistic_regression,
        "ridge_regression": _build_ridge_regression,
        "elastic_net_clf": _build_elastic_net_clf,
        "elastic_net_reg": _build_elastic_net_reg,
        "svm": _build_svm,
        "knn": _build_knn,
        "xgboost": _build_xgboost,
        "lightgbm": _build_lightgbm,
        "random_forest": _build_random_forest,
        "mlp": _build_mlp,
        "bio_cnn": _build_bio_cnn,
        "ft_transformer": _build_ft_transformer,
        "embed_xgboost": _build_embed_xgboost,
        "embed_mlp": _build_embed_mlp,
        "aido_finetune": _build_aido_finetune,
        "concat_xgboost": _build_concat_xgboost,
        "concat_mlp": _build_concat_mlp,
    }

    builder = builders.get(config.model_type)
    if builder is None:
        raise ValueError(f"Unknown model type: {config.model_type}")
    return builder(config)


# ---------------------------------------------------------------------------
# Trivial
# ---------------------------------------------------------------------------

def _build_majority_class(config: ModelConfig):
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy="most_frequent")


def _build_mean_predictor(config: ModelConfig):
    from sklearn.dummy import DummyRegressor
    return DummyRegressor(strategy="mean")


# ---------------------------------------------------------------------------
# Simple (linear)
# ---------------------------------------------------------------------------

def _build_logistic_regression(config: ModelConfig):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**config.hyperparameters)


def _build_ridge_regression(config: ModelConfig):
    from sklearn.linear_model import Ridge
    return Ridge(**config.hyperparameters)


def _build_elastic_net_clf(config: ModelConfig):
    from sklearn.linear_model import LogisticRegression
    params = dict(config.hyperparameters)
    # LogisticRegression with elasticnet penalty + saga solver
    params.pop("random_state", None)
    params["penalty"] = "elasticnet"
    params["solver"] = "saga"
    return LogisticRegression(**params)


def _build_elastic_net_reg(config: ModelConfig):
    from sklearn.linear_model import ElasticNet
    params = dict(config.hyperparameters)
    params.pop("random_state", None)  # ElasticNet doesn't use random_state
    return ElasticNet(**params)


def _build_svm(config: ModelConfig):
    from sklearn.svm import SVC, SVR
    params = dict(config.hyperparameters)
    if config.task_type == "classification":
        return SVC(probability=True, **params)
    else:
        params.pop("random_state", None)  # SVR doesn't accept random_state
        return SVR(**params)


def _build_knn(config: ModelConfig):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    params = dict(config.hyperparameters)
    params.pop("random_state", None)  # KNN doesn't use random_state
    if config.task_type == "classification":
        return KNeighborsClassifier(**params)
    else:
        return KNeighborsRegressor(**params)


# ---------------------------------------------------------------------------
# Standard (tree ensembles)
# ---------------------------------------------------------------------------

def _build_xgboost(config: ModelConfig):
    import xgboost as xgb
    params = dict(config.hyperparameters)
    if config.task_type == "classification":
        return xgb.XGBClassifier(**params, eval_metric="mlogloss")
    else:
        return xgb.XGBRegressor(**params)


def _build_lightgbm(config: ModelConfig):
    import lightgbm as lgb
    params = dict(config.hyperparameters)
    if config.task_type == "classification":
        return lgb.LGBMClassifier(**params)
    else:
        return lgb.LGBMRegressor(**params)


def _build_random_forest(config: ModelConfig):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    params = dict(config.hyperparameters)
    # YAML null → Python None, which is already what sklearn expects for max_depth
    if config.task_type == "classification":
        return RandomForestClassifier(**params)
    else:
        return RandomForestRegressor(**params)


# ---------------------------------------------------------------------------
# Advanced (neural)
# ---------------------------------------------------------------------------

def _build_mlp(config: ModelConfig):
    from .mlp import MLPClassifier, MLPRegressor
    params = dict(config.hyperparameters)
    # Convert hidden_dims from YAML list if needed
    if "hidden_dims" in params and isinstance(params["hidden_dims"], str):
        params["hidden_dims"] = [int(x) for x in params["hidden_dims"].split(",")]
    if config.task_type == "classification":
        return MLPClassifier(**params)
    else:
        return MLPRegressor(**params)


def _build_bio_cnn(config: ModelConfig):
    from .bio_cnn import BioCNNClassifier, BioCNNRegressor
    params = dict(config.hyperparameters)
    if config.task_type == "classification":
        return BioCNNClassifier(**params)
    else:
        return BioCNNRegressor(**params)



def _build_ft_transformer(config: ModelConfig):
    from .ft_transformer import FTTransformerClassifier, FTTransformerRegressor
    params = dict(config.hyperparameters)
    params.setdefault("max_epochs", 15)
    params.setdefault("patience", 2)
    if config.task_type == "classification":
        return FTTransformerClassifier(**params)
    else:
        return FTTransformerRegressor(**params)


# ---------------------------------------------------------------------------
# Foundation (embedding-based models)
# ---------------------------------------------------------------------------

def _build_embed_xgboost(config: ModelConfig):
    """XGBoost trained on foundation model embeddings."""
    import xgboost as xgb
    params = dict(config.hyperparameters)
    if config.task_type == "classification":
        return xgb.XGBClassifier(**params, eval_metric="mlogloss")
    else:
        return xgb.XGBRegressor(**params)


def _build_embed_mlp(config: ModelConfig):
    """MLP trained on foundation model embeddings."""
    from .mlp import MLPClassifier, MLPRegressor
    params = dict(config.hyperparameters)
    if "hidden_dims" in params and isinstance(params["hidden_dims"], str):
        params["hidden_dims"] = [int(x) for x in params["hidden_dims"].split(",")]
    if config.task_type == "classification":
        return MLPClassifier(**params)
    else:
        return MLPRegressor(**params)


def _build_aido_finetune(config: ModelConfig):
    """End-to-end AIDO fine-tuning with task head."""
    from .aido_finetune import AIDOFinetuneClassifier, AIDOFinetuneRegressor
    params = dict(config.hyperparameters)
    if config.task_type == "classification":
        return AIDOFinetuneClassifier(**params)
    else:
        return AIDOFinetuneRegressor(**params)


def _build_concat_xgboost(config: ModelConfig):
    """XGBoost on concatenated handcrafted features + AIDO embeddings."""
    import xgboost as xgb
    params = dict(config.hyperparameters)
    if config.task_type == "classification":
        return xgb.XGBClassifier(**params, eval_metric="mlogloss")
    else:
        return xgb.XGBRegressor(**params)


def _build_concat_mlp(config: ModelConfig):
    """MLP on concatenated handcrafted features + AIDO embeddings."""
    from .mlp import MLPClassifier, MLPRegressor
    params = dict(config.hyperparameters)
    if "hidden_dims" in params and isinstance(params["hidden_dims"], str):
        params["hidden_dims"] = [int(x) for x in params["hidden_dims"].split(",")]
    if config.task_type == "classification":
        return MLPClassifier(**params)
    else:
        return MLPRegressor(**params)
