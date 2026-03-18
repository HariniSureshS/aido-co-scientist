"""Bayesian hyperparameter optimization with Optuna."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
from rich.console import Console

from co_scientist.data.types import DatasetProfile, SplitData
from co_scientist.defaults import get_defaults
from co_scientist.evaluation.metrics import evaluate_model
from co_scientist.evaluation.types import EvalConfig, ModelResult

from .registry import build_model
from .types import ModelConfig, TrainedModel

console = Console()

# Silence Optuna's default logging — we'll print our own summaries.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_hp_search(
    base_config: ModelConfig,
    split: SplitData,
    eval_config: EvalConfig,
    profile: DatasetProfile,
    seed: int = 42,
    n_trials_override: int | None = None,
    timeout_override: int | None = None,
) -> tuple[TrainedModel, ModelResult] | None:
    """Run Optuna HP search starting from a baseline model config.

    Returns the best (TrainedModel, ModelResult) or None if search is disabled
    or doesn't improve over the baseline.

    n_trials_override/timeout_override: set by complexity scoring to scale
    search effort based on dataset difficulty.
    """
    defaults = get_defaults(profile.modality.value, profile.dataset_path)
    hp_cfg = defaults.get("hp_search", {})

    if not hp_cfg.get("enabled", True):
        return None

    search_spaces = hp_cfg.get("search_spaces", {})
    space = search_spaces.get(base_config.model_type)
    if space is None:
        console.print(f"  [dim]No search space defined for {base_config.model_type}, skipping HP search.[/dim]")
        return None

    n_trials = n_trials_override or hp_cfg.get("n_trials", 30)
    timeout = timeout_override or hp_cfg.get("timeout_seconds", 300)
    sampler_name = hp_cfg.get("sampler", "tpe")

    lower_is_better = eval_config.primary_metric in ("mse", "rmse", "mae")
    direction = "minimize" if lower_is_better else "maximize"

    # Build sampler
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    console.print(f"  [bold]Optuna HP search:[/bold] {n_trials} trials, "
                  f"timeout {timeout}s, optimizing {eval_config.primary_metric} ({direction})")

    def objective(trial: optuna.Trial) -> float:
        hp = _sample_hyperparameters(trial, space, base_config, seed)
        config = ModelConfig(
            name=f"{base_config.model_type}_trial_{trial.number}",
            tier="tuned",
            model_type=base_config.model_type,
            task_type=base_config.task_type,
            hyperparameters=hp,
        )
        try:
            model = build_model(config)
            y_train = split.y_train
            if config.model_type in _CONCAT_MODELS:
                import numpy as _np
                X_combined = _np.hstack([split.X_train, split.X_embed_train])
                model.fit(X_combined, y_train)
            elif config.model_type in _EMBEDDING_MODELS:
                model.fit(split.X_embed_train, y_train)
            elif config.model_type in _SEQUENCE_MODELS:
                model.fit(split.X_train, split.y_train, sequences=split.seqs_train)
            else:
                model.fit(split.X_train, y_train)
            trained = TrainedModel(config=config, model=model)
            result = evaluate_model(trained, split, eval_config, use_test=False)
            return result.primary_metric_value
        except Exception:
            # If a trial crashes (e.g., bad HP combo), return worst possible score
            return float("inf") if lower_is_better else float("-inf")

    study = optuna.create_study(direction=direction, sampler=sampler)

    start = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    elapsed = time.time() - start

    best_trial = study.best_trial
    console.print(f"  Completed {len(study.trials)} trials in {elapsed:.1f}s")
    console.print(f"  Best trial #{best_trial.number}: "
                  f"{eval_config.primary_metric}={best_trial.value:.4f}")

    # Retrain the best model on full training set to get the final TrainedModel
    best_hp = _sample_hyperparameters_from_dict(best_trial.params, space, base_config, seed)
    best_config = ModelConfig(
        name=f"{base_config.model_type}_tuned",
        tier="tuned",
        model_type=base_config.model_type,
        task_type=base_config.task_type,
        hyperparameters=best_hp,
    )

    model = build_model(best_config)
    y_train_final = split.y_train
    train_start = time.time()
    if best_config.model_type in _CONCAT_MODELS:
        import numpy as _np
        X_combined = _np.hstack([split.X_train, split.X_embed_train])
        model.fit(X_combined, y_train_final)
    elif best_config.model_type in _EMBEDDING_MODELS:
        model.fit(split.X_embed_train, y_train_final)
    elif best_config.model_type in _SEQUENCE_MODELS:
        model.fit(split.X_train, split.y_train, sequences=split.seqs_train)
    else:
        model.fit(split.X_train, y_train_final)
    train_time = time.time() - train_start

    trained = TrainedModel(config=best_config, model=model, train_time_seconds=train_time)
    result = evaluate_model(trained, split, eval_config, use_test=False)

    console.print(f"  Tuned model: {eval_config.primary_metric}={result.primary_metric_value:.4f}")

    return trained, result


def _sample_hyperparameters(
    trial: optuna.Trial,
    space: dict[str, dict],
    base_config: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    """Sample hyperparameters from the search space for an Optuna trial."""
    hp: dict[str, Any] = {}
    for param_name, param_spec in space.items():
        ptype = param_spec["type"]

        if ptype == "categorical":
            hp[param_name] = trial.suggest_categorical(param_name, param_spec["choices"])
        elif ptype == "int":
            hp[param_name] = trial.suggest_int(
                param_name, param_spec["low"], param_spec["high"],
                log=param_spec.get("log", False),
            )
        elif ptype == "float":
            hp[param_name] = trial.suggest_float(
                param_name, param_spec["low"], param_spec["high"],
                log=param_spec.get("log", False),
            )

    # Carry over fixed params not in search space (e.g., random_state)
    for k, v in base_config.hyperparameters.items():
        if k not in hp:
            hp[k] = v

    # Always set random_state for reproducibility
    if base_config.model_type in _SEED_MODELS:
        hp.setdefault("random_state", seed)

    # LightGBM: always suppress warnings during HP search
    if base_config.model_type == "lightgbm":
        hp["verbose"] = -1

    # MLP: convert n_layers + hidden_size → hidden_dims
    hp = _postprocess_mlp_hp(hp)

    return hp


# Models that accept random_state
_SEED_MODELS = {"xgboost", "lightgbm", "random_forest", "mlp", "bio_cnn",
                "embed_xgboost", "embed_mlp", "concat_xgboost", "concat_mlp", "aido_finetune"}

# Models that need raw sequences for fit/predict
_SEQUENCE_MODELS = {"bio_cnn", "aido_finetune"}

# Models that use foundation model embeddings
_EMBEDDING_MODELS = {"embed_xgboost", "embed_mlp"}

# Models that concatenate handcrafted features + embeddings
_CONCAT_MODELS = {"concat_xgboost", "concat_mlp"}


def _sample_hyperparameters_from_dict(
    trial_params: dict[str, Any],
    space: dict[str, dict],
    base_config: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    """Reconstruct full hyperparameters from Optuna's best_trial.params."""
    hp = dict(trial_params)

    # Carry over fixed params not in search space
    for k, v in base_config.hyperparameters.items():
        if k not in hp:
            hp[k] = v

    if base_config.model_type in _SEED_MODELS:
        hp.setdefault("random_state", seed)

    if base_config.model_type == "lightgbm":
        hp["verbose"] = -1

    hp = _postprocess_mlp_hp(hp)

    return hp


def _postprocess_mlp_hp(hp: dict[str, Any]) -> dict[str, Any]:
    """Convert n_layers + hidden_size search params into hidden_dims list."""
    if "n_layers" in hp and "hidden_size" in hp:
        n_layers = hp.pop("n_layers")
        hidden_size = hp.pop("hidden_size")
        # Decreasing layer sizes: [512, 256, 128] for n_layers=3, hidden_size=512
        hp["hidden_dims"] = [max(32, hidden_size // (2 ** i)) for i in range(n_layers)]
    return hp
