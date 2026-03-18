"""Guardrails — plausibility checks and scientific discipline enforcement.

Implements Section 10.3 of ARCHITECTURE.md:
  - Dataset validation (before modeling)
  - Task type verification (multi-signal)
  - Metric sanity checks
  - Model-data compatibility (before training)
  - Result plausibility (after training)

Inspired by:
  - Sakana AI Scientist: automated quality checks as peer review
  - CellAgent: error-feedback self-correction patterns
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from rich.console import Console

from co_scientist.data.types import DatasetProfile, Modality, SplitData, TaskType
from co_scientist.evaluation.types import EvalConfig, ModelResult
from co_scientist.modeling.types import ModelConfig, TrainedModel

console = Console()


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class GuardrailAlert:
    """A single guardrail finding."""

    def __init__(self, severity: Severity, code: str, message: str):
        self.severity = severity
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity.value, "code": self.code, "message": self.message}


# Higher-is-better metrics
_HIGHER_IS_BETTER = {"accuracy", "macro_f1", "weighted_f1", "f1", "auroc",
                     "precision", "recall", "macro_precision", "macro_recall",
                     "r2", "spearman", "pearson"}
_LOWER_IS_BETTER = {"mse", "rmse", "mae"}


def _is_lower_better(metric: str) -> bool:
    return metric in _LOWER_IS_BETTER


# ---------------------------------------------------------------------------
# Task type verification (multi-signal, Section 10.3)
# ---------------------------------------------------------------------------

# Classification metrics vs regression metrics
_CLASSIFICATION_METRICS = {"accuracy", "macro_f1", "weighted_f1", "f1", "auroc",
                           "precision", "recall", "macro_precision", "macro_recall"}
_REGRESSION_METRICS = {"mse", "rmse", "mae", "r2", "spearman", "pearson"}

# Classification model types vs regression model types
_CLASSIFICATION_MODELS = {"majority_class", "logistic_regression", "elastic_net_clf"}
_REGRESSION_MODELS = {"mean_predictor", "ridge_regression", "elastic_net_reg"}
_EITHER_MODELS = {"xgboost", "lightgbm", "random_forest", "mlp", "bio_cnn"}  # can do both
_SEQUENCE_ONLY_MODELS = {"bio_cnn"}  # require raw sequences


def verify_task_type(profile: DatasetProfile) -> list[GuardrailAlert]:
    """Multi-signal task type verification — require 2+ signals to agree.

    Signals: path parsing, target dtype, value counts, class distribution.
    Architecture Section 10.3: 'Require 2+ signals to agree'.
    """
    alerts: list[GuardrailAlert] = []

    if profile.task_type == TaskType.UNKNOWN:
        alerts.append(GuardrailAlert(
            Severity.WARNING, "unknown_task_type",
            "Could not auto-detect task type — all signals inconclusive",
        ))
        return alerts

    is_clf = profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

    # Collect signals
    signals: dict[str, str] = {}  # signal_name → "classification" or "regression"

    # Signal 1: Path hint
    hint = profile.task_hint.lower()
    if "classification" in hint or "cell_type" in hint:
        signals["path_hint"] = "classification"
    elif any(k in hint for k in ("regression", "efficiency", "expression", "abundance", "ribosome")):
        signals["path_hint"] = "regression"

    # Signal 2: Number of unique target values
    if profile.num_classes > 0 and profile.num_classes <= 50:
        signals["value_counts"] = "classification"
    elif profile.target_stats and profile.target_stats.get("std", 0) > 0:
        # Continuous target with many unique values
        signals["value_counts"] = "regression"

    # Signal 3: Class distribution present
    if profile.class_distribution:
        signals["class_distribution"] = "classification"
    elif profile.target_stats:
        signals["target_stats"] = "regression"

    # Count agreement
    detected = "classification" if is_clf else "regression"
    agreeing = sum(1 for v in signals.values() if v == detected)
    disagreeing = sum(1 for v in signals.values() if v != detected)

    if len(signals) < 2:
        alerts.append(GuardrailAlert(
            Severity.INFO, "few_task_signals",
            f"Only {len(signals)} signal(s) for task type detection ({detected}) — low confidence",
        ))
    elif disagreeing > 0 and disagreeing >= agreeing:
        conflicting = {k: v for k, v in signals.items() if v != detected}
        alerts.append(GuardrailAlert(
            Severity.WARNING, "task_type_conflict",
            f"Task type signals disagree: detected {detected}, but {conflicting} suggest otherwise",
        ))

    return alerts


# ---------------------------------------------------------------------------
# Metric sanity checks (Section 10.3)
# ---------------------------------------------------------------------------

def check_metric_sanity(
    profile: DatasetProfile,
    eval_config: EvalConfig,
) -> list[GuardrailAlert]:
    """Validate that the chosen metric makes sense for the task and data."""
    alerts: list[GuardrailAlert] = []
    metric = eval_config.primary_metric
    is_clf = eval_config.task_type in ("binary_classification", "multiclass_classification")

    # Classification metric on regression task → ERROR
    if eval_config.task_type == "regression" and metric in _CLASSIFICATION_METRICS:
        alerts.append(GuardrailAlert(
            Severity.ERROR, "metric_task_mismatch",
            f"Classification metric '{metric}' used on regression task",
        ))

    # Regression metric on classification task → ERROR
    if is_clf and metric in _REGRESSION_METRICS:
        alerts.append(GuardrailAlert(
            Severity.ERROR, "metric_task_mismatch",
            f"Regression metric '{metric}' used on classification task",
        ))

    # Accuracy on severely imbalanced data → WARNING, suggest macro F1
    if is_clf and metric == "accuracy" and profile.class_distribution:
        total = sum(profile.class_distribution.values())
        if total > 0:
            min_pct = min(profile.class_distribution.values()) / total * 100
            if min_pct < 5:
                alerts.append(GuardrailAlert(
                    Severity.WARNING, "accuracy_on_imbalanced",
                    f"Using accuracy on imbalanced data (smallest class: {min_pct:.1f}%) "
                    f"— consider macro_f1 instead",
                ))

    # AUROC on multiclass without probabilities support check
    if metric == "auroc" and eval_config.task_type == "multiclass_classification":
        if profile.num_classes and profile.num_classes > 20:
            alerts.append(GuardrailAlert(
                Severity.INFO, "auroc_many_classes",
                f"AUROC with {profile.num_classes} classes may be unreliable — consider macro_f1",
            ))

    return alerts


# ---------------------------------------------------------------------------
# Model-data compatibility checks (Section 10.3)
# ---------------------------------------------------------------------------

def check_model_data_compatibility(
    config: ModelConfig,
    profile: DatasetProfile,
    split: SplitData,
) -> list[GuardrailAlert]:
    """Validate that a model config is compatible with the data before training."""
    alerts: list[GuardrailAlert] = []
    name = config.name
    model_type = config.model_type
    is_clf = profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

    # Classification model on regression data → BLOCK
    if not is_clf and model_type in _CLASSIFICATION_MODELS:
        alerts.append(GuardrailAlert(
            Severity.ERROR, "model_task_mismatch",
            f"{name}: classification model '{model_type}' cannot be used for regression",
        ))

    # Regression model on classification data → BLOCK
    if is_clf and model_type in _REGRESSION_MODELS:
        alerts.append(GuardrailAlert(
            Severity.ERROR, "model_task_mismatch",
            f"{name}: regression model '{model_type}' cannot be used for classification",
        ))

    # Sequence model on non-sequence data → BLOCK
    if model_type in _SEQUENCE_ONLY_MODELS:
        from co_scientist.data.types import Modality
        if profile.modality not in (Modality.RNA, Modality.DNA, Modality.PROTEIN):
            alerts.append(GuardrailAlert(
                Severity.ERROR, "sequence_model_on_tabular",
                f"{name}: sequence model '{model_type}' requires sequence data, but modality is {profile.modality.value}",
            ))
        elif split.seqs_train is None:
            alerts.append(GuardrailAlert(
                Severity.ERROR, "no_sequences_for_cnn",
                f"{name}: sequence model '{model_type}' requires raw sequences but none are available",
            ))

    # Wrong num_classes in config → BLOCK
    if is_clf and "num_classes" in config.hyperparameters:
        expected = profile.num_classes
        configured = config.hyperparameters["num_classes"]
        if expected > 0 and configured != expected:
            alerts.append(GuardrailAlert(
                Severity.ERROR, "wrong_num_classes",
                f"{name}: configured for {configured} classes but data has {expected}",
            ))

    # More parameters than samples → WARN (estimate for different model types)
    n_train = len(split.y_train)
    n_features = split.X_train.shape[1] if len(split.X_train.shape) > 1 else 0
    estimated_params = _estimate_model_params(config, n_features, profile.num_classes)
    if estimated_params is not None and estimated_params > n_train:
        alerts.append(GuardrailAlert(
            Severity.WARNING, "more_params_than_samples",
            f"{name}: estimated {estimated_params:,} parameters > {n_train:,} training samples — overfitting risk",
        ))

    # FM backbone modality mismatch → BLOCK (for future foundation model tier)
    if config.tier == "foundation":
        fm_modality = config.hyperparameters.get("backbone_modality")
        if fm_modality and fm_modality != profile.modality.value:
            alerts.append(GuardrailAlert(
                Severity.ERROR, "fm_modality_mismatch",
                f"{name}: foundation model backbone is for {fm_modality} but data is {profile.modality.value}",
            ))

    return alerts


def _estimate_model_params(
    config: ModelConfig,
    n_features: int,
    n_classes: int,
) -> int | None:
    """Rough estimate of learnable parameters for a model config."""
    model_type = config.model_type
    hp = config.hyperparameters

    if model_type in ("logistic_regression", "elastic_net_clf"):
        # weights + bias per class
        n_out = max(n_classes, 2)
        return n_features * n_out + n_out

    if model_type in ("ridge_regression", "elastic_net_reg"):
        return n_features + 1

    if model_type == "mlp":
        hidden_dims = hp.get("hidden_dims", [256, 128])
        if isinstance(hidden_dims, str):
            hidden_dims = [int(x) for x in hidden_dims.split(",")]
        total = 0
        prev_dim = n_features
        for dim in hidden_dims:
            total += prev_dim * dim + dim  # weights + bias
            total += dim * 2  # batch norm (gamma + beta)
            prev_dim = dim
        n_out = max(n_classes, 1)
        total += prev_dim * n_out + n_out
        return total

    # Tree ensembles (xgboost, lightgbm, random_forest): hard to estimate, skip
    return None


# ---------------------------------------------------------------------------
# Pre-training guardrails (dataset validation, Section 10.3)
# ---------------------------------------------------------------------------

def check_pre_training(
    profile: DatasetProfile,
    split: SplitData,
) -> list[GuardrailAlert]:
    """Validate data readiness before any model training."""
    alerts: list[GuardrailAlert] = []

    # Critical issues from profiler should block
    for issue in profile.detected_issues:
        if issue.startswith("CRITICAL"):
            alerts.append(GuardrailAlert(
                Severity.ERROR, "profiler_critical", issue,
            ))

    # Train set too small for meaningful modeling
    n_train = len(split.y_train)
    n_features = split.X_train.shape[1] if len(split.X_train.shape) > 1 else 0

    if n_train < 30:
        alerts.append(GuardrailAlert(
            Severity.ERROR, "tiny_train_set",
            f"Training set has only {n_train} samples — too small for reliable modeling",
        ))

    # More features than samples (risk of overfitting)
    if n_features > 0 and n_features > n_train:
        alerts.append(GuardrailAlert(
            Severity.WARNING, "high_dimensional",
            f"More features ({n_features}) than training samples ({n_train}) — risk of overfitting",
        ))

    # Check for NaN in features
    if np.any(np.isnan(split.X_train)):
        n_nan = int(np.isnan(split.X_train).sum())
        alerts.append(GuardrailAlert(
            Severity.WARNING, "nan_in_features",
            f"Training features contain {n_nan} NaN values after preprocessing",
        ))

    # Check for infinite values
    if np.any(np.isinf(split.X_train)):
        alerts.append(GuardrailAlert(
            Severity.ERROR, "inf_in_features",
            "Training features contain infinite values after preprocessing",
        ))

    # Target checks
    if np.std(split.y_train) < 1e-12:
        alerts.append(GuardrailAlert(
            Severity.ERROR, "constant_target",
            "Training target has zero variance — nothing to learn",
        ))

    # Feature matrix is all zeros
    if n_features > 0 and np.all(split.X_train == 0):
        alerts.append(GuardrailAlert(
            Severity.ERROR, "all_zeros_features",
            "All feature values are zero after preprocessing",
        ))

    return alerts


# ---------------------------------------------------------------------------
# Post-training guardrails (per model)
# ---------------------------------------------------------------------------

def check_post_training(
    result: ModelResult,
    trained: TrainedModel,
    split: SplitData,
    eval_config: EvalConfig,
    trivial_result: ModelResult | None = None,
) -> list[GuardrailAlert]:
    """Validate a single model's results after training + evaluation."""
    alerts: list[GuardrailAlert] = []
    name = result.model_name
    metric = eval_config.primary_metric
    value = result.primary_metric_value

    # 1. Perfect score → possible data leakage
    if metric in _HIGHER_IS_BETTER and value >= 1.0:
        alerts.append(GuardrailAlert(
            Severity.WARNING, "perfect_score",
            f"{name}: perfect {metric}=1.0 — possible data leakage",
        ))
    if metric in _LOWER_IS_BETTER and value <= 0.0:
        alerts.append(GuardrailAlert(
            Severity.WARNING, "perfect_score",
            f"{name}: perfect {metric}=0.0 — possible data leakage",
        ))

    # 2. Worse than trivial baseline
    if trivial_result is not None and result.tier != "trivial":
        trivial_value = trivial_result.primary_metric_value
        if _is_lower_better(metric):
            worse = value > trivial_value * 1.1  # 10% tolerance
        else:
            worse = value < trivial_value * 0.9  # 10% tolerance
        if worse:
            alerts.append(GuardrailAlert(
                Severity.WARNING, "worse_than_trivial",
                f"{name}: {metric}={value:.4f} is worse than trivial baseline ({trivial_value:.4f})",
            ))

    # 3. All predictions identical → model collapsed (skip for trivial baselines)
    seqs_val = split.seqs_val if trained.needs_sequences else None
    from co_scientist.modeling.types import _CONCAT_MODELS
    if trained.config.model_type in _CONCAT_MODELS and split.X_embed_val is not None:
        X_val_check = np.hstack([split.X_val, split.X_embed_val])
    elif trained.needs_embeddings and split.X_embed_val is not None:
        X_val_check = split.X_embed_val
    else:
        X_val_check = split.X_val
    y_pred = trained.predict(X_val_check, sequences=seqs_val)
    if np.std(y_pred) < 1e-12 and result.tier != "trivial":
        alerts.append(GuardrailAlert(
            Severity.ERROR, "model_collapsed",
            f"{name}: all predictions are identical ({y_pred[0]:.4f}) — model collapsed",
        ))

    # 4. Train-val gap → overfitting
    seqs_train = split.seqs_train if trained.needs_sequences else None
    if trained.config.model_type in _CONCAT_MODELS and split.X_embed_train is not None:
        X_train_check = np.hstack([split.X_train, split.X_embed_train])
    elif trained.needs_embeddings and split.X_embed_train is not None:
        X_train_check = split.X_embed_train
    else:
        X_train_check = split.X_train
    y_pred_train = trained.predict(X_train_check, sequences=seqs_train)
    train_metric = _compute_single_metric(split.y_train, y_pred_train, metric, trained)
    if train_metric is not None:
        gap = abs(train_metric - value)
        if gap > 0.3:
            alerts.append(GuardrailAlert(
                Severity.WARNING, "overfitting",
                f"{name}: train-val gap = {gap:.3f} (train={train_metric:.4f}, val={value:.4f}) — likely overfitting",
            ))
        elif gap > 0.15:
            alerts.append(GuardrailAlert(
                Severity.INFO, "moderate_overfitting",
                f"{name}: train-val gap = {gap:.3f} (train={train_metric:.4f}, val={value:.4f})",
            ))

    # 5. Negative R² → worse than predicting the mean
    if "r2" in result.metrics and result.metrics["r2"] < -0.5:
        alerts.append(GuardrailAlert(
            Severity.WARNING, "negative_r2",
            f"{name}: R²={result.metrics['r2']:.4f} — substantially worse than predicting the mean",
        ))

    return alerts


def _compute_single_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    trained: TrainedModel,
) -> float | None:
    """Compute a single metric value for train-val gap checking."""
    try:
        if metric in ("mse",):
            from sklearn.metrics import mean_squared_error
            return float(mean_squared_error(y_true, y_pred))
        if metric in ("rmse",):
            from sklearn.metrics import mean_squared_error
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        if metric in ("mae",):
            from sklearn.metrics import mean_absolute_error
            return float(mean_absolute_error(y_true, y_pred))
        if metric in ("r2",):
            from sklearn.metrics import r2_score
            return float(r2_score(y_true, y_pred))
        if metric in ("spearman",):
            from scipy.stats import spearmanr
            if np.std(y_pred) < 1e-12:
                return 0.0
            val = spearmanr(y_true, y_pred).correlation
            return float(val) if np.isfinite(val) else 0.0
        if metric in ("pearson",):
            from scipy.stats import pearsonr
            if np.std(y_pred) < 1e-12:
                return 0.0
            val = pearsonr(y_true, y_pred)[0]
            return float(val) if np.isfinite(val) else 0.0
        if metric in ("accuracy",):
            from sklearn.metrics import accuracy_score
            return float(accuracy_score(y_true, y_pred))
        if metric in ("macro_f1", "weighted_f1"):
            from sklearn.metrics import f1_score
            avg = "macro" if metric == "macro_f1" else "weighted"
            return float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        if metric in ("auroc",):
            # AUROC needs probabilities — skip for train gap check
            return None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Pipeline-level summary checks
# ---------------------------------------------------------------------------

def check_pipeline_summary(
    results: list[ModelResult],
    eval_config: EvalConfig,
) -> list[GuardrailAlert]:
    """Check cross-model patterns after all baselines are trained."""
    alerts: list[GuardrailAlert] = []

    if len(results) < 2:
        return alerts

    metric = eval_config.primary_metric
    values = [r.primary_metric_value for r in results]

    # All models have same performance → something is wrong
    if np.std(values) < 1e-6:
        alerts.append(GuardrailAlert(
            Severity.WARNING, "all_models_tied",
            f"All {len(results)} models have identical {metric} — data or evaluation issue",
        ))

    # No model beats trivial
    trivial_results = [r for r in results if r.tier == "trivial"]
    non_trivial = [r for r in results if r.tier != "trivial"]
    if trivial_results and non_trivial:
        trivial_best = trivial_results[0].primary_metric_value
        if _is_lower_better(metric):
            any_better = any(r.primary_metric_value < trivial_best for r in non_trivial)
        else:
            any_better = any(r.primary_metric_value > trivial_best for r in non_trivial)
        if not any_better:
            alerts.append(GuardrailAlert(
                Severity.WARNING, "no_improvement_over_trivial",
                f"No model improves over trivial baseline ({metric}={trivial_best:.4f}) — features may not be predictive",
            ))

    return alerts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_alerts(alerts: list[GuardrailAlert], title: str = "Guardrail Checks") -> None:
    """Print guardrail alerts with colored severity."""
    if not alerts:
        console.print(f"  [green]✓[/green] {title}: all checks passed")
        return

    errors = [a for a in alerts if a.severity == Severity.ERROR]
    warnings = [a for a in alerts if a.severity == Severity.WARNING]
    infos = [a for a in alerts if a.severity == Severity.INFO]

    console.print(f"  {title}: {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} info")

    for alert in alerts:
        if alert.severity == Severity.ERROR:
            console.print(f"    [bold red]ERROR[/bold red] {alert.message}")
        elif alert.severity == Severity.WARNING:
            console.print(f"    [yellow]WARNING[/yellow] {alert.message}")
        else:
            console.print(f"    [dim]INFO {alert.message}[/dim]")


def has_blocking_errors(alerts: list[GuardrailAlert]) -> bool:
    """Check if any alerts are blocking errors."""
    return any(a.severity == Severity.ERROR for a in alerts)
