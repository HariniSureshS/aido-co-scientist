"""Metric computation for classification and regression."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table

from co_scientist.data.types import SplitData
from co_scientist.modeling.types import TrainedModel

from .types import EvalConfig, ModelResult

console = Console()


def evaluate_model(
    trained: TrainedModel,
    split: SplitData,
    eval_config: EvalConfig,
    use_test: bool = False,
) -> ModelResult:
    """Evaluate a trained model on val (default) or test set."""
    y_true = split.y_test if use_test else split.y_val
    seqs = split.seqs_test if use_test else split.seqs_val

    # Route features based on model type
    from co_scientist.modeling.types import _CONCAT_MODELS
    if trained.config.model_type in _CONCAT_MODELS:
        # Concat models use handcrafted + embeddings
        X_base = split.X_test if use_test else split.X_val
        X_embed = split.X_embed_test if use_test else split.X_embed_val
        X = np.hstack([X_base, X_embed]) if X_embed is not None else X_base
    elif trained.needs_embeddings:
        X = split.X_embed_test if use_test else split.X_embed_val
    else:
        X = split.X_test if use_test else split.X_val

    y_pred = trained.predict(X, sequences=seqs)
    y_proba = trained.predict_proba(X, sequences=seqs)

    if eval_config.task_type in ("binary_classification", "multiclass_classification"):
        metrics = _classification_metrics(y_true, y_pred, y_proba, eval_config)
    elif eval_config.task_type == "regression":
        metrics = _regression_metrics(y_true, y_pred, eval_config)
    else:
        metrics = {}

    primary_value = metrics.get(eval_config.primary_metric, 0.0)

    return ModelResult(
        model_name=trained.config.name,
        tier=trained.config.tier,
        metrics=metrics,
        primary_metric_name=eval_config.primary_metric,
        primary_metric_value=primary_value,
        train_time_seconds=trained.train_time_seconds,
    )


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    eval_config: EvalConfig,
) -> dict[str, float]:
    """Compute all classification metrics comprehensively."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {}

    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))

    # Core metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # F1 / precision / recall variants
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    if n_classes == 2:
        metrics["f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="binary", zero_division=0))

    # Probability-based metrics
    if y_proba is not None:
        try:
            if n_classes == 2:
                metrics["auroc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auroc"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro",
                ))
        except (ValueError, IndexError):
            pass

        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except (ValueError, IndexError):
            pass

    return metrics


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_config: EvalConfig,
) -> dict[str, float]:
    """Compute all regression metrics comprehensively."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    metrics: dict[str, float] = {}

    # Core error metrics
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["median_ae"] = float(median_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))
    metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))

    # MAPE — guard against zero targets
    try:
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        pass

    # Correlation metrics — require non-constant predictions
    if len(y_true) > 2 and np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        sp = spearmanr(y_true, y_pred).correlation
        pe = pearsonr(y_true, y_pred)[0]
        metrics["spearman"] = float(sp) if np.isfinite(sp) else 0.0
        metrics["pearson"] = float(pe) if np.isfinite(pe) else 0.0
    else:
        metrics["spearman"] = 0.0
        metrics["pearson"] = 0.0

    return metrics


def print_results_table(results: list[ModelResult], eval_config: EvalConfig) -> None:
    """Print a comparison table of all model results."""
    table = Table(title="Model Comparison", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Tier", style="dim")
    table.add_column(eval_config.primary_metric, style="bold green", justify="right")

    # Add secondary metric columns
    for metric in eval_config.secondary_metrics[:3]:
        table.add_column(metric, justify="right")

    table.add_column("Time", justify="right", style="dim")

    # Sort by primary metric (descending for most metrics, ascending for MSE/RMSE/MAE)
    lower_is_better = eval_config.primary_metric in ("mse", "rmse", "mae")
    sorted_results = sorted(results, key=lambda r: r.primary_metric_value, reverse=not lower_is_better)

    for r in sorted_results:
        row = [
            r.model_name,
            r.tier,
            f"{r.primary_metric_value:.4f}",
        ]
        for metric in eval_config.secondary_metrics[:3]:
            val = r.metrics.get(metric)
            row.append(f"{val:.4f}" if val is not None else "—")
        row.append(f"{r.train_time_seconds:.1f}s")
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()
