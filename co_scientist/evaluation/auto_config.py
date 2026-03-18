"""Auto-configure evaluation settings from the dataset profile."""

from __future__ import annotations

from co_scientist.data.types import DatasetProfile, TaskType
from co_scientist.defaults import get_evaluation_defaults

from .types import EvalConfig


def auto_eval_config(profile: DatasetProfile) -> EvalConfig:
    """Determine primary/secondary metrics from the dataset profile.

    Metric selection logic (from architecture Section 8.1):
      - Binary classification → AUROC
      - Multi-class balanced → accuracy
      - Multi-class imbalanced (any class < threshold%) → macro F1
      - Regression → Spearman correlation
    """
    eval_cfg = get_evaluation_defaults(profile.modality.value, profile.dataset_path)
    metrics_cfg = eval_cfg.get("metrics", {})
    imbalance_threshold = eval_cfg.get("imbalance_threshold_pct", 5)

    if profile.task_type == TaskType.BINARY_CLASSIFICATION:
        m = metrics_cfg.get("binary_classification", {})
        return EvalConfig(
            task_type="binary_classification",
            primary_metric=m.get("primary", "auroc"),
            secondary_metrics=m.get("secondary", ["accuracy", "f1", "precision", "recall"]),
        )

    if profile.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        # Check for imbalance
        is_imbalanced = False
        if profile.class_distribution:
            total = sum(profile.class_distribution.values())
            min_pct = min(profile.class_distribution.values()) / total * 100 if total > 0 else 0
            if min_pct < imbalance_threshold:
                is_imbalanced = True

        if is_imbalanced:
            m = metrics_cfg.get("multiclass_imbalanced", {})
            return EvalConfig(
                task_type="multiclass_classification",
                primary_metric=m.get("primary", "macro_f1"),
                secondary_metrics=m.get("secondary", ["accuracy", "weighted_f1", "macro_precision", "macro_recall"]),
            )
        else:
            m = metrics_cfg.get("multiclass_balanced", {})
            return EvalConfig(
                task_type="multiclass_classification",
                primary_metric=m.get("primary", "accuracy"),
                secondary_metrics=m.get("secondary", ["macro_f1", "weighted_f1", "macro_precision", "macro_recall"]),
            )

    if profile.task_type == TaskType.REGRESSION:
        m = metrics_cfg.get("regression", {})
        return EvalConfig(
            task_type="regression",
            primary_metric=m.get("primary", "spearman"),
            secondary_metrics=m.get("secondary", ["pearson", "mse", "rmse", "mae", "r2"]),
        )

    # Fallback
    return EvalConfig(
        task_type="unknown",
        primary_metric="accuracy",
        secondary_metrics=[],
    )
