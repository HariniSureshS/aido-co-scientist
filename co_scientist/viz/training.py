"""Training-stage visualizations (03_training/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console

from co_scientist.data.types import DatasetProfile, SplitData
from co_scientist.evaluation.types import EvalConfig, ModelResult
from co_scientist.modeling.types import TrainedModel

console = Console()


def generate_training_figures(
    results: list[ModelResult],
    trained_models: list[TrainedModel],
    split: SplitData,
    eval_config: EvalConfig,
    profile: DatasetProfile,
    output_dir: Path,
) -> list[Path]:
    """Generate training-stage visualizations."""
    fig_dir = output_dir / "figures" / "03_training"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Model comparison bar chart
    saved.append(_plot_model_comparison(results, eval_config, profile, fig_dir))

    # 2. Feature importance (for the best tree-based model)
    best_idx = _best_model_index(results, eval_config)
    if best_idx is not None:
        path = _plot_feature_importance(trained_models[best_idx], split, profile, fig_dir)
        if path:
            saved.append(path)

    return [p for p in saved if p is not None]


def _best_model_index(results: list[ModelResult], eval_config: EvalConfig) -> int | None:
    """Find index of the best model by primary metric."""
    if not results:
        return None
    lower_is_better = eval_config.primary_metric in ("mse", "rmse", "mae")
    return sorted(
        range(len(results)),
        key=lambda i: results[i].primary_metric_value,
        reverse=not lower_is_better,
    )[0]


def _plot_model_comparison(
    results: list[ModelResult],
    eval_config: EvalConfig,
    profile: DatasetProfile,
    fig_dir: Path,
) -> Path:
    """Bar chart comparing all models on primary metric."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by primary metric
    lower_is_better = eval_config.primary_metric in ("mse", "rmse", "mae")
    sorted_results = sorted(results, key=lambda r: r.primary_metric_value, reverse=not lower_is_better)

    names = [r.model_name for r in sorted_results]
    values = [r.primary_metric_value for r in sorted_results]
    tiers = [r.tier for r in sorted_results]

    # Color by tier
    tier_colors = {"trivial": "#95a5a6", "simple": "#3498db", "standard": "#2ecc71", "advanced": "#e74c3c", "tuned": "#9b59b6", "foundation": "#e67e22", "ensemble": "#1abc9c"}
    colors = [tier_colors.get(t, "#9b59b6") for t in tiers]

    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel(eval_config.primary_metric)
    ax.set_title(f"Model Comparison — {profile.dataset_name}")
    ax.invert_yaxis()

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    # Legend for tiers
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in tier_colors.items() if t in tiers]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    path = fig_dir / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_feature_importance(
    trained: TrainedModel, split: SplitData, profile: DatasetProfile, fig_dir: Path,
) -> Path | None:
    """Plot top feature importances for tree-based models."""
    model = trained.model

    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return None

    feature_names = split.feature_names or [f"feat_{i}" for i in range(len(importances))]

    # Top 20
    n_top = min(20, len(importances))
    top_idx = np.argsort(importances)[-n_top:]
    top_names = [feature_names[i] for i in top_idx]
    top_values = importances[top_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("viridis", n_top)
    ax.barh(range(n_top), top_values, color=colors)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {n_top} Feature Importances — {trained.config.name}")
    plt.tight_layout()

    path = fig_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
