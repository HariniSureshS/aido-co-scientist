"""Preprocessing-stage visualizations (02_preprocessing/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console

from co_scientist.data.types import DatasetProfile, SplitData, TaskType

console = Console()


def generate_preprocessing_figures(
    split: SplitData,
    profile: DatasetProfile,
    output_dir: Path,
) -> list[Path]:
    """Generate preprocessing visualizations."""
    fig_dir = output_dir / "figures" / "02_preprocessing"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Feature variance distribution (post-preprocessing)
    saved.append(_plot_feature_variance(split, profile, fig_dir))

    # 2. Split distribution verification
    saved.append(_plot_split_distribution(split, profile, fig_dir))

    return [p for p in saved if p is not None]


def _plot_feature_variance(split: SplitData, profile: DatasetProfile, fig_dir: Path) -> Path:
    """Plot per-feature variance on the training set."""
    variances = np.var(split.X_train, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(variances, bins=50, color=sns.color_palette("viridis")[1], edgecolor="white", alpha=0.8)
    ax.set_xlabel("Variance")
    ax.set_ylabel("Number of features")
    ax.set_title(f"Feature Variance Distribution (train) — {profile.dataset_name}")
    ax.axvline(np.mean(variances), color="red", linestyle="--", label=f"mean={np.mean(variances):.3f}")

    # Note how many near-zero variance features
    low_var = np.sum(variances < 0.01)
    if low_var > 0:
        ax.text(0.95, 0.95, f"{low_var} features with var < 0.01",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()
    plt.tight_layout()

    path = fig_dir / "feature_variance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_split_distribution(split: SplitData, profile: DatasetProfile, fig_dir: Path) -> Path:
    """Verify that class distributions are consistent across splits."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        # Class proportions per split
        splits = {"Train": split.y_train, "Val": split.y_val, "Test": split.y_test}
        all_classes = sorted(set(np.concatenate([split.y_train, split.y_val, split.y_test])))
        n_classes = len(all_classes)

        x = np.arange(n_classes)
        width = 0.25
        colors = sns.color_palette("viridis", 3)

        for i, (name, y) in enumerate(splits.items()):
            counts = np.array([np.sum(y == c) for c in all_classes])
            proportions = counts / len(y) * 100
            ax.bar(x + i * width, proportions, width, label=f"{name} (n={len(y)})", color=colors[i])

        # Use label names if encoder available
        if split.label_encoder is not None:
            labels = [split.label_encoder.inverse_transform([c])[0] for c in all_classes]
        else:
            labels = [str(c) for c in all_classes]

        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Proportion (%)")
        ax.set_title(f"Class Distribution per Split — {profile.dataset_name}")
        ax.legend()
    else:
        # Regression: overlapping histograms per split
        splits = {"Train": split.y_train, "Val": split.y_val, "Test": split.y_test}
        colors = sns.color_palette("viridis", 3)

        for (name, y), color in zip(splits.items(), colors):
            ax.hist(y, bins=30, alpha=0.5, label=f"{name} (n={len(y)})", color=color, edgecolor="white")

        ax.set_xlabel("Target value")
        ax.set_ylabel("Count")
        ax.set_title(f"Target Distribution per Split — {profile.dataset_name}")
        ax.legend()

    plt.tight_layout()
    path = fig_dir / "split_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
