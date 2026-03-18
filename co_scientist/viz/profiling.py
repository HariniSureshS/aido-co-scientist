"""Profiling-stage visualizations (01_profiling/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console

from co_scientist.data.types import DatasetProfile, LoadedDataset, Modality, TaskType

console = Console()


def generate_profiling_figures(
    dataset: LoadedDataset,
    profile: DatasetProfile,
    output_dir: Path,
) -> list[Path]:
    """Generate profiling visualizations. Returns list of saved figure paths."""
    fig_dir = output_dir / "figures" / "01_profiling"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Target distribution
    saved.append(_plot_target_distribution(dataset, profile, fig_dir))

    # 2. Sequence length distribution (if applicable)
    if profile.modality in (Modality.RNA, Modality.DNA, Modality.PROTEIN):
        path = _plot_sequence_lengths(dataset, profile, fig_dir)
        if path:
            saved.append(path)

    # 3. Feature sparsity overview (if expression data)
    if profile.modality == Modality.CELL_EXPRESSION:
        saved.append(_plot_expression_overview(dataset, profile, fig_dir))

    return [p for p in saved if p is not None]


def _plot_target_distribution(
    dataset: LoadedDataset, profile: DatasetProfile, fig_dir: Path,
) -> Path:
    """Plot target variable distribution — bar chart for classification, histogram for regression."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        # Class distribution bar chart
        classes = list(profile.class_distribution.keys())
        counts = list(profile.class_distribution.values())
        # Sort by count descending
        sorted_pairs = sorted(zip(counts, classes), reverse=True)
        counts, classes = zip(*sorted_pairs)

        colors = sns.color_palette("viridis", len(classes))
        bars = ax.barh(range(len(classes)), counts, color=colors)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Sample count")
        ax.set_title(f"Class Distribution — {profile.dataset_name}")
        ax.invert_yaxis()

        # Annotate counts
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=8)

    else:
        # Regression histogram
        y = np.array(dataset.y, dtype=float)
        ax.hist(y, bins=50, color=sns.color_palette("viridis")[0], edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(y), color="red", linestyle="--", label=f"mean={np.mean(y):.2f}")
        ax.axvline(np.median(y), color="orange", linestyle="--", label=f"median={np.median(y):.2f}")
        ax.set_xlabel("Target value")
        ax.set_ylabel("Count")
        ax.set_title(f"Target Distribution — {profile.dataset_name}")
        ax.legend()

    plt.tight_layout()
    path = fig_dir / "target_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_sequence_lengths(
    dataset: LoadedDataset, profile: DatasetProfile, fig_dir: Path,
) -> Path | None:
    """Plot sequence length distribution."""
    import pandas as pd

    if not isinstance(dataset.X, pd.DataFrame):
        return None

    seq_col = None
    for c in dataset.X.columns:
        if c.lower() in ("sequences", "sequence", "seq"):
            seq_col = c
            break
    if seq_col is None:
        return None

    lengths = dataset.X[seq_col].str.len()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=50, color=sns.color_palette("viridis")[2], edgecolor="white", alpha=0.8)
    ax.axvline(lengths.mean(), color="red", linestyle="--", label=f"mean={lengths.mean():.1f}")
    ax.set_xlabel("Sequence length (nt)")
    ax.set_ylabel("Count")
    ax.set_title(f"Sequence Length Distribution — {profile.dataset_name}")
    ax.legend()
    plt.tight_layout()

    path = fig_dir / "sequence_lengths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_expression_overview(
    dataset: LoadedDataset, profile: DatasetProfile, fig_dir: Path,
) -> Path:
    """Plot expression matrix overview: genes-per-cell and sparsity."""
    import pandas as pd

    X = dataset.X.values if isinstance(dataset.X, pd.DataFrame) else np.array(dataset.X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Genes detected per cell
    genes_per_cell = np.sum(X > 0, axis=1)
    axes[0].hist(genes_per_cell, bins=50, color=sns.color_palette("viridis")[0], edgecolor="white")
    axes[0].set_xlabel("Genes detected per cell")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title("Genes per cell")
    axes[0].axvline(np.mean(genes_per_cell), color="red", linestyle="--",
                    label=f"mean={np.mean(genes_per_cell):.0f}")
    axes[0].legend()

    # Total counts per cell
    counts_per_cell = np.sum(X, axis=1)
    axes[1].hist(counts_per_cell, bins=50, color=sns.color_palette("viridis")[3], edgecolor="white")
    axes[1].set_xlabel("Total counts per cell")
    axes[1].set_ylabel("Number of cells")
    axes[1].set_title("Total counts per cell")
    axes[1].axvline(np.mean(counts_per_cell), color="red", linestyle="--",
                    label=f"mean={np.mean(counts_per_cell):.0f}")
    axes[1].legend()

    plt.suptitle(f"Expression Overview — {profile.dataset_name}", fontsize=12)
    plt.tight_layout()

    path = fig_dir / "expression_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
