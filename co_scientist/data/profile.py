"""Dataset profiler — produces a DatasetProfile without any LLM involvement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .types import DatasetProfile, LoadedDataset, Modality, TaskType

console = Console()


def profile_dataset(dataset: LoadedDataset, dataset_path: str) -> DatasetProfile:
    """Analyze a loaded dataset and produce a structured profile."""
    profile = DatasetProfile(
        dataset_name=dataset.info.name,
        dataset_path=dataset_path,
        has_predefined_splits=dataset.info.has_predefined_splits,
    )

    # Parse task hint from path
    parts = dataset_path.strip("/").split("/")
    if len(parts) >= 2:
        profile.task_hint = parts[-1]

    # Detect modality
    profile.modality = _detect_modality(dataset, dataset_path)

    # Detect input type
    profile.input_type = _infer_input_type(profile.modality, dataset)

    # Basic size stats
    profile.num_samples = len(dataset.y)
    profile.input_columns = list(dataset.X.columns) if isinstance(dataset.X, pd.DataFrame) else []
    profile.num_features = dataset.X.shape[1] if len(dataset.X.shape) > 1 else 0

    # Target analysis
    profile.target_column = dataset.y.name if hasattr(dataset.y, "name") and dataset.y.name else "target"
    profile.task_type = _detect_task_type(dataset, profile)
    _analyze_target(dataset, profile)

    # Feature quality
    _analyze_features(dataset, profile)

    # Sequence stats (if applicable)
    if profile.modality in (Modality.RNA, Modality.DNA, Modality.PROTEIN):
        _analyze_sequences(dataset, profile)

    # Split info
    if dataset.info.has_predefined_splits:
        _analyze_splits(dataset, profile)

    # Detect issues
    _detect_issues(profile)

    return profile


# ---------------------------------------------------------------------------
# Modality detection cascade: path → column names → content → dimensionality
# ---------------------------------------------------------------------------

def _detect_modality(dataset: LoadedDataset, dataset_path: str) -> Modality:
    """Detect data modality using a cascade of signals."""
    path_lower = dataset_path.lower()

    # 1. Path parsing
    if any(k in path_lower for k in ["rna/", "rna_", "rna-"]):
        return Modality.RNA
    if any(k in path_lower for k in ["dna/", "dna_", "dna-", "promoter"]):
        return Modality.DNA
    if any(k in path_lower for k in ["protein/", "protein_", "protein-"]):
        return Modality.PROTEIN
    if any(k in path_lower for k in ["expression/", "cell_type", "scrna", "cell-downstream"]):
        return Modality.CELL_EXPRESSION
    if any(k in path_lower for k in ["spatial/", "tissue/"]):
        return Modality.SPATIAL

    # 2. Column names
    if isinstance(dataset.X, pd.DataFrame):
        cols_lower = {c.lower() for c in dataset.X.columns}
        if "sequences" in cols_lower or "sequence" in cols_lower or "seq" in cols_lower:
            # Sequence data — further disambiguate
            return _disambiguate_sequence_type(dataset)
        if any("gene" in c for c in cols_lower) or len(dataset.X.columns) > 500:
            return Modality.CELL_EXPRESSION

    # 3. Content inspection: check if values look like sequences
    if isinstance(dataset.X, pd.DataFrame) and len(dataset.X.columns) == 1:
        sample = dataset.X.iloc[:5, 0]
        if sample.dtype == object and all(isinstance(v, str) for v in sample):
            return _classify_sequence_content(sample)

    # 4. Source format hint
    if dataset.info.source_format == "h5ad":
        return Modality.CELL_EXPRESSION

    # 5. Dimensionality patterns
    if isinstance(dataset.X, pd.DataFrame):
        n_features = dataset.X.shape[1]
        if n_features > 1000:
            return Modality.CELL_EXPRESSION
        if n_features < 50:
            return Modality.TABULAR

    return Modality.UNKNOWN


def _disambiguate_sequence_type(dataset: LoadedDataset) -> Modality:
    """Given that we have sequence data, determine if it's DNA, RNA, or protein."""
    if isinstance(dataset.X, pd.DataFrame):
        seq_col = None
        for c in dataset.X.columns:
            if c.lower() in ("sequences", "sequence", "seq"):
                seq_col = c
                break
        if seq_col:
            sample = dataset.X[seq_col].iloc[:10]
            return _classify_sequence_content(sample)
    return Modality.RNA  # default for genbio


def _classify_sequence_content(sample: pd.Series) -> Modality:
    """Classify sequence content as DNA, RNA, or protein based on character frequency."""
    all_chars = set("".join(str(v).upper() for v in sample))
    # DNA: A, T, G, C (+ N)
    if all_chars <= {"A", "T", "G", "C", "N", "U"}:
        if "U" in all_chars:
            return Modality.RNA
        return Modality.DNA
    # RNA: A, U, G, C
    if all_chars <= {"A", "U", "G", "C", "N"}:
        return Modality.RNA
    # Protein: 20 standard amino acids
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    if all_chars <= amino_acids | {"X", "B", "Z", "J", "O", "U"}:
        return Modality.PROTEIN
    return Modality.UNKNOWN


def _infer_input_type(modality: Modality, dataset: LoadedDataset) -> str:
    """Map modality to input type string."""
    if modality in (Modality.RNA, Modality.DNA, Modality.PROTEIN):
        return "sequence"
    if modality == Modality.CELL_EXPRESSION:
        return "expression_matrix"
    if modality == Modality.SPATIAL:
        return "spatial_expression"
    return "tabular"


# ---------------------------------------------------------------------------
# Task type detection
# ---------------------------------------------------------------------------

def _detect_task_type(dataset: LoadedDataset, profile: DatasetProfile) -> TaskType:
    """Detect whether this is classification or regression."""
    y = dataset.y

    # Path hints
    hint = profile.task_hint.lower()
    if "classification" in hint or "cell_type" in hint:
        n_classes = y.nunique() if hasattr(y, "nunique") else len(set(y))
        return TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
    if any(k in hint for k in ["regression", "efficiency", "expression", "abundance", "ribosome"]):
        return TaskType.REGRESSION

    # Data-driven detection
    if hasattr(y, "dtype"):
        if y.dtype == object or y.dtype.name == "category":
            n_classes = y.nunique()
            return TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
        if np.issubdtype(y.dtype, np.floating):
            n_unique = y.nunique() if hasattr(y, "nunique") else len(set(y))
            # If fewer than 20 unique float values, likely classification with numeric labels
            if n_unique <= 20:
                return TaskType.BINARY_CLASSIFICATION if n_unique == 2 else TaskType.MULTICLASS_CLASSIFICATION
            return TaskType.REGRESSION
        if np.issubdtype(y.dtype, np.integer):
            n_unique = y.nunique() if hasattr(y, "nunique") else len(set(y))
            if n_unique <= 50:
                return TaskType.BINARY_CLASSIFICATION if n_unique == 2 else TaskType.MULTICLASS_CLASSIFICATION
            return TaskType.REGRESSION

    return TaskType.UNKNOWN


# ---------------------------------------------------------------------------
# Target analysis
# ---------------------------------------------------------------------------

def _analyze_target(dataset: LoadedDataset, profile: DatasetProfile) -> None:
    """Populate target-related fields in the profile."""
    y = pd.Series(dataset.y) if not isinstance(dataset.y, pd.Series) else dataset.y

    if profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        vc = y.value_counts()
        profile.num_classes = len(vc)
        profile.class_distribution = {str(k): int(v) for k, v in vc.items()}
    elif profile.task_type == TaskType.REGRESSION:
        y_numeric = pd.to_numeric(y, errors="coerce")
        profile.target_stats = {
            "mean": float(y_numeric.mean()),
            "std": float(y_numeric.std()),
            "min": float(y_numeric.min()),
            "max": float(y_numeric.max()),
            "median": float(y_numeric.median()),
        }


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

def _analyze_features(dataset: LoadedDataset, profile: DatasetProfile) -> None:
    """Analyze feature quality: missing values, sparsity."""
    X = dataset.X
    if isinstance(X, pd.DataFrame):
        # Only analyze numeric columns for sparsity/missing
        numeric_df = X.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            total_cells = numeric_df.size
            profile.missing_value_pct = float(numeric_df.isna().sum().sum() / total_cells * 100) if total_cells > 0 else 0.0
            profile.feature_sparsity = float((numeric_df == 0).sum().sum() / total_cells * 100) if total_cells > 0 else 0.0
        else:
            # Non-numeric (e.g. sequences) — check for nulls
            profile.missing_value_pct = float(X.isna().any(axis=1).sum() / len(X) * 100)
    elif isinstance(X, np.ndarray):
        total_cells = X.size
        if total_cells > 0:
            profile.missing_value_pct = float(np.isnan(X).sum() / total_cells * 100) if np.issubdtype(X.dtype, np.floating) else 0.0
            profile.feature_sparsity = float((X == 0).sum() / total_cells * 100)


def _analyze_sequences(dataset: LoadedDataset, profile: DatasetProfile) -> None:
    """Compute sequence length statistics."""
    X = dataset.X
    if isinstance(X, pd.DataFrame):
        # Find the sequence column
        seq_col = None
        for c in X.columns:
            if c.lower() in ("sequences", "sequence", "seq"):
                seq_col = c
                break
        if seq_col is None and len(X.columns) == 1:
            seq_col = X.columns[0]
        if seq_col and X[seq_col].dtype == object:
            lengths = X[seq_col].str.len()
            profile.sequence_length_stats = {
                "mean": float(lengths.mean()),
                "std": float(lengths.std()),
                "min": float(lengths.min()),
                "max": float(lengths.max()),
                "median": float(lengths.median()),
            }


# ---------------------------------------------------------------------------
# Split analysis
# ---------------------------------------------------------------------------

def _analyze_splits(dataset: LoadedDataset, profile: DatasetProfile) -> None:
    """Record info about predefined splits."""
    if dataset.fold_ids is not None:
        unique_folds = np.unique(dataset.fold_ids)
        profile.split_info = {
            "type": "cv_folds",
            "num_folds": len(unique_folds),
            "fold_sizes": {str(f): int((dataset.fold_ids == f).sum()) for f in unique_folds},
        }
    elif dataset.raw_data is not None:
        if isinstance(dataset.raw_data, pd.DataFrame) and "_split" in dataset.raw_data.columns:
            vc = dataset.raw_data["_split"].value_counts()
            profile.split_info = {
                "type": "predefined",
                "splits": {str(k): int(v) for k, v in vc.items()},
            }
        elif isinstance(dataset.raw_data, dict) and "splits" in dataset.raw_data:
            splits_arr = dataset.raw_data["splits"]
            unique, counts = np.unique(splits_arr, return_counts=True)
            profile.split_info = {
                "type": "predefined",
                "splits": {str(u): int(c) for u, c in zip(unique, counts)},
            }


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------

def _detect_issues(profile: DatasetProfile) -> None:
    """Run validation checks and flag issues."""
    issues = []

    if profile.num_samples == 0:
        issues.append("CRITICAL: Dataset has 0 samples")

    if profile.num_samples < 50:
        issues.append(f"WARNING: Very small dataset ({profile.num_samples} samples)")

    if profile.missing_value_pct > 10:
        issues.append(f"WARNING: High missing value rate ({profile.missing_value_pct:.1f}%)")

    if profile.feature_sparsity > 90:
        issues.append(f"INFO: Very sparse features ({profile.feature_sparsity:.1f}% zeros)")

    if profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        if profile.class_distribution:
            counts = list(profile.class_distribution.values())
            total = sum(counts)
            min_pct = min(counts) / total * 100 if total > 0 else 0
            max_pct = max(counts) / total * 100 if total > 0 else 0
            if min_pct < 5:
                issues.append(f"WARNING: Severe class imbalance (smallest class: {min_pct:.1f}%)")
            elif min_pct < 10:
                issues.append(f"INFO: Moderate class imbalance (smallest class: {min_pct:.1f}%)")

            if profile.num_classes > 50:
                issues.append(f"INFO: Large number of classes ({profile.num_classes})")

    if profile.task_type == TaskType.REGRESSION and profile.target_stats:
        std = profile.target_stats.get("std", 0)
        if std == 0:
            issues.append("CRITICAL: Target has zero variance")

    if profile.task_type == TaskType.UNKNOWN:
        issues.append("WARNING: Could not auto-detect task type")

    if profile.modality == Modality.UNKNOWN:
        issues.append("WARNING: Could not auto-detect data modality")

    profile.detected_issues = issues


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fallback profiling — when standard profiling fails
# ---------------------------------------------------------------------------

def fallback_profile(dataset: LoadedDataset, dataset_path: str) -> DatasetProfile:
    """Minimal profiling that shouldn't fail — used when standard profiling raises."""
    profile = DatasetProfile(
        dataset_name=dataset.info.name,
        dataset_path=dataset_path,
        has_predefined_splits=dataset.info.has_predefined_splits,
    )

    # Safe modality detection from path only
    path_lower = dataset_path.lower()
    if any(k in path_lower for k in ["rna", "dna", "splice", "ncrna", "ribosome"]):
        profile.modality = Modality.RNA
    elif any(k in path_lower for k in ["protein", "fitness", "dms"]):
        profile.modality = Modality.PROTEIN
    elif any(k in path_lower for k in ["expression", "cell_type", "scrna", "h5ad"]):
        profile.modality = Modality.CELL_EXPRESSION
    elif dataset.info.source_format == "h5ad":
        profile.modality = Modality.CELL_EXPRESSION
    else:
        profile.modality = Modality.TABULAR

    # Safe size stats
    try:
        profile.num_samples = len(dataset.y)
        profile.num_features = dataset.X.shape[1] if len(dataset.X.shape) > 1 else 0
        profile.input_columns = list(dataset.X.columns) if isinstance(dataset.X, pd.DataFrame) else []
    except Exception:
        profile.num_samples = 0
        profile.num_features = 0

    # Safe task type detection
    try:
        y = np.array(dataset.y)
        if y.dtype == object or str(y.dtype) == "category":
            n_unique = len(set(y))
            profile.task_type = TaskType.BINARY_CLASSIFICATION if n_unique <= 2 else TaskType.MULTICLASS_CLASSIFICATION
            profile.num_classes = n_unique
        elif np.issubdtype(y.dtype, np.integer):
            n_unique = len(set(y))
            profile.task_type = TaskType.BINARY_CLASSIFICATION if n_unique <= 2 else (
                TaskType.MULTICLASS_CLASSIFICATION if n_unique <= 50 else TaskType.REGRESSION
            )
            if profile.task_type != TaskType.REGRESSION:
                profile.num_classes = n_unique
        else:
            profile.task_type = TaskType.REGRESSION
    except Exception:
        profile.task_type = TaskType.REGRESSION  # safe default

    profile.target_column = dataset.y.name if hasattr(dataset.y, "name") and dataset.y.name else "target"
    console.print(f"  [yellow]Used fallback profiling: {profile.modality.value} / {profile.task_type.value}[/yellow]")
    return profile


# Pretty printing
# ---------------------------------------------------------------------------

def print_profile(profile: DatasetProfile) -> None:
    """Print a rich-formatted summary of the dataset profile."""
    console.print()
    console.print(f"[bold]Dataset Profile: {profile.dataset_name}[/bold]")
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Path", profile.dataset_path)
    table.add_row("Modality", profile.modality.value)
    table.add_row("Task type", profile.task_type.value)
    table.add_row("Input type", profile.input_type)
    table.add_row("Samples", f"{profile.num_samples:,}")
    table.add_row("Features", f"{profile.num_features:,}" if profile.num_features > 0 else "N/A (sequence)")
    table.add_row("Target column", profile.target_column)

    if profile.num_classes > 0:
        table.add_row("Classes", str(profile.num_classes))
    if profile.target_stats:
        stats = profile.target_stats
        table.add_row("Target range", f"[{stats['min']:.4f}, {stats['max']:.4f}]")
        table.add_row("Target mean/std", f"{stats['mean']:.4f} / {stats['std']:.4f}")

    table.add_row("Missing values", f"{profile.missing_value_pct:.2f}%")
    if profile.feature_sparsity > 0:
        table.add_row("Sparsity", f"{profile.feature_sparsity:.1f}% zeros")

    if profile.sequence_length_stats:
        sl = profile.sequence_length_stats
        table.add_row("Seq length", f"{sl['min']:.0f}–{sl['max']:.0f} (mean {sl['mean']:.1f})")

    if profile.has_predefined_splits:
        table.add_row("Splits", str(profile.split_info))

    console.print(table)

    if profile.detected_issues:
        console.print()
        for issue in profile.detected_issues:
            if issue.startswith("CRITICAL"):
                console.print(f"  [bold red]{issue}[/bold red]")
            elif issue.startswith("WARNING"):
                console.print(f"  [yellow]{issue}[/yellow]")
            else:
                console.print(f"  [dim]{issue}[/dim]")
    console.print()
