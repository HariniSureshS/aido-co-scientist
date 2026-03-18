"""Train/val/test splitting strategies."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from sklearn.model_selection import train_test_split

from co_scientist.defaults import get_splitting_defaults

from .types import DatasetProfile, LoadedDataset, PreprocessingResult, SplitData, TaskType

console = Console()


def split_dataset(
    dataset: LoadedDataset,
    preprocessed: PreprocessingResult,
    profile: DatasetProfile,
    seed: int = 42,
) -> SplitData:
    """Split into train/val/test. Uses predefined splits when available, else 70/15/15."""

    # Strategy 1: Predefined train/valid/test splits (e.g. h5ad datasets)
    if _has_predefined_tvt(dataset):
        split = _split_predefined(dataset, preprocessed, profile, seed=seed)
    # Strategy 2: Fold-based CV → use one fold as test, one as val, rest as train
    elif dataset.fold_ids is not None:
        split = _split_from_folds(dataset, preprocessed, profile, seed)
    # Strategy 3: Random split 70/15/15
    else:
        split = _split_random(preprocessed, profile, seed)

    return split


def _has_predefined_tvt(dataset: LoadedDataset) -> bool:
    """Check if the dataset has predefined train/valid/test splits."""
    if dataset.raw_data is None:
        return False
    if isinstance(dataset.raw_data, dict) and "splits" in dataset.raw_data:
        splits = set(dataset.raw_data["splits"])
        return "train" in splits and ("valid" in splits or "test" in splits)
    if hasattr(dataset.raw_data, "columns") and "_split" in dataset.raw_data.columns:
        splits = set(dataset.raw_data["_split"].unique())
        return "train" in splits and ("valid" in splits or "test" in splits)
    return False


def _split_predefined(
    dataset: LoadedDataset,
    preprocessed: PreprocessingResult,
    profile: DatasetProfile,
    seed: int = 42,
) -> SplitData:
    """Use predefined train/valid/test assignments."""
    split_cfg = get_splitting_defaults(profile.modality.value, profile.dataset_path)
    fallback_val = split_cfg.get("fallback_val_fraction", 0.2)

    if isinstance(dataset.raw_data, dict) and "splits" in dataset.raw_data:
        split_labels = np.array(dataset.raw_data["splits"])
    else:
        split_labels = dataset.raw_data["_split"].values

    X, y = preprocessed.X, preprocessed.y

    train_mask = split_labels == "train"
    val_mask = split_labels == "valid"
    test_mask = split_labels == "test"

    # If no val split, carve from train
    if not val_mask.any():
        console.print(f"  [yellow]No validation split found. Carving {fallback_val:.0%} from train.[/yellow]")
        train_indices = np.where(train_mask)[0]
        n_val = max(1, int(len(train_indices) * fallback_val))
        rng = np.random.RandomState(seed)
        rng.shuffle(train_indices)
        val_indices = train_indices[:n_val]
        train_indices_final = train_indices[n_val:]
        val_mask = np.zeros(len(X), dtype=bool)
        val_mask[val_indices] = True
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[train_indices_final] = True

    seqs = preprocessed.raw_sequences
    embed = preprocessed.X_embed
    split = SplitData(
        X_train=X[train_mask], y_train=y[train_mask],
        X_val=X[val_mask], y_val=y[val_mask],
        X_test=X[test_mask], y_test=y[test_mask],
        split_method="predefined",
        feature_names=preprocessed.feature_names,
        label_encoder=preprocessed.label_encoder,
        seqs_train=_mask_list(seqs, train_mask) if seqs else None,
        seqs_val=_mask_list(seqs, val_mask) if seqs else None,
        seqs_test=_mask_list(seqs, test_mask) if seqs else None,
        X_embed_train=embed[train_mask] if embed is not None else None,
        X_embed_val=embed[val_mask] if embed is not None else None,
        X_embed_test=embed[test_mask] if embed is not None else None,
    )
    _print_split_summary(split)
    return split


def _split_from_folds(
    dataset: LoadedDataset,
    preprocessed: PreprocessingResult,
    profile: DatasetProfile,
    seed: int,
) -> SplitData:
    """Use fold_id column: last fold = test, second-to-last = val, rest = train."""
    fold_ids = dataset.fold_ids
    unique_folds = sorted(np.unique(fold_ids))
    X, y = preprocessed.X, preprocessed.y

    test_fold = unique_folds[-1]
    val_fold = unique_folds[-2] if len(unique_folds) > 1 else None

    test_mask = fold_ids == test_fold
    val_mask = fold_ids == val_fold if val_fold is not None else np.zeros(len(X), dtype=bool)
    train_mask = ~(test_mask | val_mask)

    # If val is empty (only 1 fold), carve from train
    split_cfg = get_splitting_defaults(profile.modality.value, profile.dataset_path)
    fallback_val = split_cfg.get("fallback_val_fraction", 0.2)
    if not val_mask.any():
        console.print(f"  [yellow]Only 1 fold. Splitting train {1-fallback_val:.0%}/{fallback_val:.0%} for val.[/yellow]")
        train_idx = np.where(train_mask)[0]
        rng = np.random.RandomState(seed)
        rng.shuffle(train_idx)
        n_val = max(1, int(len(train_idx) * fallback_val))
        val_mask[train_idx[:n_val]] = True
        train_mask[train_idx[:n_val]] = False

    console.print(f"  Fold strategy: train=folds {[f for f in unique_folds if f != test_fold and f != val_fold]}, "
                  f"val=fold {val_fold}, test=fold {test_fold}")

    seqs = preprocessed.raw_sequences
    embed = preprocessed.X_embed
    split = SplitData(
        X_train=X[train_mask], y_train=y[train_mask],
        X_val=X[val_mask], y_val=y[val_mask],
        X_test=X[test_mask], y_test=y[test_mask],
        split_method=f"fold_based (test={test_fold}, val={val_fold})",
        feature_names=preprocessed.feature_names,
        label_encoder=preprocessed.label_encoder,
        seqs_train=_mask_list(seqs, train_mask) if seqs else None,
        seqs_val=_mask_list(seqs, val_mask) if seqs else None,
        seqs_test=_mask_list(seqs, test_mask) if seqs else None,
        X_embed_train=embed[train_mask] if embed is not None else None,
        X_embed_val=embed[val_mask] if embed is not None else None,
        X_embed_test=embed[test_mask] if embed is not None else None,
    )
    _print_split_summary(split)
    return split


def _split_random(
    preprocessed: PreprocessingResult,
    profile: DatasetProfile,
    seed: int,
) -> SplitData:
    """Random split using configured ratios. Stratified for classification."""
    split_cfg = get_splitting_defaults(profile.modality.value, profile.dataset_path)
    test_size = split_cfg.get("test_size", 0.15)
    val_size = split_cfg.get("val_size", 0.15)
    do_stratify = split_cfg.get("stratify_classification", True)
    temp_size = test_size + val_size  # fraction held out from train

    X, y = preprocessed.X, preprocessed.y
    seqs = preprocessed.raw_sequences
    indices = np.arange(len(X))
    is_classification = profile.task_type in (
        TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION
    )

    stratify = y if (is_classification and do_stratify) else None

    # First split: train vs temp (val + test) — split indices to track sequences
    try:
        idx_train, idx_temp = train_test_split(
            indices, test_size=temp_size, random_state=seed, stratify=stratify,
        )
    except ValueError:
        console.print("  [yellow]Stratified split failed, falling back to random.[/yellow]")
        idx_train, idx_temp = train_test_split(
            indices, test_size=temp_size, random_state=seed,
        )
        stratify = None

    # Second split: divide temp into val and test
    test_fraction_of_temp = test_size / temp_size
    stratify_temp = y[idx_temp] if (is_classification and stratify is not None) else None
    try:
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=test_fraction_of_temp, random_state=seed, stratify=stratify_temp,
        )
    except ValueError:
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=test_fraction_of_temp, random_state=seed,
        )

    train_pct = int((1 - temp_size) * 100)
    val_pct = int(val_size * 100)
    test_pct = int(test_size * 100)
    method = f"stratified {train_pct}/{val_pct}/{test_pct}" if is_classification else f"random {train_pct}/{val_pct}/{test_pct}"

    embed = preprocessed.X_embed
    split = SplitData(
        X_train=X[idx_train], y_train=y[idx_train],
        X_val=X[idx_val], y_val=y[idx_val],
        X_test=X[idx_test], y_test=y[idx_test],
        split_method=method,
        feature_names=preprocessed.feature_names,
        label_encoder=preprocessed.label_encoder,
        seqs_train=[seqs[i] for i in idx_train] if seqs else None,
        seqs_val=[seqs[i] for i in idx_val] if seqs else None,
        seqs_test=[seqs[i] for i in idx_test] if seqs else None,
        X_embed_train=embed[idx_train] if embed is not None else None,
        X_embed_val=embed[idx_val] if embed is not None else None,
        X_embed_test=embed[idx_test] if embed is not None else None,
    )
    _print_split_summary(split)
    return split


def _mask_list(lst: list, mask: np.ndarray) -> list:
    """Apply a boolean mask to a list."""
    return [lst[i] for i in range(len(lst)) if mask[i]]


def _print_split_summary(split: SplitData) -> None:
    """Print split sizes."""
    s = split.summary()
    total = sum(s.values())
    console.print(
        f"  Split ({split.split_method}): "
        f"train={s['train']} ({s['train']/total*100:.0f}%), "
        f"val={s['val']} ({s['val']/total*100:.0f}%), "
        f"test={s['test']} ({s['test']/total*100:.0f}%)"
    )
