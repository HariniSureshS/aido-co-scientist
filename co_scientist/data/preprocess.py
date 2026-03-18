"""Modality-specific preprocessing transforms."""

from __future__ import annotations

import os
from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.preprocessing import LabelEncoder, StandardScaler

from co_scientist.defaults import get_defaults, get_preprocessing_defaults

from .types import DatasetProfile, LoadedDataset, Modality, PreprocessingResult, TaskType

console = Console()


def preprocess(dataset: LoadedDataset, profile: DatasetProfile) -> PreprocessingResult:
    """Apply modality-specific preprocessing. Returns numeric arrays ready for modeling."""
    console.print(f"  Modality: [cyan]{profile.modality.value}[/cyan] → routing to preprocessor")

    dispatcher = {
        Modality.RNA: _preprocess_sequence,
        Modality.DNA: _preprocess_sequence,
        Modality.PROTEIN: _preprocess_protein,
        Modality.CELL_EXPRESSION: _preprocess_expression,
        Modality.TABULAR: _preprocess_tabular,
    }

    fn = dispatcher.get(profile.modality, _preprocess_tabular)
    X, feature_names, steps = fn(dataset, profile)

    # Extract raw sequences for CNN models (sequence modalities only)
    raw_sequences = None
    if profile.modality in (Modality.RNA, Modality.DNA, Modality.PROTEIN):
        try:
            seq_col = _find_seq_column(dataset.X)
            raw_sequences = dataset.X[seq_col].astype(str).tolist()
        except (ValueError, KeyError):
            pass

    # Encode target
    y, label_encoder = _encode_target(dataset.y, profile)

    # --- Foundation model embeddings (GPU-gated) ---
    X_embed = _maybe_extract_embeddings(dataset, profile)

    # Validate embedding dimensions match feature matrix
    if X_embed is not None and X_embed.shape[0] != X.shape[0]:
        console.print(
            f"  [yellow]Embedding dimension mismatch: X={X.shape[0]} vs X_embed={X_embed.shape[0]} "
            f"— discarding embeddings[/yellow]"
        )
        X_embed = None

    console.print(f"  Output shape: [bold]{X.shape}[/bold]  ({len(steps)} transforms applied)")
    if X_embed is not None:
        console.print(f"  Embeddings: [bold]{X_embed.shape}[/bold]")
    return PreprocessingResult(
        X=X, y=y, feature_names=feature_names,
        steps_applied=steps, label_encoder=label_encoder,
        raw_sequences=raw_sequences,
        X_embed=X_embed,
    )


# ---------------------------------------------------------------------------
# Sequence preprocessing (DNA / RNA)
# ---------------------------------------------------------------------------

def _preprocess_sequence(dataset: LoadedDataset, profile: DatasetProfile) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract k-mer frequency features from nucleotide sequences."""
    cfg = get_preprocessing_defaults(profile.modality.value, profile.dataset_path)
    seq_cfg = cfg.get("sequence", {})
    steps = []

    # Find the sequence column
    seq_col = _find_seq_column(dataset.X)
    sequences = dataset.X[seq_col].astype(str).values

    features = []
    feature_names = []

    # 1. K-mer frequencies
    kmer_sizes = seq_cfg.get("kmer_sizes", [3, 4])
    for k in kmer_sizes:
        kmer_matrix, kmer_names = _kmer_frequencies(sequences, k=k)
        features.append(kmer_matrix)
        feature_names.extend(kmer_names)
        steps.append(f"{k}-mer frequencies ({len(kmer_names)} features)")

    # 2. Sequence length
    if seq_cfg.get("include_seq_length", True):
        lengths = np.array([len(s) for s in sequences], dtype=np.float64).reshape(-1, 1)
        features.append(lengths)
        feature_names.append("seq_length")
        steps.append("sequence length")

    # 3. GC content (for DNA/RNA)
    if seq_cfg.get("include_gc_content", True):
        gc = np.array([_gc_content(s) for s in sequences], dtype=np.float64).reshape(-1, 1)
        features.append(gc)
        feature_names.append("gc_content")
        steps.append("GC content")

    # 4. Nucleotide composition
    if seq_cfg.get("include_nuc_composition", True):
        nuc_matrix, nuc_names = _nucleotide_composition(sequences)
        features.append(nuc_matrix)
        feature_names.extend(nuc_names)
        steps.append("nucleotide composition")

    X = np.hstack(features)

    # 5. Standard scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    steps.append("standard scaling")

    return X, feature_names, steps


def _find_seq_column(X: pd.DataFrame) -> str:
    """Find the sequence column in the dataframe."""
    for c in X.columns:
        if c.lower() in ("sequences", "sequence", "seq"):
            return c
    # Fallback: first string column
    for c in X.columns:
        if X[c].dtype == object:
            return c
    raise ValueError(f"No sequence column found in {list(X.columns)}")


def _kmer_frequencies(sequences: np.ndarray, k: int) -> tuple[np.ndarray, list[str]]:
    """Compute k-mer frequency vectors for an array of sequences."""
    alphabet = "ACGT"  # works for DNA; RNA U→T mapping happens implicitly
    all_kmers = ["".join(kmer) for kmer in product(alphabet, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    n_kmers = len(all_kmers)

    matrix = np.zeros((len(sequences), n_kmers), dtype=np.float64)
    for i, seq in enumerate(sequences):
        seq_upper = seq.upper().replace("U", "T")
        counts = Counter(seq_upper[j:j+k] for j in range(len(seq_upper) - k + 1))
        total = sum(counts.values())
        if total > 0:
            for kmer, count in counts.items():
                idx = kmer_to_idx.get(kmer)
                if idx is not None:
                    matrix[i, idx] = count / total

    names = [f"kmer_{k}_{km}" for km in all_kmers]
    return matrix, names


def _gc_content(seq: str) -> float:
    """Compute GC content of a nucleotide sequence."""
    seq_upper = seq.upper().replace("U", "T")
    if len(seq_upper) == 0:
        return 0.0
    gc = sum(1 for c in seq_upper if c in ("G", "C"))
    return gc / len(seq_upper)


def _nucleotide_composition(sequences: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compute single-nucleotide frequencies."""
    nucs = ["A", "C", "G", "T"]
    matrix = np.zeros((len(sequences), len(nucs)), dtype=np.float64)
    for i, seq in enumerate(sequences):
        seq_upper = seq.upper().replace("U", "T")
        length = len(seq_upper)
        if length > 0:
            for j, nuc in enumerate(nucs):
                matrix[i, j] = seq_upper.count(nuc) / length
    return matrix, [f"nuc_freq_{n}" for n in nucs]


# ---------------------------------------------------------------------------
# Protein preprocessing
# ---------------------------------------------------------------------------

def _preprocess_protein(dataset: LoadedDataset, profile: DatasetProfile) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract amino acid composition + physicochemical features from protein sequences."""
    steps = []
    seq_col = _find_seq_column(dataset.X)
    sequences = dataset.X[seq_col].astype(str).values

    features = []
    feature_names = []

    # 1. Amino acid composition (20 standard AAs)
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_matrix = np.zeros((len(sequences), len(aa_list)), dtype=np.float64)
    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()
        length = len(seq_upper)
        if length > 0:
            for j, aa in enumerate(aa_list):
                aa_matrix[i, j] = seq_upper.count(aa) / length
    features.append(aa_matrix)
    feature_names.extend([f"aa_freq_{aa}" for aa in aa_list])
    steps.append("amino acid composition (20 features)")

    # 2. Sequence length
    lengths = np.array([len(s) for s in sequences], dtype=np.float64).reshape(-1, 1)
    features.append(lengths)
    feature_names.append("seq_length")
    steps.append("sequence length")

    # 3. Physicochemical properties (mean per sequence)
    # Molecular weight approximation per AA
    mw = {"A": 89, "C": 121, "D": 133, "E": 147, "F": 165, "G": 75, "H": 155,
           "I": 131, "K": 146, "L": 131, "M": 149, "N": 132, "P": 115, "Q": 146,
           "R": 174, "S": 105, "T": 119, "V": 117, "W": 204, "Y": 181}
    # Hydrophobicity (Kyte-Doolittle)
    hydro = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
             "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
             "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3}

    physchem = np.zeros((len(sequences), 2), dtype=np.float64)
    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()
        if len(seq_upper) > 0:
            physchem[i, 0] = np.mean([mw.get(c, 110) for c in seq_upper])
            physchem[i, 1] = np.mean([hydro.get(c, 0) for c in seq_upper])
    features.append(physchem)
    feature_names.extend(["avg_molecular_weight", "avg_hydrophobicity"])
    steps.append("physicochemical properties")

    X = np.hstack(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    steps.append("standard scaling")

    return X, feature_names, steps


# ---------------------------------------------------------------------------
# Cell expression preprocessing
# ---------------------------------------------------------------------------

def _preprocess_expression(dataset: LoadedDataset, profile: DatasetProfile) -> tuple[np.ndarray, list[str], list[str]]:
    """Preprocess cell/scRNA-seq expression matrix: log1p → HVG → scale."""
    cfg = get_preprocessing_defaults(profile.modality.value, profile.dataset_path)
    expr_cfg = cfg.get("expression", {})
    steps = []

    if isinstance(dataset.X, pd.DataFrame):
        X = dataset.X.values.astype(np.float64)
        gene_names = list(dataset.X.columns)
    else:
        X = np.array(dataset.X, dtype=np.float64)
        gene_names = [f"gene_{i}" for i in range(X.shape[1])]

    # 1. Normalization
    normalization = expr_cfg.get("normalization", "log1p")
    if normalization == "log1p":
        X = np.log1p(X)
        steps.append("log1p normalization")

    # 2. Highly variable gene (HVG) selection
    n_hvg = min(expr_cfg.get("n_hvg", 2000), X.shape[1])
    if X.shape[1] > n_hvg:
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-n_hvg:]
        top_indices = np.sort(top_indices)  # keep original order
        X = X[:, top_indices]
        gene_names = [gene_names[i] for i in top_indices]
        steps.append(f"HVG selection (top {n_hvg} of {profile.num_features})")
    else:
        steps.append(f"kept all {X.shape[1]} genes (< {n_hvg} HVG threshold)")

    # 3. Standard scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    steps.append("standard scaling")

    return X, gene_names, steps


# ---------------------------------------------------------------------------
# Tabular / fallback preprocessing
# ---------------------------------------------------------------------------

def _preprocess_tabular(dataset: LoadedDataset, profile: DatasetProfile) -> tuple[np.ndarray, list[str], list[str]]:
    """Generic preprocessing for tabular data."""
    cfg = get_preprocessing_defaults(profile.modality.value, profile.dataset_path)
    steps = []

    if isinstance(dataset.X, pd.DataFrame):
        # Separate numeric and non-numeric
        numeric_cols = dataset.X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in dataset.X.columns if c not in numeric_cols]

        # Encode non-numeric columns
        X_parts = []
        feature_names = []

        if numeric_cols:
            X_num = dataset.X[numeric_cols].fillna(0).values.astype(np.float64)
            X_parts.append(X_num)
            feature_names.extend(numeric_cols)

        for col in non_numeric_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(dataset.X[col].fillna("missing").astype(str))
            X_parts.append(encoded.reshape(-1, 1).astype(np.float64))
            feature_names.append(col)
            steps.append(f"label-encoded '{col}'")

        X = np.hstack(X_parts) if X_parts else np.zeros((len(dataset.y), 0))
    else:
        X = np.array(dataset.X, dtype=np.float64)
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    # Fill NaN
    nan_fill = cfg.get("nan_fill", 0.0)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=nan_fill)
        steps.append(f"filled NaN with {nan_fill}")

    # Standard scaling
    if X.shape[1] > 0:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        steps.append("standard scaling")

    return X, feature_names, steps


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def _maybe_extract_embeddings(
    dataset: LoadedDataset, profile: DatasetProfile,
) -> np.ndarray | None:
    """Extract foundation model embeddings if GPU is available and modality is supported."""
    from co_scientist.modeling.foundation import should_use_foundation, extract_embeddings

    defaults = get_defaults(profile.modality.value, profile.dataset_path)
    fm_config = defaults.get("foundation_models", {})

    if not should_use_foundation(profile.modality.value, fm_config):
        return None

    console.print("  [bold cyan]Foundation model embeddings:[/bold cyan] extracting...")

    # Gather raw data for embedding extraction
    if profile.modality in (Modality.RNA, Modality.DNA, Modality.PROTEIN):
        try:
            seq_col = _find_seq_column(dataset.X)
            raw_data = dataset.X[seq_col].astype(str).tolist()
        except (ValueError, KeyError):
            console.print("  [yellow]Could not find sequence column for embeddings[/yellow]")
            return None
    elif profile.modality == Modality.CELL_EXPRESSION:
        if isinstance(dataset.X, pd.DataFrame):
            raw_data = dataset.X.values.astype(np.float64)
        else:
            raw_data = np.array(dataset.X, dtype=np.float64)
    else:
        return None

    # Use a cache directory based on dataset path to avoid re-extracting
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "co_scientist_embeddings")
    return extract_embeddings(raw_data, profile.modality.value, fm_config, cache_dir=cache_dir)


def _encode_target(y: pd.Series | np.ndarray, profile: DatasetProfile) -> tuple[np.ndarray, LabelEncoder | None]:
    """Encode target variable. Returns (encoded_array, encoder_or_None)."""
    y_arr = np.array(y)

    if profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        # String/categorical labels → integer encoding
        if y_arr.dtype == object or not np.issubdtype(y_arr.dtype, np.number):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_arr)
            console.print(f"  Encoded {len(le.classes_)} classes: {list(le.classes_[:5])}{'...' if len(le.classes_) > 5 else ''}")
            return y_encoded.astype(np.int64), le
        return y_arr.astype(np.int64), None

    # Regression
    return y_arr.astype(np.float64), None
