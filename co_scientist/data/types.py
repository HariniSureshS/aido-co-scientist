"""Data types for the data layer."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class Modality(str, Enum):
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"
    CELL_EXPRESSION = "cell_expression"
    SPATIAL = "spatial"
    MULTIMODAL = "multimodal"
    TABULAR = "tabular"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    MULTILABEL = "multilabel"
    UNKNOWN = "unknown"


class DatasetInfo(BaseModel):
    """Metadata about a loaded dataset."""

    name: str
    hf_repo: str | None = None
    hf_subset: str | None = None
    source_format: str = "unknown"  # "hf_dataset", "h5ad", "csv", "parquet"
    has_predefined_splits: bool = False
    split_column: str | None = None  # e.g. "fold_id"
    num_raw_samples: int = 0


class DatasetProfile(BaseModel):
    """Structured profile of a dataset — produced without any LLM involvement."""

    model_config = {"arbitrary_types_allowed": True}

    # Identity
    dataset_name: str
    dataset_path: str
    task_hint: str = ""  # parsed from the path, e.g. "translation_efficiency"

    # Detected types
    modality: Modality = Modality.UNKNOWN
    task_type: TaskType = TaskType.UNKNOWN

    # Schema
    input_columns: list[str] = Field(default_factory=list)
    target_column: str = ""
    input_type: str = ""  # "sequence", "expression_matrix", "tabular"

    # Size
    num_samples: int = 0
    num_features: int = 0  # 0 for sequence data (variable length)
    num_classes: int = 0  # 0 for regression

    # Distribution
    class_distribution: dict[str, int] = Field(default_factory=dict)
    target_stats: dict[str, float] = Field(default_factory=dict)  # mean, std, min, max for regression
    missing_value_pct: float = 0.0
    feature_sparsity: float = 0.0  # fraction of zeros

    # Sequence-specific
    sequence_length_stats: dict[str, float] = Field(default_factory=dict)

    # Quality
    detected_issues: list[str] = Field(default_factory=list)

    # Splitting
    has_predefined_splits: bool = False
    split_info: dict[str, Any] = Field(default_factory=dict)


class LoadedDataset:
    """Container for a loaded dataset with its features and labels."""

    def __init__(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        info: DatasetInfo,
        raw_data: Any = None,
        fold_ids: np.ndarray | None = None,
    ):
        self.X = X
        self.y = y
        self.info = info
        self.raw_data = raw_data  # original data before any transforms
        self.fold_ids = fold_ids  # for CV-based datasets


class PreprocessingResult:
    """Output of the preprocessing step."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        steps_applied: list[str],
        label_encoder: Any | None = None,  # for classification with string labels
        raw_sequences: list[str] | None = None,  # raw seqs for CNN models
        X_embed: np.ndarray | None = None,  # foundation model embeddings (GPU-only)
    ):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.steps_applied = steps_applied
        self.label_encoder = label_encoder
        self.raw_sequences = raw_sequences
        self.X_embed = X_embed


class SplitData:
    """Train/val/test split container."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        split_method: str,
        feature_names: list[str] | None = None,
        label_encoder: Any | None = None,
        # Raw sequences for CNN models (only populated for sequence modalities)
        seqs_train: list[str] | None = None,
        seqs_val: list[str] | None = None,
        seqs_test: list[str] | None = None,
        # Foundation model embeddings (GPU-only, None on CPU)
        X_embed_train: np.ndarray | None = None,
        X_embed_val: np.ndarray | None = None,
        X_embed_test: np.ndarray | None = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.split_method = split_method
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.seqs_train = seqs_train
        self.seqs_val = seqs_val
        self.seqs_test = seqs_test
        self.X_embed_train = X_embed_train
        self.X_embed_val = X_embed_val
        self.X_embed_test = X_embed_test

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.y_train),
            "val": len(self.y_val),
            "test": len(self.y_test),
        }
