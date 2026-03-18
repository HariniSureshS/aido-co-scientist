"""Stacking ensemble — builds a new model from base model predictions.

Architecture Section 7.6: The stacking ensemble combines all trained base models
via a meta-learner that learns which models to trust for which types of inputs.

How it works:
  1. Generate out-of-fold predictions from each base model (cross-validation)
  2. Train a meta-learner (Ridge/Logistic) on the stacked predictions
  3. At inference: run all base models → feed predictions to meta-learner → output
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from co_scientist.data.types import SplitData

from .types import ModelConfig, TrainedModel

console = Console()


class StackingEnsemble:
    """Meta-learner stacking ensemble for regression."""

    def __init__(
        self,
        base_models: list[TrainedModel],
        split: SplitData,
        n_folds: int = 5,
        seed: int = 42,
    ):
        self.base_models = base_models
        self.split = split
        self.n_folds = n_folds
        self.seed = seed
        self._meta_model: Ridge | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """Train the meta-learner on out-of-fold base model predictions."""
        # Generate out-of-fold predictions for each base model
        oof_preds = self._generate_oof_predictions(X, y)

        if oof_preds.shape[1] < 2:
            raise ValueError("Stacking requires at least 2 base models")

        # Train meta-learner on stacked OOF predictions
        self._meta_model = Ridge(alpha=1.0)
        self._meta_model.fit(oof_preds, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        """Run all base models, stack predictions, feed to meta-learner."""
        if not self._is_fitted:
            raise RuntimeError("StackingEnsemble not fitted yet")
        stacked = self._get_base_predictions(X, sequences)
        return self._meta_model.predict(stacked)

    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate out-of-fold predictions using cross-validation.

        This avoids data leakage: each base model predicts on data it wasn't trained on.
        """
        n_models = len(self.base_models)
        oof_matrix = np.zeros((len(X), n_models))
        seqs = self.split.seqs_train

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]
            seqs_fold_train = [seqs[i] for i in train_idx] if seqs else None
            seqs_fold_val = [seqs[i] for i in val_idx] if seqs else None

            for model_idx, trained in enumerate(self.base_models):
                # Re-train base model on this fold
                from .registry import build_model
                fold_model = build_model(trained.config)
                try:
                    if trained.needs_sequences:
                        fold_model.fit(X_fold_train, y_fold_train, sequences=seqs_fold_train)
                        oof_matrix[val_idx, model_idx] = fold_model.predict(
                            X_fold_val, sequences=seqs_fold_val
                        )
                    else:
                        fold_model.fit(X_fold_train, y_fold_train)
                        preds = fold_model.predict(X_fold_val)
                        if preds.ndim == 2 and preds.shape[1] == 1:
                            preds = preds.ravel()
                        oof_matrix[val_idx, model_idx] = preds
                except Exception:
                    # If a model fails on a fold, use mean prediction as fallback
                    oof_matrix[val_idx, model_idx] = np.mean(y_fold_train)

        return oof_matrix

    def _get_base_predictions(
        self, X: np.ndarray, sequences: list[str] | None = None
    ) -> np.ndarray:
        """Get predictions from all base models for new data."""
        preds = np.zeros((len(X), len(self.base_models)))
        for i, trained in enumerate(self.base_models):
            try:
                if trained.needs_sequences:
                    preds[:, i] = trained.predict(X, sequences=sequences)
                else:
                    preds[:, i] = trained.predict(X)
            except Exception:
                preds[:, i] = 0.0  # fallback
        return preds


class StackingEnsembleClassifier:
    """Meta-learner stacking ensemble for classification."""

    def __init__(
        self,
        base_models: list[TrainedModel],
        split: SplitData,
        n_folds: int = 5,
        seed: int = 42,
    ):
        self.base_models = base_models
        self.split = split
        self.n_folds = n_folds
        self.seed = seed
        self._meta_model: LogisticRegression | None = None
        self._classes: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsembleClassifier":
        """Train the meta-learner on out-of-fold base model predictions."""
        self._classes = np.unique(y)
        oof_preds = self._generate_oof_predictions(X, y)

        if oof_preds.shape[1] < 2:
            raise ValueError("Stacking requires at least 2 base models")

        self._meta_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self._meta_model.fit(oof_preds, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("StackingEnsembleClassifier not fitted yet")
        stacked = self._get_base_predictions(X, sequences)
        return self._meta_model.predict(stacked)

    def predict_proba(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("StackingEnsembleClassifier not fitted yet")
        stacked = self._get_base_predictions(X, sequences)
        return self._meta_model.predict_proba(stacked)

    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate out-of-fold predictions (class predictions, not probabilities)."""
        n_models = len(self.base_models)
        oof_matrix = np.zeros((len(X), n_models))
        seqs = self.split.seqs_train

        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]
            seqs_fold_train = [seqs[i] for i in train_idx] if seqs else None
            seqs_fold_val = [seqs[i] for i in val_idx] if seqs else None

            for model_idx, trained in enumerate(self.base_models):
                from .registry import build_model
                fold_model = build_model(trained.config)
                try:
                    if trained.needs_sequences:
                        fold_model.fit(X_fold_train, y_fold_train, sequences=seqs_fold_train)
                        oof_matrix[val_idx, model_idx] = fold_model.predict(
                            X_fold_val, sequences=seqs_fold_val
                        )
                    else:
                        fold_model.fit(X_fold_train, y_fold_train)
                        oof_matrix[val_idx, model_idx] = fold_model.predict(X_fold_val)
                except Exception:
                    # Majority class fallback
                    from collections import Counter
                    most_common = Counter(y_fold_train).most_common(1)[0][0]
                    oof_matrix[val_idx, model_idx] = most_common

        return oof_matrix

    def _get_base_predictions(
        self, X: np.ndarray, sequences: list[str] | None = None
    ) -> np.ndarray:
        """Get predictions from all base models for new data."""
        preds = np.zeros((len(X), len(self.base_models)))
        for i, trained in enumerate(self.base_models):
            try:
                if trained.needs_sequences:
                    preds[:, i] = trained.predict(X, sequences=sequences)
                else:
                    preds[:, i] = trained.predict(X)
            except Exception:
                preds[:, i] = 0
        return preds


def build_stacking_ensemble(
    trained_models: list[TrainedModel],
    split: SplitData,
    task_type: str,
    seed: int = 42,
) -> tuple[TrainedModel, Any] | None:
    """Build and train a stacking ensemble from the trained base models.

    Skips trivial baselines. Requires at least 2 non-trivial models.
    Returns (TrainedModel, ensemble_object) or None if not enough base models.
    """
    import time

    # Filter: skip trivial baselines, existing ensembles, custom models,
    # and foundation-tier models (embedding models need X_embed which isn't
    # available per-fold, and aido_finetune is too expensive for OOF CV)
    _SKIP_TYPES = {"stacking", "custom", "embed_xgboost", "embed_mlp",
                    "embed_ridge", "embed_logistic", "aido_finetune",
                    "concat_xgboost", "concat_mlp"}
    base_models = [
        m for m in trained_models
        if m.config.tier != "trivial"
        and m.config.model_type not in _SKIP_TYPES
    ]

    if len(base_models) < 2:
        console.print("  [dim]Skipping stacking: need at least 2 non-trivial base models[/dim]")
        return None

    console.print(f"  Training [cyan]stacking_ensemble[/cyan] (ensemble) with {len(base_models)} base models...")
    base_names = [m.config.name for m in base_models]
    console.print(f"    Base models: {', '.join(base_names)}")

    start = time.time()
    try:
        if task_type == "classification":
            ensemble = StackingEnsembleClassifier(
                base_models=base_models, split=split, seed=seed,
            )
        else:
            ensemble = StackingEnsemble(
                base_models=base_models, split=split, seed=seed,
            )
        ensemble.fit(split.X_train, split.y_train)
    except Exception as e:
        console.print(f"    [yellow]Stacking failed: {e}[/yellow]")
        return None

    elapsed = time.time() - start
    console.print(f"    [green]done[/green] ({elapsed:.1f}s)")

    config = ModelConfig(
        name="stacking_ensemble",
        tier="ensemble",
        model_type="stacking",
        task_type=task_type,
        hyperparameters={"n_base_models": len(base_models), "base_models": base_names},
    )
    trained = TrainedModel(config=config, model=ensemble, train_time_seconds=elapsed)
    return trained
