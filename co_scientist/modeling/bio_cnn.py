"""Multi-Scale BioCNN — custom 1D CNN for biological sequences.

Parallel convolutional branches with different kernel sizes capture biological
motifs at multiple scales simultaneously (codons, binding sites, regulatory elements).
Inspired by Inception-style networks, adapted for 1D bio sequences.

Architecture (Section 7.4):
  Input (one-hot) → [Conv1D(k=3) | Conv1D(k=5) | Conv1D(k=7) | Conv1D(k=9)]
  → Concat → GlobalPool → FC + Residual → Output
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# Nucleotide vocabulary for one-hot encoding
_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
_N_CHANNELS = 4  # A, C, G, T


def _one_hot_encode(sequences: list[str], max_len: int) -> np.ndarray:
    """One-hot encode sequences into (N, 4, max_len) arrays.

    Unknown characters (N, etc.) are encoded as all zeros.
    Sequences shorter than max_len are zero-padded on the right.
    """
    result = np.zeros((len(sequences), _N_CHANNELS, max_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq_upper = seq.upper().replace("U", "T")
        for j, c in enumerate(seq_upper[:max_len]):
            idx = _NUC_TO_IDX.get(c)
            if idx is not None:
                result[i, idx, j] = 1.0
    return result


class _ConvBranch(nn.Module):
    """Single convolutional branch with a specific kernel size."""

    def __init__(self, in_channels: int, n_filters: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, n_filters, kernel_size,
            padding=kernel_size // 2,  # same padding
        )
        self.bn = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _MultiScaleCNN(nn.Module):
    """Multi-scale 1D CNN with parallel branches and residual FC layers."""

    def __init__(
        self,
        n_filters: int = 64,
        kernel_sizes: list[int] | None = None,
        fc_dims: list[int] | None = None,
        dropout: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 5, 7, 9]
        fc_dims = fc_dims or [256, 128]

        # Parallel conv branches
        self.branches = nn.ModuleList([
            _ConvBranch(_N_CHANNELS, n_filters, k) for k in kernel_sizes
        ])

        # After concat + global pool: n_filters * n_branches * 2 (avg + max)
        pool_dim = n_filters * len(kernel_sizes) * 2

        # FC layers with residual connections
        fc_layers = []
        prev_dim = pool_dim
        for dim in fc_dims:
            fc_layers.append(_ResidualBlock(prev_dim, dim, dropout))
            prev_dim = dim
        self.fc = nn.Sequential(*fc_layers)
        self.head = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 4, seq_len)
        branch_outs = [branch(x) for branch in self.branches]
        # Concat along channel dim: (batch, n_filters * n_branches, seq_len)
        combined = torch.cat(branch_outs, dim=1)

        # Global pooling: both average and max for richer representation
        avg_pool = combined.mean(dim=2)   # (batch, n_filters * n_branches)
        max_pool = combined.max(dim=2).values
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, n_filters * n_branches * 2)

        # FC + residual
        features = self.fc(pooled)
        return self.head(features)


class _ResidualBlock(nn.Module):
    """FC block with skip connection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Project skip connection if dimensions differ
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.skip(x)


def _iter_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Yield batches without DataLoader (avoids OpenMP issues on macOS)."""
    n = len(X)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


class BioCNNRegressor:
    """Sklearn-compatible multi-scale BioCNN for regression on biological sequences."""

    def __init__(
        self,
        n_filters: int = 64,
        kernel_sizes: list[int] | None = None,
        fc_dims: list[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes or [3, 5, 7, 9]
        self.fc_dims = fc_dims or [256, 128]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._model: _MultiScaleCNN | None = None
        self._max_len: int = 0
        self._device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray, sequences: list[str] | None = None) -> "BioCNNRegressor":
        """Train the CNN. X is ignored if sequences are provided (uses one-hot encoding)."""
        if sequences is None:
            raise ValueError("BioCNN requires raw sequences. Pass sequences= to fit().")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._max_len = max(len(s) for s in sequences)
        X_encoded = _one_hot_encode(sequences, self._max_len)

        self._model = _MultiScaleCNN(
            n_filters=self.n_filters,
            kernel_sizes=self.kernel_sizes,
            fc_dims=self.fc_dims,
            dropout=self.dropout,
            output_dim=1,
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        # Split off 10% for early stopping
        n_val = max(1, int(len(X_encoded) * 0.1))
        perm = np.random.permutation(len(X_encoded))
        X_t = torch.tensor(X_encoded[perm[n_val:]], dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y[perm[n_val:]], dtype=torch.float32).unsqueeze(1).to(self._device)
        X_v = torch.tensor(X_encoded[perm[:n_val]], dtype=torch.float32).to(self._device)
        y_v = torch.tensor(y[perm[:n_val]], dtype=torch.float32).unsqueeze(1).to(self._device)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            for xb, yb in _iter_batches(X_t, y_t, self.batch_size):
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = criterion(self._model(X_v), y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        """Predict. X is ignored if sequences are provided."""
        if sequences is None:
            raise ValueError("BioCNN requires raw sequences for prediction.")
        X_encoded = _one_hot_encode(sequences, self._max_len)
        self._model.eval()
        X_t = torch.tensor(X_encoded, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            preds = self._model(X_t).squeeze(1).cpu().numpy()
        return preds


class BioCNNClassifier:
    """Sklearn-compatible multi-scale BioCNN for classification on biological sequences."""

    def __init__(
        self,
        n_filters: int = 64,
        kernel_sizes: list[int] | None = None,
        fc_dims: list[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes or [3, 5, 7, 9]
        self.fc_dims = fc_dims or [256, 128]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._model: _MultiScaleCNN | None = None
        self._max_len: int = 0
        self._classes: np.ndarray | None = None
        self._device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray, sequences: list[str] | None = None) -> "BioCNNClassifier":
        if sequences is None:
            raise ValueError("BioCNN requires raw sequences. Pass sequences= to fit().")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._max_len = max(len(s) for s in sequences)
        X_encoded = _one_hot_encode(sequences, self._max_len)

        self._model = _MultiScaleCNN(
            n_filters=self.n_filters,
            kernel_sizes=self.kernel_sizes,
            fc_dims=self.fc_dims,
            dropout=self.dropout,
            output_dim=n_classes,
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        n_val = max(1, int(len(X_encoded) * 0.1))
        perm = np.random.permutation(len(X_encoded))
        X_t = torch.tensor(X_encoded[perm[n_val:]], dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y[perm[n_val:]], dtype=torch.long).to(self._device)
        X_v = torch.tensor(X_encoded[perm[:n_val]], dtype=torch.float32).to(self._device)
        y_v = torch.tensor(y[perm[:n_val]], dtype=torch.long).to(self._device)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            for xb, yb in _iter_batches(X_t, y_t, self.batch_size):
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = criterion(self._model(X_v), y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        proba = self.predict_proba(X, sequences=sequences)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        if sequences is None:
            raise ValueError("BioCNN requires raw sequences for prediction.")
        X_encoded = _one_hot_encode(sequences, self._max_len)
        self._model.eval()
        X_t = torch.tensor(X_encoded, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba
