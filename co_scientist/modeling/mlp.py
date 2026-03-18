"""PyTorch MLP with sklearn-compatible interface for tabular data.

Uses manual batching instead of DataLoader to avoid OpenMP/threading
conflicts with XGBoost on macOS.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

# Prevent OpenMP deadlock when XGBoost and PyTorch coexist (macOS).
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)


class _MLPModule(nn.Module):
    """Simple feedforward network with configurable hidden layers."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _iter_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Yield (X_batch, y_batch) without DataLoader."""
    n = len(X)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


class MLPClassifier:
    """Sklearn-compatible MLP classifier."""

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._model: _MLPModule | None = None
        self._classes: np.ndarray | None = None
        self._device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        input_dim = X.shape[1]

        self._model = _MLPModule(input_dim, n_classes, self.hidden_dims, self.dropout).to(self._device)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Split off 10% for early stopping
        n_val = max(1, int(len(X) * 0.1))
        perm = np.random.permutation(len(X))
        X_t = torch.tensor(X[perm[n_val:]], dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y[perm[n_val:]], dtype=torch.long).to(self._device)
        X_v = torch.tensor(X[perm[:n_val]], dtype=torch.float32).to(self._device)
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba


class MLPRegressor:
    """Sklearn-compatible MLP regressor."""

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._model: _MLPModule | None = None
        self._device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = X.shape[1]
        self._model = _MLPModule(input_dim, 1, self.hidden_dims, self.dropout).to(self._device)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        # Split off 10% for early stopping
        n_val = max(1, int(len(X) * 0.1))
        perm = np.random.permutation(len(X))
        X_t = torch.tensor(X[perm[n_val:]], dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y[perm[n_val:]], dtype=torch.float32).unsqueeze(1).to(self._device)
        X_v = torch.tensor(X[perm[:n_val]], dtype=torch.float32).to(self._device)
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            preds = self._model(X_t).squeeze(1).cpu().numpy()
        return preds
