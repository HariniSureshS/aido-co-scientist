"""FT-Transformer (Feature Tokenizer + Transformer) with sklearn-compatible interface.

A transformer architecture for tabular data that tokenizes each feature
into an embedding, adds a [CLS] token, and passes through transformer
encoder layers. The [CLS] token representation is used for prediction.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)

Uses manual batching (like mlp.py) to avoid OpenMP/threading conflicts.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)


class _FTTransformerModule(nn.Module):
    """Feature Tokenizer + Transformer Encoder."""

    def __init__(
        self,
        n_features: int,
        output_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            d_model = n_heads * (d_model // n_heads or 1)

        self.d_model = d_model

        # Feature tokenizer: one linear projection per feature → d_model
        self.feature_tokenizer = nn.Linear(n_features, n_features * d_model)
        self.n_features = n_features

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm + output head
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Tokenize: (batch, n_features) → (batch, n_features, d_model)
        tokens = self.feature_tokenizer(x).view(batch_size, self.n_features, self.d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, 1+n_features, d_model)

        # Transformer
        out = self.transformer(tokens)

        # Take [CLS] output
        cls_out = out[:, 0]
        cls_out = self.ln(cls_out)

        return self.head(cls_out)


def _iter_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Yield (X_batch, y_batch) without DataLoader."""
    n = len(X)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


class FTTransformerClassifier:
    """Sklearn-compatible FT-Transformer classifier."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._model: _FTTransformerModule | None = None
        self._classes: np.ndarray | None = None
        self._device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FTTransformerClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        n_features = X.shape[1]

        self._model = _FTTransformerModule(
            n_features=n_features,
            output_dim=n_classes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
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


class FTTransformerRegressor:
    """Sklearn-compatible FT-Transformer regressor."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._model: _FTTransformerModule | None = None
        self._device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FTTransformerRegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_features = X.shape[1]
        self._model = _FTTransformerModule(
            n_features=n_features,
            output_dim=1,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

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
