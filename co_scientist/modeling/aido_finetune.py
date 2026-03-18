"""AIDO Foundation Model Fine-Tuning — end-to-end fine-tuning with task head.

Unfreezes the last N layers of a pre-trained AIDO backbone and trains with
a lightweight task head. Requires GPU + modelgenerator package.

Architecture (Section 7.8):
  Input sequences → AIDO tokenizer
      ↓
  AIDO backbone (frozen layers 0..L-N)
      ↓
  AIDO backbone (unfrozen layers L-N+1..L)  ← fine-tuned
      ↓
  Task head (Linear → ReLU → Dropout → Linear → output)
      ↓
  Loss (MSE for regression, CrossEntropy for classification)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger(__name__)


class _TaskHead(nn.Module):
    """Lightweight task-specific head on top of backbone embeddings."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_aido_backbone(model_name: str):
    """Load an AIDO backbone model via modelgenerator.

    Returns (embed_model, transform_fn, embed_dim).

    API: Embed.from_config({"model.backbone": model_name})
         model.transform({"sequences": [...]}) → tokenized batch
         model(transformed_batch) → embeddings
    """
    from modelgenerator.tasks import Embed

    embed_model = Embed.from_config({"model.backbone": model_name}).eval()
    transform_fn = embed_model.transform

    # Probe embedding dimension with a dummy input
    embed_dim = _probe_embed_dim(embed_model, transform_fn)

    return embed_model, transform_fn, embed_dim


def _probe_embed_dim(embed_model, transform_fn) -> int:
    """Probe the embedding dimension by running a dummy forward pass."""
    with torch.no_grad():
        try:
            transformed = transform_fn({"sequences": ["ACGT"]})
            if isinstance(transformed, dict):
                transformed = {k: v.to(next(embed_model.parameters()).device) if hasattr(v, "to") else v for k, v in transformed.items()}
            out = embed_model(transformed)
            if isinstance(out, torch.Tensor):
                return out.shape[-1]
            elif hasattr(out, "last_hidden_state"):
                return out.last_hidden_state.shape[-1]
            elif isinstance(out, dict):
                for key in ("embeddings", "last_hidden_state"):
                    if key in out:
                        return out[key].shape[-1]
        except Exception:
            pass
    return 768  # safe default for most transformer models


def _unfreeze_last_n_layers(model: nn.Module, n: int) -> None:
    """Freeze all parameters, then unfreeze the last n transformer layers."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Find transformer layers (common naming patterns)
    layers = None
    for attr in ("encoder.layer", "layers", "transformer.layer", "blocks"):
        parts = attr.split(".")
        obj = model
        try:
            for part in parts:
                obj = getattr(obj, part)
            if hasattr(obj, "__len__"):
                layers = obj
                break
        except AttributeError:
            continue

    if layers is not None and len(layers) > 0:
        # Unfreeze last n layers
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info("Unfroze last %d of %d transformer layers", n, len(layers))
    else:
        # Fallback: unfreeze all parameters in the last portion of named parameters
        all_params = list(model.named_parameters())
        n_unfreeze = max(1, len(all_params) // 4)  # unfreeze last 25%
        for name, param in all_params[-n_unfreeze:]:
            param.requires_grad = True
        logger.warning("Could not find transformer layers, unfroze last %d parameters", n_unfreeze)


class _AIDOFinetuneModel(nn.Module):
    """Wraps AIDO backbone + task head for end-to-end fine-tuning."""

    def __init__(
        self,
        embed_model: Any,
        transform_fn: Any,
        embed_dim: int,
        output_dim: int,
        unfreeze_layers: int = 2,
        head_hidden: int = 256,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_model = embed_model
        self.transform_fn = transform_fn  # model.transform — tokenizes sequences
        self.head = _TaskHead(embed_dim, output_dim, head_hidden, head_dropout)

        # Freeze backbone then unfreeze last N layers
        _unfreeze_last_n_layers(self.embed_model, unfreeze_layers)

        # Always unfreeze the task head
        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, batch_seqs: list[str], device: Any = None) -> torch.Tensor:
        """Forward pass: sequences → transform → backbone → mean pool → head."""
        # Tokenize using the model's transform pipeline
        transformed = self.transform_fn({"sequences": batch_seqs})

        # Move to device
        if device is not None and isinstance(transformed, dict):
            transformed = {k: v.to(device) if hasattr(v, "to") else v for k, v in transformed.items()}
        elif device is not None and hasattr(transformed, "to"):
            transformed = transformed.to(device)

        # Run through AIDO backbone
        embeddings = self.embed_model(transformed)

        # Handle different output formats → pooled vector
        # Output may be Tensor, dict, or SequenceBackboneOutput (dataclass with .last_hidden_state)
        if isinstance(embeddings, torch.Tensor):
            pooled = embeddings.mean(dim=1) if embeddings.ndim == 3 else embeddings
        elif hasattr(embeddings, "last_hidden_state"):
            # SequenceBackboneOutput from modelgenerator
            emb = embeddings.last_hidden_state
            pooled = emb.mean(dim=1) if emb.ndim == 3 else emb
        elif isinstance(embeddings, dict):
            for key in ("embeddings", "last_hidden_state", "hidden_states"):
                if key in embeddings:
                    emb = embeddings[key]
                    if isinstance(emb, torch.Tensor):
                        pooled = emb.mean(dim=1) if emb.ndim == 3 else emb
                        break
            else:
                raise ValueError(f"Unexpected model output keys: {list(embeddings.keys())}")
        else:
            pooled = torch.tensor(np.array(embeddings), dtype=torch.float32)

        return self.head(pooled)

    def get_save_dict(self) -> dict:
        """Return a serializable dict of fine-tuned weights."""
        return {
            "head_state_dict": self.head.state_dict(),
            "full_state_dict": self.state_dict(),
        }


class AIDOFinetuneRegressor:
    """Sklearn-compatible AIDO fine-tuning regressor. Requires GPU."""

    def __init__(
        self,
        model_name: str = "",
        unfreeze_layers: int = 2,
        head_hidden: int = 256,
        head_dropout: float = 0.3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        max_epochs: int = 20,
        patience: int = 5,
        max_length: int = 1024,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.unfreeze_layers = unfreeze_layers
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.max_length = max_length
        self.random_state = random_state
        self._model: _AIDOFinetuneModel | None = None
        self._device = torch.device("cpu")

    def fit(
        self, X: np.ndarray, y: np.ndarray, sequences: list[str] | None = None,
    ) -> "AIDOFinetuneRegressor":
        if sequences is None:
            raise ValueError("aido_finetune requires raw sequences (sequences= parameter)")
        if not torch.cuda.is_available():
            raise RuntimeError("aido_finetune requires a CUDA-capable GPU")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._device = torch.device("cuda")

        # Load AIDO backbone
        embed_model, transform_fn, embed_dim = _load_aido_backbone(self.model_name)

        # Build model
        self._model = _AIDOFinetuneModel(
            embed_model=embed_model,
            transform_fn=transform_fn,
            embed_dim=embed_dim,
            output_dim=1,
            unfreeze_layers=self.unfreeze_layers,
            head_hidden=self.head_hidden,
            head_dropout=self.head_dropout,
        ).to(self._device)

        # Carve 10% for validation (early stopping)
        n = len(sequences)
        idx = np.random.permutation(n)
        n_val = max(1, int(n * 0.1))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        seqs_train = [sequences[i] for i in train_idx]
        y_train = y[train_idx]
        seqs_val = [sequences[i] for i in val_idx]
        y_val = y[val_idx]

        # Optimizer with differential learning rates
        backbone_params = [p for p in self._model.embed_model.parameters() if p.requires_grad]
        head_params = list(self._model.head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.learning_rate},
            {"params": head_params, "lr": self.learning_rate * 10},
        ], weight_decay=self.weight_decay)

        loss_fn = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # --- Train ---
            self._model.train()
            train_loss = 0.0
            n_batches = 0

            for start in range(0, len(seqs_train), self.batch_size):
                batch_seqs = [s[:self.max_length] for s in seqs_train[start:start + self.batch_size]]
                batch_y = torch.tensor(
                    y_train[start:start + self.batch_size], dtype=torch.float32,
                ).unsqueeze(1).to(self._device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    preds = self._model(batch_seqs, device=self._device)
                    loss = loss_fn(preds, batch_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                n_batches += 1

            # --- Validate ---
            self._model.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for start in range(0, len(seqs_val), self.batch_size):
                    batch_seqs = [s[:self.max_length] for s in seqs_val[start:start + self.batch_size]]
                    batch_y = torch.tensor(
                        y_val[start:start + self.batch_size], dtype=torch.float32,
                    ).unsqueeze(1).to(self._device)

                    with torch.amp.autocast("cuda"):
                        preds = self._model(batch_seqs, device=self._device)
                        loss = loss_fn(preds, batch_y)
                    val_loss += loss.item()
                    n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)
            avg_train_loss = train_loss / max(n_batches, 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.patience)
                    break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Move to CPU to free GPU memory for other models
        self._model = self._model.cpu()
        self._device = torch.device("cpu")
        torch.cuda.empty_cache()

        return self

    def predict(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        if sequences is None:
            raise ValueError("aido_finetune requires raw sequences for prediction")
        if self._model is None:
            raise RuntimeError("Model not fitted")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        self._model.eval()

        all_preds = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch_seqs = [s[:self.max_length] for s in sequences[start:start + self.batch_size]]
                with torch.amp.autocast("cuda") if device.type == "cuda" else _nullcontext():
                    preds = self._model(batch_seqs, device=device)
                all_preds.append(preds.cpu().squeeze(1).numpy())

        self._model = self._model.cpu()
        torch.cuda.empty_cache()

        return np.concatenate(all_preds)


class AIDOFinetuneClassifier:
    """Sklearn-compatible AIDO fine-tuning classifier. Requires GPU."""

    def __init__(
        self,
        model_name: str = "",
        unfreeze_layers: int = 2,
        head_hidden: int = 256,
        head_dropout: float = 0.3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        max_epochs: int = 20,
        patience: int = 5,
        max_length: int = 1024,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.unfreeze_layers = unfreeze_layers
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.max_length = max_length
        self.random_state = random_state
        self._model: _AIDOFinetuneModel | None = None
        self._device = torch.device("cpu")
        self._classes: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sequences: list[str] | None = None,
    ) -> "AIDOFinetuneClassifier":
        if sequences is None:
            raise ValueError("aido_finetune requires raw sequences (sequences= parameter)")
        if not torch.cuda.is_available():
            raise RuntimeError("aido_finetune requires a CUDA-capable GPU")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._device = torch.device("cuda")
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Load AIDO backbone
        embed_model, transform_fn, embed_dim = _load_aido_backbone(self.model_name)

        # Build model
        self._model = _AIDOFinetuneModel(
            embed_model=embed_model,
            transform_fn=transform_fn,
            embed_dim=embed_dim,
            output_dim=n_classes,
            unfreeze_layers=self.unfreeze_layers,
            head_hidden=self.head_hidden,
            head_dropout=self.head_dropout,
        ).to(self._device)

        # Carve 10% for validation
        n = len(sequences)
        idx = np.random.permutation(n)
        n_val = max(1, int(n * 0.1))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        seqs_train = [sequences[i] for i in train_idx]
        y_train = y[train_idx]
        seqs_val = [sequences[i] for i in val_idx]
        y_val = y[val_idx]

        # Optimizer with differential learning rates
        backbone_params = [p for p in self._model.embed_model.parameters() if p.requires_grad]
        head_params = list(self._model.head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.learning_rate},
            {"params": head_params, "lr": self.learning_rate * 10},
        ], weight_decay=self.weight_decay)

        loss_fn = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # --- Train ---
            self._model.train()
            train_loss = 0.0
            n_batches = 0

            for start in range(0, len(seqs_train), self.batch_size):
                batch_seqs = [s[:self.max_length] for s in seqs_train[start:start + self.batch_size]]
                batch_y = torch.tensor(
                    y_train[start:start + self.batch_size], dtype=torch.long,
                ).to(self._device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    logits = self._model(batch_seqs, device=self._device)
                    loss = loss_fn(logits, batch_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                n_batches += 1

            # --- Validate ---
            self._model.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for start in range(0, len(seqs_val), self.batch_size):
                    batch_seqs = [s[:self.max_length] for s in seqs_val[start:start + self.batch_size]]
                    batch_y = torch.tensor(
                        y_val[start:start + self.batch_size], dtype=torch.long,
                    ).to(self._device)

                    with torch.amp.autocast("cuda"):
                        logits = self._model(batch_seqs, device=self._device)
                        loss = loss_fn(logits, batch_y)
                    val_loss += loss.item()
                    n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.patience)
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._model = self._model.cpu()
        self._device = torch.device("cpu")
        torch.cuda.empty_cache()

        return self

    def predict(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        if sequences is None:
            raise ValueError("aido_finetune requires raw sequences for prediction")
        if self._model is None:
            raise RuntimeError("Model not fitted")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        self._model.eval()

        all_preds = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch_seqs = [s[:self.max_length] for s in sequences[start:start + self.batch_size]]
                with torch.amp.autocast("cuda") if device.type == "cuda" else _nullcontext():
                    logits = self._model(batch_seqs, device=device)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())

        self._model = self._model.cpu()
        torch.cuda.empty_cache()

        return np.concatenate(all_preds)

    def predict_proba(self, X: np.ndarray, sequences: list[str] | None = None) -> np.ndarray:
        if sequences is None:
            raise ValueError("aido_finetune requires raw sequences for prediction")
        if self._model is None:
            raise RuntimeError("Model not fitted")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        self._model.eval()

        all_probs = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch_seqs = [s[:self.max_length] for s in sequences[start:start + self.batch_size]]
                with torch.amp.autocast("cuda") if device.type == "cuda" else _nullcontext():
                    logits = self._model(batch_seqs, device=device)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        self._model = self._model.cpu()
        torch.cuda.empty_cache()

        return np.concatenate(all_probs)


class _nullcontext:
    """Minimal no-op context manager for CPU fallback."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
