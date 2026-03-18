"""Model trainer — trains models and returns TrainedModel objects."""

from __future__ import annotations

import threading
import time

import numpy as np
from rich.console import Console

from co_scientist.data.types import SplitData

from .registry import build_model
from .types import ModelConfig, TrainedModel

console = Console()

# Model types that need raw sequences instead of/in addition to features
_SEQUENCE_MODELS = {"bio_cnn", "aido_finetune"}

# Model types that use foundation model embeddings instead of handcrafted features
_EMBEDDING_MODELS = {"embed_xgboost", "embed_mlp"}

# Model types that concatenate handcrafted features + embeddings
_CONCAT_MODELS = {"concat_xgboost", "concat_mlp"}

# Per-model training timeout (seconds)
_TRAINING_TIMEOUT = 120  # 2 minutes


def _train_in_thread(fn, result_holder: list, error_holder: list) -> None:
    """Run training function in a thread, storing result or exception."""
    try:
        result_holder.append(fn())
    except Exception as e:
        error_holder.append(e)


def train_model(config: ModelConfig, split: SplitData) -> TrainedModel | None:
    """Train a single model and return a TrainedModel, or None if skipped/timed out."""
    console.print(f"  Training [cyan]{config.name}[/cyan] ({config.tier})...")

    if config.model_type in _EMBEDDING_MODELS or config.model_type in _CONCAT_MODELS:
        if split.X_embed_train is None:
            console.print(f"    [yellow]skipping {config.name} — no embeddings available[/yellow]")
            return None

    model = build_model(config)
    y_train = split.y_train

    # Build the training function
    def do_train():
        if config.model_type in _CONCAT_MODELS:
            X_combined = np.hstack([split.X_train, split.X_embed_train])
            model.fit(X_combined, y_train)
        elif config.model_type in _EMBEDDING_MODELS:
            model.fit(split.X_embed_train, y_train)
        elif config.model_type in _SEQUENCE_MODELS:
            model.fit(split.X_train, split.y_train, sequences=split.seqs_train)
        else:
            model.fit(split.X_train, y_train)

    # Run with timeout using threading (safe with C extensions unlike SIGALRM)
    start = time.time()
    result_holder: list = []
    error_holder: list = []
    thread = threading.Thread(target=_train_in_thread, args=(do_train, result_holder, error_holder))
    thread.daemon = True
    thread.start()
    thread.join(timeout=_TRAINING_TIMEOUT)

    if thread.is_alive():
        elapsed = time.time() - start
        console.print(f"    [yellow]timed out after {elapsed:.0f}s — skipping {config.name}[/yellow]")
        # Thread will eventually finish in background; we move on
        return None

    if error_holder:
        elapsed = time.time() - start
        console.print(f"    [yellow]training failed ({elapsed:.1f}s): {error_holder[0]}[/yellow]")
        return None

    elapsed = time.time() - start
    console.print(f"    [green]done[/green] ({elapsed:.1f}s)")

    return TrainedModel(config=config, model=model, train_time_seconds=elapsed)


def train_baselines(configs: list[ModelConfig], split: SplitData) -> list[TrainedModel]:
    """Train all baseline models in order (trivial → simple → standard → advanced → foundation)."""
    models = []
    for config in configs:
        trained = train_model(config, split)
        if trained is not None:
            models.append(trained)
    return models
