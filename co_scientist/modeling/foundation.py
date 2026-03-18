"""Foundation model embedding extraction via AIDO models (GPU-gated).

This module provides GPU detection and embedding extraction using GenBio's AIDO
foundation models. All functions degrade gracefully — returning None on failure
so the pipeline can continue with handcrafted features only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# Maps modality → default AIDO model name (overridable via config)
_DEFAULT_MODELS = {
    "rna": "aido_rna_1b600m",
    "dna": "aido_dna_300m",
    "protein": "aido_protein_16b",
    "cell_expression": "aido_cell_100m",
}

# Modalities that support foundation model embeddings
_SUPPORTED_MODALITIES = set(_DEFAULT_MODELS.keys())


def gpu_available() -> bool:
    """Check if a CUDA-capable GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def should_use_foundation(modality: str, config: dict[str, Any] | None = None) -> bool:
    """Decide whether to use foundation model embeddings.

    Parameters
    ----------
    modality : str
        Dataset modality (e.g. "rna", "dna", "protein", "cell_expression").
    config : dict, optional
        The ``foundation_models`` section from defaults.yaml.

    Returns
    -------
    bool
        True if embeddings should be extracted.
    """
    if modality not in _SUPPORTED_MODALITIES:
        return False

    if config is None:
        config = {}

    enabled = config.get("enabled", "auto")

    if enabled is False or enabled == "false":
        return False
    if enabled is True or enabled == "true":
        if not gpu_available():
            console.print("  [yellow]foundation_models.enabled=true but no GPU — skipping[/yellow]")
            return False
        return True
    # "auto" — use if GPU available
    return gpu_available()


def get_foundation_model_name(modality: str, config: dict[str, Any] | None = None) -> str:
    """Return the AIDO model identifier for a given modality."""
    if config and "models" in config:
        name = config["models"].get(modality)
        if name:
            return name
    return _DEFAULT_MODELS.get(modality, "")


def extract_embeddings(
    sequences_or_data: list[str] | np.ndarray,
    modality: str,
    config: dict[str, Any] | None = None,
    cache_dir: str | None = None,
) -> np.ndarray | None:
    """Extract embeddings from an AIDO foundation model.

    Parameters
    ----------
    sequences_or_data : list[str] or np.ndarray
        Raw sequences (DNA/RNA/protein) or expression vectors.
    modality : str
        Dataset modality.
    config : dict, optional
        The ``foundation_models`` section from defaults.yaml.
    cache_dir : str, optional
        Directory for caching embeddings to disk. If provided, embeddings are
        saved as .npy and reused on subsequent calls with the same data.

    Returns
    -------
    np.ndarray or None
        Embedding matrix of shape (n_samples, embedding_dim), or None on failure.
    """
    if config is None:
        config = {}

    model_name = get_foundation_model_name(modality, config)
    if not model_name:
        console.print(f"  [yellow]No foundation model configured for modality '{modality}'[/yellow]")
        return None

    batch_size = config.get("batch_size", 32)
    max_length = config.get("max_length", 1024)

    # Check disk cache
    cache_path = _get_cache_path(sequences_or_data, modality, model_name, cache_dir)
    if cache_path and cache_path.exists():
        try:
            cached = np.load(cache_path)
            console.print(f"  [green]Loaded cached embeddings:[/green] {cache_path.name} ({cached.shape})")
            return cached
        except Exception:
            pass  # Cache corrupt, re-extract

    try:
        result = _extract_embeddings_impl(
            sequences_or_data, model_name, modality, batch_size, max_length,
        )
        # Save to cache
        if result is not None and cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, result)
                console.print(f"  [dim]Cached embeddings to {cache_path.name}[/dim]")
            except Exception as ce:
                logger.debug("Could not cache embeddings: %s", ce)
        return result
    except Exception as e:
        console.print(f"  [yellow]Foundation embedding extraction failed: {e}[/yellow]")
        logger.warning("Foundation embedding extraction failed for modality=%s model=%s: %s", modality, model_name, e)
        return None


def _get_cache_path(
    data: list[str] | np.ndarray,
    modality: str,
    model_name: str,
    cache_dir: str | None,
) -> Any:
    """Compute a cache file path based on a hash of the input data."""
    if cache_dir is None:
        return None

    import hashlib
    from pathlib import Path

    # Hash the data to detect changes
    if isinstance(data, list):
        data_str = "\n".join(str(s)[:200] for s in data[:100])  # sample for speed
        data_hash = hashlib.md5(f"{len(data)}:{data_str}".encode()).hexdigest()[:12]
    elif isinstance(data, np.ndarray):
        data_hash = hashlib.md5(f"{data.shape}:{data.sum():.6f}".encode()).hexdigest()[:12]
    else:
        data_hash = "unknown"

    filename = f"embeddings_{modality}_{model_name}_{data_hash}.npy"
    return Path(cache_dir) / filename


def _extract_embeddings_impl(
    sequences_or_data: list[str] | np.ndarray,
    model_name: str,
    modality: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Internal implementation — imports heavy deps and runs on GPU.

    Uses the modelgenerator API:
        model = Embed.from_config({"model.backbone": model_name}).eval()
        transformed = model.transform({"sequences": [...]})
        embedding = model(transformed)
    """
    import torch
    from modelgenerator.tasks import Embed

    console.print(f"  Loading AIDO model [cyan]{model_name}[/cyan] for {modality} embeddings...")

    # Load the embedding model — follows official API exactly:
    #   model = Embed.from_config({"model.backbone": name}).eval()
    #   transformed = model.transform({"sequences": [...]})
    #   embedding = model(transformed)
    model = Embed.from_config({"model.backbone": model_name}).eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare data as list of strings
    if isinstance(sequences_or_data, np.ndarray):
        data_list = sequences_or_data.tolist() if sequences_or_data.ndim == 1 else [str(row) for row in sequences_or_data]
    else:
        data_list = list(sequences_or_data)

    n_samples = len(data_list)
    all_embeddings = []

    console.print(f"  Extracting embeddings for {n_samples} samples (batch_size={batch_size})...")

    n_failed = 0
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            batch = data_list[start : start + batch_size]

            # Truncate sequences if needed
            if modality in ("rna", "dna", "protein"):
                batch = [s[:max_length] if isinstance(s, str) else s for s in batch]

            try:
                # Step 1: Transform (tokenize) — may return CPU tensors
                transformed = model.transform({"sequences": batch})

                # Step 2: Move to device
                if isinstance(transformed, dict):
                    transformed = {k: v.to(device) if hasattr(v, "to") else v for k, v in transformed.items()}

                # Step 3: Forward pass
                embeddings = model(transformed)

                # Step 4: Extract numpy array, mean-pool if 3D
                # The model may return: Tensor, dict, or SequenceBackboneOutput (dataclass)
                emb_tensor = None

                if isinstance(embeddings, torch.Tensor):
                    emb_tensor = embeddings
                elif hasattr(embeddings, "last_hidden_state"):
                    # SequenceBackboneOutput from modelgenerator
                    emb_tensor = embeddings.last_hidden_state
                elif isinstance(embeddings, dict):
                    for key in ("embeddings", "last_hidden_state", "predictions"):
                        if key in embeddings and isinstance(embeddings[key], torch.Tensor):
                            emb_tensor = embeddings[key]
                            break

                if emb_tensor is not None:
                    if emb_tensor.ndim == 3:
                        emb_np = emb_tensor.mean(dim=1).cpu().float().numpy()
                    else:
                        emb_np = emb_tensor.cpu().float().numpy()
                else:
                    raise ValueError(
                        f"Cannot extract embeddings from output type {type(embeddings).__name__}. "
                        f"Attrs: {[a for a in dir(embeddings) if not a.startswith('_')]}"
                    )

                # Verify batch output matches input batch size
                if emb_np.shape[0] != len(batch):
                    console.print(
                        f"  [yellow]Batch size mismatch at {start}: input={len(batch)}, "
                        f"output={emb_np.shape[0]} — skipping[/yellow]"
                    )
                    n_failed += len(batch)
                    continue

                all_embeddings.append(emb_np)

            except Exception as batch_err:
                console.print(f"  [yellow]Batch {start}-{start+len(batch)} failed: {batch_err}[/yellow]")
                logger.warning("Embedding batch %d-%d failed: %s", start, start + len(batch), batch_err)
                n_failed += len(batch)
                continue

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    if not all_embeddings:
        raise RuntimeError("All embedding batches failed")

    result = np.vstack(all_embeddings)

    # Verify we got embeddings for all samples
    if result.shape[0] != n_samples:
        if n_failed > 0:
            console.print(
                f"  [yellow]Warning: {n_failed}/{n_samples} samples failed embedding extraction. "
                f"Got {result.shape[0]} embeddings.[/yellow]"
            )
        raise RuntimeError(
            f"Embedding count mismatch: expected {n_samples}, got {result.shape[0]}. "
            f"({n_failed} samples failed)"
        )

    console.print(f"  [green]Embeddings extracted:[/green] shape {result.shape}")
    return result
