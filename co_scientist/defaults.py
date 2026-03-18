"""Load and merge layered YAML defaults.

Resolution order (later wins):
  1. Base defaults (defaults.yaml top-level keys)
  2. Modality overrides (modality_overrides.<modality>)
  3. Dataset-specific overrides (dataset_overrides.<dataset_path>)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"

# Cache the parsed YAML so we only read the file once per process.
_cache: dict[str, Any] | None = None


def _load_raw() -> dict[str, Any]:
    global _cache
    if _cache is None:
        with open(_DEFAULTS_PATH) as f:
            _cache = yaml.safe_load(f)
    return _cache


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def get_defaults(modality: str | None = None, dataset_path: str | None = None) -> dict[str, Any]:
    """Return the fully-merged defaults dict for a given modality + dataset.

    Parameters
    ----------
    modality : str, optional
        E.g. "rna", "cell_expression". Used to apply modality overrides.
    dataset_path : str, optional
        E.g. "RNA/translation_efficiency_muscle". Used to apply dataset overrides.

    Returns
    -------
    dict
        Merged config with keys: preprocessing, splitting, models, evaluation.
    """
    raw = _load_raw()

    # Start with base keys (everything except override sections)
    base = {
        k: copy.deepcopy(v)
        for k, v in raw.items()
        if k not in ("modality_overrides", "dataset_overrides")
    }

    # Layer 2: modality overrides
    if modality and "modality_overrides" in raw:
        mod_overrides = raw["modality_overrides"].get(modality, {})
        if mod_overrides:
            base = _deep_merge(base, mod_overrides)

    # Layer 3: dataset-specific overrides
    if dataset_path and "dataset_overrides" in raw:
        ds_overrides = raw["dataset_overrides"].get(dataset_path, {})
        if ds_overrides:
            base = _deep_merge(base, ds_overrides)

    return base


# ── Convenience accessors ────────────────────────────────────────────────────

def get_preprocessing_defaults(modality: str | None = None, dataset_path: str | None = None) -> dict[str, Any]:
    return get_defaults(modality, dataset_path).get("preprocessing", {})


def get_splitting_defaults(modality: str | None = None, dataset_path: str | None = None) -> dict[str, Any]:
    return get_defaults(modality, dataset_path).get("splitting", {})


def get_model_defaults(
    task_family: str,
    modality: str | None = None,
    dataset_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return model configs for a task family ('classification' or 'regression').

    Returns dict keyed by tier: {"trivial": {...}, "simple": {...}, "standard": {...}}
    """
    models = get_defaults(modality, dataset_path).get("models", {})
    return models.get(task_family, {})


def get_evaluation_defaults(modality: str | None = None, dataset_path: str | None = None) -> dict[str, Any]:
    return get_defaults(modality, dataset_path).get("evaluation", {})


def get_foundation_defaults(modality: str | None = None, dataset_path: str | None = None) -> dict[str, Any]:
    return get_defaults(modality, dataset_path).get("foundation_models", {})
