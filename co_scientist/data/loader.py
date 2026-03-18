"""Dataset loading from genbio-leaderboard (HuggingFace).

Resilient loading: if a subset name is wrong, auto-discovers available configs
and picks the closest match. Handles arbitrary split structures gracefully.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console

from .types import DatasetInfo, LoadedDataset

console = Console()

# ---------------------------------------------------------------------------
# Registry: maps user-facing paths to HuggingFace repos + subsets/files
# ---------------------------------------------------------------------------
# The user passes paths like "RNA/translation_efficiency_muscle".
# We resolve these to the correct HuggingFace dataset repo and config.
#
# Structure:
#   "<modality_prefix>": {
#       "repo": "genbio-ai/<repo-name>",
#       "format": "hf_dataset" | "h5ad",
#       "tasks": {
#           "<task_name>": "<hf_subset_or_subdir>"
#       }
#   }
# ---------------------------------------------------------------------------

GENBIO_REGISTRY: dict[str, dict[str, Any]] = {
    "RNA": {
        "repo": "genbio-ai/rna-downstream-tasks",
        "format": "hf_dataset",
        "tasks": {
            # translation efficiency
            "translation_efficiency_muscle": "translation_efficiency_Muscle",
            "translation_efficiency_hek": "translation_efficiency_HEK",
            "translation_efficiency_pc3": "translation_efficiency_pc3",
            # expression level
            "expression_muscle": "expression_Muscle",
            "expression_hek": "expression_HEK",
            "expression_pc3": "expression_pc3",
            # other RNA tasks
            "mean_ribosome_load": "mean_ribosome_load",
            "splice_site_prediction": "splice_site_acceptor",
            "splice_site_acceptor": "splice_site_acceptor",
            "splice_site_donor": "splice_site_donor",
            "ncrna_family_classification": "ncrna_family_bnoise0",
            "ncrna_family_bnoise0": "ncrna_family_bnoise0",
            "ncrna_family_bnoise200": "ncrna_family_bnoise200",
        },
    },
    "expression": {
        "repo": "genbio-ai/cell-downstream-tasks",
        "format": "h5ad",
        "tasks": {
            "cell_type_classification_segerstolpe": "Segerstolpe",
            "cell_type_classification_zheng": "Zheng",
        },
    },
    "protein": {
        "repo": "genbio-ai/ProteinGYM-DMS",
        "format": "hf_dataset",
        "tasks": {},  # will be populated as needed
    },
    "tissue": {
        "repo": "genbio-ai/tissue-downstream-tasks",
        "format": "h5ad",
        "tasks": {},
    },
}

# Case-insensitive lookup helper
_MODALITY_ALIASES: dict[str, str] = {}
for key in GENBIO_REGISTRY:
    _MODALITY_ALIASES[key.lower()] = key


def resolve_dataset_path(dataset_path: str) -> tuple[str, str, str, str]:
    """Resolve a user-facing dataset path to (hf_repo, subset_or_subdir, format, task_name).

    Supports:
      - Registry paths: "RNA/translation_efficiency_muscle"
      - Direct HF paths: "genbio-ai/rna-downstream-tasks:translation_efficiency_Muscle"
      - Local file paths: "/path/to/data.csv"
    """
    # Local file
    if Path(dataset_path).exists():
        suffix = Path(dataset_path).suffix.lstrip(".")
        fmt = {"csv": "csv", "parquet": "parquet", "h5ad": "h5ad", "tsv": "csv"}.get(suffix, "unknown")
        return dataset_path, "", fmt, Path(dataset_path).stem

    # Direct HF path with colon separator: "genbio-ai/repo:subset"
    if ":" in dataset_path and "/" in dataset_path.split(":")[0]:
        repo, subset = dataset_path.split(":", 1)
        return repo, subset, "hf_dataset", subset

    # Registry lookup: "RNA/translation_efficiency_muscle"
    parts = dataset_path.strip("/").split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid dataset path '{dataset_path}'. "
            "Expected format: '<modality>/<task_name>' (e.g. 'RNA/translation_efficiency_muscle') "
            "or a direct HuggingFace path like 'genbio-ai/rna-downstream-tasks:subset_name'."
        )

    modality_key, task_name = parts
    modality_key_resolved = _MODALITY_ALIASES.get(modality_key.lower())
    if modality_key_resolved is None:
        raise ValueError(
            f"Unknown modality '{modality_key}'. "
            f"Known modalities: {list(GENBIO_REGISTRY.keys())}. "
            "You can also pass a direct HuggingFace path like 'genbio-ai/repo:subset'."
        )

    entry = GENBIO_REGISTRY[modality_key_resolved]
    task_name_lower = task_name.lower()

    # Exact match
    if task_name_lower in {k.lower(): k for k in entry["tasks"]}:
        matched_key = next(k for k in entry["tasks"] if k.lower() == task_name_lower)
        return entry["repo"], entry["tasks"][matched_key], entry["format"], matched_key

    # Fuzzy: try as a direct subset name
    console.print(
        f"[yellow]Task '{task_name}' not in registry for {modality_key_resolved}. "
        f"Trying as a direct subset name on {entry['repo']}...[/yellow]"
    )
    return entry["repo"], task_name, entry["format"], task_name


def load_dataset(dataset_path: str) -> LoadedDataset:
    """Load a dataset from a user-facing path. Returns a LoadedDataset.

    Resilient: tries multiple strategies, auto-discovers HF configs, and
    provides helpful error messages on failure.
    """
    hf_repo, subset, fmt, task_name = resolve_dataset_path(dataset_path)

    console.print(f"  Loading: [bold]{hf_repo}[/bold] / [cyan]{subset}[/cyan] (format: {fmt})")

    loaders = {
        "hf_dataset": lambda: _load_hf_dataset(hf_repo, subset, task_name),
        "h5ad": lambda: _load_h5ad_dataset(hf_repo, subset, task_name),
        "csv": lambda: _load_csv(hf_repo, task_name),
        "parquet": lambda: _load_parquet(hf_repo, task_name),
    }

    loader = loaders.get(fmt)
    if loader is None:
        raise ValueError(f"Unsupported format: {fmt}")

    try:
        result = loader()
    except Exception as e:
        console.print(f"  [yellow]Primary load failed: {e}[/yellow]")

        # If HF dataset failed, maybe it's actually h5ad or vice versa
        if fmt == "hf_dataset":
            console.print("  [yellow]Retrying as h5ad format...[/yellow]")
            try:
                result = _load_h5ad_dataset(hf_repo, subset, task_name)
                console.print("  [green]Successfully loaded as h5ad![/green]")
                return result
            except Exception:
                pass

        # Re-raise original with helpful context
        raise ValueError(
            f"Failed to load '{dataset_path}' (repo={hf_repo}, subset={subset}, format={fmt}). "
            f"Error: {e}\n"
            f"Tip: Try passing a direct HuggingFace path like 'genbio-ai/repo:subset_name'"
        ) from e

    # Validate the loaded dataset isn't empty
    if result.info.num_raw_samples == 0:
        raise ValueError(f"Dataset loaded but is empty (0 samples): {dataset_path}")

    return result


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_hf_dataset(repo: str, subset: str, task_name: str) -> LoadedDataset:
    """Load a HuggingFace dataset (arrow/parquet).

    Resilient: if the subset name doesn't match, auto-discovers available
    configs and picks the closest match via fuzzy matching.
    """
    from datasets import load_dataset as hf_load

    ds = _hf_load_with_fallback(hf_load, repo, subset)

    # Convert to DataFrame, normalizing split names
    df_all = _hf_splits_to_dataframe(ds)

    # Identify target and input columns
    target_col, input_cols, fold_col = _detect_columns_hf(df_all)

    X = df_all[input_cols]
    y = df_all[target_col]
    fold_ids = df_all[fold_col].values if fold_col else None

    has_splits = "_split" in df_all.columns
    split_col = "_split" if has_splits else fold_col

    info = DatasetInfo(
        name=task_name,
        hf_repo=repo,
        hf_subset=subset,
        source_format="hf_dataset",
        has_predefined_splits=has_splits or fold_col is not None,
        split_column=split_col,
        num_raw_samples=len(df_all),
    )

    return LoadedDataset(X=X, y=y, info=info, raw_data=df_all, fold_ids=fold_ids)


def _hf_load_with_fallback(hf_load, repo: str, subset: str):
    """Try to load the dataset; on failure, discover available configs and fuzzy-match."""
    # First attempt: direct load
    try:
        return hf_load(repo, name=subset if subset else None)
    except (ValueError, FileNotFoundError, Exception) as e:
        error_msg = str(e)

    # Extract available configs from the error message if possible
    available_configs = _extract_available_configs(error_msg)

    # If we couldn't parse configs from error, try to discover them
    if not available_configs:
        available_configs = _discover_hf_configs(repo)

    if not available_configs:
        raise ValueError(
            f"Cannot load dataset '{repo}' with subset '{subset}'. "
            f"Original error: {error_msg}"
        )

    console.print(f"  [yellow]Subset '{subset}' not found. Available: {available_configs}[/yellow]")

    # Fuzzy match: find closest config name
    matched = _fuzzy_match_config(subset, available_configs)
    if matched:
        console.print(f"  [green]Auto-matched to: '{matched}'[/green]")
        try:
            return hf_load(repo, name=matched)
        except Exception as e2:
            console.print(f"  [red]Fuzzy match '{matched}' also failed: {e2}[/red]")

    # Last resort: try each config that contains keywords from the subset name
    keywords = set(re.split(r"[_\-\s]+", subset.lower()))
    for config in available_configs:
        config_words = set(re.split(r"[_\-\s]+", config.lower()))
        if keywords & config_words:  # any overlap
            console.print(f"  [yellow]Trying keyword-matched config: '{config}'...[/yellow]")
            try:
                return hf_load(repo, name=config)
            except Exception:
                continue

    raise ValueError(
        f"Cannot load dataset '{repo}' with subset '{subset}'. "
        f"Available configs: {available_configs}. None matched."
    )


def _extract_available_configs(error_msg: str) -> list[str]:
    """Parse available config names from a HuggingFace error message."""
    # Pattern: "Available: ['config1', 'config2', ...]"
    match = re.search(r"Available:\s*\[([^\]]+)\]", error_msg)
    if match:
        raw = match.group(1)
        return [s.strip().strip("'\"") for s in raw.split(",")]
    return []


def _discover_hf_configs(repo: str) -> list[str]:
    """Discover available configs by querying the HuggingFace API."""
    try:
        from huggingface_hub import dataset_info
        info = dataset_info(repo)
        if hasattr(info, "config_names") and info.config_names:
            return list(info.config_names)
        # Fallback: try loading without subset to see configs in error
        from datasets import get_dataset_config_names
        return list(get_dataset_config_names(repo))
    except Exception:
        return []


def _fuzzy_match_config(target: str, available: list[str]) -> str | None:
    """Find the best fuzzy match for a config name."""
    target_lower = target.lower()

    # Exact case-insensitive match
    for cfg in available:
        if cfg.lower() == target_lower:
            return cfg

    # Substring match (target is part of config or vice versa)
    for cfg in available:
        if target_lower in cfg.lower() or cfg.lower() in target_lower:
            return cfg

    # difflib fuzzy match
    matches = get_close_matches(target_lower, [c.lower() for c in available], n=1, cutoff=0.5)
    if matches:
        # Map back to original case
        for cfg in available:
            if cfg.lower() == matches[0]:
                return cfg

    return None


def _hf_splits_to_dataframe(ds) -> pd.DataFrame:
    """Convert a HuggingFace DatasetDict to a single DataFrame with normalized split names."""
    if isinstance(ds, dict):
        split_names = list(ds.keys())
        if len(split_names) > 1:
            frames = []
            test_splits_found = []
            for split_name, split_ds in ds.items():
                df = split_ds.to_pandas()
                if split_name.startswith("test"):
                    df["_split"] = "test"
                    test_splits_found.append(split_name)
                elif split_name in ("validation", "valid", "val"):
                    df["_split"] = "valid"
                else:
                    df["_split"] = split_name
                frames.append(df)
            df_all = pd.concat(frames, ignore_index=True)
            if len(test_splits_found) > 1:
                console.print(f"  Merged {len(test_splits_found)} test splits: {test_splits_found}")
            return df_all
        else:
            return ds[split_names[0]].to_pandas()
    else:
        return ds.to_pandas()


def _detect_columns_hf(df: pd.DataFrame) -> tuple[str, list[str], str | None]:
    """Detect target column, input columns, and fold column from a HuggingFace dataframe.

    Resilient: tries common names, then case-insensitive matching, then heuristics.
    """
    # Common target column names (ordered by likelihood)
    target_candidates = [
        "labels", "label", "target", "y", "class", "category",
        "output", "response", "score", "value", "fitness",
    ]
    target_col = None

    # Exact match
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break

    # Case-insensitive match
    if target_col is None:
        col_lower_map = {c.lower(): c for c in df.columns}
        for cand in target_candidates:
            if cand.lower() in col_lower_map:
                target_col = col_lower_map[cand.lower()]
                break

    # Heuristic: last column that isn't a sequence/fold/split column
    if target_col is None:
        skip = {"_split", "fold_id", "fold", "cv_fold", "sequences", "sequence", "seq"}
        for col in reversed(df.columns.tolist()):
            if col.lower() not in skip and not col.startswith("_"):
                # Prefer columns with low cardinality or numeric type
                n_unique = df[col].nunique()
                if n_unique <= 200 or df[col].dtype in ("float64", "float32", "int64", "int32"):
                    target_col = col
                    console.print(f"  [yellow]Auto-detected target column: '{target_col}' (heuristic)[/yellow]")
                    break

    if target_col is None:
        raise ValueError(
            f"Cannot detect target column. Columns: {list(df.columns)}. "
            f"Expected one of: {target_candidates}"
        )

    # Fold column
    fold_col = None
    for col in ["fold_id", "fold", "cv_fold"]:
        if col in df.columns:
            fold_col = col
            break

    # Input columns: everything except target, fold, and internal columns
    exclude = {target_col, fold_col, "_split"} - {None}
    input_cols = [c for c in df.columns if c not in exclude]

    return target_col, input_cols, fold_col


def _load_h5ad_dataset(repo: str, subdir: str, task_name: str) -> LoadedDataset:
    """Load h5ad files from a HuggingFace repo (cell/tissue datasets).

    Resilient: tries multiple filename patterns, lists the repo on failure,
    and fuzzy-matches subdirectory names.
    """
    import anndata as ad
    from huggingface_hub import hf_hub_download

    # Try multiple naming conventions for each split
    split_files = {}
    for split_name in ["train", "valid", "test"]:
        # Common patterns across different HF repos
        subdir_lower = subdir.lower()
        patterns = [
            f"{subdir}_{split_name}.h5ad",
            f"{subdir}/{subdir}_{split_name}.h5ad",
            f"{subdir_lower}_{split_name}.h5ad",
            f"{subdir_lower}/{subdir_lower}_{split_name}.h5ad",
            f"{subdir}/{split_name}.h5ad",
            f"{subdir_lower}/{split_name}.h5ad",
            f"{split_name}/{subdir}.h5ad",
            f"{split_name}/{subdir_lower}.h5ad",
            f"{split_name}.h5ad",
        ]
        # Also try "validation" spelling
        if split_name == "valid":
            patterns.extend([
                f"{subdir}_validation.h5ad",
                f"{subdir}/{subdir}_validation.h5ad",
                f"{subdir_lower}_validation.h5ad",
                f"{subdir_lower}/{subdir_lower}_validation.h5ad",
            ])
        for pattern in patterns:
            try:
                path = hf_hub_download(repo_id=repo, filename=pattern, repo_type="dataset")
                split_files[split_name] = path
                break
            except Exception:
                continue

    if not split_files:
        # Discovery fallback: list repo and fuzzy-match
        split_files = _discover_h5ad_files(repo, subdir)

    if not split_files:
        raise FileNotFoundError(f"Could not find any h5ad files for '{subdir}' in {repo}")

    console.print(f"  Downloaded splits: {list(split_files.keys())}")

    # Load and concatenate
    adatas = {}
    for split_name, path in split_files.items():
        adatas[split_name] = ad.read_h5ad(path)

    # Concatenate all splits, tracking which split each cell belongs to
    for split_name, adata in adatas.items():
        adata.obs["_split"] = split_name
    adata_all = ad.concat(list(adatas.values()), join="outer")

    # Extract expression matrix and labels
    X_raw = adata_all.X
    if hasattr(X_raw, "toarray"):
        X_array = X_raw.toarray()  # sparse → dense
    else:
        X_array = np.array(X_raw)

    gene_names = list(adata_all.var_names) if adata_all.var_names is not None else [f"gene_{i}" for i in range(X_array.shape[1])]
    X_df = pd.DataFrame(X_array, columns=gene_names)

    # Detect label column in obs
    target_col = _detect_label_column_h5ad(adata_all)
    y = adata_all.obs[target_col].values

    info = DatasetInfo(
        name=task_name,
        hf_repo=repo,
        hf_subset=subdir,
        source_format="h5ad",
        has_predefined_splits=True,
        split_column="_split",
        num_raw_samples=len(adata_all),
    )

    # Store split info in raw_data
    raw_data = {"adata": adata_all, "splits": adata_all.obs["_split"].values}

    return LoadedDataset(X=X_df, y=pd.Series(y, name=target_col), info=info, raw_data=raw_data)


def _discover_h5ad_files(repo: str, subdir: str) -> dict[str, str]:
    """List all h5ad files in a HF repo and match them to splits via fuzzy matching."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    try:
        # list_repo_tree returns RepoFile and RepoFolder objects with .path attribute
        # Also search inside the subdir folder directly
        all_files = []
        for search_path in [None, subdir, subdir.lower()]:
            try:
                kwargs = {"repo_type": "dataset"}
                if search_path:
                    kwargs["path_in_repo"] = search_path
                items = list_repo_tree(repo, **kwargs)
                for f in items:
                    if hasattr(f, "path"):
                        all_files.append(f.path)
            except Exception:
                continue
    except Exception as e:
        console.print(f"  [red]Cannot list repo {repo}: {e}[/red]")
        return {}

    h5ad_files = [f for f in all_files if f.endswith(".h5ad")]
    if not h5ad_files:
        console.print(f"  [red]No .h5ad files found in {repo}[/red]")
        return {}

    # Try exact subdir match first, then fuzzy
    matched_files = [f for f in h5ad_files if subdir.lower() in f.lower()]
    if not matched_files:
        # Fuzzy match on directory/file components
        subdir_words = set(re.split(r"[_\-\s/]+", subdir.lower()))
        for f in h5ad_files:
            f_words = set(re.split(r"[_\-\s/]+", f.lower().replace(".h5ad", "")))
            if subdir_words & f_words:
                matched_files.append(f)

    if not matched_files:
        console.print(
            f"  [yellow]No h5ad files matched '{subdir}'. "
            f"Available: {h5ad_files}[/yellow]"
        )
        return {}

    console.print(f"  [green]Discovered h5ad files: {matched_files}[/green]")

    split_files = {}
    for f in matched_files:
        path = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset")
        f_lower = f.lower()
        for split_name in ["train", "valid", "validation", "test"]:
            if split_name in f_lower:
                norm = "valid" if split_name == "validation" else split_name
                split_files[norm] = path
                break
        else:
            # No split keyword — treat as full dataset
            if "train" not in split_files:
                split_files["train"] = path

    return split_files


def _detect_label_column_h5ad(adata) -> str:
    """Find the cell type / label column in an AnnData obs dataframe."""
    candidates = [
        "cell_type", "celltype", "cell_label", "cell_type_label",
        "label", "labels", "CellType", "Cell_type", "cell_type_id",
        "cluster", "class",
    ]
    for col in candidates:
        if col in adata.obs.columns:
            return col
    # Heuristic: find a categorical or string column with reasonable cardinality
    for col in adata.obs.columns:
        if col.startswith("_"):
            continue
        series = adata.obs[col]
        if series.dtype == "category" or series.dtype == object:
            n_unique = series.nunique()
            if 2 <= n_unique <= 200:
                return col
    raise ValueError(f"Cannot detect label column in h5ad obs. Columns: {list(adata.obs.columns)}")


def _load_csv(path: str, task_name: str) -> LoadedDataset:
    """Load a local CSV file."""
    df = pd.read_csv(path)
    target_col, input_cols, fold_col = _detect_columns_hf(df)
    info = DatasetInfo(name=task_name, source_format="csv", num_raw_samples=len(df))
    return LoadedDataset(
        X=df[input_cols], y=df[target_col], info=info, raw_data=df,
        fold_ids=df[fold_col].values if fold_col else None,
    )


def _load_parquet(path: str, task_name: str) -> LoadedDataset:
    """Load a local parquet file."""
    df = pd.read_parquet(path)
    target_col, input_cols, fold_col = _detect_columns_hf(df)
    info = DatasetInfo(name=task_name, source_format="parquet", num_raw_samples=len(df))
    return LoadedDataset(
        X=df[input_cols], y=df[target_col], info=info, raw_data=df,
        fold_ids=df[fold_col].values if fold_col else None,
    )
