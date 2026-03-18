# Step 7: Predefined Defaults YAML — Detailed Walkthrough

## Overview

Step 7 answers: **"How do we make the pipeline configurable without touching Python code?"**

Previously, every parameter was hardcoded: k-mer sizes `[3, 4]`, HVG count `2000`, XGBoost `n_estimators=100`, split ratio `70/15/15`, imbalance threshold `5%`. Changing any of these meant editing Python files. Step 7 externalizes all tunable parameters into a single `defaults.yaml` file with a layered override system.

---

## Architecture

```
co_scientist/
├── defaults.yaml       ← all tunable parameters in one file
├── defaults.py         ← loader with layered merge logic
├── data/
│   ├── preprocess.py   ← reads preprocessing params from YAML
│   └── split.py        ← reads split ratios from YAML
├── modeling/
│   └── registry.py     ← reads model configs from YAML
└── evaluation/
    └── auto_config.py  ← reads metric choices from YAML
```

## The Layered Override System

The config resolves in three layers — each layer merges on top of the previous:

```
1. Base defaults          (defaults.yaml top-level keys)
2. Modality overrides     (modality_overrides.<modality>)
3. Dataset overrides      (dataset_overrides.<dataset_path>)
```

### Example: XGBoost for cell expression classification

**Base default:**
```yaml
models:
  classification:
    standard:
      hyperparameters:
        n_estimators: 100
        max_depth: 6
```

**Modality override (cell_expression):**
```yaml
modality_overrides:
  cell_expression:
    models:
      classification:
        standard:
          hyperparameters:
            n_estimators: 200
            max_depth: 8
```

**Result after merge:** `n_estimators=200, max_depth=8, learning_rate=0.1` — the override replaces `n_estimators` and `max_depth` but `learning_rate` is inherited from base.

**RNA/DNA modality overrides** also include FT-Transformer configurations (not just the base config). This means sequence-specific hyperparameters for this advanced model are set at the modality level, allowing RNA and DNA datasets to use tuned defaults for FT-Transformer out of the box.

### Why this design?

The future LLM agents (Phase C) will need to modify pipeline behavior. Instead of generating Python code (risky, hard to validate), an agent can output a dataset-specific YAML override. The pipeline reads it the same way it reads the base defaults. This is the "agent action interface" for configuration.

---

## What's in the YAML

### Preprocessing

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `scaling` | `standard` | Scaler type (standard, minmax, none) |
| `nan_fill` | `0.0` | Fill value for missing data |
| `sequence.kmer_sizes` | `[3, 4]` | Which k-mer sizes to extract |
| `sequence.include_gc_content` | `true` | Whether to add GC content feature |
| `sequence.include_nuc_composition` | `true` | Whether to add nucleotide frequencies |
| `sequence.include_seq_length` | `true` | Whether to add sequence length feature |
| `expression.normalization` | `log1p` | Normalization for expression data |
| `expression.n_hvg` | `2000` | Number of highly variable genes to keep |

### Splitting

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `test_size` | `0.15` | Fraction for test set |
| `val_size` | `0.15` | Fraction for validation set |
| `stratify_classification` | `true` | Use stratified splitting for classification |
| `fallback_val_fraction` | `0.2` | Carve from train if no val split exists |

### Models (per task type)

Each task type (`classification`, `regression`) has four tiers (`trivial`, `simple`, `standard`, `advanced`), each specifying `name`, `model_type`, and `hyperparameters`.

| Tier | Models |
|------|--------|
| Trivial | majority_class, mean_predictor |
| Simple | logistic_regression, ridge_regression, elastic_net |
| Standard | xgboost, lightgbm, random_forest, svm, knn |
| Advanced | mlp, ft_transformer, bio_cnn (sequence data) |

### Evaluation

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `imbalance_threshold_pct` | `5` | Below this % triggers imbalanced metrics |
| `metrics.*.primary` | varies | Primary metric per task type |
| `metrics.*.secondary` | varies | Secondary metrics list |

---

## What's NOT in the YAML

Biological constants stay in code:
- **Amino acid molecular weights** — these are physical constants, not tunable parameters
- **Kyte-Doolittle hydrophobicity values** — same, reference data
- **Nucleotide alphabet (ACGT)** — fundamental, not configurable
- **Standard amino acid list** — biochemistry, not ML tuning

The rule: if changing a value would make results *wrong* rather than *different*, it belongs in code. If changing it makes results *different but potentially better*, it belongs in YAML.

---

## Deep Merge Algorithm

The `_deep_merge(base, override)` function recursively merges dictionaries:

```python
# base = {"models": {"classification": {"standard": {"hp": {"n_est": 100, "lr": 0.1}}}}}
# override = {"models": {"classification": {"standard": {"hp": {"n_est": 200}}}}}
# result = {"models": {"classification": {"standard": {"hp": {"n_est": 200, "lr": 0.1}}}}}
```

Key behavior: nested dicts are merged recursively, but scalar values and lists are replaced entirely. So overriding `kmer_sizes: [5, 6]` replaces the whole list — it doesn't append to `[3, 4]`. This is intentional: if you want different k-mers, you want to replace, not accumulate.

---

## How Each Module Uses Defaults

### `preprocess.py`
```python
cfg = get_preprocessing_defaults(profile.modality.value, profile.dataset_path)
seq_cfg = cfg.get("sequence", {})
kmer_sizes = seq_cfg.get("kmer_sizes", [3, 4])
```

Each preprocessor calls `get_preprocessing_defaults()` once at the top, then reads specific keys with sensible fallbacks. The fallbacks match the YAML defaults — this means the code works even if the YAML file is missing.

### `registry.py`
```python
tier_defaults = get_model_defaults(task_family, modality=..., dataset_path=...)
for tier in ("trivial", "simple", "standard"):
    td = tier_defaults[tier]
    configs.append(ModelConfig(name=td["name"], ...))
```

Instead of hardcoded `ModelConfig(hyperparameters={"max_iter": 1000, ...})`, the registry reads the YAML and builds configs dynamically. Adding a new model tier is now just adding a YAML block.

### `auto_config.py`
```python
eval_cfg = get_evaluation_defaults(modality, dataset_path)
imbalance_threshold = eval_cfg.get("imbalance_threshold_pct", 5)
```

Metric selection logic stays in Python (the conditional branching), but the actual metric names and thresholds come from YAML.

---

## Test Results

Both datasets produce identical results to the hardcoded version (except expression dataset, which now uses the modality override of `n_estimators=200, max_depth=8` — a deliberate change for cell expression data).

---

## API Key Configuration

The Anthropic API key can be configured in two ways:

1. **Environment variable:** `ANTHROPIC_API_KEY` (existing method)
2. **config.yaml file:** Set the `anthropic_api_key` field in `config.yaml`

The environment variable takes precedence if both are set. The `config.yaml` approach is useful for persistent local setups where exporting an environment variable each session is inconvenient.
