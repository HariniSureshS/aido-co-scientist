# Phase 1: Frozen Embeddings (embed_xgboost, embed_mlp)

## Idea

The simplest way to use a foundation model: extract embeddings once and train standard downstream models on them. The AIDO backbone is **frozen** — no gradients flow through it. This is fast, deterministic, and cacheable.

## Why This Works

Biological foundation models learn general-purpose representations during pre-training. For example, AIDO.RNA-1.6B was trained on millions of RNA sequences to predict masked tokens — in doing so, it learns that:
- Certain codon patterns correlate with translation efficiency
- UTR structures affect mRNA stability
- Sequence context matters beyond what a 4-mer can capture

By mean-pooling the hidden states, we compress this knowledge into a fixed-size vector per sequence. A simple XGBoost or MLP can then learn the mapping from this rich representation to the target variable.

## Technical Implementation

### Embedding Extraction (`modeling/foundation.py`)

```python
# GPU detection
gpu_available() -> bool  # torch.cuda.is_available()

# Modality → AIDO model mapping (from defaults.yaml)
get_foundation_model_name("rna") -> "aido_rna_1b600m"

# Extraction with caching
extract_embeddings(sequences, modality, config, cache_dir)
  → loads model: Embed.from_config({"model.backbone": name}).eval()
  → tokenizes: model.transform({"sequences": batch})
  → batched forward pass on tokenized tensors (batch_size=32)
  → mean-pool over sequence dimension
  → returns np.ndarray (n_samples, embed_dim)
  → caches to disk as .npy (keyed by data hash)
```

### Data Flow

```
preprocess.py::preprocess()
    ├── existing k-mer/HVG preprocessing → X (handcrafted)
    └── _maybe_extract_embeddings()      → X_embed (or None)
            ↓
PreprocessingResult(X=X, X_embed=X_embed)
            ↓
split.py: split X_embed in parallel with X using same indices
            ↓
SplitData(X_train, X_embed_train, X_val, X_embed_val, ...)
            ↓
trainer.py: route embedding models to X_embed_train
            ↓
evaluate_model: route embedding models to X_embed_val
```

### Models

| Model | Builder | Downstream Model | Input |
|-------|---------|-----------------|-------|
| `embed_xgboost` | `_build_embed_xgboost` | XGBClassifier/Regressor | `X_embed_train` |
| `embed_mlp` | `_build_embed_mlp` | MLPClassifier/Regressor | `X_embed_train` |

Both use the same underlying models as their standard counterparts — the only difference is the input features.

### Routing

The trainer identifies embedding models via `_EMBEDDING_MODELS = {"embed_xgboost", "embed_mlp"}`:

```python
if config.model_type in _EMBEDDING_MODELS:
    if split.X_embed_train is None:
        return None  # skip gracefully
    model.fit(split.X_embed_train, y_train)
```

The evaluator checks `trained.needs_embeddings` to route val/test data.

### Embedding Caching

Embeddings are cached to `{tempdir}/co_scientist_embeddings/embeddings_{modality}_{model}_{hash}.npy`. The hash is computed from the first 100 sequences + total count, so:
- Same data → cache hit (instant)
- Different data → cache miss (re-extract)
- Pipeline re-runs → cached (saves 8-15 min per dataset)

### Graceful Degradation

1. No GPU → `should_use_foundation()` returns False → X_embed stays None → embed_* models skipped
2. GPU but no `modelgenerator` → `extract_embeddings()` catches ImportError → returns None
3. GPU but model fails → `extract_embeddings()` catches all exceptions → returns None, warning logged
4. In all failure cases, the standard CPU pipeline runs unchanged

### Benefits

- **Fast**: Extract once, train many models on the embeddings
- **Cacheable**: Embeddings saved to disk, reused across pipeline iterations
- **Low risk**: If anything fails, CPU pipeline is unaffected
- **Competitive**: AIDO embeddings + XGBoost often beats handcrafted features + XGBoost by 5-15%

### Limitations

- Embeddings are **frozen** — they don't adapt to the specific task
- Mean-pooling discards position-specific information
- For small datasets, the high-dimensional embeddings may overfit (XGBoost regularization helps)

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/modeling/foundation.py` | **New** — GPU detection, AIDO loading, embedding extraction, caching |
| `co_scientist/data/types.py` | Added `X_embed` to `PreprocessingResult`, `X_embed_train/val/test` to `SplitData` |
| `co_scientist/data/preprocess.py` | Added `_maybe_extract_embeddings()` after existing preprocessing |
| `co_scientist/data/split.py` | All 3 split strategies propagate `X_embed` through splits |
| `co_scientist/modeling/registry.py` | Added `embed_xgboost`, `embed_mlp` builders; foundation tier in `get_baseline_configs` |
| `co_scientist/modeling/trainer.py` | Embedding model routing in `train_model()` |
| `co_scientist/modeling/types.py` | Added `_EMBEDDING_MODELS` set, `needs_embeddings` property |
| `co_scientist/evaluation/metrics.py` | Routes embedding models to `X_embed_val/test` |
| `co_scientist/defaults.yaml` | Foundation tier configs + HP search spaces + `foundation_models` section |
| `co_scientist/export/exporter.py` | Foundation-aware `train.py`/`predict.py` with inline embedding extraction |
