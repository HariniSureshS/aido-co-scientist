# Phase 2: End-to-End Fine-Tuning (aido_finetune)

## Idea

Frozen embeddings are a strong baseline, but they can't adapt to the specific task. Fine-tuning unfreezes the last N layers of the AIDO backbone, allowing the model to **specialize its representations** for the target variable. Combined with a task-specific head, this is typically the strongest single-model approach.

## Why Fine-Tuning Outperforms Frozen Embeddings

Pre-training teaches the model general biological language. But each downstream task has its own signal:
- **Translation efficiency**: The model needs to attend to codon optimality, UTR structure, and ribosome stalling patterns
- **Cell type classification**: The model needs to focus on marker genes and cell-type-specific expression patterns
- **Protein function**: The model needs to capture domain architecture and active site motifs

By unfreezing the last few layers, the backbone can shift its attention to the features most relevant to the task — while still leveraging the vast knowledge from pre-training in the frozen lower layers.

## Technical Implementation

### Architecture (`modeling/aido_finetune.py`)

```
AIDOFinetuneRegressor / AIDOFinetuneClassifier
    │
    ├── _AIDOFinetuneModel (nn.Module)
    │       ├── embed_model (AIDO backbone, partially frozen)
    │       └── head (_TaskHead: Linear → ReLU → Dropout → Linear)
    │
    ├── fit(X, y, sequences=...)
    │       ├── Load AIDO backbone via modelgenerator.tasks.Embed
    │       ├── Probe embedding dimension
    │       ├── Freeze all → unfreeze last N layers
    │       ├── AdamW optimizer (differential LR: backbone=2e-5, head=2e-4)
    │       ├── Mixed precision (torch.amp.autocast + GradScaler)
    │       ├── Training loop with gradient clipping
    │       ├── Early stopping on validation loss (10% train carved for val)
    │       ├── Restore best checkpoint
    │       └── Move model to CPU, free GPU memory
    │
    ├── predict(X, sequences=...)
    │       ├── Move to GPU, forward pass in batches
    │       └── Move back to CPU, return numpy
    │
    └── predict_proba(X, sequences=...)  [classifier only]
```

### Key Design Decisions

**1. Differential Learning Rates**
The backbone uses 10x lower learning rate than the head. This prevents catastrophic forgetting — the backbone weights shift slowly while the head adapts quickly. This is standard practice for fine-tuning pre-trained models.

**2. Mixed Precision Training**
`torch.amp.autocast("cuda")` + `GradScaler` reduces memory by ~40% and speeds training by ~2x on the H100. Essential when the backbone is 1B+ parameters.

**3. Gradient Clipping**
`nn.utils.clip_grad_norm_(params, max_norm=1.0)` prevents exploding gradients that can destabilize fine-tuning, especially in the early epochs.

**4. Internal Validation Split**
The model carves 10% of training data for early stopping. This avoids overfitting — fine-tuning large models on small biological datasets is prone to it.

**5. Memory Management**
After training, the model is moved to CPU and GPU memory is cleared. For prediction, it temporarily moves back to GPU. This ensures the H100's memory is available for other models in the pipeline.

### Layer Unfreezing Strategy

The `_unfreeze_last_n_layers` function searches for transformer layers using common naming patterns:
- `encoder.layer` (BERT-style)
- `layers` (generic)
- `transformer.layer`
- `blocks` (vision transformer style)

If no recognizable layer structure is found, it falls back to unfreezing the last 25% of parameters by name.

### Sequence Model Routing

`aido_finetune` is registered as a **sequence model** (like `bio_cnn`), not an embedding model. This means:
- The trainer passes `sequences=split.seqs_train` to `fit()`
- The evaluator passes `sequences=split.seqs_val` to `predict()`
- It processes raw sequences directly — no pre-extracted embeddings needed

### Export

Fine-tuned models can't be pickled (too large, custom modules). Instead:
- `best_model.pt`: `torch.save()` with head state_dict + full model state_dict
- `train.py`: Standalone script with inline AIDO loading, layer unfreezing, training loop
- `predict.py`: Standalone script that reconstructs model from checkpoint

### HP Search

The HP search space for `aido_finetune` (in `defaults.yaml`):
```yaml
aido_finetune:
  unfreeze_layers: { type: int, low: 1, high: 4 }
  head_hidden:     { type: categorical, choices: [128, 256, 512] }
  head_dropout:    { type: float, low: 0.1, high: 0.5 }
  learning_rate:   { type: float, low: 5e-6, high: 1e-4, log: true }
  weight_decay:    { type: float, low: 0.001, high: 0.1, log: true }
  batch_size:      { type: categorical, choices: [8, 16, 32] }
```

## When Fine-Tuning Wins vs Embeddings

| Condition | Best Approach | Why |
|-----------|--------------|-----|
| Small dataset (<500 samples) | embed_xgboost | XGBoost regularization prevents overfitting on limited data |
| Medium dataset (500-5000) | concat_xgboost or aido_finetune | Both competitive; concat is faster, finetune adapts better |
| Large dataset (>5000) | aido_finetune | Enough data to learn task-specific representations |
| High-dimensional expression | embed_mlp | Fine-tuning cell expression models is less mature |
| Sequence data + clear motifs | aido_finetune | Model can learn to attend to task-relevant motifs |

## Files Created/Modified

| File | Change |
|------|--------|
| `co_scientist/modeling/aido_finetune.py` | **New** — AIDOFinetuneClassifier/Regressor with sklearn interface |
| `co_scientist/modeling/registry.py` | Added `_build_aido_finetune`, AIDO model name injection |
| `co_scientist/modeling/types.py` | Added `aido_finetune` to `_SEQUENCE_MODELS` |
| `co_scientist/modeling/trainer.py` | Added `aido_finetune` to `_SEQUENCE_MODELS` |
| `co_scientist/modeling/hp_search.py` | Added `aido_finetune` to sequence models + HP search space |
| `co_scientist/defaults.yaml` | Foundation tier config + HP search space |
| `co_scientist/export/exporter.py` | state_dict export, standalone train.py/predict.py templates |
