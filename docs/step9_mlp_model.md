# Step 9: Advanced Models (MLP) — Detailed Walkthrough

## Overview

Step 9 answers: **"Can a neural network outperform tree-based models on biological data?"**

This adds a PyTorch MLP (Multi-Layer Perceptron) as the "advanced" tier in the baseline progression: trivial → simple → standard → **advanced**. The MLP provides a different inductive bias than XGBoost — it learns smooth, continuous decision boundaries rather than axis-aligned splits, which can capture different patterns in the data.

---

## Architecture

```
co_scientist/modeling/
├── mlp.py          ← MLPClassifier + MLPRegressor (sklearn-compatible)
├── registry.py     ← updated: builds MLP from YAML config
├── hp_search.py    ← updated: MLP search space with n_layers/hidden_size
└── types.py        ← unchanged
```

### The MLP Design

```
Input (326 features)
  → Linear(326, 256) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(256, 128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(128, 1)   [regression] or Linear(128, n_classes) [classification]
```

**Why this specific architecture?**

- **BatchNorm after each linear layer:** Stabilizes training by normalizing activations. Especially important for biological data where feature scales vary wildly (k-mer frequencies ~0.001-0.01, sequence length ~50-200).
- **ReLU activation:** Simple, fast, avoids vanishing gradients. No need for fancier activations on tabular data.
- **Dropout(0.3):** Prevents overfitting on small datasets (1K-10K samples is small for neural nets).
- **[256, 128] hidden dims:** Decreasing layer widths create an information bottleneck that forces the network to learn compressed representations.

### Sklearn-Compatible Interface

Both `MLPClassifier` and `MLPRegressor` implement:
- `fit(X, y)` — trains the network with early stopping
- `predict(X)` — returns predictions
- `predict_proba(X)` — returns class probabilities (classifier only)

This means the rest of the pipeline (trainer, evaluator, exporter, HP search) works with MLPs exactly as it does with XGBoost or sklearn models. No special-casing needed.

---

## Training Details

### Early Stopping

The MLP holds out 10% of the training data as an internal validation set. After each epoch, it evaluates validation loss. If validation loss doesn't improve for `patience` epochs (default 10), training stops and the best weights are restored.

**Why internal validation, not the pipeline's val set?** Using the pipeline's validation set for early stopping would leak information — the same data would influence both training (when to stop) and evaluation (reporting metrics). The internal 10% split keeps evaluation clean.

### Manual Batching (No DataLoader)

We use a simple `_iter_batches()` function instead of PyTorch's DataLoader:

```python
def _iter_batches(X, y, batch_size, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield X[idx[start:start+batch_size]], y[idx[start:start+batch_size]]
```

**Why not DataLoader?** XGBoost uses OpenMP for parallel tree building. On macOS, XGBoost's OpenMP thread pool deadlocks with PyTorch's DataLoader threading when both run in the same process. Manual batching avoids this entirely — it's single-threaded and equally fast for small tabular data.

### Thread Safety

The MLP module sets `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` at import time. This prevents the XGBoost/PyTorch OpenMP deadlock. The performance impact is negligible — for datasets under 100K samples, single-threaded PyTorch is already fast enough (1-5 seconds per MLP training run).

---

## Optuna Search Space for MLP

```yaml
mlp:
  dropout:       { type: float, low: 0.1,   high: 0.5 }
  learning_rate: { type: float, low: 0.0001, high: 0.01, log: true }
  weight_decay:  { type: float, low: 0.00001, high: 0.01, log: true }
  batch_size:    { type: categorical, choices: [32, 64, 128, 256] }
  n_layers:      { type: int,   low: 1,     high: 3 }
  hidden_size:   { type: categorical, choices: [64, 128, 256, 512] }
```

**Special handling:** `n_layers` and `hidden_size` are search parameters but the MLP expects `hidden_dims` (a list). The HP search post-processes: `n_layers=3, hidden_size=512` → `hidden_dims=[512, 256, 128]` (decreasing by half per layer).

---

## Results

### RNA/translation_efficiency_muscle (regression)

| Model | Tier | Spearman |
|-------|------|----------|
| mean_predictor | trivial | 0.0000 |
| ridge_regression | simple | 0.4058 |
| xgboost_default | standard | 0.6279 |
| **mlp** | **advanced** | **0.6229** |
| xgboost_tuned | tuned | 0.7044 |

MLP nearly matches XGBoost (0.62 vs 0.63). On small tabular/k-mer data, tree-based models and MLPs are competitive — neither has a strong structural advantage.

### expression/cell_type_classification_segerstolpe (classification)

| Model | Tier | macro_f1 |
|-------|------|----------|
| majority_class | trivial | 0.0445 |
| logistic_regression | simple | 0.8041 |
| **mlp** | **advanced** | **0.6820** |
| xgboost_default | standard | 0.8594 |

MLP underperforms here (0.68 vs 0.86). This is expected:
- **13 classes, severe imbalance** (smallest class: 5 samples). The MLP's internal 10% validation split may not contain any samples from rare classes, making early stopping unreliable.
- **2000 features, 1279 training samples.** Neural nets typically need more data per feature than tree-based models. XGBoost's built-in feature selection (via splits) handles high-dimensional sparse data better.
- **Small dataset.** MLPs shine with >10K samples. At ~1K, the regularization (dropout, weight decay) fights against the model having enough capacity to learn.

---

## Design Decisions

### Why not skip MLP if XGBoost usually wins?

1. **Not always true.** On some datasets (especially with continuous features and smooth targets), MLPs outperform tree models.
2. **Different inductive bias = more information.** If MLP and XGBoost agree, we're more confident. If they disagree, it tells us something about the problem structure.
3. **Phase C benefit.** The LLM agents can analyze MLP vs XGBoost performance to decide whether to invest in deeper neural architectures.

### Why not use a more complex architecture (CNN, Transformer)?

This is the "advanced" tier, not the "foundation model" tier. The MLP is the simplest neural architecture that adds value over tree-based models. CNNs and Transformers for sequence data are Phase D territory and require significantly more training time and data.

### Why max_epochs=50 instead of 100?

Early stopping typically triggers within 15-25 epochs on small datasets. 50 epochs provides enough headroom without wasting time. The patience=10 setting means training stops if no improvement for 10 epochs — so effective training is usually 15-30 epochs.
