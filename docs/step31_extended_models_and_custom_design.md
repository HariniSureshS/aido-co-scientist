# Step 31: Extended Models & Custom Model Design

## Overview

Expands the model registry with 3 new model types (SVM, KNN, FT-Transformer) and adds a `design_model` tool that lets the ReAct agent **design custom PyTorch architectures at runtime** using the LLM.

**Problem:** The pipeline was limited to 11 model types. When standard models plateau, there's no way to try novel architectures tailored to the specific dataset. Users working on diverse biological datasets need more variety, and the agent should be able to innovate beyond a fixed menu.

**Solution:**
1. **3 new standard models** — SVM, KNN (sklearn, CPU-only), FT-Transformer (custom PyTorch)
2. **`design_model` tool** — LLM generates a full PyTorch model class based on dataset characteristics + architecture hint, code is validated and trained

---

## New Models

### Model Registry (14 total)

| Tier | Model | Library | GPU needed? |
|------|-------|---------|-------------|
| Trivial | majority_class | sklearn | No |
| Trivial | mean_predictor | sklearn | No |
| Simple | logistic_regression | sklearn | No |
| Simple | ridge_regression | sklearn | No |
| Simple | elastic_net | sklearn | No |
| Standard | xgboost | xgboost | No |
| Standard | lightgbm | lightgbm | No |
| Standard | random_forest | sklearn | No |
| **Standard** | **svm** | **sklearn** | **No** |
| **Standard** | **knn** | **sklearn** | **No** |
| Advanced | mlp | custom PyTorch | No (CPU) |
| Advanced | bio_cnn | custom PyTorch | No (CPU) |
| **Advanced** | **ft_transformer** | **custom PyTorch** | **No (CPU)** |
| Custom | LLM-designed | dynamic PyTorch | No (CPU) |

### SVM (Support Vector Machine)

```python
# Classification: SVC with probability=True (for predict_proba)
# Regression: SVR
# Default HP: kernel=rbf, C=1.0, gamma=scale
```

Good for small-to-medium datasets. Kernel trick captures non-linear relationships without feature engineering.

### KNN (K-Nearest Neighbors)

```python
# Classification: KNeighborsClassifier
# Regression: KNeighborsRegressor
# Default HP: n_neighbors=5, weights=uniform, metric=minkowski
```

Non-parametric baseline. Useful for datasets where local structure matters.

### FT-Transformer

```python
# Custom implementation in co_scientist/modeling/ft_transformer.py
# Feature Tokenizer: each feature → d_model embedding via nn.Linear
# [CLS] token prepended, processed by nn.TransformerEncoder
# Output: [CLS] representation → linear head
# Default HP: d_model=64, n_heads=4, n_layers=3, d_ff=128
```

Based on Gorishniy et al. (NeurIPS 2021). Tokenizes each numerical feature independently, then uses self-attention to capture feature interactions. Uses AdamW optimizer and early stopping, following the same pattern as `mlp.py`. Default `max_epochs` reduced from 100 to 30, `patience` from 10 to 3 (in `registry.py`).

---

## Custom Model Design (`design_model` tool)

### How It Works

```
Agent thinks: "Standard models plateau at 0.82. Let me try a custom architecture."
  │
  ▼
Agent calls: design_model({"architecture_hint": "Tabular ResNet with skip connections"})
  │
  ▼
┌─────────────────────────────────────────────┐
│ 1. Build prompt with dataset characteristics │
│    (n_features, n_samples, modality, etc.)   │
│                                              │
│ 2. LLM generates PyTorch model class         │
│                                              │
│ 3. AST validation                            │
│    - Syntax check                            │
│    - Block disallowed imports (os, sys, etc.) │
│    - Block disallowed builtins (exec, eval)   │
│    - Verify fit/predict methods exist         │
│                                              │
│ 4. Dynamic loading via exec()                │
│    - Create synthetic module                  │
│    - Instantiate model class                  │
│    - Verify sklearn interface                 │
│                                              │
│ 5. Train on X_train/y_train                  │
│                                              │
│ 6. Evaluate on validation set                │
└─────────────────────────────────────────────┘
  │
  ▼
Observation: "Designed and trained custom_tabresnet_a3f2: spearman=0.8500 (12.3s)"
```

### Tool Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `architecture_hint` | Yes | Natural language description of desired architecture |
| `hyperparameters` | No | Optional HP overrides passed to the generated model |

### Code Validation (Safety)

Generated code is validated via AST analysis before execution:

| Check | What it blocks |
|-------|---------------|
| Disallowed imports | `os`, `subprocess`, `sys`, `shutil`, `pathlib`, `socket`, `http`, `urllib`, `requests` |
| Disallowed builtins | `exec()`, `eval()`, `__import__()`, `compile()`, `open()` |
| Class required | Code must define at least one class |
| Methods required | Class must have `fit()` and `predict()` methods |
| Allowed: `model.eval()` | Calls to `model.eval()` (PyTorch inference mode) are permitted and not blocked by the AST validator. The disallowed `eval()` check applies only to the built-in `eval()` function, not to method calls on objects. |

**Note:** This is a basic safety net for a research tool, not a full sandbox. The generated code runs in the same process.

### LLM Prompt

The prompt (`CUSTOM_MODEL_PROMPT`) includes:
- Dataset characteristics (n_features, n_samples, n_classes, modality, metric)
- Architecture hint from the agent
- Requirements: sklearn interface, PyTorch only, early stopping, reproducibility
- Output format: single Python class in a code fence

### Example Architectures the LLM Might Generate

| Hint | What the LLM generates |
|------|----------------------|
| "ResNet-style with skip connections" | Tabular ResNet with residual blocks + batch norm |
| "Attention over codon features" | Multi-head attention layer focusing on sequence-derived features |
| "Dual pathway for numeric and categorical" | Two-branch network merging at a fusion layer |
| "Custom loss combining MSE and correlation" | Model with a Pearson correlation loss component |

### Integration with ReAct Agent

The `design_model` tool is registered in both `build_default_registry()` and `build_tree_search_registry()`. The agent is guided to use it **after standard models have been tried** (costs an extra LLM call).

In tree search mode, the agent is prompted to dedicate a branch to custom model exploration.

**HP tuning:** Custom models cannot be tuned via `tune_hyperparameters` (the code is ephemeral). Instead, the agent calls `design_model` again with a revised hint or different hyperparameters.

**Checkpoint pickling:** Custom (LLM-designed) models cannot be pickled because they are dynamically loaded via `exec()`. Checkpoint saving now catches `PicklingError` and saves state without unpicklable model objects. Results and scores are always preserved even if model objects cannot be serialized.

### Deterministic Model Selection

The ML Engineer's deterministic model selection logic now includes the new models with sample-size thresholds:

- **FT-Transformer** is included for datasets with **>800 samples** (transformer architectures require more data to avoid overfitting)
- **KNN** and **SVM** are explicitly included in the ML Engineer's selection with reasoning (e.g., KNN for local-structure sensitivity, SVM with RBF kernel for non-linear decision boundaries on small-to-medium data)

Additionally, FT-Transformer is added to the **RNA modality override** in `defaults.yaml`, so RNA datasets automatically include this advanced model in the candidate set.

The ReAct agent's system prompt (`REACT_AGENT_SYSTEM`) explicitly mentions trying `ft_transformer` as part of the exploration strategy, encouraging the agent to evaluate it alongside tree-based and linear baselines.

---

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/modeling/ft_transformer.py` | FT-Transformer module + sklearn-compatible Classifier/Regressor wrappers |
| `co_scientist/modeling/custom_model.py` | `validate_model_code()`, `load_custom_model()`, `extract_code_from_response()`, `CUSTOM_MODEL_PROMPT` |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/modeling/registry.py` | Added builders for svm, knn, ft_transformer. Updated `_SEED_MODELS`. |
| `co_scientist/defaults.yaml` | Added default configs (standard: svm, knn; advanced: ft_transformer) and HP search spaces for all 3 |
| `co_scientist/agents/tools.py` | Added `DesignModelTool`. Updated `TrainModelTool` description with new model types. Registered in both registries. |
| `co_scientist/agents/react.py` | Added `llm_client: Any = None` to `ReactState` |
| `co_scientist/agents/coordinator.py` | Passes `self.client` as `llm_client` to `ReactState` |
| `co_scientist/llm/prompts.py` | Updated strategy guidelines in both `REACT_AGENT_SYSTEM` and `REACT_TREE_SEARCH_SYSTEM` to mention new models and `design_model` |

---

## HP Search Spaces

All 3 new models have Optuna search spaces in `defaults.yaml`:

```yaml
svm:
  C:       { type: float, low: 0.01, high: 100.0, log: true }
  gamma:   { type: categorical, choices: ["scale", "auto"] }
  kernel:  { type: categorical, choices: ["rbf", "linear", "poly"] }

knn:
  n_neighbors: { type: int, low: 3, high: 25 }
  weights:     { type: categorical, choices: ["uniform", "distance"] }
  metric:      { type: categorical, choices: ["minkowski", "euclidean", "manhattan"] }

ft_transformer:
  d_model:       { type: categorical, choices: [32, 64, 128] }
  n_heads:       { type: categorical, choices: [2, 4, 8] }
  n_layers:      { type: int, low: 1, high: 6 }
  dropout:       { type: float, low: 0.0, high: 0.5 }
  learning_rate: { type: float, low: 0.0001, high: 0.01, log: true }
  weight_decay:  { type: float, low: 0.00001, high: 0.01, log: true }
```

**Note:** `ft_transformer` requires `d_model` divisible by `n_heads`. The `_FTTransformerModule` auto-adjusts `d_model` if this constraint is violated.

---

## Verification

```bash
# Train new model types via ReAct agent:
co-scientist run RNA/translation_efficiency_muscle --budget 15

# The agent should now try svm, knn, ft_transformer in addition to tree models.
# With --tree-search, it may dedicate a branch to design_model.

co-scientist run RNA/translation_efficiency_muscle --tree-search --budget 15

# Check that custom models appear in the report:
grep "custom_" outputs/RNA__translation_efficiency_muscle/report.md
```

---

## Design Decisions

### Why SVM and KNN in the Standard tier, not Simple?

SVM with RBF kernel and KNN are non-linear models — they capture complex patterns that linear models cannot. They sit logically between linear baselines and tree ensembles.

### Why not deep copy custom model code for HP tuning?

Custom model code could be stored in `ModelConfig.hyperparameters["_code"]` to enable rebuilding for HP tuning. However, HP tuning for LLM-generated code is fragile — the code may have implicit dependencies between hyperparameters. Instead, the agent calls `design_model` again with a revised hint, which is more natural and produces better results.

### Why AST validation instead of a sandbox?

A proper sandbox (subprocess, Docker) adds complexity for minimal benefit in a research tool. AST validation blocks the most dangerous patterns (file access, network, code injection) while keeping the implementation simple. Users running this tool trust the LLM output to the same degree they trust the rest of the pipeline.

### Why FT-Transformer as custom code instead of a library?

No drop-in sklearn-compatible FT-Transformer library exists. The implementation follows the exact same pattern as `mlp.py` — a PyTorch module + sklearn wrapper with early stopping and manual batching. This avoids an external dependency and gives us full control over the training loop.
