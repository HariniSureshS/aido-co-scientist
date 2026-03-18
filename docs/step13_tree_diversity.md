# Step 13: Tree Model Diversity — Detailed Walkthrough

## Overview

Step 13 expands the model registry from a single tree model (XGBoost) to three diverse tree ensembles plus Elastic Net, providing real model diversity at the standard and simple tiers. This implements Architecture Section 7.1.

**Before:** 4 models (mean/ridge/xgboost/mlp) — XGBoost doing all the heavy lifting.
**After:** 8 base models across 4 tiers with 3 different tree families + 2 linear approaches.

---

## Why Model Diversity Matters

Different model families have different inductive biases:

| Model | Inductive Bias | Strength |
|-------|---------------|----------|
| **XGBoost** | Sequential boosting, additive corrections | Best on structured/tabular, handles missing data |
| **LightGBM** | Histogram-based, leaf-wise growth | Faster training, better on high-cardinality features |
| **Random Forest** | Bagging, independent trees | Less prone to overfitting, better OOB estimates |
| **Elastic Net** | L1+L2 regularization | Feature selection, handles multicollinearity in k-mer features |

On the RNA dataset, Random Forest (0.6941) beat XGBoost (0.6279) by 10% — showing that model diversity directly improves results.

---

## Multi-Model YAML Structure

The defaults YAML now supports **lists of models per tier**, not just a single model:

```yaml
# Before: single model per tier
models:
  regression:
    standard:
      name: xgboost_default
      model_type: xgboost
      hyperparameters: { ... }

# After: multiple models per tier
models:
  regression:
    standard:
      - name: xgboost_default
        model_type: xgboost
        hyperparameters: { ... }
      - name: lightgbm_default
        model_type: lightgbm
        hyperparameters: { ... }
      - name: random_forest
        model_type: random_forest
        hyperparameters: { ... }
```

The registry's `get_baseline_configs()` detects whether a tier value is a dict (single model) or list (multiple models) and handles both.

---

## New Models

### Random Forest
- **sklearn** `RandomForestClassifier` / `RandomForestRegressor`
- Default: 200 trees, unlimited depth, min 2 samples per leaf
- **Key advantage:** Bagging reduces variance. Independent trees mean individual tree errors cancel out rather than compound (unlike boosting).
- HP search space: n_estimators [50-500], max_depth [3-20], min_samples_split [2-20], min_samples_leaf [1-10], max_features [sqrt, log2, None]

### LightGBM
- Microsoft's gradient boosting library
- Default: 100 trees, depth 6, learning rate 0.1, 31 leaves
- **Key advantage:** Histogram-based splitting is 5-10x faster than XGBoost on high-dimensional data. Leaf-wise growth can find better splits.
- HP search space: same as XGBoost plus num_leaves [15-127] and min_child_samples [5-50]
- `verbose: -1` suppresses LightGBM's chatty output

### Elastic Net (Classification)
- **sklearn** `LogisticRegression` with `penalty="elasticnet"` and `solver="saga"`
- Combines L1 (sparsity) and L2 (regularization) penalties
- **Key advantage:** Performs feature selection (L1 zeros out irrelevant k-mers) while keeping correlated features stable (L2)
- Good for high-dimensional k-mer features where many k-mers are irrelevant

### Elastic Net (Regression)
- **sklearn** `ElasticNet`
- Default: alpha=0.1, l1_ratio=0.5
- Same L1+L2 benefit as classification version

---

## Registry Changes

### Seed Injection
Models that accept `random_state` are tracked in `_SEED_MODELS`:

```python
_SEED_MODELS = {"xgboost", "lightgbm", "random_forest", "mlp", "bio_cnn"}
```

The registry automatically injects the pipeline seed into these models for reproducibility.

### Builder Functions
Each model type has a builder that instantiates the sklearn-compatible object:

```python
builders = {
    "majority_class":       _build_majority_class,
    "mean_predictor":       _build_mean_predictor,
    "logistic_regression":  _build_logistic_regression,
    "ridge_regression":     _build_ridge_regression,
    "elastic_net_clf":      _build_elastic_net_clf,
    "elastic_net_reg":      _build_elastic_net_reg,
    "xgboost":              _build_xgboost,
    "lightgbm":             _build_lightgbm,
    "random_forest":        _build_random_forest,
    "mlp":                  _build_mlp,
    "bio_cnn":              _build_bio_cnn,
}
```

---

## Resilience Fallbacks

Each new model type has fallback configurations for error recovery:

| Error Type | LightGBM Fallback | Random Forest Fallback |
|-----------|-------------------|----------------------|
| MemoryError | n_estimators: 50, max_depth: 4, num_leaves: 15 | n_estimators: 50, max_depth: 6 |
| ValueError | reg_alpha: 1.0, reg_lambda: 5.0 | min_samples_leaf: 5, max_depth: 8 |
| Default | n_estimators: 50, max_depth: 3, num_leaves: 15 | n_estimators: 50, max_depth: 5 |

---

## Guardrail Updates

- `_CLASSIFICATION_MODELS` now includes `elastic_net_clf`
- `_REGRESSION_MODELS` now includes `elastic_net_reg`
- `_EITHER_MODELS` now includes `lightgbm` and `random_forest`
- Parameter estimation (`_estimate_model_params`) recognizes elastic net models
- Tree ensembles return `None` for parameter estimation (overfitting controlled by tree constraints, not parameter count)

---

## HP Search Spaces

All new models have defined Optuna search spaces in `defaults.yaml`:

```yaml
hp_search:
  search_spaces:
    lightgbm:
      n_estimators:  { type: int,   low: 50,    high: 500 }
      max_depth:     { type: int,   low: 3,     high: 10 }
      learning_rate: { type: float, low: 0.01,  high: 0.3,  log: true }
      num_leaves:    { type: int,   low: 15,    high: 127 }
      # ... plus subsample, colsample, regularization

    random_forest:
      n_estimators:  { type: int,   low: 50,    high: 500 }
      max_depth:     { type: int,   low: 3,     high: 20 }
      max_features:  { type: categorical, choices: ["sqrt", "log2", null] }
      # ... plus min_samples_split, min_samples_leaf

    elastic_net_clf:
      C:             { type: float, low: 0.001, high: 100.0, log: true }
      l1_ratio:      { type: float, low: 0.1,   high: 0.9 }

    elastic_net_reg:
      alpha:         { type: float, low: 0.001, high: 100.0, log: true }
      l1_ratio:      { type: float, low: 0.1,   high: 0.9 }
```

The HP search sampler was also fixed to handle `categorical` parameters correctly (they don't have `low`/`high` fields).

---

## Dependencies

Added `lightgbm>=4.0.0` to `pyproject.toml`. Random Forest and Elastic Net use sklearn (already a dependency).

---

## Results

### RNA/translation_efficiency_muscle (regression)

| Model | Spearman | Notes |
|-------|----------|-------|
| random_forest | **0.6941** | Best — bagging wins on this dataset |
| lightgbm | 0.6604 | Strong second |
| xgboost | 0.6279 | Previous best |
| mlp | 0.6229 | |
| ridge | 0.4058 | |
| elastic_net | 0.3913 | Sparse solution, decent for linear |
| mean_predictor | 0.0000 | Floor |

### expression/cell_type_classification_segerstolpe (classification)

| Model | Macro F1 | Notes |
|-------|----------|-------|
| lightgbm | **0.9106** | Best baseline (tuned to 0.9706) |
| xgboost | 0.8594 | |
| logistic_regression | 0.8041 | |
| mlp | 0.6820 | |
| random_forest | 0.6125 | |
| elastic_net_clf | 0.0445 | Collapsed — high-dim expression data |

---

## File Changes

```
co_scientist/
├── modeling/
│   └── registry.py      ← multi-model tier support, new builders
├── defaults.yaml        ← lists per tier, new HP search spaces
├── resilience.py        ← fallback configs for new models
├── guardrails.py        ← updated model type sets
└── pyproject.toml       ← lightgbm dependency
```
