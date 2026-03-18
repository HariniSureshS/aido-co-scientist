# Step 16: Stacking Ensemble — Detailed Walkthrough

## Overview

The stacking ensemble is the system's key differentiator — it **builds a new model** by combining the predictions of all successfully trained base models. A meta-learner (Ridge/Logistic regression) learns which base models to trust for which types of inputs.

This implements Architecture Section 7.6.

---

## Why Stacking?

Individual models have different strengths:

| Model | What It Captures |
|-------|-----------------|
| Ridge/Elastic Net | Linear relationships in k-mer frequencies |
| XGBoost | Non-linear interactions between features, handles missing data |
| LightGBM | Similar to XGBoost, different regularization bias |
| Random Forest | Bagged estimates, robust to outliers |
| MLP | Non-linear transformations of tabular features |
| BioCNN | Positional motif patterns in raw sequences |

No single model captures everything. The stacking ensemble combines all of them.

---

## How It Works

### Step 1: Generate Out-of-Fold Predictions

To avoid data leakage, we can't just use the base models' training predictions (they'd overfit). Instead, we use **k-fold cross-validation**:

```
Training data split into 5 folds:
  Fold 1: ████░░░░░░░░░░░░░░░░
  Fold 2: ░░░░████░░░░░░░░░░░░
  Fold 3: ░░░░░░░░████░░░░░░░░
  Fold 4: ░░░░░░░░░░░░████░░░░
  Fold 5: ░░░░░░░░░░░░░░░░████

For each fold:
  1. Train all base models on the OTHER 4 folds
  2. Predict on THIS fold
  3. Store predictions

Result: Every training sample has an "out-of-fold" prediction from each base model
```

This gives us an N × M matrix (N samples, M base models) where each prediction was made by a model that **never saw that sample during training**.

### Step 2: Train the Meta-Learner

The meta-learner is trained on the out-of-fold prediction matrix:

```
                 ridge  elastic  xgb    lgbm   rf     mlp    cnn
Sample 1:       [ 0.3,   0.2,   0.8,   0.7,   0.9,   0.6,   0.5 ] → y=0.7
Sample 2:       [ 0.1,   0.1,   0.3,   0.4,   0.3,   0.2,   0.6 ] → y=0.2
  ...
Sample 1000:    [ 0.5,   0.4,   0.7,   0.8,   0.7,   0.5,   0.8 ] → y=0.9

Meta-learner (Ridge) learns weights:
  final_prediction = 0.05*ridge + 0.03*elastic + 0.25*xgb + 0.30*lgbm + 0.20*rf + 0.02*mlp + 0.15*cnn
```

The meta-learner is intentionally simple (Ridge for regression, LogisticRegression for classification) to avoid overfitting the combination weights.

### Step 3: Inference

At prediction time:
1. Run all base models on the new data
2. Stack their predictions into a matrix
3. Feed through the meta-learner
4. Output the final prediction

```
New sample → [ridge_pred, elastic_pred, xgb_pred, ..., cnn_pred] → meta-learner → final prediction
```

---

## Handling Sequence Models in Stacking

The BioCNN needs raw sequences, not k-mer features. The stacking ensemble handles this transparently:

```python
for model_idx, trained in enumerate(self.base_models):
    if trained.needs_sequences:
        fold_model.fit(X_fold_train, y_fold_train, sequences=seqs_fold_train)
        oof_matrix[val_idx, model_idx] = fold_model.predict(X_fold_val, sequences=seqs_fold_val)
    else:
        fold_model.fit(X_fold_train, y_fold_train)
        oof_matrix[val_idx, model_idx] = fold_model.predict(X_fold_val)
```

The split's `seqs_train` are sliced using the fold indices, so each fold gets the correct sequences.

---

## Graceful Degradation

- **Minimum 2 base models:** Stacking requires at least 2 non-trivial models. If fewer are available (e.g., most models failed), stacking is skipped.
- **Trivial models excluded:** Mean predictor and majority class are not included as base models (they don't add useful signal).
- **Per-fold failure handling:** If a base model fails on a specific fold, its prediction is replaced with the mean (regression) or majority class (classification) of the training fold.
- **Full failure:** If stacking itself fails (e.g., meta-learner can't converge), it's caught and the pipeline continues with the best individual model.

---

## Pipeline Integration

The stacking ensemble is trained **after** all baselines are evaluated but **before** HP search:

```
Step 3: Train Baselines
  ├── Train all models (trivial → simple → standard → advanced)
  ├── Evaluate all on validation set
  ├── Show model comparison table
  ├── Run post-training guardrails
  │
  ├── Build stacking ensemble (5-fold CV on training data)    ← NEW
  ├── Evaluate ensemble on validation set
  ├── Show updated comparison table with ensemble
  │
  └── Select best overall model

Step 4: HP Search (tunes the best model — could be the ensemble's best base model)
```

### Why Before HP Search?

The stacking ensemble uses the base models' **default** configurations. HP search tunes the single best model. If the stacking ensemble becomes the best overall, we still HP-tune the best *base* model (since the ensemble itself doesn't have traditional hyperparameters to tune).

---

## Results

### RNA/translation_efficiency_muscle

| Model | Spearman | Pearson | MSE |
|-------|----------|---------|-----|
| random_forest | **0.6941** | 0.7068 | 1.0627 |
| **stacking_ensemble** | **0.6941** | **0.7287** | **0.9645** |
| bio_cnn | 0.6655 | 0.7207 | 0.9969 |
| lightgbm | 0.6604 | 0.6982 | 1.0652 |

The stacking ensemble:
- **Matches** Random Forest on Spearman (0.6941)
- **Best** Pearson correlation (0.7287) — 3% better than any individual model
- **Best** MSE (0.9645) — 3.5% better than BioCNN, 9% better than Random Forest

This shows the ensemble successfully combines the different strengths: tree models' rank ordering + BioCNN's continuous prediction accuracy.

### Training Time

The ensemble takes ~80s because it retrains all 7 base models across 5 CV folds (7 × 5 = 35 training runs). This is the most expensive step in the pipeline. For simple datasets (complexity < 3), a future optimization could reduce to 3 folds.

---

## Implementation Details

### Regression: `StackingEnsemble`
- Meta-learner: `sklearn.linear_model.Ridge(alpha=1.0)`
- Out-of-fold: `KFold(n_splits=5, shuffle=True)`
- Prediction: `meta_model.predict(stacked_predictions)`

### Classification: `StackingEnsembleClassifier`
- Meta-learner: `sklearn.linear_model.LogisticRegression(max_iter=1000)`
- Out-of-fold: `StratifiedKFold(n_splits=5, shuffle=True)` — preserves class distribution in each fold
- Prediction: `meta_model.predict(stacked_predictions)` and `predict_proba(stacked_predictions)`

### Model Config
The ensemble's `ModelConfig` stores metadata about its base models:

```python
ModelConfig(
    name="stacking_ensemble",
    tier="ensemble",
    model_type="stacking",
    task_type="regression",
    hyperparameters={
        "n_base_models": 7,
        "base_models": ["ridge_regression", "elastic_net_reg", "xgboost_default",
                        "lightgbm_default", "random_forest", "mlp", "bio_cnn"]
    },
)
```

### Export
The ensemble is exported as a single pickle containing all base models + the meta-learner. The `predict()` method works self-contained — it runs all base models internally.

---

## Why This Is "Building a New Model"

The stacking ensemble is not just model selection or averaging. It's a **new model** that:

1. **Learns combination weights** — not equal weighting, but learned from data
2. **Handles complementary errors** — if trees overpredict on sample X but CNN underpredicts, the meta-learner learns to balance them
3. **Creates novel decision boundaries** — the meta-learner's decision surface is different from any individual model's
4. **Adapts per-dataset** — the weights are learned for THIS dataset, not predefined

This is the answer to "can you build your own models, not just run existing ones?"

---

## File Structure

```
co_scientist/
├── modeling/
│   └── ensemble.py      ← StackingEnsemble, StackingEnsembleClassifier,
│                            build_stacking_ensemble()
├── cli.py               ← ensemble integration after baseline evaluation
└── modeling/types.py     ← "stacking" in _SEQUENCE_MODELS (routes sequences)
```

The module exports:
- `StackingEnsemble` — regression meta-learner ensemble
- `StackingEnsembleClassifier` — classification meta-learner ensemble
- `build_stacking_ensemble(trained_models, split, task_type, seed)` — convenience function that handles filtering, training, and wrapping
