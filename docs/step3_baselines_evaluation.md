# Step 3: Baselines + Evaluation — Detailed Walkthrough

## Overview

Step 3 answers: **"How hard is this problem, and how well can simple approaches do?"**

This is the first time we actually train ML models. We train three tiers of baselines — trivial, simple, and standard — and evaluate them on the validation set. This establishes:
1. A **floor** (how well can you do by guessing?)
2. A **signal check** (do the features contain any useful information?)
3. A **competitive reference** (how well does a solid off-the-shelf model do?)

These baselines are critical benchmarks. Every future model in the pipeline must beat the standard baseline to justify its additional complexity.

---

## Part A: Evaluation Auto-Configuration (`auto_config.py`)

### The problem: which metric should we optimize?

Different ML tasks need different metrics. Accuracy is meaningless for imbalanced classification. MSE is meaningless for comparing models on different scales. We need to pick the right primary metric automatically.

### Metric selection logic (architecture Section 8.1)

```python
def auto_eval_config(profile: DatasetProfile) -> EvalConfig:
```

The rules:

| Task type | Condition | Primary metric | Why |
|---|---|---|---|
| Binary classification | — | AUROC | Threshold-independent; works even with class imbalance |
| Multi-class, balanced | All classes ≥ 5% | Accuracy | Simple, interpretable, all classes matter equally |
| Multi-class, imbalanced | Any class < 5% | Macro F1 | Averages per-class F1; prevents ignoring rare classes |
| Regression | — | Spearman correlation | Measures monotonic relationship; robust to scale/outliers |

**Why Spearman over Pearson for regression?** Spearman measures rank correlation — if the model correctly ranks samples from lowest to highest, Spearman is high even if the absolute values are off. This is more useful in biology because:
- We often care about relative rankings (which genes are most/least efficient?)
- Biological measurements have noise; exact values are less reliable than orderings
- Spearman is robust to outliers and non-linear relationships

**Why macro F1 for imbalanced classification?** Consider Segerstolpe with 13 cell types. If the model predicts everything as "alpha cell" (the most common), it gets ~40% accuracy. That looks terrible. But if we use per-class accuracy, it gets 100% on alpha cells and 0% on everything else. Macro F1:
- Computes F1 for each class independently
- Averages them equally (a class with 4 samples weighs the same as one with 870)
- Forces the model to perform well on *every* class, not just common ones

### The `EvalConfig` object

```python
EvalConfig(
    task_type="regression",
    primary_metric="spearman",
    secondary_metrics=["pearson", "mse", "rmse", "mae", "r2"],
)
```

Secondary metrics are computed for completeness (they appear in the results table and final report) but model selection decisions are based solely on the primary metric.

---

## Part B: Model Registry (`registry.py`)

### The three baseline tiers

The registry defines what models to train at each tier. This is the "LLM-as-strategist" principle from the architecture — the code defines *what's available*; later (Phase C), the LLM decides *what to try*. For baselines, the choices are fixed:

#### Tier 1: Trivial baseline — the floor

**Classification: `DummyClassifier(strategy="most_frequent")`**
Always predicts the most common class. For Segerstolpe: always predicts "alpha cell".
- Expected accuracy: ~40% (proportion of the majority class)
- Expected macro F1: ~0.044 (1/13 × F1 for one class, 0 for all others)
- Any real model **must** beat this. If it doesn't, something is broken.

**Regression: `DummyRegressor(strategy="mean")`**
Always predicts the training set mean. For RNA translation efficiency: always predicts -0.255.
- Expected Spearman: 0.0 (constant predictions have no correlation)
- Expected MSE: ~variance of the target (≈2.13)
- This establishes the "no-information" baseline — predicting the average.

**Why bother training a trivial baseline?** It seems useless, but it serves two purposes:
1. **Sanity check:** If a "real" model performs *worse* than this, we have a bug (wrong labels, data leakage from the wrong direction, etc.)
2. **Baseline for the report:** The report says "XGBoost improves X% over trivial baseline" — this contextualizes the result for readers.

#### Tier 2: Simple baseline — the signal check

**Classification: `LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")`**
A linear model that fits a hyperplane separating classes in feature space.

- `max_iter=1000`: increases from default 100 because high-dimensional data (2000 genes) can take more iterations to converge
- `C=1.0`: regularization strength (default). Lower C = more regularization = simpler decision boundary
- `solver="lbfgs"`: efficient for multi-class; uses L2 penalty

**Why logistic regression as the simple baseline?**
- If logistic regression gets high accuracy, the features are linearly separable (good preprocessing!)
- If it does poorly, the problem needs non-linear models
- It's fast, interpretable, and well-understood
- For Segerstolpe: 0.804 macro F1 — the features clearly have strong signal

**Regression: `Ridge(alpha=1.0)`**
Linear regression with L2 regularization.

- `alpha=1.0`: regularization strength. Prevents overfitting on high-dimensional data.
- Ridge is preferred over plain linear regression because with 326 features and 1000 samples, we're in the "more features than expected by a simple model" regime

**Why not Lasso?** Ridge (L2) shrinks all coefficients; Lasso (L1) sets some to zero. For k-mer features where many features contribute small amounts, Ridge is more appropriate — we don't expect a sparse solution.

For RNA: 0.406 Spearman — k-mer features have some linear signal, but there's room for improvement.

#### Tier 3: Standard baseline — the competitive reference

**Both tasks: XGBoost with defaults**
```python
XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
```

XGBoost is a gradient-boosted tree ensemble. It builds 100 decision trees sequentially, where each tree corrects the errors of the previous ones.

**Why XGBoost as the standard baseline?**
- Consistently wins or ties in tabular/structured data competitions
- Handles non-linear relationships (unlike logistic/ridge)
- Handles feature interactions automatically (tree splits can capture "if GC content > 0.5 AND 3-mer AAA > 0.1")
- Robust to feature scaling (trees only care about rank order)
- Fast to train
- Low hyperparameter sensitivity (defaults are good)

**Key hyperparameters:**
- `n_estimators=100`: number of trees. Default is reasonable; more trees = more capacity but diminishing returns
- `max_depth=6`: maximum tree depth. Controls model complexity. Deeper = more capacity = more overfitting risk
- `learning_rate=0.1`: shrinkage factor. Each tree's contribution is multiplied by this. Lower = more trees needed but less overfitting

**Results:**
- RNA: 0.628 Spearman (55% improvement over ridge) — XGBoost captures non-linear k-mer interactions
- Segerstolpe: 0.962 macro F1 (20% improvement over logistic regression) — XGBoost handles the class imbalance better

#### Tier 4: Advanced baseline — deep tabular models

FT-Transformer is now part of the advanced model tier. This deep learning architecture for tabular data can capture complex feature interactions that tree-based models may miss. It sits alongside MLP and bio_cnn as an optional advanced baseline that the pipeline can train when the compute budget allows. (TabNet was removed from the pipeline due to excessive CPU training time.)

### The `build_model` function

```python
def build_model(config: ModelConfig) -> Any:
    builders = {
        "majority_class": _build_majority_class,
        "logistic_regression": _build_logistic_regression,
        ...
    }
```

A dispatcher pattern: model type → builder function. Each builder imports its library lazily (XGBoost only imported when needed) and returns an sklearn-compatible object with `fit()` and `predict()` methods.

**Why sklearn-compatible?** The trainer (`trainer.py`) calls `model.fit(X_train, y_train)` and `model.predict(X_val)` regardless of the model type. This uniform interface means adding a new model is just adding a new builder function — no changes to training/evaluation code.

---

## Part C: Training (`trainer.py`)

### The training loop

```python
def train_model(config: ModelConfig, split: SplitData) -> TrainedModel:
    model = build_model(config)
    model.fit(split.X_train, split.y_train)
    return TrainedModel(config=config, model=model, train_time_seconds=elapsed)
```

This is intentionally simple:
1. Build the model from config
2. Fit on training data
3. Time it
4. Return the trained model

**What `model.fit(X_train, y_train)` does internally depends on the model:**

- **DummyClassifier:** Finds the most frequent class in `y_train`. Done.
- **LogisticRegression:** Runs L-BFGS optimization to find weights `w` that minimize cross-entropy loss + L2 penalty. For 2000 features and 13 classes, this is a (2000 × 13) weight matrix.
- **Ridge:** Solves the closed-form solution `w = (X'X + αI)^{-1} X'y`. For 326 features this is near-instant.
- **XGBoost:** Iteratively builds 100 trees using gradient boosting. Each tree fits the negative gradient of the loss function on the residuals of the current ensemble.

### `TrainedModel` wrapper

Wraps the sklearn model with:
- `config`: records what model this is and its hyperparameters (for the report)
- `model`: the actual trained object
- `train_time_seconds`: for the results table
- `predict(X)`: delegates to `model.predict(X)`
- `predict_proba(X)`: delegates to `model.predict_proba(X)` if available (needed for AUROC)

---

## Part D: Metrics (`metrics.py`)

### Classification metrics

**Accuracy** (`accuracy_score`):
Fraction of correct predictions. Simple but misleading for imbalanced data.
- Segerstolpe majority class: 0.408 (only predicts "alpha cell")
- Segerstolpe XGBoost: 0.981

**Macro F1** (`f1_score(average="macro")`):
Compute F1 for each class, then average. F1 = harmonic mean of precision and recall.
- "Macro" means each class contributes equally regardless of size
- A model that ignores rare classes gets low macro F1 even if accuracy is high

**Weighted F1** (`f1_score(average="weighted")`):
Like macro F1 but weighted by class size. More lenient toward ignoring rare classes.

**AUROC** (`roc_auc_score`):
Area Under the ROC Curve. Measures how well the model *ranks* positive examples above negative ones, across all classification thresholds.
- Needs probability predictions (`predict_proba`)
- For multi-class: computed as one-vs-rest and macro-averaged
- Not all models produce calibrated probabilities (XGBoost does; dummy doesn't)

### Regression metrics

**Spearman correlation** (`spearmanr`):
Correlation between ranks. If the model perfectly orders samples from lowest to highest predicted value, Spearman = 1.0, even if the absolute predictions are wrong.
- Range: [-1, 1]. 0 = no correlation.
- Our primary metric for regression.

**Pearson correlation** (`pearsonr`):
Linear correlation between predicted and actual values. Assumes a linear relationship; sensitive to outliers.
- RNA XGBoost: Pearson 0.682 > Spearman 0.628 — the relationship is somewhat linear

**MSE / RMSE** (`mean_squared_error`):
Mean squared error and its square root. Heavily penalizes large errors.
- RNA mean predictor MSE: 2.11 (≈ variance of y)
- RNA XGBoost MSE: 1.10 (48% reduction — explains ~48% of variance)

**MAE** (`mean_absolute_error`):
Mean absolute error. Less sensitive to outliers than MSE.

**R²** (`r2_score`):
Fraction of variance explained. R² = 1 − MSE/Var(y).
- R² = 0: model is as good as predicting the mean
- R² = 1: perfect predictions
- R² < 0: worse than the mean (broken model)

### Handling edge cases

**Constant predictions (lines 113–119):**
The mean predictor always outputs the same value. Spearman/Pearson are undefined for constant input. We check `np.std(y_pred) > 1e-12` and return 0.0 if predictions are constant.

**Missing classes in AUROC (line 82–87):**
If a model never predicts certain classes (common with trivial baselines), AUROC computation can fail. We wrap it in try/except.

---

## Part E: Results Table (`print_results_table`)

```
                              Model Comparison
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ Model            ┃ Tier     ┃ spearman ┃ pearson ┃    mse ┃   rmse ┃ Time ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ xgboost_default  │ standard │   0.6279 │  0.6816 │ 1.1035 │ 1.0505 │ 0.3s │
│ ridge_regression │ simple   │   0.4058 │  0.3685 │ 2.2056 │ 1.4851 │ 0.0s │
│ mean_predictor   │ trivial  │   0.0000 │  0.0000 │ 2.1062 │ 1.4513 │ 0.0s │
└──────────────────┴──────────┴──────────┴─────────┴────────┴────────┴──────┘
```

Models are sorted by primary metric (best first). This gives an immediate visual of the baseline progression: trivial → simple → standard, with clear improvement at each tier.

---

## What the results tell us

### RNA translation efficiency

| Model | Spearman | Interpretation |
|---|---|---|
| Mean predictor | 0.000 | No signal (by definition) |
| Ridge regression | 0.406 | K-mer features have moderate linear signal |
| XGBoost | 0.628 | Non-linear interactions improve substantially |

**Key insight:** The 55% jump from ridge to XGBoost means there are important non-linear patterns — likely interactions between k-mers (e.g., a specific combination of codon usage patterns, not just individual codons). This suggests hyperparameter tuning and feature engineering could push further.

### Cell type classification (Segerstolpe)

| Model | Macro F1 | Interpretation |
|---|---|---|
| Majority class | 0.044 | Floor (13 classes, so random ≈ 1/13 ≈ 0.077) |
| Logistic regression | 0.804 | Strong linear separability — HVG features are excellent |
| XGBoost | 0.962 | Near-perfect; non-linear boundaries help rare classes |

**Key insight:** Logistic regression already gets 0.804 — the log1p + HVG preprocessing produces features where cell types are mostly linearly separable. XGBoost's improvement comes from better handling of rare classes (epsilon, mast cells) that sit in non-linear pockets of gene expression space. 0.962 macro F1 is very strong — there may not be much room for improvement.

---

## Engineering notes

### Why evaluation happens on validation set, never test

```python
result = evaluate_model(trained, split, eval_config, use_test=False)
```

The `use_test=False` flag is explicit and intentional. Test set evaluation is gated behind `use_test=True` which will only be called once, in Step 5 (Phase 5: Analyze). All model selection, hyperparameter tuning, and iteration decisions use validation performance.

### Held-out test-set evaluation after model selection

After the best model is selected based on validation performance, a final held-out test-set evaluation step runs automatically. Both validation and test metrics are reported side by side, and the report includes a **Validation vs Test comparison table** so readers can assess generalization at a glance. This ensures that the winning model's performance is confirmed on data it has never seen during training or model selection.

### Why we compute all metrics even though only one is "primary"

Secondary metrics appear in the report and help diagnose issues:
- High accuracy but low macro F1 → model ignoring rare classes
- High Spearman but high MSE → good ranking but poor absolute predictions
- Large gap between train and val metrics → overfitting (not computed yet; coming in Step 11)

### File structure after Step 3

```
co_scientist/
├── data/
│   ├── types.py          # DatasetProfile, LoadedDataset, SplitData, etc.
│   ├── loader.py         # HuggingFace loading
│   ├── preprocess.py     # Modality-specific transforms
│   └── split.py          # Train/val/test splitting
├── modeling/
│   ├── types.py          # ModelConfig, TrainedModel
│   ├── registry.py       # Model definitions + build_model()
│   └── trainer.py        # train_model(), train_baselines()
├── evaluation/
│   ├── types.py          # EvalConfig, ModelResult
│   ├── auto_config.py    # Metric selection from profile
│   └── metrics.py        # Metric computation + results table
├── cli.py                # Pipeline orchestration
└── config.py             # RunConfig
```

This is the **first working pipeline** — data goes in, a trained model and metrics come out.
