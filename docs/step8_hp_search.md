# Step 8: Hyperparameter Search (Optuna) — Detailed Walkthrough

## Overview

Step 8 answers: **"Can we do better than default hyperparameters?"**

After Step 3 identifies the best baseline model (usually XGBoost with default HPs), Step 8 runs Bayesian hyperparameter optimization using Optuna to find a better configuration. This is the single biggest lever for improving model performance without changing the model architecture or features.

---

## Why Bayesian Optimization, Not Grid Search?

Three common HP search strategies:

1. **Grid search** — try every combination. For 8 hyperparameters with 5 values each, that's 5^8 = 390,625 trials. Impossible in our time budget.

2. **Random search** — sample randomly from ranges. Surprisingly effective (Bergstra & Bengio, 2012 showed it outperforms grid search). But it doesn't learn from past trials.

3. **Bayesian optimization (TPE)** — Optuna's default. Builds a probabilistic model of "which HP regions give good results" and focuses sampling there. After 10-15 trials it starts converging on good regions. In 30 trials it typically finds configurations that grid search would need hundreds of trials to reach.

We use **TPE (Tree-structured Parzen Estimator)** — Optuna's default sampler. It models the distribution of "good" and "bad" hyperparameters separately, then samples from regions where the good/bad ratio is highest.

---

## Architecture

```
co_scientist/modeling/hp_search.py   ← Optuna search loop
co_scientist/defaults.yaml           ← search spaces + config
```

### Search Spaces (from YAML)

```yaml
hp_search:
  n_trials: 30
  timeout_seconds: 180
  sampler: tpe

  search_spaces:
    xgboost:
      n_estimators:     { type: int,   low: 50,   high: 500 }
      max_depth:        { type: int,   low: 3,    high: 10 }
      learning_rate:    { type: float, low: 0.01, high: 0.3,  log: true }
      subsample:        { type: float, low: 0.6,  high: 1.0 }
      colsample_bytree: { type: float, low: 0.5,  high: 1.0 }
      min_child_weight: { type: int,   low: 1,    high: 10 }
      reg_alpha:        { type: float, low: 0.0,  high: 1.0 }
      reg_lambda:       { type: float, low: 0.5,  high: 5.0 }

    logistic_regression:
      C:        { type: float, low: 0.001, high: 100.0, log: true }
      max_iter: { type: int,   low: 500,   high: 3000 }

    ridge_regression:
      alpha: { type: float, low: 0.001, high: 100.0, log: true }
```

### Why These Specific Ranges?

**XGBoost hyperparameters explained:**

| Parameter | Range | Why |
|-----------|-------|-----|
| `n_estimators` | 50–500 | More trees = more capacity, but slower. 500 is generous for small bio datasets. |
| `max_depth` | 3–10 | Controls tree complexity. Deeper trees capture more interactions but overfit faster. |
| `learning_rate` | 0.01–0.3 (log) | Log-scale because the effect is multiplicative. Lower rates need more trees. |
| `subsample` | 0.6–1.0 | Row sampling per tree. < 0.6 loses too much data; 1.0 = no sampling. |
| `colsample_bytree` | 0.5–1.0 | Column sampling per tree. Regularization via feature dropout. |
| `min_child_weight` | 1–10 | Minimum samples per leaf. Higher = more conservative, prevents overfitting rare classes. |
| `reg_alpha` | 0.0–1.0 | L1 regularization. Encourages sparse feature usage. |
| `reg_lambda` | 0.5–5.0 | L2 regularization. Prevents any single tree from dominating. |

**Log-scale for `learning_rate` and `C`:** These parameters span orders of magnitude. Linear sampling would waste most trials in the upper range. Log-scale samples uniformly across magnitudes (0.01, 0.03, 0.1, 0.3 get equal attention).

---

## How the Search Works

### The Objective Function

Each Optuna trial:
1. **Samples** hyperparameters from the search space (TPE-guided after initial random trials)
2. **Builds** a model with those HPs
3. **Trains** on `X_train, y_train`
4. **Evaluates** on `X_val, y_val` using the primary metric
5. **Returns** the metric value (Optuna maximizes or minimizes based on the metric)

```python
def objective(trial):
    hp = sample_hyperparameters(trial, search_space)
    model = build_model(config_with_hp)
    model.fit(X_train, y_train)
    result = evaluate_on_val(model)
    return result.primary_metric_value
```

### Error Handling

If a trial crashes (e.g., a bad HP combination causes numerical instability), it returns the worst possible score (`-inf` for maximize, `+inf` for minimize) instead of killing the whole search. Optuna learns to avoid that region.

### After Search Completes

1. Extract the best trial's hyperparameters
2. **Retrain** the model with those HPs on the training set (the trial's model was already trained, but we retrain cleanly to get timing info)
3. **Evaluate** on validation set
4. **Compare** to the baseline: if the tuned model improves, it replaces the best model; otherwise we keep the baseline

This comparison is important — HP search can sometimes overfit to the validation set, especially with small datasets. If tuning doesn't help, the safer choice is the default.

---

## Integration with the Pipeline

The search runs as **Step 3b** — between baseline evaluation and model export:

```
Step 3:  Train baselines → Evaluate → Identify best baseline
Step 3b: HP search on best baseline → Compare → Update best if improved
Step 4:  Export best model (baseline or tuned)
Step 6:  Generate report (includes tuned model in results)
```

The tuned model gets a "tuned" tier label, so it shows up distinctly in the results table and report. It's added to the results list so the report shows all models including the tuned one.

---

## Configuration

All search parameters are in `defaults.yaml` under `hp_search:`:

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `enabled` | `true` | Master switch for HP search |
| `n_trials` | `30` | Number of Optuna trials |
| `timeout_seconds` | `180` | Max wall-clock time (whichever limit hits first) |
| `sampler` | `tpe` | Optuna sampler (tpe or random) |

The search only runs if:
1. `hp_search.enabled` is `true`
2. A search space is defined for the best model's `model_type`
3. The best baseline is not a trivial model (no point tuning a DummyClassifier)

---

## Test Results

### RNA/translation_efficiency_muscle (regression)
- Baseline XGBoost: Spearman = **0.6279**
- After 30 Optuna trials (30s): Spearman = **0.7044**
- **+12% improvement** — the tuned model found that lower learning rate + more trees + regularization helps

### expression/cell_type_classification_segerstolpe (classification)
- Baseline XGBoost: macro_f1 = **0.8594**
- After 30 Optuna trials (223s): macro_f1 = **0.9620**
- **+12% improvement** — tuning especially helped with rare cell types (macro F1 is sensitive to minority classes)

---

## Design Decisions

### Why 30 trials?

Empirically a good balance for TPE on 8-dimensional search spaces. The first ~10 trials are essentially random (TPE needs data to build its model). Trials 10-30 are where Bayesian optimization starts converging. More trials give diminishing returns unless the search space is very large.

### Why timeout of 180 seconds?

For small datasets (1K-10K samples), 30 XGBoost trials finish in 30-60 seconds. For larger datasets or more complex models, the timeout prevents runaway searches. The 3-minute default fits within the global pipeline deadline (30 min) while leaving time for other steps. Previously 300s, reduced to stay within the tighter time budget.

### Why search the best baseline, not all models?

Time efficiency. Searching 3 models × 30 trials = 90 model fits. Since the standard tier (XGBoost) almost always wins the baseline comparison, we only tune the winner. If the simple model won the baseline, we'd tune *that* instead — which is the right choice (the data is linearly separable, so tuning XGBoost would be wasted effort).

### Why not tune preprocessing?

Preprocessing search (k-mer sizes, HVG counts, scaling methods) is a much bigger search space because it requires re-preprocessing the entire dataset for each trial. That's Phase C territory — where the LLM agents can reason about "should I try 5-mers?" and make informed decisions rather than brute-forcing.
