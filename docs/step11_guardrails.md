# Step 11: Guardrails — Detailed Walkthrough

## Overview

Guardrails are automated plausibility checks that catch data quality problems, model failures, and scientific methodology issues. They implement **Architecture Section 10.3** — the full set of deterministic scientific decision guardrails.

The design is inspired by two publications:
- **Sakana AI Scientist** (Section 2.2): "Our guardrail system serves a similar role to their automated peer review, catching errors before they reach the final output"
- **CellAgent** (Section 2.3): Error-feedback patterns — when something goes wrong, the system captures structured information about *what* failed and *why*, enabling intelligent recovery (deterministic now, LLM-driven in Phase C)

---

## Guardrail Categories (from Architecture Section 10.3)

The architecture defines five categories. All are now implemented:

### 1. Dataset Validation (before modeling)
### 2. Task Type Verification (multi-signal)
### 3. Metric Sanity
### 4. Model-Data Compatibility (before training)
### 5. Result Plausibility (after training)

Each category runs at a specific point in the pipeline, ordered from earliest to latest.

---

## Category 1: Dataset Validation (Pre-training Checks)

Runs after Step 2 (preprocess + split), before Step 3 (baselines). Validates that the data is ready for modeling.

| Check | Severity | Condition | Why |
|-------|----------|-----------|-----|
| Profiler criticals | ERROR | Any CRITICAL issue from Step 1 profiler | Data didn't load or is fundamentally broken |
| Tiny training set | ERROR | < 30 training samples | Not enough data for train/val/test to be meaningful |
| High dimensional | WARNING | More features than samples | Severe overfitting risk; regularization essential |
| NaN in features | WARNING | Any NaN after preprocessing | Preprocessing missed something; models may crash |
| Inf in features | ERROR | Any infinite values | Guaranteed model failures |
| Constant target | ERROR | Training target std ≈ 0 | Nothing to learn — likely a data loading bug |
| All-zero features | ERROR | Entire feature matrix is 0 | Preprocessing likely stripped all information |

If any ERROR fires, the pipeline halts with exit code 1. Better to fail fast than train on broken data.

---

## Category 2: Task Type Verification (Multi-Signal)

Runs after Step 1 (profiling). Architecture Section 10.3: *"Multi-signal confirmation: path parsing + target analysis + value counts + predefined defaults + LLM confirmation. Require 2+ signals to agree."*

The deterministic version uses three signals:

| Signal | Classification indicator | Regression indicator |
|--------|------------------------|---------------------|
| **Path hint** | "classification", "cell_type" in path | "regression", "efficiency", "expression" in path |
| **Value counts** | ≤ 50 unique target values | Continuous with many unique values |
| **Distribution** | Class distribution present | Target stats (mean/std) present |

The check counts how many signals agree with the detected task type:
- **2+ signals agree** → high confidence, no alert
- **Only 1 signal** → INFO: low confidence
- **Signals disagree** → WARNING: task type conflict detected

The LLM confirmation signal (4th signal from architecture) is added in Phase C.

### Example

For `RNA/translation_efficiency_muscle`:
- Path hint: "efficiency" → regression ✓
- Value counts: many unique floats → regression ✓
- Target stats: mean=-0.25, std=1.46 → regression ✓
- **3 signals agree**: no alert needed

---

## Category 3: Metric Sanity

Runs at the start of Step 3, right after `auto_eval_config()` selects the metric. Catches metric-task mismatches per Section 10.3.

| Check | Severity | Condition | Why |
|-------|----------|-----------|-----|
| Classification metric on regression | ERROR | e.g., accuracy/F1 used for regression task | Metric is undefined for continuous targets |
| Regression metric on classification | ERROR | e.g., MSE/spearman used for classification | Metric doesn't capture classification quality |
| Accuracy on imbalanced data | WARNING | accuracy chosen but smallest class < 5% | Accuracy is misleading when one class dominates. Suggests macro_f1 |
| AUROC on many classes | INFO | AUROC with > 20 classes | OvR AUROC becomes unstable with many classes |

### Why this matters

The auto_eval_config system already picks good metrics, but guardrails catch cases where:
- YAML overrides introduce a mismatch (e.g., someone puts `primary: accuracy` in a regression dataset override)
- A future LLM agent overrides the metric incorrectly
- The task type detection was wrong and the metric followed

---

## Category 4: Model-Data Compatibility

Runs after config generation, before each model trains. Filters out incompatible models and warns about risky configurations.

| Check | Severity | Condition | Why |
|-------|----------|-----------|-----|
| Classification model on regression | ERROR (BLOCK) | e.g., majority_class/logistic on regression data | Model will crash or produce nonsense |
| Regression model on classification | ERROR (BLOCK) | e.g., mean_predictor/ridge on classification data | Model will crash or produce nonsense |
| Wrong num_classes | ERROR (BLOCK) | Config says N classes but data has M | Training will fail or waste a class slot |
| More params than samples | WARNING | Estimated parameters > training samples | High overfitting risk — may still work with regularization |
| FM modality mismatch | ERROR (BLOCK) | Foundation model backbone modality ≠ data modality | Wrong embeddings would be meaningless |

### Parameter estimation

For the "more params than samples" check, we estimate learnable parameters per model type:

- **Logistic regression**: `n_features × n_classes + n_classes`
- **Ridge regression**: `n_features + 1`
- **MLP**: sum of `(input × output + bias + batch_norm)` per layer
- **XGBoost**: skipped (ensemble of trees, not straightforward to count)

### Blocking behavior

Models that trigger ERROR-level compatibility alerts are **skipped** — they're removed from the baseline list before training. The pipeline continues with the remaining compatible models.

### Real-world output

For the expression dataset (2000 HVG features, 1279 training samples):
```
Model-data compatibility: 0 error(s), 2 warning(s), 0 info
  WARNING logistic_regression: estimated 26,013 parameters > 1,279 training samples
  WARNING mlp: estimated 547,597 parameters > 1,279 training samples
```

Both still train (warnings don't block), but the information surfaces overfitting risk before training even starts.

---

## Category 5: Result Plausibility (Post-training Checks)

Runs after each model is evaluated on the validation set. Catches problems that only become visible through model behavior.

### Per-model checks

| Check | Severity | Condition | Why |
|-------|----------|-----------|-----|
| Perfect score | WARNING | Primary metric = 1.0 (higher-is-better) or 0.0 (lower-is-better) | Possible data leakage — target information leaked into features |
| Worse than trivial | WARNING | Model score < 90% of trivial baseline | Something is broken — a real model should beat random/mean prediction |
| Model collapsed | ERROR | All predictions identical (std < 1e-12) | Model learned nothing; outputs a constant regardless of input. Skipped for trivial-tier models (mean/majority predictors are *supposed* to be constant) |
| Overfitting (severe) | WARNING | Train-val gap > 0.3 | Model memorized training data; likely won't generalize |
| Overfitting (moderate) | INFO | Train-val gap > 0.15 | Some overfitting but may still be useful |
| Negative R² | WARNING | R² < -0.5 | Model is substantially worse than predicting the mean |

### Pipeline-level checks

Run once after all baselines, looking at cross-model patterns:

| Check | Severity | Condition | Why |
|-------|----------|-----------|-----|
| All models tied | WARNING | All models have identical primary metric (std < 1e-6) | Something wrong with evaluation or data |
| No improvement over trivial | WARNING | No non-trivial model beats the trivial baseline | Features may not be predictive of the target |

---

## Train-val Gap Calculation

The overfitting check computes the primary metric on both training and validation sets:

```python
y_pred_train = model.predict(X_train)
train_metric = compute_metric(y_train, y_pred_train)
gap = abs(train_metric - val_metric)
```

This works for any metric. A gap > 0.3 means the model loses more than 30 percentage points when moving from seen to unseen data.

### Why overfitting is expected on these datasets

For the RNA dataset, all models show train-val gaps around 0.33-0.36:
- K-mer features capture local sequence patterns that models can memorize
- XGBoost with default depth=6 has enough capacity to partially memorize 1000 training samples
- The HP search (Step 8) addresses this by tuning regularization parameters

The guardrail doesn't block — it surfaces information for the user or the Phase C ML Engineer agent.

---

## Severity Levels

| Level | Display | Effect |
|-------|---------|--------|
| ERROR | Bold red | Blocks pipeline (pre-training, metric sanity) or skips model (compatibility) |
| WARNING | Yellow | Surfaced to user; logged for agent reasoning |
| INFO | Dim | Informational; no action needed |

---

## When Each Check Runs

```
Step 1 (Profile)     → Task type verification (multi-signal)
Step 2 (Split)       → Dataset validation (pre-training checks)
Step 3 (Baselines)   → Metric sanity → Model-data compatibility → [training] → Result plausibility
```

---

## Integration with Experiment Log

Every guardrail alert is logged as a `guardrail` event:

```json
{"timestamp": "...", "elapsed_seconds": 10.85, "event": "guardrail",
 "data": {"severity": "warning", "code": "overfitting",
          "message": "xgboost_default: train-val gap = 0.357 ..."}}
```

This enables:
- **Phase C agents** to read alerts and adjust strategy
- **Report generation** to include model health information
- **Post-hoc analysis** to compare guardrail patterns across datasets

---

## Connection to Publications

### From Sakana AI Scientist (Section 2.2)
The guardrail system serves the same role as Sakana's automated peer review — a quality gate that catches errors before they reach the output. Sakana applies this to paper review; we apply it to model validation. The key insight: automated quality checks at multiple stages catch more issues than a single final review.

### From CellAgent (Section 2.3)
CellAgent's error-feedback self-correction pattern: when something produces implausible results, capture structured information (what failed, the exact error/metric, context) and feed it back. Our guardrail alerts are this structured feedback — each alert has a severity, code, and message that Phase C agents can parse and act on. The deterministic version logs the alerts; the LLM version (Phase C) will use them to drive intelligent recovery.

### Novel: Trivial Baseline as Sanity Check
The "worse than trivial" check uses the trivial baseline (mean predictor / majority class) as a scientific sanity check. This isn't from any paper — it's a core scientific discipline principle. If a sophisticated model can't beat predicting the mean, either the features don't contain signal or something is fundamentally wrong.

---

## What the Guardrails Enable for Phase C

The agent layer (Steps 13-19) will use guardrail alerts as input signals:

- **Data Analyst agent**: reads pre-training alerts → adjusts preprocessing (e.g., dimensionality reduction for high-dimensional data)
- **ML Engineer agent**: reads post-training alerts → adjusts model configuration (e.g., reduce max_depth for overfitting, increase regularization for high params-to-samples ratio)
- **Biology Specialist agent**: reads "worse than trivial" → questions whether features capture biological signal
- **Coordinator**: reads metric sanity alerts → can override metric choice

Without guardrails, agents would need to independently discover these issues, wasting LLM calls and iteration budget.

---

## File Structure

```
co_scientist/
└── guardrails.py        ← all checks, severity types, display helpers
```

The module exports:
- `verify_task_type(profile)` → task type multi-signal verification
- `check_metric_sanity(profile, eval_config)` → metric-task compatibility
- `check_model_data_compatibility(config, profile, split)` → per-model compatibility
- `check_pre_training(profile, split)` → dataset validation
- `check_post_training(result, trained, split, eval_config, trivial_result)` → result plausibility
- `check_pipeline_summary(results, eval_config)` → cross-model patterns
- `print_alerts(alerts, title)` → rich-formatted console output
- `has_blocking_errors(alerts)` → bool for pipeline halt decision
