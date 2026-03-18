# Step 6: Report Generation — Detailed Walkthrough

## Overview

Step 6 answers: **"What did the pipeline do, and what did it find?"**

This is the final step of Phase A — the deterministic pipeline. After this, every run produces a complete, self-contained output directory with a trained model, standalone code, figures, and now a comprehensive markdown report that ties it all together.

The report follows the template structure from ARCHITECTURE.md Section 12.2, adapted for the current Phase A scope (no LLM, no biological interpretation, no active learning — those come in Phase C).

---

## Architecture

```
co_scientist/report/
├── __init__.py
├── generator.py          ← main report generation logic
└── template.md.jinja    ← Jinja2 markdown template
```

**Why Jinja2 templating instead of f-strings?**

We used f-strings for code generation in Step 4 (train.py, predict.py), and it worked fine for small, structured Python files. But a report is different:

1. **Conditional sections:** Classification reports need a class distribution table; regression reports need target statistics. Jinja2's `{% if %}` blocks handle this cleanly.
2. **Loops:** Iterating over preprocessing steps, model results, and figure paths is natural in Jinja2 but ugly with f-strings.
3. **Separation of concerns:** The template defines _structure_ (what sections appear, in what order). The generator defines _content_ (what data goes into each section). Changing the report layout doesn't require touching Python code.
4. **Readability:** A markdown file with Jinja2 tags is readable as-is — you can see the report structure by opening the template. An f-string version would be an unreadable wall of Python.

---

## Report Sections

### 1. Executive Summary

A quick-reference table plus a one-paragraph summary. The table gives the key facts at a glance: dataset, modality, task type, sample count, feature count, classes (if classification), primary metric, best model, best score.

The paragraph is generated programmatically by `_build_executive_summary()`:
- States the dataset, task type, modality, and sample count
- Reports the best model's score and improvement over the trivial baseline
- Notes any detected data issues

**Why compare to the trivial baseline?** Because raw metric values are meaningless without context. A Spearman of 0.63 sounds mediocre — but if the trivial baseline (mean predictor) gets 0.00, that's a +0.63 improvement. The improvement tells you how much the model actually learned.

### 2. Dataset Profile

- **Overview:** Natural-language summary of the dataset (samples, features, sequence lengths, sparsity)
- **Class Distribution** (classification only): Table with class names, counts, and proportions
- **Target Statistics** (regression only): Mean, std, min, max, median
- **Detected Issues:** Any warnings from the profiling stage (class imbalance, high sparsity, etc.)
- **Profiling Figures:** Embedded images from `figures/01_profiling/`

### 3. Preprocessing

- **Steps Applied:** Numbered list of transforms (e.g., "3-mer frequencies", "log1p normalization")
- **Feature Representation:** Final dimensionality after preprocessing
- **Split Strategy:** Method used (predefined, fold-based, random) and sample counts per split
- **Preprocessing Figures:** Feature variance and split distribution verification

### 4. Model Development

- **4.0 Model Selection Strategy:** Explains _why_ the agent chose specific models for this dataset, based on dataset characteristics — modality (e.g., RNA-seq expression vs. DNA k-mers), sample size, and feature-space dimensionality. This section is generated from the agent's decision context so the reader understands the reasoning, not just the outcome.
- **Baseline Progression:** Table of all models with tier, primary metric, and training time, sorted by performance
- **"Why [model]?" subsection:** Appears immediately after the best model is identified. Explains why the winning model won: margin over the runner-up, comparison to the trivial baseline, suitability of the model type for the data characteristics, and the effect of hyperparameter tuning (tuned vs. default score delta).
- **"Why [metric]?" subsection:** Explains the rationale for the primary metric choice. Covers why the selected metric is appropriate for the task type and biological context (e.g., why Spearman correlation is preferred over MSE for translation efficiency prediction, or why macro F1 is used for imbalanced cell type classification). This helps domain scientists understand the evaluation criteria.
- **Validation vs Test Metrics (4.2):** A side-by-side table showing validation and held-out test scores for each metric of the best model. This replaces the old validation-only full metrics table and makes it easy to spot overfitting or distribution shift.
- **Full Metrics:** Complete metrics table for the best model (all primary + secondary metrics)
- **Training Figures:** Model comparison bar chart and feature importance

### 5. Reproducibility

Step-by-step commands to retrain, predict, and evaluate using the exported standalone code from Step 4. Also records the random seed, Python version, and pipeline version.

### 6. Appendix

- Full model configuration JSON (hyperparameters, model type, tier)
- List of all generated figures with their paths

---

## How the Template Rendering Works

The generator collects all data from the pipeline into a flat context dictionary and passes it to Jinja2:

```python
template.render(
    version=__version__,
    timestamp=...,
    profile=profile,           # DatasetProfile object
    eval_config=eval_config,   # EvalConfig object
    best_result=best_result,   # ModelResult object
    results_sorted=...,        # List[ModelResult] sorted by metric
    split=split,               # SplitData object
    split_sizes=split.summary(),
    preprocessing_steps=...,
    profiling_figures=...,
    preprocessing_figures=...,
    training_figures=...,
    test_metrics=...,          # Dict of held-out test set metrics (may be None)
    agent_decisions=...,       # Natural-language decision summaries
    ...
)
```

**Agent reasoning format:** Agent decisions (model selection rationale, preprocessing choices, etc.) are rendered as natural-language definition lists rather than raw JSON parameter dumps. The `_summarize_params()` helper converts internal state into readable text, e.g.:

```
Selected models
:   xgboost, lightgbm (+4 more). Priority: bio_cnn

Sample size assessment
:   1 284 samples — sufficient for tree ensembles, marginal for deep models
```

This makes the report accessible to domain scientists who should not need to parse Python dicts.

Jinja2 can access object attributes directly in the template (e.g., `{{ profile.dataset_name }}`), iterate over lists (`{% for r in results_sorted %}`), and conditionally include sections (`{% if profile.class_distribution %}`).

**Figure embedding:** Figures are embedded as standard markdown images with relative paths:
```markdown
![target_distribution](figures/01_profiling/target_distribution.png)
```
This works because the report lives at `outputs/DATASET/report.md` and figures are at `outputs/DATASET/figures/...` — the relative path resolves correctly.

---

## What the Report Looks Like

For the RNA/translation_efficiency_muscle regression dataset:
- Executive summary shows Spearman 0.6279, +0.6279 over trivial
- Target statistics table (mean, std, min, max, median)
- 6 preprocessing steps (k-mer features + scaling)
- 3 models compared (mean_predictor → ridge_regression → xgboost_default)
- Full metrics: MSE, RMSE, MAE, R², Spearman, Pearson

For the expression/cell_type_classification_segerstolpe classification dataset:
- Executive summary shows macro_f1 0.9620, +0.9175 over trivial
- 13-class distribution table with proportions
- Warning about severe class imbalance
- 3 preprocessing steps (log1p → HVG → scaling)
- 3 models compared, XGBoost dominates

---

## Design Decisions

### Why markdown, not HTML/PDF?

1. **Markdown is universal:** GitHub renders it, VS Code previews it, pandoc converts it to anything
2. **Figures as relative links:** No need to base64-encode images; they're just file references
3. **Version control friendly:** Plain text diffs show exactly what changed between runs
4. **Easy to extend:** Adding a new section = adding a block in the template, not wrestling with CSS

### Test-set evaluation

The report now includes a final held-out test evaluation step that runs _after_ model selection is complete. The best model (chosen on validation metrics) is evaluated once on the test split, and both validation and test scores are shown side by side in section 4.2. This keeps the selection process honest (decisions are still driven by validation) while giving the reader an unbiased estimate of generalization performance in the same report.

### Report reviewer fixes

The automated report reviewer (which validates that the generated report is internally consistent) now uses **word-boundary matching** when checking model names, preventing false positives such as `random_forest` matching inside `random_forest_tuned`. Additionally, the reviewer extracts only the baseline progression table when performing model-count checks, so supplementary tables (e.g., the validation-vs-test table) no longer inflate the count.

### Duplicate section fix

The Jinja2 template previously rendered a duplicate "4.9 Agent Decision Log" section when agent decisions were present. The template now guards this block so it appears exactly once.

### What's missing (future steps)?

The ARCHITECTURE.md template has sections we skip in Phase A:
- **Section 3: Biological Context** — requires LLM (Phase C, Step 17)
- **Section 7: Biological Interpretation** — requires Biology Specialist agent (Phase C)
- **Section 8: Recommendations for Further Data Collection** — requires active learning analysis (Phase C, Step 18)

These will be added as the pipeline grows. The current report is complete for what the deterministic pipeline can produce.

---

## Phase A Complete

With Step 6 done, the full Phase A deterministic pipeline is complete. Running `co-scientist run DATASET` now produces:

```
outputs/DATASET/
├── report.md              ← NEW: comprehensive markdown report
├── requirements.txt
├── models/
│   ├── best_model.pkl
│   ├── model_config.json
│   └── label_encoder.pkl  (classification only)
├── code/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
└── figures/
    ├── 01_profiling/
    ├── 02_preprocessing/
    └── 03_training/
```

This is a submittable output — a reviewer can read the report to understand what was done, inspect the figures, retrain the model, and run predictions on new data.
