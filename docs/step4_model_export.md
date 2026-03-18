# Step 4: Model Export — Detailed Walkthrough

## Overview

Step 4 answers: **"How does someone use what we built without our pipeline?"**

After Step 3 identifies the best model, Step 4 exports everything a reviewer or user needs to:
1. **Load and run** the trained model on new data
2. **Retrain** the model from scratch
3. **Evaluate** predictions against ground truth
4. **Understand** what was done (config, preprocessing steps, metrics)

This is a key requirement from the architecture (Section 7.3, Section 12.5): "The reviewer can copy any task output to a new machine, run `pip install -r requirements.txt`, and immediately use `python predict.py --input new_data.csv`."

---

## What gets exported

```
outputs/RNA__translation_efficiency_muscle/
├── models/
│   ├── best_model.pkl          ← serialized sklearn/XGBoost model
│   ├── model_config.json       ← full metadata (hyperparameters, metrics, etc.)
│   └── label_encoder.pkl       ← only for classification (maps int→string labels)
├── code/
│   ├── train.py                ← standalone retraining script
│   ├── predict.py              ← standalone prediction script
│   └── evaluate.py             ← standalone evaluation script
└── requirements.txt            ← pinned Python dependencies
```

---

## Part A: Model serialization

### `best_model.pkl` — the trained model (lines 44–48)

```python
with open(model_path, "wb") as f:
    pickle.dump(trained.model, f)
```

**What pickle does:** Serializes the entire Python object — the XGBoost/sklearn model with all its learned parameters — to a binary file. Anyone can load it with `pickle.load()` and call `.predict()` immediately, no retraining needed.

**Why pickle and not a model-specific format?**
- **Universality:** Works for any sklearn-compatible model (XGBoost, logistic regression, ridge, future MLP). One format, one loading pattern.
- **Completeness:** Pickle captures the entire object state — hyperparameters, learned weights, internal tree structures (for XGBoost), everything.
- **Trade-off acknowledged:** Pickle files are Python-version-sensitive and not human-readable. For production deployment, you'd convert to ONNX or a model-specific format. For a research submission, pickle is the standard.

**What's actually inside the pickle for each model type:**
- **DummyClassifier:** Just the majority class label and class prior probabilities (~bytes)
- **LogisticRegression:** Weight matrix (n_features × n_classes) + bias vector (~16KB for Segerstolpe)
- **Ridge:** Weight vector (n_features) + bias (~2.6KB for RNA)
- **XGBoost:** Serialized tree ensemble — 100 trees with splits, thresholds, and leaf values (~100KB–1MB depending on complexity)

### `model_config.json` — complete metadata (lines 50–77)

```json
{
  "model_name": "xgboost_default",
  "model_type": "xgboost",
  "tier": "standard",
  "task_type": "regression",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42
  },
  "dataset": {
    "name": "translation_efficiency_muscle",
    "path": "RNA/translation_efficiency_muscle",
    "modality": "rna",
    "task_type": "regression",
    "num_samples": 1257,
    "num_features": 326,
    "num_classes": 0
  },
  "evaluation": {
    "primary_metric": "spearman",
    "primary_value": 0.6279,
    "all_metrics": { "mse": 1.1035, "rmse": 1.0505, ... },
    "evaluated_on": "validation"
  },
  "preprocessing_steps": [
    "3-mer frequencies (64 features)",
    "4-mer frequencies (256 features)",
    "sequence length",
    "GC content",
    "nucleotide composition",
    "standard scaling"
  ],
  "feature_names": ["kmer_3_AAA", "kmer_3_AAC", ...]
}
```

**Why JSON and not pickle?** This is the human-readable counterpart to `best_model.pkl`. A reviewer can open this and immediately understand:
- What model was trained (XGBoost regressor with these exact hyperparameters)
- On what data (1,257 RNA sequences, 326 k-mer features)
- How it performed (Spearman 0.6279 on validation)
- What preprocessing was applied

**Why store `feature_names`?** If someone wants to inspect feature importance (XGBoost provides this), they need to map feature index → name. Without `feature_names`, feature 73 means nothing; with it, they know it's `kmer_4_ACGT`.

### `label_encoder.pkl` — class label mapping (lines 80–84)

Only saved for classification tasks. Maps integer predictions back to human-readable labels:
```
0 → "acinar cell"
1 → "alpha cell"
2 → "beta cell"
...
12 → "stellate cell"
```

Without this, the model predicts `7` and the reviewer doesn't know what cell type that is.

---

## Part B: Code generation

### The design principle

The generated scripts must work **completely independently** of the co-scientist package. A reviewer should be able to:
```bash
cd outputs/RNA__translation_efficiency_muscle/
pip install -r requirements.txt
python predict.py --input new_data.csv
```

No `import co_scientist` anywhere in the generated code. This is the "portable, reproducible codebase" requirement from architecture Section 1.

### `train.py` — reproduce training (lines 100–184)

**What it generates (for XGBoost regression):**

```python
import xgboost as xgb

def train(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model
```

**How the code generation works:**

The function builds the model construction line dynamically based on model type (lines 113–129):
- `model_type == "xgboost"` → `xgb.XGBRegressor(...)` or `xgb.XGBClassifier(...)`
- `model_type == "logistic_regression"` → `LogisticRegression(...)`
- `model_type == "ridge_regression"` → `Ridge(...)`
- Unknown type → falls back to loading the pickle

`_format_params(hparams)` (lines 334–342) converts the hyperparameter dict into Python keyword argument syntax: `{"n_estimators": 100, "max_depth": 6}` → `n_estimators=100, max_depth=6`.

**Why not just load the pickle in train.py?** Because that's not reproducible. The pickle is a black box. The generated `train.py` has the exact model class and hyperparameters in readable source code. A researcher can modify and re-run it.

**Why is `load_data` a stub?** The preprocessing pipeline (k-mer extraction, log1p, HVG selection) is dataset-specific. Generating a fully working data loader for every modality would duplicate `preprocess.py`. Instead, the script documents what preprocessing was applied and what shape the model expects. The reviewer fills in their own data loading. In later steps (Step 6: Report), we'll provide more detailed reproduction instructions.

### `predict.py` — run inference on new data (lines 187–252)

```python
python predict.py --input new_data.csv --output predictions.csv --model models/best_model.pkl
```

**Step by step:**
1. Load the pickle model
2. Read input CSV as a numpy array
3. Call `model.predict(X)`
4. For classification: load `label_encoder.pkl` and convert integer predictions back to class names
5. Save predictions to CSV

**Why check for label encoder with try/except (line 234)?** Regression models don't have a label encoder. Rather than branching on task type at generation time, the classification script tries to load the encoder and gracefully handles its absence.

### `evaluate.py` — evaluate predictions against ground truth (lines 255–315)

```python
python evaluate.py --predictions predictions.csv --ground-truth labels.csv
```

Generates different metric code for classification vs. regression:
- **Classification:** accuracy, macro F1, weighted F1, full classification report (per-class precision/recall/F1)
- **Regression:** Spearman, Pearson, MSE, RMSE, MAE, R²

### `requirements.txt` — pinned dependencies (lines 318–331)

```
scikit-learn>=1.6.1
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
xgboost>=2.1.4
```

**Why pin with `>=` instead of `==`?** Exact pinning (`==`) breaks across platforms and Python versions. `>=` ensures compatibility while avoiding known-broken older versions. The versions are read from the currently installed packages at export time.

**Why only include what's needed?** The model type determines whether XGBoost is in requirements. A logistic regression model doesn't need XGBoost installed. This keeps the dependency footprint minimal.

---

## Part C: How it connects to the pipeline

### Which model gets exported?

In `cli.py`, after Step 3 evaluates all baselines, we sort by primary metric:

```python
ranked = sorted(
    zip(results, trained_models),
    key=lambda pair: pair[0].primary_metric_value,
    reverse=not lower_is_better,
)
best_result, best_trained = ranked[0]
```

This pairs each `ModelResult` with its `TrainedModel` and sorts them. `lower_is_better` flips the sort for metrics like MSE where smaller is better.

Currently this always selects XGBoost (the standard baseline beats the others). In later steps when we add hyperparameter tuning and advanced models, the same ranking mechanism selects the best model regardless of type.

### Output directory convention

```python
# In config.py
@property
def task_output_dir(self) -> Path:
    sanitized = self.dataset_path.replace("/", "__")
    return self.output_dir / sanitized
```

`RNA/translation_efficiency_muscle` → `outputs/RNA__translation_efficiency_muscle/`

The double-underscore convention avoids nested directories while preserving the modality/task structure in the name.

---

## What the exported output matches in the architecture

Architecture Section 12.1 specifies:
```
outputs/RNA__translation_efficiency_muscle/
├── models/
│   ├── best_model.pkl        ✓ done
│   └── model_config.json     ✓ done
├── code/
│   ├── train.py              ✓ done
│   ├── predict.py            ✓ done
│   └── evaluate.py           ✓ done
├── requirements.txt          ✓ done
├── README.md                 ← Step 6 (report generation)
├── report.md                 ← Step 6
├── figures/                  ← Step 5 (visualization)
└── logs/                     ← Step 10 (experiment log)
```

Steps 5 and 6 will fill in the remaining output files.

---

## Engineering notes

### Why code generation instead of templating?

The architecture mentions `co_scientist/export/templates/` with Jinja templates. We chose inline code generation instead because:
1. The generated scripts are short (< 50 lines each)
2. The logic that determines *what* to generate (model type dispatch) is tightly coupled with the templates
3. For this complexity level, f-strings are more readable than Jinja

If the generated scripts grow significantly (e.g., full preprocessing pipelines per modality), Jinja templates would be worth it. This can be refactored in Step 7+ without changing the interface.

### The pickle security note

Pickle files can execute arbitrary code when loaded. This is fine for our use case (the reviewer generates and loads their own models), but the generated `predict.py` should not be used to load untrusted pickle files from the internet. This is a known Python ecosystem limitation, not specific to our tool.

### What happens when we add more model types?

Adding a new model (e.g., MLP in Step 9) requires:
1. A new builder in `registry.py`
2. A new branch in `_generate_train_py` for the model construction code
3. Possibly new entries in `requirements.txt`

The export interface (`export_model()`) doesn't change — it already accepts any `TrainedModel`.
