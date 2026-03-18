# Step 32: Validation Agent — Step-Level Validation + Auto-Repair

## Overview

The Validation Agent is a dedicated pipeline component that runs **after every step** (data loading, profiling, preprocessing, splitting, model training, export) to detect silent errors and fix them before they propagate downstream. Unlike the other agents (Data Analyst, ML Engineer, Biology Specialist) which provide strategic advice, the Validation Agent is a **quality enforcer** — it validates outputs structurally and repairs them deterministically.

**Key principle:** The pipeline always moves forward with clean data. If an error can be fixed, fix it. If it can't, fail fast with a clear message.

**File:** `co_scientist/validation.py`

---

## Validate-and-Fix Pattern

Every validator follows a 4-step protocol:

```
Step completes → Validator receives output
    ↓
1. DETECT — inspect for problems (NaN, Inf, empty, shape mismatch, syntax error)
    ↓
2. DETERMINISTIC FIX — apply rule-based repair if possible
    ↓
3. LLM FIX — if deterministic fix insufficient, invoke LLM to diagnose + repair
    ↓
4. RETURN — pass repaired object back to pipeline
    ↓
Pipeline continues with clean data
```

---

## Validation Steps

### Step 1: Data Loading (`validate_and_fix_loaded_data`)

**What it checks:**
- Dataset is non-empty (X and y have rows)
- X and y have same number of rows
- Target column is not all NaN/null
- Predefined splits exist and have a "train" split
- Dataset has at least 10 samples (warning if not)

**Fixes applied:**
- **Shape mismatch**: Truncates X and y to minimum length
- **Null targets**: Drops rows with null target values
- **Missing train split**: Renames largest split to "train"

### Step 2: Profiling (`validate_and_fix_profile`)

**What it checks:**
- Modality is not UNKNOWN
- Task type is not UNKNOWN
- num_samples > 0
- Classification tasks have >= 2 classes

**Fixes applied:**
- **Unknown modality**: Infers from content — checks for sequence columns (ACGTN → RNA, amino acid chars → PROTEIN), high numeric column count (>100 → CELL_EXPRESSION), else TABULAR
- **Unknown task type**: Infers from target dtype — object/categorical → classification, integer with low cardinality → classification, float → regression

### Step 3: Preprocessing (`validate_and_fix_preprocessing`)

**What it checks:**
- X has > 0 rows and > 0 features
- X and y have same length
- No NaN values in feature matrix
- No Inf values in feature matrix
- Variance of features (warns if >50% zero-variance)
- No NaN in target variable

**Fixes applied:**
- **Shape mismatch**: Truncates to minimum length
- **NaN in features**: Replaces with column mean (or 0 if entire column is NaN)
- **Inf in features**: Clips to ±(10 × max finite value)
- **NaN in target**: Drops affected rows

### Step 4: Splitting (`validate_and_fix_split`)

**What it checks:**
- No empty splits (train, val, test)
- Feature count matches across splits
- No NaN values in any split
- No data leakage (overlap between train/test feature vectors)
- Train split is not unusually small (<10% of data)

**Fixes applied:**
- **Empty val/test**: Carves 15% from training data (stratified if possible)
- **Feature count mismatch**: Pads with zeros or truncates to match train
- **NaN in splits**: Replaces with train column means (never leaks val/test stats into imputation)

### Step 5: Model Validation (`validate_trained_model`)

**What it checks:**
- Model's `predict()` returns correct number of predictions
- Predictions don't contain NaN
- Predictions don't contain Inf
- Model doesn't predict constant values for all samples (warning)

**Foundation model routing** (added in GPU integration):
```python
if trained.needs_embeddings and split.X_embed_val is not None:
    # Embedding models (embed_xgboost, embed_mlp) use AIDO embeddings
    X_check = split.X_embed_val[:n_check]
    y_pred = trained.model.predict(X_check)
elif trained.needs_sequences and split.seqs_val:
    # Sequence models (bio_cnn, aido_finetune) use raw sequences
    X_check = split.X_val[:n_check]
    seqs_check = split.seqs_val[:n_check]
    y_pred = trained.model.predict(X_check, sequences=seqs_check)
else:
    # Standard models use handcrafted features
    X_check = split.X_val[:n_check]
    y_pred = trained.model.predict(X_check)
```

Without this routing, the validator would feed wrong-shaped data to foundation models and incorrectly flag them as broken.

**Note:** Model validation is detection-only — no auto-fix. A model that produces bad predictions is reported, not repaired. The pipeline continues with other models.

### Step 6: Export Validation (`validate_and_fix_export`)

**What it checks:**
- `train.py` is syntactically valid (`ast.parse`)
- `predict.py` is syntactically valid
- `train.py` actually **runs** in a subprocess (integration test)

**Fixes applied:**
- **Syntax errors**: Common patterns auto-fixed (missing imports, f-string issues, indentation)
- **LLM fix**: If deterministic fix fails, sends the error + code to LLM for repair
- **Retry**: Fixed script is re-parsed and optionally re-executed

This is the most aggressive validation step — it actually executes the generated code to verify it works end-to-end.

---

## Dashboard Integration

The Validation Agent reports its results to the live dashboard:

```
─── Validation ──────────────────────────────────────
✓ data_loading:    PASS
✓ profiling:       FIXED (inferred RNA modality)
✓ preprocessing:   FIXED (replaced 42 NaN values)
✓ splitting:       PASS
✓ model_xgboost:   PASS
✓ model_embed_mlp: PASS
✓ export:          PASS
```

Each step shows PASS, FIXED (with what was fixed), or FAIL (with the issue).

---

## How It Integrates with the Pipeline

The validation agent is called in `cli.py` after each step:

```python
# After loading
dataset, v_load = validate_and_fix_loaded_data(dataset, profile)

# After profiling
profile, v_profile = validate_and_fix_profile(profile, dataset)

# After preprocessing
preprocessed, v_preproc = validate_and_fix_preprocessing(preprocessed, profile)

# After splitting
split, v_split = validate_and_fix_split(split, preprocessed, profile, seed)

# After each model training
v_model = validate_trained_model(trained, split)

# After export
v_export = validate_and_fix_export(output_dir, profile, client)
```

The returned objects replace the originals — the pipeline always continues with the validated (and possibly repaired) data.

---

## Design Decisions

### Why Not Just Fail?

Silent errors are the enemy of automated pipelines. A NaN in one feature doesn't mean the entire run should abort — it means that feature needs imputation. An empty test split doesn't mean the dataset is broken — it means we need to carve from train. By fixing what we can and continuing, the pipeline produces results for 95%+ of datasets instead of failing on edge cases.

### Why Deterministic Fixes First?

LLM calls cost money and time. Most issues (NaN, Inf, empty splits, shape mismatches) have obvious deterministic fixes. The LLM is only invoked when the issue is genuinely ambiguous (e.g., a generated script with a logic error that can't be pattern-matched).

### Why Validate After Every Step?

Because errors compound. A NaN introduced during preprocessing becomes a NaN in every model prediction, which becomes a NaN in the evaluation metrics, which crashes the report generator. Catching it at the preprocessing step is 1 fix. Catching it at the report step requires debugging the entire chain.

---

## Relationship to Other Components

| Component | Validation Agent's Role |
|-----------|------------------------|
| **Guardrails** (`guardrails.py`) | Guardrails enforce scientific discipline (no inflated metrics, proper baselines). Validation Agent enforces structural correctness (no NaN, proper shapes). |
| **Error Recovery** (`resilience.py`) | Resilience handles catastrophic failures (model crashes, OOM). Validation Agent handles silent corruption (NaN creep, empty splits). |
| **ReAct Agent** | The ReAct agent drives strategy. The Validation Agent verifies the results of each action. |
| **Foundation Models** | Validation Agent correctly routes foundation model predictions to the right feature set (X_embed, sequences, or X). |
