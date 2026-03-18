# Step 1: Data Loading + Profiling — Detailed Walkthrough

## Overview

Step 1 answers the question: **"What do we have?"** Before any ML can happen, we need to load raw data from wherever it lives and understand its structure, quality, and characteristics. This step produces a `DatasetProfile` — a structured summary of everything downstream steps need to make decisions — entirely without LLM involvement.

---

## Part A: Data Loading (`loader.py`)

### The problem being solved

The genbio-leaderboard hosts ~20 datasets across different biological modalities (RNA, DNA, protein, cell expression, tissue). These datasets live in different HuggingFace repos, use different file formats, have different column naming conventions, and organize their train/test splits differently. The user just types:

```bash
co-scientist run RNA/translation_efficiency_muscle
```

The loader must figure out: Where does this data actually live? What format is it in? How do I get it into a standard in-memory representation?

### Step-by-step: Path resolution (`resolve_dataset_path`)

The function takes a user-facing path and returns `(hf_repo, subset, format, task_name)`. It tries three strategies in order:

**1. Local file check (line 89)**
```python
if Path(dataset_path).exists():
```
If the path is a real file on disk (e.g. `/data/my_experiment.csv`), we detect the format from the extension and skip HuggingFace entirely. This is important for running on hidden/custom datasets.

**2. Direct HuggingFace path (line 95)**
```python
if ":" in dataset_path and "/" in dataset_path.split(":")[0]:
```
If the user passes `genbio-ai/rna-downstream-tasks:translation_efficiency_Muscle`, we split on the colon. The part before is the HF repo, the part after is the subset. This is the escape hatch — any HF dataset works, no registry needed.

**3. Registry lookup (line 100–130)**
The common case. We split `RNA/translation_efficiency_muscle` into `modality_key="RNA"` and `task_name="translation_efficiency_muscle"`. Then:

- Look up `"RNA"` in `GENBIO_REGISTRY` (case-insensitive via `_MODALITY_ALIASES`)
- Find the matching task: `"translation_efficiency_muscle"` → `"translation_efficiency_Muscle"` (note the capitalization — HF subset names are case-sensitive)
- Return `("genbio-ai/rna-downstream-tasks", "translation_efficiency_Muscle", "hf_dataset", ...)`

If the task name isn't in the registry, we don't fail — we try it as a direct subset name. This means new genbio tasks work without code changes.

**Why a registry at all?** HuggingFace naming conventions are inconsistent. The RNA dataset uses `translation_efficiency_Muscle` (capital M), the cell dataset uses `Segerstolpe` as a subdirectory name. The registry absorbs this inconsistency so the user doesn't have to know internal HF naming.

### Step-by-step: HuggingFace dataset loading (`_load_hf_dataset`)

This path handles RNA, DNA, and protein datasets that are stored as parquet/arrow files on HuggingFace.

**Line 159: Download the data**
```python
ds = hf_load(repo, name=subset if subset else None)
```
The `datasets` library downloads and caches the data locally. On first run this hits the network; subsequent runs use the cache. The `name` parameter selects a specific subset (e.g. `translation_efficiency_Muscle` from the larger `rna-downstream-tasks` repo).

**Lines 162–177: Handle split structure**
HuggingFace datasets can have explicit splits (`{"train": ..., "test": ...}`) or a single blob. The RNA translation efficiency dataset has only a `"train"` split — all 1,257 samples are in one split, with a `fold_id` column for cross-validation. We handle both cases:
- If multiple splits exist: merge them into one DataFrame with a `_split` column tracking origin
- If single split: just convert to pandas

**Lines 180–184: Detect columns**
`_detect_columns_hf` identifies three things:
- **Target column**: tries `labels`, `label`, `target`, `y` in order. The genbio RNA datasets use `labels`.
- **Fold column**: tries `fold_id`, `fold`, `cv_fold`. RNA datasets have `fold_id` (integers 0–9).
- **Input columns**: everything that's not target, fold, or internal (`_split`). For RNA, this is just `sequences`.

**Lines 189–199: Package into `LoadedDataset`**
We wrap everything in a `LoadedDataset` — a container that holds `X` (features), `y` (target), `fold_ids`, and the raw DataFrame for later reference.

### Step-by-step: H5AD loading (`_load_h5ad_dataset`)

This path handles cell expression and tissue datasets stored as H5AD files (the standard format for single-cell genomics data).

**Why H5AD?** Single-cell RNA-seq produces a matrix of (cells × genes) — typically thousands of cells and ~20,000 genes. H5AD is the standard container format built on HDF5, used by scanpy/anndata. It stores:
- `X`: the expression count matrix (usually sparse — most genes are not expressed in most cells)
- `obs`: per-cell metadata (cell type labels, batch, donor, etc.)
- `var`: per-gene metadata (gene names, variance stats, etc.)

**Lines 234–246: Download split files**
The Segerstolpe dataset stores three separate files: `Segerstolpe_train.h5ad`, `Segerstolpe_valid.h5ad`, `Segerstolpe_test.h5ad`. We try multiple filename patterns because different datasets organize their files differently:
- `{subdir}_train.h5ad` (flat)
- `{subdir}/{subdir}_train.h5ad` (nested in a subdirectory)

`hf_hub_download` downloads each file from the HuggingFace repo and caches it locally.

**Lines 248–264: Fallback discovery**
If the expected patterns don't match, we list all files in the repo and search for h5ad files containing the dataset name. This handles datasets with non-standard naming.

**Lines 272–279: Load and concatenate**
Each split file is loaded into an AnnData object. We add a `_split` column to each, then concatenate into a single AnnData. This gives us one unified matrix where we can still identify which cells belong to which split.

**Lines 282–289: Extract the expression matrix**
```python
X_raw = adata_all.X
if hasattr(X_raw, "toarray"):
    X_array = X_raw.toarray()  # sparse → dense
```
The expression matrix is typically stored as a scipy sparse matrix (CSR format) because ~70% of entries are zero (a gene not expressed in a cell). We convert to dense here because sklearn/XGBoost work with dense arrays. For the Segerstolpe dataset, this produces a (2,133 × 19,264) matrix.

Gene names come from `adata_all.var_names` — these are the column labels (e.g. "TP53", "BRCA1", etc.).

**Lines 292–293: Detect the label column**
Cell type labels are stored in `adata.obs`, but different datasets use different column names. `_detect_label_column_h5ad` tries known names (`cell_type`, `celltype`, `cell_type_label`, etc.), then falls back to a heuristic: find a categorical column with 2–200 unique values. Segerstolpe uses `cell_type_label`.

### What comes out

A `LoadedDataset` with:

| Dataset | X shape | X type | y example | Splits |
|---|---|---|---|---|
| RNA translation efficiency | (1257, 1) | DataFrame with `sequences` column | -0.255 (float) | fold_id 0–9 |
| Cell type Segerstolpe | (2133, 19264) | DataFrame with gene columns | "alpha cell" (string) | train/valid/test |

---

## Part B: Profiling (`profile.py`)

### The problem being solved

The loaded data is raw — we have a matrix and labels but don't yet know what kind of ML problem this is. The profiler answers:
- What biological modality is this? (RNA? cell expression?)
- What ML task type? (regression? classification?)
- How big, how clean, how balanced is the data?
- Are there any red flags?

This is all done deterministically, no LLM needed. The architecture calls this Phase 0/1 (Initialize + Understand).

### Step-by-step: Modality detection (`_detect_modality`)

This is a **cascade** — a series of increasingly expensive checks, each more reliable than the last. We stop at the first confident answer.

**Level 1 — Path parsing (lines 69–79)**
The cheapest check. If the user typed `RNA/...`, it's RNA. If `expression/...`, it's cell expression. This handles 90% of genbio datasets.

**Why it's not sufficient alone:** A user might pass a direct HF path or a local file where the path gives no hint.

**Level 2 — Column names (lines 82–88)**
Look at the DataFrame columns. A column named `sequences` means sequence data. Columns named like genes (or >500 columns total) suggest expression data. For sequence data, we need to further disambiguate DNA vs RNA vs protein, which leads to...

**Level 3 — Content inspection (lines 91–94, `_classify_sequence_content`)**
Sample a few sequences and check the character alphabet:
- Only A, T, G, C, N → DNA
- Contains U → RNA (uracil instead of thymine)
- Contains amino acid letters (D, E, F, H, I, K, L, M, etc.) → Protein

**Why this works:** DNA/RNA use a 4-letter alphabet; proteins use 20. There's almost no ambiguity. The only edge case is very short sequences that happen to only use letters shared between nucleotides and amino acids, but for real datasets the distinction is clear.

**Level 4 — Source format (line 97)**
H5AD files are always cell expression data. This is a convention of the single-cell community.

**Level 5 — Dimensionality (lines 101–106)**
Expression matrices have >1,000 features (genes). Tabular datasets typically have <50.

### Step-by-step: Task type detection (`_detect_task_type`)

**Path hints first (lines 163–168)**
- "classification" or "cell_type" in the task name → classification
- "efficiency", "expression", "abundance", "ribosome" → regression

**Data-driven fallback (lines 171–185)**
If the path gives no hint, look at the target variable:
- String/categorical dtype → classification
- Float with many unique values → regression
- Float with ≤20 unique values → probably classification with numeric labels (e.g. 0/1)
- Integer with ≤50 unique values → classification

**Why the 20/50 thresholds?** These are heuristics. A target with values [0, 1, 2, 3] is almost certainly classification. A target with 10,000 unique floats is almost certainly regression. The thresholds are conservative — they may misclassify edge cases, but the architecture allows LLM override later (Phase C).

### Step-by-step: Target analysis (`_analyze_target`)

For **classification**: compute class distribution via `value_counts()`. This tells us:
- How many classes (Segerstolpe: 13 cell types)
- How balanced they are (Segerstolpe's smallest class has just 4 cells = 0.2%)
This information drives metric selection (imbalanced → macro F1 instead of accuracy) and whether we need class weighting or SMOTE.

For **regression**: compute mean, std, min, max, median. This tells us:
- The scale of the target (RNA efficiency: range [-5.78, 3.61])
- Whether the target is approximately normal or skewed
- Whether there are outliers

### Step-by-step: Feature analysis (`_analyze_features`)

**Missing values (lines 222–229)**
For numeric features: count NaN cells / total cells. For non-numeric (sequences): count rows with any null.

**Why it matters for ML:** Many sklearn models crash on NaN. We need to know whether imputation is needed and how much.

**Sparsity (line 226)**
Fraction of zeros in numeric features. The Segerstolpe expression matrix is 70.5% zeros. This matters because:
- Very sparse data benefits from sparse-aware algorithms
- Some features (genes) may be all-zero and should be removed
- High sparsity is normal for scRNA-seq (not a problem to fix, just a characteristic to know)

### Step-by-step: Sequence analysis (`_analyze_sequences`)

For sequence data only. Computes length statistics (min, max, mean, median, std). The RNA translation efficiency dataset has sequences ranging 45–100 nucleotides (mean 91.1).

**Why it matters:** Sequence length determines what ML approaches are viable. Very short sequences (< 50 nt) have limited k-mer diversity. Very long sequences (> 10,000 nt) are expensive to embed. Variable-length sequences need special handling (padding, pooling, or feature extraction).

### Step-by-step: Split analysis (`_analyze_splits`)

Records what predefined splits exist so the splitter (Step 2) knows what to use:
- **Fold-based** (RNA): 10 folds, 125–132 samples each
- **Predefined** (cell expression): train=1279, valid=427, test=427

### Step-by-step: Issue detection (`_detect_issues`)

A final pass that flags problems. These are the "guardrails" from the architecture (Section 10.3). Current checks:

| Check | Threshold | Why |
|---|---|---|
| Zero samples | 0 | Data didn't load |
| Very small dataset | < 50 | Not enough for train/val/test split |
| High missing values | > 10% | Need imputation strategy |
| Very sparse | > 90% zeros | Normal for scRNA-seq, but flag it |
| Severe class imbalance | smallest class < 5% | Need weighted loss / resampling |
| Moderate class imbalance | smallest class < 10% | May want macro F1 over accuracy |
| Many classes | > 50 | Higher model complexity needed |
| Zero-variance target | std = 0 | Nothing to predict |
| Unknown task/modality | — | Auto-detection failed |

The Segerstolpe dataset triggers "Severe class imbalance" because some cell types (e.g. epsilon cells) have only 4 samples out of 2,133.

### What comes out

A `DatasetProfile` with all fields populated:

**RNA translation efficiency:**
```
modality=rna, task_type=regression, input_type=sequence
samples=1257, features=1, target_range=[-5.78, 3.61]
seq_length=45-100, splits=10-fold CV
issues: (none)
```

**Cell type Segerstolpe:**
```
modality=cell_expression, task_type=multiclass_classification, input_type=expression_matrix
samples=2133, features=19264, classes=13
sparsity=70.5%, splits=predefined train/valid/test
issues: Severe class imbalance (smallest class: 0.2%)
```

---

## Engineering notes

### Why these two files and not one?

Loading and profiling are separate because:
1. **Different datasets need different loaders** (HF vs h5ad vs CSV) but the **same profiler** works for all. Keeping them separate avoids a god-function.
2. **Testability** — you can test the profiler with synthetic data without needing network access.
3. **Resumability** — if profiling fails, we still have the loaded data cached.

### Why `LoadedDataset` is a plain class, not Pydantic?

It holds numpy arrays and pandas DataFrames — Pydantic's serialization overhead is unnecessary for in-memory objects that never get saved to JSON. `DatasetProfile` *is* Pydantic because it gets logged and may be serialized later.

### The HuggingFace caching behavior

Both `datasets.load_dataset` and `hf_hub_download` cache to `~/.cache/huggingface/`. First run downloads data; subsequent runs are instant. This means the expensive network call only happens once.
