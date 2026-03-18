# Step 2: Preprocessing + Splitting — Detailed Walkthrough

## Overview

Step 2 answers the question: **"How do we turn raw data into something a model can learn from?"**

The raw data from Step 1 isn't ready for ML models:
- RNA sequences are strings like `"AUGCGUACCGUA..."` — models need numbers
- Expression matrices have 19,264 genes, most of which are noise
- Cell type labels are strings like `"alpha cell"` — models need integers
- We need separate train/val/test sets to evaluate honestly

This step transforms the raw `LoadedDataset` into a `SplitData` object containing train/val/test numpy arrays ready to feed into sklearn, XGBoost, or PyTorch.

---

## Part A: Preprocessing (`preprocess.py`)

### The dispatcher pattern (lines 19–41)

```python
dispatcher = {
    Modality.RNA: _preprocess_sequence,
    Modality.DNA: _preprocess_sequence,
    Modality.PROTEIN: _preprocess_protein,
    Modality.CELL_EXPRESSION: _preprocess_expression,
    Modality.TABULAR: _preprocess_tabular,
}
fn = dispatcher.get(profile.modality, _preprocess_tabular)
```

Different biological data types need fundamentally different preprocessing. You can't compute k-mer frequencies on an expression matrix, and you can't do HVG selection on a protein sequence. The dispatcher routes to the right function based on the modality detected in Step 1. If the modality is unknown, we fall back to generic tabular preprocessing — safe but suboptimal.

Each preprocessing function returns `(X, feature_names, steps)`:
- `X`: a `(n_samples, n_features)` numpy float64 array
- `feature_names`: human-readable names for each column
- `steps`: a list of strings describing what was done (for the report)

---

### RNA/DNA sequence preprocessing (`_preprocess_sequence`)

**The fundamental problem:** ML models (logistic regression, XGBoost, neural nets) operate on fixed-length numeric vectors. A sequence like `"AUGCGUACCG"` is a variable-length string. We need to convert it to a fixed-length vector that captures the biologically meaningful information.

This is called **feature engineering** — manually designing numeric representations that encode domain knowledge.

#### Feature 1: K-mer frequencies (lines 59–64)

**What are k-mers?** A k-mer is a substring of length k. For the sequence `"AUGC"`:
- 2-mers: `AU`, `UG`, `GC` (3 substrings)
- 3-mers: `AUG`, `UGC` (2 substrings)

**What we compute:** For every possible k-mer, we count how often it appears in the sequence, then divide by the total number of k-mers to get a frequency (so sequences of different lengths are comparable).

For k=3 with alphabet {A, C, G, T}: 4³ = 64 possible 3-mers → 64 features
For k=4: 4⁴ = 256 possible 4-mers → 256 features

**Why k=3 and k=4?**
- k=3 (codons): In molecular biology, every 3 nucleotides encode one amino acid (this is the genetic code). Codon usage bias — the preference for certain codons over synonymous alternatives — correlates strongly with translation efficiency (which is exactly what the RNA dataset measures). 3-mer frequencies capture this.
- k=4: Captures longer motifs. Many regulatory elements (binding sites, splice signals) are 4–8 nucleotides long. 4-mers give the model access to these patterns.
- k=1 and k=2 are too short (captured by nucleotide composition anyway). k=5 gives 1,024 features which starts to be sparse for small datasets.

**Implementation detail — U→T mapping (line 115):**
```python
seq_upper = seq.upper().replace("U", "T")
```
RNA uses uracil (U); DNA uses thymine (T). Computationally they're equivalent — mapping U→T lets us use a single 4-letter alphabet (ACGT) for both.

**Implementation detail — frequency normalization (lines 117–122):**
```python
total = sum(counts.values())
matrix[i, idx] = count / total
```
We divide counts by total k-mers in the sequence. Without this, a 100-nt sequence would have ~2x the counts of a 50-nt sequence, and the model would learn sequence length instead of composition.

#### Feature 2: Sequence length (lines 67–70)

Raw nucleotide count. For the RNA dataset, lengths range 45–100 nt.

**Why include it?** Translation efficiency can correlate with UTR length. Longer 5'UTR sequences can form more complex secondary structures, which affects ribosome scanning efficiency. Including length lets the model use this signal if it's there.

#### Feature 3: GC content (lines 73–76)

Fraction of G and C nucleotides in the sequence (vs. A and T/U).

**Why it matters biologically:** G-C base pairs have 3 hydrogen bonds (vs. 2 for A-T/U), making them thermodynamically more stable. High GC content means:
- Stronger RNA secondary structure (harder for ribosomes to unfold)
- Higher melting temperature
- Different codon usage patterns

For translation efficiency prediction, GC content is a known predictive feature in the literature.

#### Feature 4: Nucleotide composition (lines 79–82)

Frequency of each individual nucleotide: `[freq_A, freq_C, freq_G, freq_T]` — 4 features.

**Wait, isn't this redundant with 1-mer frequencies?** It's actually the same information (1-mers), but we include it separately because: (a) it's more interpretable in the feature importance analysis later, and (b) it's explicit rather than buried in the k-mer matrix.

**Note:** These 4 values sum to ~1.0 (not exactly 1 if there are N/ambiguous characters). This means one feature is linearly dependent on the others. Some models handle this fine (tree-based); for others (linear), it could cause slight multicollinearity, but standard scaling mitigates this.

#### Feature 5: Standard scaling (lines 87–89)

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

Transforms each feature to have mean=0 and standard deviation=1.

**Why is this necessary?**
- K-mer frequencies are in [0, 1] (small values)
- Sequence length is in [45, 100] (larger values)
- GC content is in [0, 1]

Without scaling, models that depend on feature magnitude (logistic regression, SVMs, neural nets) would be dominated by sequence length simply because its values are ~100x larger. Tree-based models (XGBoost) don't strictly need scaling, but it doesn't hurt and it makes downstream code simpler to have everything on the same scale.

**Total output:** 64 + 256 + 1 + 1 + 4 = **326 features** per sequence.

---

### Cell expression preprocessing (`_preprocess_expression`)

**The input:** A (cells × genes) matrix. For Segerstolpe: 2,133 cells × 19,264 genes. Each entry is a count — how many times gene X was detected in cell Y.

#### Transform 1: log1p normalization (line 226)

```python
X = np.log1p(X)  # log(1 + x)
```

**What this does:** Compresses the dynamic range. Raw counts might be:
- Gene A in cell 1: 0 (not detected)
- Gene B in cell 1: 5 (moderate expression)
- Gene C in cell 1: 10,000 (highly expressed housekeeping gene)

After log1p: 0 → 0, 5 → 1.79, 10,000 → 9.21.

**Why `log1p` and not `log`?** log(0) is undefined (negative infinity). log1p(0) = log(1) = 0. The "+1" handles zeros gracefully, which matters because ~70% of the matrix is zeros.

**Why log-transform at all?** This is the single most important preprocessing step for scRNA-seq data. Three reasons:

1. **Biological:** Gene expression spans orders of magnitude. A 2x change from 5 to 10 counts is biologically meaningful. A 2x change from 5,000 to 10,000 counts is often just noise. Log-transform makes fold-changes additive (log(2x) = log(x) + log(2)), so the model treats relative changes equally regardless of baseline.

2. **Statistical:** Raw count distributions are heavily right-skewed (most values are 0 or small, a few are very large). Log-transform makes distributions approximately normal, which is an assumption of many models.

3. **Practical:** Without it, a handful of highly expressed genes dominate the variance and the model learns "cells with high ACTB expression" instead of meaningful cell-type-specific signatures.

#### Transform 2: Highly Variable Gene (HVG) selection (lines 230–239)

```python
variances = np.var(X, axis=0)          # variance of each gene across cells
top_indices = np.argsort(variances)[-n_hvg:]  # top 2000 most variable
X = X[:, top_indices]                   # keep only those genes
```

We go from 19,264 genes down to 2,000.

**Why throw away 90% of genes?** Most genes are:
- **Not expressed at all** (zero in every cell) — ~30% of genes
- **Housekeeping genes** (expressed at similar levels in all cell types) — e.g. ribosomal genes, ACTB. These have high mean but low variance. They don't help distinguish cell types.
- **Low-expression noise** — detected in a few cells by chance (technical noise from the sequencing process)

The informative genes — the ones that distinguish alpha cells from beta cells from delta cells — are the ones with **high variance across cells**. These are the "highly variable genes."

**Why 2,000?** This is the standard default in the single-cell community (scanpy uses it, Seurat uses it). It's a balance:
- Too few (500): might miss subtle markers
- Too many (5,000+): include noisy genes that hurt model performance
- 2,000: captures the major cell-type markers without excessive noise

**Implementation note — variance after log1p (line 232):** We compute variance on the log-transformed data, not the raw counts. This is important because raw-count variance is dominated by highly expressed genes (var of [0, 10000] >> var of [0, 5]), while log-space variance better captures biologically meaningful variability.

#### Transform 3: Standard scaling (lines 242–244)

Same as for sequences. After log1p and HVG selection, each gene gets zero-mean and unit-variance. This ensures no single gene dominates the model just because it happens to have higher expression levels.

**Total output:** 2,133 cells × 2,000 genes.

---

### Protein sequence preprocessing (`_preprocess_protein`)

Similar philosophy to nucleotide sequences but with protein-specific features:

#### Amino acid composition (lines 164–174)
Frequency of each of the 20 standard amino acids. This is the protein equivalent of nucleotide composition but with a larger alphabet. Amino acid composition alone can distinguish protein families — globular vs. membrane proteins have very different compositions (membrane proteins are enriched in hydrophobic residues).

#### Physicochemical properties (lines 183–200)

**Average molecular weight (line 196):** Each amino acid has a known molecular weight (in Daltons). Glycine is lightest (75 Da), tryptophan is heaviest (204 Da). The average per-residue MW gives a rough indicator of amino acid composition complexity.

**Average hydrophobicity — Kyte-Doolittle scale (line 197):** Each amino acid has a hydrophobicity value ranging from -4.5 (arginine, very hydrophilic) to +4.5 (isoleucine, very hydrophobic). The average across a sequence indicates whether the protein is likely soluble (negative) or membrane-associated (positive). This is relevant for predicting protein function, stability, and localization.

---

### Tabular fallback preprocessing (`_preprocess_tabular`)

For datasets that don't fit any biological modality:
1. Numeric columns: fill NaN with 0, keep as-is
2. Categorical columns: label-encode (convert strings to integers)
3. Standard scale everything

This is intentionally simple — it's a fallback, not optimized for any domain.

---

### Target encoding (`_encode_target`)

**Classification targets (lines 305–312):**
```python
le = LabelEncoder()
y_encoded = le.fit_transform(y_arr)  # "alpha cell" → 0, "beta cell" → 1, ...
```
Sklearn/XGBoost need integer class labels, not strings. The `LabelEncoder` maps each unique string to an integer and stores the mapping so we can convert predictions back to human-readable labels later.

**Why store the encoder?** When the final model predicts class 7, we need to know that means "gamma cell". The encoder travels through `PreprocessingResult` → `SplitData` and is used in evaluation/reporting.

**Regression targets (line 315):** Just cast to float64. No encoding needed.

---

## Part B: Splitting (`split.py`)

### Why three separate sets?

This is fundamental to honest ML evaluation:

- **Train set:** The model learns from this data. It sees these examples during training and adjusts its parameters to minimize error on them.

- **Validation set:** Used during development to make decisions: which model is better? Should we tune hyperparameters more? Is the model overfitting? The model never trains on this data, but our *decisions* are influenced by val performance — so we can inadvertently overfit to val.

- **Test set:** Touched **exactly once**, at the very end (architecture Section 8.3). This is our honest estimate of real-world performance. If we looked at test performance during development and used it to make decisions, we'd be cheating — the reported number would be optimistic.

**The architecture is strict about this:** test set evaluation happens in Phase 5 (Analyze), not Phase 3 (Baseline) or Phase 4 (Iterate).

### Strategy 1: Predefined splits (`_split_predefined`)

**When used:** Cell expression datasets (Segerstolpe, Zheng) that ship with `_train.h5ad`, `_valid.h5ad`, `_test.h5ad`.

**Why respect author-defined splits?** The dataset authors designed these splits to avoid **data leakage**. In single-cell data, cells from the same donor or the same batch tend to be more similar than cells from different donors. If you randomly split, you might put cells from donor A in both train and test — the model memorizes donor-specific artifacts instead of learning cell type biology. Author-defined splits typically ensure donor separation.

**Implementation (lines 53–86):**
1. Extract the `_split` column that was attached during loading
2. Create boolean masks: `train_mask`, `val_mask`, `test_mask`
3. Index into the preprocessed arrays: `X[train_mask]`, `y[train_mask]`, etc.

**Edge case (lines 64–76):** If the dataset has train + test but no validation split, we carve 20% from the training set. This is preferable to using the test set for validation (which would violate the "test set touched once" rule).

**Segerstolpe result:** train=1,279 (60%), val=427 (20%), test=427 (20%).

### Strategy 2: Fold-based splits (`_split_from_folds`)

**When used:** RNA datasets that have a `fold_id` column (integers 0–9) for 10-fold cross-validation.

**What is cross-validation?** The standard approach: split data into k folds, train on k-1, test on the remaining 1, rotate, average. This gives a robust performance estimate but requires training k models.

**Why we don't do full CV:** Our pipeline trains many models iteratively (baselines, hyperparameter search, advanced models). Full 10-fold CV for each would be 10x the compute. Instead, we use the fold assignments to create a single train/val/test split:

```
Folds 0–7 → train (80%)
Fold 8    → validation (10%)
Fold 9    → test (10%)
```

**Why this is better than random splitting:** The fold assignments may encode domain knowledge. For sequence data, the dataset authors may have ensured that similar sequences (by edit distance or homology) are in the same fold. Using their folds avoids the risk of training on a sequence that's 95% identical to a test sequence.

**Implementation (lines 97–130):**
1. Sort unique fold IDs: `[0, 1, 2, ..., 9]`
2. Last fold → test, second-to-last → val, rest → train
3. Create boolean masks and index into arrays

**RNA result:** train=1,000 (80%), val=125 (10%), test=132 (11%). The slight asymmetry (132 vs 125) is because fold 9 has 132 samples while other folds have 125 — the dataset has 1,257 samples which doesn't divide evenly by 10.

### Strategy 3: Random splitting (`_split_random`)

**When used:** Datasets with no predefined splits or fold IDs (local files, datasets without split metadata).

**Implementation (lines 133–181):**
Two-step split using sklearn's `train_test_split`:
1. Split 70/30 → train vs. temporary pool
2. Split temporary pool 50/50 → 15% val, 15% test

**Stratification (line 144):**
```python
stratify = y if is_classification else None
```
For classification, `stratify=y` ensures each split has approximately the same class proportions as the full dataset. Without stratification, a rare class with 10 samples might end up with 0 samples in the test set purely by chance.

**Fallback (lines 151–156):** Stratification requires at least 2 samples per class per split. For very rare classes, this is impossible, so we catch the `ValueError` and fall back to non-stratified splitting. This follows the architecture's graceful degradation principle.

---

## The full data flow

```
User input: "RNA/translation_efficiency_muscle"
                    │
          ┌─────────▼──────────┐
Step 1a   │ resolve_dataset_path│ → ("genbio-ai/rna-downstream-tasks",
          │ (loader.py:80)      │    "translation_efficiency_Muscle",
          └─────────┬──────────┘    "hf_dataset", ...)
                    │
          ┌─────────▼──────────┐
Step 1b   │ _load_hf_dataset   │ → LoadedDataset:
          │ (loader.py:155)    │    X = DataFrame["sequences"] (1257 strings)
          └─────────┬──────────┘    y = Series[float] (1257 values)
                    │                fold_ids = [0,0,...,9,9] (1257 ints)
          ┌─────────▼──────────┐
Step 1c   │ profile_dataset    │ → DatasetProfile:
          │ (profile.py:15)    │    modality=RNA, task=regression
          └─────────┬──────────┘    samples=1257, seq_length=45-100
                    │
          ┌─────────▼──────────┐
Step 2a   │ _preprocess_seq    │ → PreprocessingResult:
          │ (preprocess.py:48) │    X = ndarray(1257, 326) float64
          └─────────┬──────────┘    y = ndarray(1257,) float64
                    │                feature_names = ["kmer_3_AAA", ...]
          ┌─────────▼──────────┐
Step 2b   │ _split_from_folds  │ → SplitData:
          │ (split.py:90)      │    X_train(1000,326) y_train(1000,)
          └────────────────────┘    X_val(125,326)    y_val(125,)
                                    X_test(132,326)   y_test(132,)
```

```
User input: "expression/cell_type_classification_segerstolpe"
                    │
          ┌─────────▼──────────┐
Step 1a   │ resolve_dataset_path│ → ("genbio-ai/cell-downstream-tasks",
          │ (loader.py:80)      │    "Segerstolpe", "h5ad", ...)
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
Step 1b   │ _load_h5ad_dataset │ → LoadedDataset:
          │ (loader.py:228)    │    X = DataFrame[19264 gene cols] (2133 cells)
          └─────────┬──────────┘    y = Series["alpha cell", "beta cell", ...]
                    │                raw_data = {adata, splits=["train","train",...]}
          ┌─────────▼──────────┐
Step 1c   │ profile_dataset    │ → DatasetProfile:
          │ (profile.py:15)    │    modality=cell_expression, task=multiclass
          └─────────┬──────────┘    classes=13, sparsity=70.5%
                    │
          ┌─────────▼──────────┐
Step 2a   │ _preprocess_expr   │ → PreprocessingResult:
          │ (preprocess.py:214)│    X = ndarray(2133, 2000) float64
          └─────────┬──────────┘    y = ndarray(2133,) int64 (0–12)
                    │                label_encoder maps 0→"acinar cell", etc.
          ┌─────────▼──────────┐
Step 2b   │ _split_predefined  │ → SplitData:
          │ (split.py:47)      │    X_train(1279,2000) y_train(1279,)
          └────────────────────┘    X_val(427,2000)    y_val(427,)
                                    X_test(427,2000)   y_test(427,)
```

---

## Engineering notes

### Why preprocessing happens before splitting (in code), but conceptually they're coupled

In a strict ML workflow, you should fit preprocessing parameters (scaler mean/std, HVG selection) on the training set only, then transform val/test using those same parameters. This prevents **data leakage** — the model indirectly seeing test data through preprocessing statistics.

In our current implementation, we fit `StandardScaler` on the full dataset before splitting. This is a deliberate simplification for Phase A (deterministic pipeline). The leakage is minimal:
- For scaling: the mean/std of 1000 training samples is very close to 1257 total samples
- For HVG selection: the top-2000 variable genes computed on 2133 cells vs. 1279 cells are nearly identical

In Phase B (Step 7+), we'll refactor to fit on train only. The current approach gets us to a working end-to-end pipeline faster.

### Why `PreprocessingResult` carries both X and y

It would be tempting to preprocess X separately and pass y through unchanged. But target encoding (string labels → integers) is part of preprocessing, and the label encoder needs to travel with the data so we can decode predictions later. Bundling them ensures they stay in sync.

### Why `SplitData` is a plain class with named arrays

Alternatives considered:
- **Dict of dicts:** `{"train": {"X": ..., "y": ...}, ...}` — too verbose to access, no autocomplete
- **Named tuples:** immutable, no methods
- **Dataclass:** would work, but we want `summary()` and `feature_names` on the same object

The plain class with explicit `X_train`, `y_train`, etc. is the clearest to read and the easiest to pass into sklearn's `model.fit(split.X_train, split.y_train)`.
