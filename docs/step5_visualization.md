# Step 5: Visualization — Detailed Walkthrough

## Overview

Step 5 answers: **"What does the data and model performance actually look like?"**

Visualization is not an afterthought — it's a core deliverable. Every figure serves a specific diagnostic purpose: catching data issues, verifying preprocessing correctness, and communicating model quality. The figures are embedded in the final report (Step 6) and saved as standalone PNGs for inspection.

---

## Architecture: Three Stages of Figures

Figures are organized by pipeline stage, mirroring the output structure from ARCHITECTURE.md Section 12.1:

```
figures/
├── 01_profiling/          ← generated after Step 1 (data loading + profiling)
├── 02_preprocessing/      ← generated after Step 2 (preprocess + split)
└── 03_training/           ← generated after Step 3 (baselines + evaluation)
```

Each stage has its own module in `co_scientist/viz/`:
- `profiling.py` → `01_profiling/`
- `preprocessing.py` → `02_preprocessing/`
- `training.py` → `03_training/`

The `__init__.py` sets `matplotlib.use("Agg")` — the non-interactive backend — because we're a CLI tool that runs headless. Without this, matplotlib would try to open GUI windows and crash on servers.

---

## Stage 1: Profiling Figures (`01_profiling/`)

### Figure: Target Distribution

**What it shows:** The distribution of the thing we're trying to predict.

- **Classification:** Horizontal bar chart of class counts, sorted descending, color-coded by class. Immediately reveals class imbalance.
- **Regression:** Histogram with mean (red dashed) and median (orange dashed) vertical lines. Shows skewness, outliers, and whether the target needs transformation.

**Why it matters in ML:** Class imbalance directly affects model performance and metric choice. A model that always predicts the majority class gets high accuracy but is useless. Seeing the distribution up front informs:
- Whether to use stratified splitting (we do)
- Whether to use class weights
- Which metric to trust (macro F1 for imbalanced, accuracy for balanced)

For regression, a heavily skewed target suggests log-transforming it before training.

### Figure: Sequence Length Distribution (RNA/DNA/Protein only)

**What it shows:** Histogram of input sequence lengths with mean line.

**Why it matters:** K-mer feature extraction (Step 2) normalizes by sequence length, but extreme length variation can still cause issues. Very short sequences may not have enough k-mers to be informative. Very long sequences dominate certain k-mer counts. This figure helps diagnose unexpected model behavior later.

### Figure: Expression Overview (Cell Expression only)

**What it shows:** Two side-by-side histograms:
1. **Genes detected per cell** — how many genes have non-zero expression in each cell
2. **Total counts per cell** — total read count per cell

**Why it matters:** These are the two most important quality metrics in single-cell RNA-seq:
- Cells with very few detected genes might be empty droplets or dead cells
- Cells with extremely high total counts might be doublets (two cells captured together)
- The overall distribution shape tells you about sequencing depth and data quality
- After log1p + HVG preprocessing, these distributions change — comparing before/after catches bugs

---

## Stage 2: Preprocessing Figures (`02_preprocessing/`)

### Figure: Feature Variance Distribution

**What it shows:** Histogram of per-feature variance across the training set, with annotation for near-zero-variance features (var < 0.01).

**Why this specific diagnostic:** After k-mer extraction or HVG selection, some features may have near-zero variance — they're essentially constant across all samples. These features:
- Add noise without signal
- Can cause numerical issues in some models (division by near-zero in normalization)
- Slow down training without benefit

The count annotation ("N features with var < 0.01") is a quick health check. A large number suggests the preprocessing may need adjustment (e.g., more aggressive feature selection).

**Engineering detail:** We compute variance on the training set only, not the full dataset. This prevents data leakage — we don't want validation/test statistics influencing our understanding of the training data.

### Figure: Split Distribution

**What it shows:** Verifies that the train/val/test splits have consistent target distributions.

- **Classification:** Grouped bar chart showing class proportions (%) per split. All three bars for each class should be roughly equal height.
- **Regression:** Overlapping histograms of target values per split. The three distributions should overlap closely.

**Why it matters:** This is a **correctness check**, not a model diagnostic. If the splits have very different distributions, something went wrong:
- Random splitting without stratification on an imbalanced dataset
- Predefined splits that aren't representative (e.g., time-based splits where the target distribution shifts)
- A bug in the splitting code

For classification, the label encoder is used to show actual class names (e.g., "alpha", "beta") instead of integer codes — making the figure interpretable.

---

## Stage 3: Training Figures (`03_training/`)

### Figure: Model Comparison

**What it shows:** Horizontal bar chart comparing all baseline models on the primary metric, color-coded by tier (trivial=gray, simple=blue, standard=green, advanced=red).

**Why this figure design:**
- **Horizontal bars** because model names can be long ("XGBClassifier" vs "DummyClassifier")
- **Sorted by metric** so the best model is always at top
- **Color by tier** to visually confirm that higher tiers outperform lower tiers — if they don't, it's a red flag
- **Annotated values** on each bar for precise comparison

**What to look for:**
- **Trivial ≈ Simple:** The simple model (logistic regression / ridge) isn't learning much beyond the baseline. May indicate features aren't informative, or the problem is genuinely hard.
- **Simple ≈ Standard:** XGBoost isn't helping over linear models. The decision boundary is likely linear, or features are already well-separated. Good news: simpler model is sufficient.
- **Big jump Trivial → Simple:** Features are informative. The problem is learnable.
- **Big jump Simple → Standard:** Non-linear relationships matter. Tree-based models capture interactions that linear models miss.

### Figure: Feature Importance (tree-based models only)

**What it shows:** Top 20 features ranked by importance from the best tree-based model (XGBoost's `feature_importances_`).

**Why it matters biologically:** For RNA datasets with k-mer features, high-importance k-mers may correspond to known regulatory motifs. For cell expression data, high-importance genes are potential biomarkers. This bridges ML performance and biological interpretability — a key requirement of the architecture.

**Engineering detail:** Only generated when the best model has a `feature_importances_` attribute (tree-based models). Linear models have `coef_` instead (not visualized yet — that's a future improvement). Dummy models have neither, so this figure is skipped if the trivial baseline happens to "win."

---

## Integration with the Pipeline

The viz calls are wired directly into `cli.py` at the natural pipeline boundaries:

```python
# After Step 1: profile
profiling_figs = generate_profiling_figures(dataset, profile, config.task_output_dir)

# After Step 2: split
preproc_figs = generate_preprocessing_figures(split, profile, config.task_output_dir)

# After Step 3: evaluate
training_figs = generate_training_figures(results, trained_models, split, eval_config, profile, config.task_output_dir)
```

Each function returns a list of saved file paths, so the CLI can report how many figures were generated. This is important because the number varies by modality (RNA gets sequence length plots, expression gets the overview plot, etc.).

---

## Design Decisions

### Why matplotlib + seaborn, not plotly/altair?

1. **Static output:** We save PNGs for embedding in markdown reports. No need for interactivity.
2. **Server-friendly:** matplotlib with Agg backend runs headless. No browser needed.
3. **Publication quality:** matplotlib produces figures suitable for scientific papers.
4. **seaborn on top:** Provides nicer defaults (color palettes, statistical plot types) without replacing matplotlib.

### Why DPI 150?

Balances file size and readability. DPI 72 looks blurry on retina displays. DPI 300 is publication-quality but creates large files that slow down report rendering. DPI 150 is sharp enough for screen viewing and reasonable for print.

### Why `plt.close(fig)` after every save?

matplotlib accumulates figures in memory. In a long-running pipeline generating many figures, this causes memory leaks. Closing each figure after saving is defensive programming.

### Why color by tier in model comparison?

The tier system (trivial → simple → standard → advanced) is a core architectural concept. Coloring by tier makes the expected progression visible. If a "trivial" gray bar is competitive with a "standard" green bar, that's immediately visually jarring — as it should be, because it means the more complex model isn't earning its complexity.

---

## Test Results

Both test datasets generate figures successfully:

**RNA/translation_efficiency_muscle (regression):**
- `01_profiling/target_distribution.png` — histogram of MRL values
- `01_profiling/sequence_lengths.png` — UTR length distribution
- `02_preprocessing/feature_variance.png` — k-mer feature variances
- `02_preprocessing/split_distribution.png` — overlapping target histograms per split
- `03_training/model_comparison.png` — Spearman comparison bar chart
- `03_training/feature_importance.png` — top k-mer features from XGBoost

**expression/cell_type_classification_segerstolpe (classification):**
- `01_profiling/target_distribution.png` — cell type class distribution bar chart
- `01_profiling/expression_overview.png` — genes/cell and counts/cell histograms
- `02_preprocessing/feature_variance.png` — gene expression feature variances
- `02_preprocessing/split_distribution.png` — class proportion bars per split
- `03_training/model_comparison.png` — accuracy comparison bar chart
- `03_training/feature_importance.png` — top genes from XGBoost
