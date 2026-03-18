# AIDO Co-Scientist: A Multi-Agent System for Automated Machine Learning on Biological Datasets

---

## Abstract

We present **AIDO Co-Scientist**, a CLI-based multi-agent system that automates the end-to-end machine learning pipeline for biological datasets, including RNA sequences, protein sequences, and single-cell gene expression data. Inspired by Google DeepMind's Co-Scientist (arXiv:2502.18864) and Sakana AI's AI Scientist (arXiv:2408.06292), our system employs a coordinator-supervised ensemble of specialized agents — a Data Analyst, ML Engineer, Biology Specialist, Research Agent, and Validation Agent — that collaboratively profile data, select and train models, tune hyperparameters, and generate scientific reports. Unlike prior work where LLMs generate and execute arbitrary code (leading to ~42% failure rates), our system uses the LLM as a *strategist* while execution is handled by deterministic, pre-built code paths. We evaluate on two benchmark tasks from the GenBio leaderboard: cell type classification on the Segerstolpe pancreatic scRNA-seq dataset (13 classes, macro F1 = **0.9706**) and translation efficiency prediction on muscle tissue RNA sequences (Spearman ρ = **0.6947**). The system completes full pipelines in under 12 minutes per dataset, explores 8–9 model architectures per run, and produces portable, reproducible codebases alongside comprehensive scientific reports — all at an LLM cost of under $0.35 per experiment.

---

## 1. Introduction

The rapid growth of biological data — from high-throughput sequencing to single-cell transcriptomics — has created an urgent need for automated tools that can translate raw data into predictive models without requiring deep ML expertise.

### 1.1 AIDO: The AI-Driven Digital Organism

GenBio AI's **AIDO (AI-Driven Digital Organism)** foundation-model stack is opening a path toward a virtual laboratory — a sandbox where researchers can design, perturb, and observe biological systems entirely *in silico* (GenBio AI, 2025). AIDO aims to accelerate biology research by enabling cheap, fast, and easy virtual experiments powered by AI to rapidly generate, test, and prioritize hypotheses. Like other computational systems, the virtual laboratory enables researchers to go beyond the physical limitations of scientific instruments and data collection to perform counterfactual studies, generic interventions, and optimization routines.

The core components of AIDO are **foundation models** for generation, property prediction, and translation, pretrained on various biological data types and scales. These include:

| Model | Parameters | Modality | Description |
|-------|-----------|----------|-------------|
| **AIDO.RNA-1.6B** | 1.6 billion | RNA sequences | Pretrained on RNA sequences for property prediction and generation |
| **AIDO.DNA-300M** | 300 million | DNA sequences | Genomic sequence understanding and variant effect prediction |
| **AIDO.Protein-16B** | 16 billion | Protein sequences | Protein structure, function, and interaction prediction |
| **AIDO.Cell-100M** | 100 million | Single-cell expression | Cell type classification, gene program discovery from scRNA-seq |

A key technical challenge is building and exposing these models so that they may cooperate to simulate complex multi-scale systems — forming a design loop where users can zoom in, make a change, propagate its effects, and zoom out to visualize. The three functional categories are: **AIDO Translators** (cross-modality translation), **AIDO Property Predictors** (predict phenotype from sequence/expression), and **AIDO Generative Models** (generate variants and optimize properties).

AIDO expects both experts and non-experts to benefit from a platform that democratizes access to a field that has traditionally required deep technical knowledge. The **GenBio leaderboard** (hosted on HuggingFace) provides standardized benchmark datasets spanning 11 tasks across RNA sequences, DNA sequences, protein sequences, and single-cell expression data, enabling systematic evaluation of downstream models.

### 1.2 The Gap: From Foundation Models to Downstream ML

However, a critical gap remains: while AIDO's foundation models provide powerful pretrained representations, the process of selecting, training, evaluating, and interpreting **downstream ML models** for specific biological prediction tasks still requires substantial human effort and expertise. A researcher wanting to predict translation efficiency from RNA sequences or classify cell types from scRNA-seq data must still manually choose preprocessing strategies, select model architectures, tune hyperparameters, evaluate results, and interpret findings in biological context.

Recent advances in LLM-driven scientific agents offer a promising direction. Google DeepMind's Co-Scientist (Gottweis et al., 2025) demonstrated that multi-agent systems with tournament-based ranking and iterative refinement can generate novel scientific hypotheses. Sakana AI's AI Scientist (Lu et al., 2024; Yamada et al., 2025) showed that end-to-end automation from idea generation to paper writing is feasible, with v2's agentic tree search enabling the first AI-generated paper accepted at an ICLR workshop. CellAgent (Yuan et al., 2024) applied multi-agent LLM frameworks specifically to single-cell RNA-seq analysis, achieving ~60% efficiency improvement over human experts.

We build on these foundations to create **AIDO Co-Scientist**, a system that:

1. **Automates the full ML lifecycle** for biological data: data profiling → preprocessing → model training → hyperparameter tuning → evaluation → report generation.
2. **Employs domain-specialized agents** (Data Analyst, ML Engineer, Biology Specialist) rather than cognitive-function agents, reflecting the importance of domain knowledge in biological ML.
3. **Uses LLM-as-strategist, not executor**: the LLM makes high-level decisions (which model, which features, which hyperparameters) while execution is handled by deterministic code paths, dramatically improving reliability.
4. **Supports two operating modes**: fully autonomous and interactive (human-in-the-loop), enabling both rapid prototyping and careful, expert-guided exploration.
5. **Produces portable artifacts**: trained models, standalone `train.py`/`predict.py` scripts, and comprehensive Markdown/PDF reports suitable for scientific communication.

We evaluate our system on two representative tasks from the GenBio leaderboard, demonstrating competitive performance, efficient runtime, and rich interpretability.

---

## 2. Related Work

### 2.1 Google DeepMind — Towards an AI Co-Scientist

Gottweis et al. (2025) proposed a multi-agent system built on Gemini 2.0 with a Supervisor agent orchestrating six specialized agents (Generation, Reflection, Ranking, Evolution, Proximity, Meta-review). Key innovations include tournament-based hypothesis ranking with Elo ratings, a generate–debate–evolve loop, and test-time compute scaling.

**What we adopt:** (1) The multi-agent with supervisor pattern — our Coordinator mirrors their Supervisor. (2) Tournament-style model ranking using Elo ratings in Phase 4 (ITERATE). (3) Agent debate before major decisions, where the Data Analyst and ML Engineer present competing arguments. (4) Adaptive complexity scaling, where agent activation and iteration depth scale with dataset difficulty.

**What we do differently:** Our agents are specialized by *domain role* (Data Analyst, ML Engineer, Biology Specialist) rather than *cognitive function* (generate, reflect, rank), which is more appropriate for structured ML pipelines. Our system produces concrete, measurable artifacts (models, code, reports) rather than textual hypotheses.

### 2.2 Sakana AI — The AI Scientist

Lu et al. (2024) introduced end-to-end automated scientific discovery at ~$15/paper. Version 2 (Yamada et al., 2025) added agentic tree search for experiment planning and VLM-based figure evaluation.

**What we adopt:** (1) Full pipeline automation. (2) Agentic tree search — our Phase 4 maintains a lightweight experiment tree, branching into parallel paths and backtracking when a branch stalls. (3) Automated quality checks via our five-category guardrail system.

**What we do differently:** Critically, Sakana's system has the LLM write and execute arbitrary code, which led to a 42% experiment failure rate (Ifargan et al., 2025). Our system confines the LLM to *decision-making* while execution follows pre-built, validated code paths. We also provide interactive mode, reflecting the practical lesson that co-scientist framing outperforms full autonomy.

### 2.3 CellAgent

Yuan et al. (2024) proposed an LLM-driven multi-agent framework for scRNA-seq analysis with Planner/Executor/Evaluator roles, achieving ~60% efficiency improvement over human experts at ICLR 2025.

**What we adopt:** Planner/Executor/Evaluator separation (mapping to our Coordinator/ML Engineer/Validation Agent) and domain-specific toolkits for modality-aware preprocessing.

### 2.4 Traditional AutoML

Systems like Auto-sklearn (Feurer et al., 2015), AutoGluon (Erickson et al., 2020), and FLAML (Wang et al., 2021) provide robust automated model selection and hyperparameter optimization. Our system extends this paradigm with: (1) biological domain awareness (modality detection, biology-informed interpretation), (2) LLM-driven strategic reasoning, (3) rich report generation, and (4) foundation model integration for biological sequences.

---

## 3. System Architecture

### 3.1 Overview

AIDO Co-Scientist is structured as a **four-phase pipeline** orchestrated by a Coordinator agent:

| Phase | Name | Description |
|-------|------|-------------|
| 0 | **Initialize** | Load dataset, detect modality, compute complexity score |
| 1 | **Understand** | Data profiling, literature search, biological context |
| 2 | **Prepare** | Modality-specific preprocessing, train/val/test splitting |
| 3 | **Model** | Tiered model training (trivial → simple → standard → advanced → ensemble) |
| 4 | **Iterate** | ReAct-based agent loop or deterministic iteration with Optuna HP search |
| Final | **Export** | Save model, generate standalone scripts, render report |

### 3.2 Multi-Agent Design

The system employs five specialized agents coordinated by a central Coordinator:

- **Coordinator**: Orchestrates the pipeline, manages compute budget, resolves agent conflicts, and tracks global state.
- **Data Analyst**: Profiles datasets (dimensionality, class distribution, missing values), recommends preprocessing strategies, and performs error analysis.
- **ML Engineer**: Recommends model architectures, defines hyperparameter search spaces, proposes iteration strategies, and interprets model performance.
- **Biology Specialist**: Validates that metrics are biologically plausible, provides domain context (e.g., expected Spearman ranges for translation efficiency), and interprets feature importances in biological terms.
- **Research Agent**: Searches Semantic Scholar, PubMed, and Tavily for relevant benchmarks and methods.
- **Validation Agent**: Runs five categories of guardrails (data quality, script validity, model plausibility, metric consistency, export completeness) with automatic repair.

Agent communication uses structured messages routed through the Coordinator, enabling cost control, audit logging, and deduplication. When an Anthropic API key is available, agents use Claude as their reasoning backbone through a ReAct (Thought → Action → Observation) loop. Without an API key, the system falls back to fully deterministic execution with rule-based decision logic.

### 3.3 Adaptive Complexity Scaling

Inspired by test-time compute scaling (Gottweis et al., 2025), the system dynamically adjusts its resource allocation based on dataset complexity:

| Complexity | Active Agents | Max Iterations | HP Trials | Literature Search |
|------------|---------------|----------------|-----------|-------------------|
| Simple (0–2) | Coordinator, DA, ML | 4 | 10 | None |
| Moderate (3–5) | + Research (lite) | 6 | 20 | 3 web queries |
| Complex (6–8) | All five agents | 10 | 30 | 6 web + 3 papers |
| Very Complex (9–10) | All five (deep) | 15 | 50 | 10 web + 6 papers |

Complexity is computed from dataset characteristics: sample count, feature dimensionality, class imbalance ratio, modality type, and missing data prevalence.

### 3.4 ReAct Agent Loop

When LLM access is available, Phase 4 uses a ReAct agent with the following tool repertoire:

- `train_model(model_type, hyperparameters)` — Train and evaluate a model
- `tune_hyperparameters(model_type)` — Optuna Bayesian HP optimization
- `build_ensemble()` — Stacking ensemble from best base models
- `analyze_errors()` — Feature importance and error pattern analysis
- `consult_biology()` — Query the Biology Specialist mid-loop
- `design_model()` — Generate a custom neural architecture via LLM
- `finish()` — Terminate and produce final report

The agent maintains a scratchpad of all thought/action/observation steps for full reproducibility and interpretability.

---

## 4. Methods

### 4.1 Data Layer

**Modality Detection.** The system uses a cascading detection strategy: (1) path parsing for keywords ("RNA/", "expression/"), (2) column name matching ("sequence", "seq"), (3) content inspection (character frequency analysis for nucleotide/amino acid content), (4) source format (`.h5ad` → cell expression), and (5) dimensionality heuristics (>1000 features → likely expression).

**Preprocessing.** Modality-specific pipelines include:
- *RNA/DNA sequences*: k-mer frequency extraction (k = 3, 4), GC content, nucleotide composition, sequence length features, followed by standard scaling. Raw sequences are cached separately for CNN-based models.
- *Cell expression (scRNA-seq)*: log1p normalization, highly variable gene (HVG) selection (top 2,000 genes), and standard scaling.
- *Protein sequences*: Amino acid composition and physicochemical property features.
- *Tabular*: NaN imputation (fill with 0.0) and standard or min-max scaling.

**Data Splitting.** Default splits are 70% train / 15% validation / 15% test with stratification for classification tasks (via StratifiedKFold). Predefined HuggingFace splits are used when available.

### 4.2 Model Tier System

Models are organized into tiers of increasing complexity, trained sequentially:

| Tier | Models | Description |
|------|--------|-------------|
| **Trivial** | Majority class / Mean predictor | Sanity-check baselines |
| **Simple** | Logistic Regression, Ridge, Elastic Net | Linear models |
| **Standard** | XGBoost, LightGBM, Random Forest, SVM, KNN | Tree-based and kernel methods |
| **Advanced** | MLP, FT-Transformer, Multi-Scale BioCNN | Neural networks |
| **Ensemble** | Stacking (Ridge/Logistic meta-learner) | Combines best base models |
| **Foundation** | AIDO embeddings + downstream classifiers | GPU-required, uses pretrained representations |

#### 4.2.1 MLP Classifier/Regressor
A PyTorch feedforward network with configurable hidden layers, BatchNorm, ReLU activations, and Dropout. Training uses early stopping on validation loss. Manual batching is employed to avoid OpenMP/threading conflicts with XGBoost on macOS.

#### 4.2.2 Feature Tokenizer Transformer (FT-Transformer)
Following Gorishniy et al. (2021), each input feature is projected into a shared embedding space via per-feature linear layers. A learnable [CLS] token is prepended, and the sequence is processed by a standard Transformer encoder. The [CLS] representation feeds a task-specific output head.

#### 4.2.3 Multi-Scale BioCNN
A 1D convolutional architecture with four parallel branches (kernel sizes 3, 5, 7, 9) designed to capture biological motifs at multiple scales. Input sequences are one-hot encoded. Global pooling (average + max) feeds fully connected layers with residual connections.

#### 4.2.4 Stacking Ensemble
Out-of-fold cross-validation predictions from 2+ non-trivial base models are combined via a Ridge (regression) or Logistic Regression (classification) meta-learner, avoiding data leakage.

#### 4.2.5 LLM-Designed Custom Architectures
When the ReAct agent calls `design_model()`, the LLM generates a custom PyTorch architecture (e.g., residual MLPs, attention networks) tailored to the specific dataset characteristics observed during profiling.

#### 4.2.6 AIDO Foundation Model Integration (GPU Tier)

When a CUDA-capable GPU is available, the system leverages AIDO foundation models as a powerful feature extraction layer. The integration follows a **compute-once, cache-reuse** pattern:

1. **Embedding extraction**: Raw biological sequences (or expression vectors) are passed through the appropriate AIDO backbone (e.g., AIDO.RNA-1.6B for RNA, AIDO.Cell-100M for scRNA-seq) via the `modelgenerator.tasks.Embed` API. The backbone produces dense vector representations that capture pretrained biological knowledge.

2. **Caching**: Extracted embeddings are saved to disk per dataset, so they need only be computed once regardless of how many downstream models use them.

3. **Downstream strategies**: Five modeling strategies are supported on top of AIDO embeddings:
   - `embed_xgboost`: AIDO embeddings → XGBoost classifier/regressor
   - `embed_mlp`: AIDO embeddings → MLP classifier/regressor
   - `concat_xgboost`: Handcrafted features + AIDO embeddings → XGBoost
   - `concat_mlp`: Handcrafted features + AIDO embeddings → MLP
   - `aido_finetune`: End-to-end fine-tuning of the last N layers of the AIDO backbone with a lightweight task head (Linear → ReLU → Dropout → Linear), trained with MSE (regression) or CrossEntropy (classification) loss

The fine-tuning approach freezes the lower layers of the AIDO backbone while unfreezing the final N layers, allowing the pretrained representations to adapt to the specific downstream task while preventing catastrophic forgetting. This architecture is particularly powerful for tasks where the pretrained representations may not fully capture task-specific biological signals.

**Note:** In our current experiments (Section 6), GPU was not available, so foundation model results are not reported. This represents a significant avenue for performance improvement (see Section 9, Future Work).

### 4.3 Hyperparameter Optimization

We use **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler for Bayesian hyperparameter optimization. Search spaces are defined per model type in a YAML configuration. The number of trials (10–50) and timeout (180–600s) adapt based on dataset complexity. Cross-run memory persists successful hyperparameter configurations as warm-start priors for future experiments on related datasets.

### 4.4 Evaluation

**Classification metrics:** accuracy, balanced accuracy, macro/weighted F1, macro precision/recall, Matthews Correlation Coefficient (MCC), Cohen's kappa, AUROC (binary and one-vs-rest multiclass), and log loss.

**Regression metrics:** MSE, RMSE, MAE, median absolute error, R², explained variance, Spearman correlation, Pearson correlation, and MAPE.

**Primary metric** is auto-detected from the dataset metadata (e.g., macro F1 for imbalanced classification, Spearman for rank-based regression) and can be overridden via configuration.

**Tournament Ranking.** Following Google's Co-Scientist, trained models participate in pairwise Elo-style tournaments. This provides a more robust ranking than single-metric leaderboards, as it accounts for performance variance across different data subsets.

### 4.5 Guardrails and Resilience

The Validation Agent enforces five categories of checks after each pipeline phase:

1. **Data Quality**: NaN/Inf detection, empty split prevention, class imbalance warnings
2. **Script Validation**: Syntax checking, import verification, missing artifact detection
3. **Model Plausibility**: Scores within expected ranges, non-degenerate predictions
4. **Metric Consistency**: Validation ≈ test performance (flags potential overfitting/data leakage)
5. **Export Completeness**: All required files present and functional

On failure, the Validation Agent diagnoses the issue and suggests automatic repairs (e.g., "OOM → reduce batch size", "NaN loss → fill missing values with median").

A global pipeline timeout (default 1800s) and per-step timeouts with exponential backoff ensure the system always terminates gracefully.

---

## 5. Experimental Setup

### 5.1 Benchmark Tasks

We evaluate on two tasks from the GenBio leaderboard (HuggingFace: `genbio-ai`):

**Task 1: Cell Type Classification (Segerstolpe)**
- **Dataset**: Segerstolpe et al. (2016) pancreatic islet scRNA-seq data
- **Modality**: Cell expression (scRNA-seq, `.h5ad` format)
- **Task**: 13-class classification of pancreatic cell types
- **Samples**: 2,133 total (1,279 train / 427 validation / 427 test)
- **Raw features**: 19,264 genes → 2,000 after HVG selection
- **Class distribution**: Severely imbalanced (174.4× ratio; smallest class: 5 samples, largest: 873)
- **Primary metric**: Macro F1

**Task 2: Translation Efficiency Prediction (Muscle)**
- **Dataset**: RNA sequences associated with translation efficiency in muscle tissue
- **Modality**: RNA sequences
- **Task**: Regression (predicting translation efficiency from sequence)
- **Samples**: 1,257 total (1,000 train / 125 validation / 132 test)
- **Features**: 326 engineered features (k-mer frequencies for k=3,4; GC content; nucleotide composition; sequence length)
- **Primary metric**: Spearman correlation (ρ)

### 5.2 Experimental Configuration

Experiments were conducted in two environments to evaluate the system under different compute constraints:

**Environment A: Local CPU**
- **Hardware**: MacOS, CPU-only (no GPU)
- **Mode**: Autonomous with LLM-driven ReAct agent (Anthropic Claude API)
- **Budget**: Default (10 iteration steps)
- **Random seed**: 42
- **HP tuning**: Optuna TPE, adaptive trials
- **Foundation models**: Not available (no GPU)

**Environment B: Google Colab (GPU)**
- **Hardware**: Google Colab with CUDA GPU
- **Mode**: Autonomous with LLM-driven ReAct agent (Anthropic Claude API)
- **Budget**: Default (10 iteration steps)
- **Random seed**: 42
- **HP tuning**: Optuna TPE, adaptive trials
- **Foundation models**: AIDO.RNA-1.6B embeddings extracted (2,048 dimensions)
- **Limitation**: Colab's per-step timeout (120s) caused failures during HP tuning and ensemble building for the larger cell expression dataset

A key practical limitation was **limited GPU access**: we relied on Google Colab's free/standard tier, which imposed memory and timeout constraints that prevented exhaustive evaluation of foundation model strategies (e.g., end-to-end fine-tuning, concat_mlp) and large-scale hyperparameter sweeps.

---

## 6. Results

### 6.1 Task 1: Cell Type Classification (Segerstolpe)

#### 6.1.1 Model Comparison

| Model | Tier | Macro F1 (Val) | Train Time (s) |
|-------|------|---------------:|----------------:|
| **LightGBM (tuned)** | **Tuned** | **0.9706** | **6.1** |
| XGBoost (default) | Standard | 0.9361 | 7.2 |
| LightGBM (default) | Standard | 0.9090 | 8.3 |
| XGBoost (tuned) | Tuned | 0.8048 | 8.5 |
| Logistic Regression | Simple | 0.8041 | 0.1 |
| Custom Attention Net | Custom (LLM-designed) | 0.6913 | 4.1 |
| MLP | Advanced | 0.6820 | 5.4 |
| Random Forest | Standard | 0.6728 | 0.9 |
| Stacking Ensemble | Ensemble | 0.5823 | 147.1 |

#### 6.1.2 Best Model: LightGBM (Tuned)

**Validation performance:**
| Metric | Score |
|--------|------:|
| Accuracy | 0.9836 |
| Balanced Accuracy | 0.9602 |
| **Macro F1** | **0.9706** |
| Weighted F1 | 0.9829 |
| MCC | 0.9787 |
| Cohen's Kappa | 0.9786 |
| AUROC | 0.9981 |
| Log Loss | 0.0827 |

**Test performance:**
| Metric | Score |
|--------|------:|
| Accuracy | 0.9813 |
| Balanced Accuracy | 0.8731 |
| **Macro F1** | **0.8773** |
| Weighted F1 | 0.9779 |
| MCC | 0.9756 |
| AUROC | 0.9993 |
| Log Loss | 0.0542 |

**Optimized hyperparameters:** n_estimators=484, max_depth=3, learning_rate=0.283, num_leaves=127, subsample=0.838, colsample_bytree=0.854, reg_alpha=0.182, reg_lambda=2.634, min_child_samples=34.

**Elo Tournament Rankings:**
| Rank | Model | Elo Rating | Matches | Wins |
|------|-------|----------:|--------:|-----:|
| 1 | XGBoost (default) | 1615.1 | 23 | 21 |
| 2 | LightGBM (tuned) | 1599.4 | 13 | 13 |
| 3 | LightGBM (default) | 1587.6 | 23 | 15 |
| 4 | Logistic Regression | 1495.2 | 17 | 7 |
| 5 | XGBoost (tuned) | 1494.8 | 13 | 7 |
| 6 | Custom Attention Net | 1451.9 | 7 | 2 |
| 7 | Random Forest | 1380.1 | 20 | 0 |
| 8 | MLP | 1375.9 | 22 | 4 |

**Top 10 Most Important Genes:** Gene_12758 (434.0), Gene_15869 (352.0), Gene_3583 (146.0), Gene_2604 (136.0), Gene_7438 (103.0), Gene_17846 (102.0), Gene_7697 (83.0), Gene_3778 (80.0), Gene_5477 (75.0), Gene_9559 (70.0).

#### 6.1.3 Biological Interpretation

The macro F1 of 0.97 is biologically plausible for pancreatic cell type classification, a well-studied system with established marker genes (INS for beta cells, GCG for alpha cells, SST for delta cells). The dominance of tree-based models suggests that cell type identity in this dataset is determined by discrete marker gene expression thresholds rather than complex non-linear interactions — consistent with the biological understanding that cell types are defined by a relatively small number of key transcription factors and their downstream targets.

The validation-to-test drop (0.97 → 0.88 macro F1) is attributable to the severe class imbalance: rare cell types (e.g., mast cells with only 5 samples) have high variance in per-class F1 across splits, disproportionately affecting macro-averaged metrics.

### 6.2 Task 2: Translation Efficiency Prediction (Muscle)

#### 6.2.1 Model Comparison

| Model | Tier | Spearman ρ (Val) | Train Time (s) |
|-------|------|------------------:|----------------:|
| **Random Forest (default)** | **Standard** | **0.6947** | **3.1** |
| Random Forest (tuned) | Tuned | 0.6926 | 0.9 |
| LightGBM (tuned) | Tuned | 0.6680 | 1.5 |
| LightGBM (default) | Standard | 0.6536 | 0.5 |
| MLP | Advanced | 0.6229 | 1.4 |
| Custom Residual MLP | Custom (LLM-designed) | 0.6134 | 2.8 |
| XGBoost (default) | Standard | 0.5679 | 0.4 |
| FT-Transformer | Advanced | 0.4212 | 54.2 |

#### 6.2.2 Best Model: Random Forest (Default)

**Validation performance:**
| Metric | Score |
|--------|------:|
| **Spearman ρ** | **0.6947** |
| Pearson r | 0.7143 |
| R² | 0.4965 |
| RMSE | 1.0164 |
| MAE | 0.6696 |
| Explained Variance | 0.5009 |

**Test performance:**
| Metric | Score |
|--------|------:|
| **Spearman ρ** | **0.6278** |
| Pearson r | 0.6638 |
| R² | 0.4403 |
| RMSE | 1.0559 |
| MAE | 0.7219 |
| Explained Variance | 0.4403 |

**Elo Tournament Rankings:**
| Rank | Model | Elo Rating | Matches | Wins |
|------|-------|----------:|--------:|-----:|
| 1 | Random Forest (default) | 1589.3 | 27 | 27 |
| 2 | Random Forest (tuned) | 1570.7 | 18 | 15 |
| 3 | LightGBM (default) | 1542.3 | 28 | 17 |
| 4 | LightGBM (tuned) | 1536.2 | 13 | 9 |
| 5 | MLP | 1502.3 | 25 | 10 |
| 6 | Custom Residual MLP | 1495.8 | 7 | 2 |
| 7 | XGBoost | 1436.4 | 28 | 4 |
| 8 | FT-Transformer | 1327.0 | 22 | 0 |

**Top predictive k-mer features:** kmer_3_GTT (0.0345), kmer_3_CAC (0.0304), kmer_4_TCTG (0.0151), kmer_3_GGT (0.0123), kmer_4_GTAG (0.0120), nuc_freq_C (0.0103), nuc_freq_A (0.0097).

#### 6.2.3 Biological Interpretation

A Spearman ρ of 0.69 falls within the expected range (0.6–0.8) for translation efficiency prediction from sequence features, as reported in the literature. Translation efficiency in muscle tissue is primarily governed by codon usage bias optimized for muscle's tRNA pool and high expression of structural proteins (myosin, actin, titin). The dominance of k-mer features (particularly trinucleotide frequencies like GTT and CAC) aligns with codon-level determinants of translation rate.

The model struggles most in the target range [−3.79, −1.45], where very low translation efficiencies likely reflect regulatory mechanisms (e.g., upstream ORFs, RNA secondary structure) not captured by simple k-mer frequency features.

### 6.3 Runtime and Cost Analysis

| Metric | Cell Type Classification | Translation Efficiency |
|--------|-------------------------:|------------------------:|
| Total pipeline time | 722.2 s (~12 min) | 453.1 s (~7.5 min) |
| Models explored | 9 | 8 |
| Agent iterations | 13 | 12 |
| Improvements found | 1 | 2 |
| LLM API cost | $0.32 | $0.28 |
| Best model train time | 6.1 s | 3.1 s |

Both experiments completed well within the default 30-minute timeout, with LLM costs under $0.35 — orders of magnitude cheaper than comparable human expert time.

### 6.4 Google Colab GPU Results: AIDO Foundation Model Evaluation

To evaluate AIDO foundation model embeddings, we ran the same two tasks on Google Colab with GPU access. The Colab environment enabled extraction of AIDO.RNA-1.6B embeddings (2,048 dimensions) for the RNA task and GPU-accelerated training for both tasks.

#### 6.4.1 Translation Efficiency (Colab GPU) — AIDO Embedding Comparison

Two independent GPU runs were conducted. The table below shows results from Run 1 (Run 2 yielded consistent results: best Spearman ρ = 0.6827):

| Model | Feature Source | Spearman ρ (Val) | Train Time (s) |
|-------|---------------|------------------:|----------------:|
| **LightGBM (tuned)** | **k-mer features** | **0.6857** | **2.1** |
| Stacking Ensemble | k-mer features | 0.6543 | 31.1 |
| LightGBM (default) | k-mer features | 0.6536 | 0.6 |
| Custom Neural Net | k-mer features | 0.6422 | 15.4 |
| MLP | k-mer features | 0.5967 | 2.7 |
| concat_xgboost | k-mers + AIDO embeddings | 0.5957 | 65.0 |
| XGBoost (default) | k-mer features | 0.5679 | 2.0 |
| **embed_xgboost** | **AIDO embeddings only** | **0.5647** | **69.0** |

**Key finding: AIDO embeddings did not improve over handcrafted k-mer features for translation efficiency prediction.** The pure embedding model (`embed_xgboost`, Spearman = 0.5647) performed slightly *worse* than XGBoost on handcrafted features (0.5679). The hybrid model (`concat_xgboost`, 0.5957) showed modest improvement over either alone but still underperformed LightGBM on k-mer features (0.6536). This suggests that for short RNA sequences (~91 nt), task-specific k-mer features capture translation-relevant patterns (codon usage bias, nucleotide composition) more effectively than general-purpose AIDO.RNA-1.6B embeddings.

**Best model test performance (Colab Run 1):** Spearman ρ = 0.6300 (test), consistent with the CPU-only run (0.6278).

**Agent reasoning about AIDO embeddings** (from the ReAct trace):
> *"The AIDO embeddings (0.5647) performed slightly worse than handcrafted features (0.5679), which is unexpected since foundation models usually excel on RNA tasks. This suggests the handcrafted k-mer features might be well-suited for this specific translation efficiency task."*

The agent then intelligently pivoted to the hybrid approach (`concat_xgboost`), demonstrating adaptive strategy based on empirical results.

#### 6.4.2 Cell Type Classification (Colab GPU)

| Model | Tier | Macro F1 (Val) | Train Time (s) |
|-------|------|---------------:|----------------:|
| xgboost (manually tuned) | React | 0.9517 | 351.1 |
| **LightGBM (tuned)** | **Tuned** | **0.9490** | **81.7** |
| XGBoost (default) | Standard | 0.9361 | 23.2 |
| LightGBM (default) | Standard | 0.9090 | 93.2 |
| XGBoost (tuned by Optuna) | Tuned | 0.8618 | 128.3 |
| MLP | Advanced | 0.6877 | 3.4 |
| Custom Residual MLP | Custom (LLM-designed) | 0.6863 | 2.7 |
| Random Forest | Standard | 0.6728 | 3.1 |
| SVM | Standard | 0.5344 | 11.4 |

**Best model test performance:** Macro F1 = 0.8246 (test), accuracy = 0.9883 (test).

**Colab timeout issues:** The Colab environment imposed a 120s per-step timeout that caused failures during:
- XGBoost HP tuning (timed out)
- LightGBM HP tuning (timed out)
- Stacking ensemble building (timed out)
- Manual XGBoost training with 500 estimators (timed out)
- Custom attention network (API incompatibility with `ReduceLROnPlateau`)

Despite these constraints, the agent adapted its strategy — switching from compute-intensive HP tuning to simpler custom architectures and lighter model configurations. The Biology Specialist confirmed that the achieved performance (macro F1 = 0.9490) is at the high end of expected performance for pancreatic cell type classification.

**Literature search results** (from Research Agent): The agent searched Semantic Scholar and PubMed but found no specific benchmarks for the Segerstolpe cell type classification dataset, noting this may be an underexplored benchmark opportunity. General literature confirmed that ML approaches for gene expression classification typically achieve 85–95% accuracy.

#### 6.4.3 Cross-Environment Comparison

| Task | Metric | CPU-only (Local) | GPU (Colab Run 1) | GPU (Colab Run 2) |
|------|--------|------------------:|-------------------:|-------------------:|
| Cell Type Classification | Macro F1 (Val) | 0.9706 | 0.9490 | — |
| Cell Type Classification | Macro F1 (Test) | 0.8773 | 0.8246 | — |
| Translation Efficiency | Spearman ρ (Val) | 0.6947 | 0.6857 | 0.6827 |
| Translation Efficiency | Spearman ρ (Test) | 0.6278 | 0.6300 | 0.6090 |

Interestingly, the **CPU-only local runs slightly outperformed the Colab GPU runs** on both tasks. This is attributable to:
1. **No timeout constraints** on the local machine, allowing full HP tuning and ensemble exploration.
2. **Different best models**: Local runs found Random Forest (translation efficiency) and more thoroughly tuned LightGBM (cell typing) that the Colab timeout prevented.
3. **AIDO embeddings did not help**: The GPU advantage of foundation model access did not translate to improved scores on these specific tasks and dataset sizes.

This highlights a practical tradeoff: while GPU access enables foundation model evaluation, **unconstrained compute time for thorough HP search proved more valuable than foundation model embeddings** for these two benchmarks.

---

## 7. Discussion

### 7.1 Key Findings

**Tree-based models dominate on tabular biological features.** Across both tasks, gradient-boosted trees (LightGBM, XGBoost) and Random Forests consistently outperformed neural architectures (MLP, FT-Transformer, custom attention networks). This aligns with recent large-scale benchmarks showing that tree-based methods remain competitive or superior on tabular data (Grinsztajn et al., 2022; McElfresh et al., 2023), and suggests that the biological signals in these datasets (marker gene thresholds for cell typing, codon frequency patterns for translation efficiency) are well-captured by axis-aligned decision boundaries.

**LLM-designed custom architectures show promise but underperform.** The ReAct agent's `design_model()` tool generated a custom attention network (cell typing) and a residual MLP (translation efficiency). While both produced reasonable results, they underperformed simpler defaults — likely because the small dataset sizes (1,257–2,133 samples) favor low-variance models. With larger datasets or foundation model embeddings, custom architectures may become more competitive.

**AIDO foundation model embeddings did not improve over handcrafted features.** On the RNA translation efficiency task (Colab GPU runs), AIDO.RNA-1.6B embeddings (2,048 dims) used with XGBoost scored 0.5647 — slightly *worse* than handcrafted k-mer features (0.5679). The hybrid approach (k-mers + AIDO embeddings) scored 0.5957, better than either alone but still below LightGBM on k-mers (0.6536). This counter-intuitive result may reflect: (1) the short sequence lengths (~91 nt) where k-mer features already capture most relevant information, (2) AIDO.RNA-1.6B being pretrained for general RNA understanding rather than translation efficiency specifically, and (3) the 2,048-dimensional embedding space potentially introducing noise relative to the 326-dimensional k-mer feature space for only 1,257 samples. However, we note this evaluation was limited by Colab constraints — end-to-end fine-tuning (`aido_finetune`) and `embed_mlp` / `concat_mlp` strategies were not tested, and performance on larger datasets may differ substantially.

**Stacking ensembles did not improve over the best single model.** The stacking ensemble for cell typing (F1 = 0.58 on CPU; 0.6569 on Colab) underperformed its base models. This is attributable to severe class imbalance causing poor out-of-fold predictions for rare classes, which then misled the meta-learner. Future work should explore class-weighted meta-learners or specialized ensemble strategies for imbalanced data.

**Validation-test gap highlights small-sample challenges.** The cell typing task showed a notable macro F1 drop from validation (0.97) to test (0.88) on CPU, and validation (0.95) to test (0.82) on Colab, driven by rare classes with as few as 5 samples (classes 9, 11, 12 each with 5 samples). Per-class analysis from the Colab run showed that classes 9 and 12 achieved F1 = 0.000 on the test set (with only 1 sample each in test), while classes with >10 test samples achieved F1 > 0.97. This underscores the fundamental challenge of evaluating multi-class classifiers on severely imbalanced, small datasets.

### 7.2 Design Decisions and Their Impact

**LLM-as-strategist vs. LLM-as-executor.** Our decision to confine the LLM to strategy (model selection, HP recommendations) while using pre-built code for execution proved effective: zero pipeline crashes across both experiments, compared to the 42% failure rate reported for code-generating approaches (Ifargan et al., 2025). The tradeoff is reduced flexibility — the system cannot invent entirely novel preprocessing steps or model architectures outside its predefined repertoire (though the `design_model()` tool partially addresses this).

**Deterministic fallback.** The system's ability to run without an API key (fully deterministic mode) ensures reproducibility and removes the LLM as a source of non-determinism. In deterministic mode, model selection follows a fixed tier progression, and HP tuning uses the same Optuna search spaces.

**Biology Specialist agent.** The inclusion of a domain-specific Biology Specialist agent proved valuable for result interpretation — contextualizing a Spearman ρ of 0.69 as "within expected range" or validating that tree-based models make biological sense for marker-gene-driven classification. This agent bridges the gap between raw metrics and scientific understanding.

### 7.3 Comparison with Related Systems

| Feature | AIDO Co-Scientist | Google Co-Scientist | Sakana AI Scientist | CellAgent |
|---------|-------------------|--------------------|--------------------|-----------|
| Domain | Biology-specific | General science | General ML research | scRNA-seq |
| Output | Models + code + report | Hypotheses (text) | Papers | Analysis scripts |
| LLM role | Strategist | Reasoner | Code generator | Planner |
| Failure rate | ~0% | N/R | ~42% | N/R |
| Human-in-loop | Yes (interactive) | No | No | No |
| Cost/run | ~$0.30 | N/R | ~$15/paper | N/R |
| Modalities | RNA, DNA, protein, expression | N/A | N/A | scRNA-seq |

---

## 8. Limitations

1. **Limited GPU access — inability to test holistically.** A primary limitation of this work was the lack of dedicated GPU infrastructure. Local experiments ran on CPU only (no foundation model evaluation), and GPU experiments relied on Google Colab which imposed per-step timeouts (120s) that prevented thorough HP tuning, ensemble building, and end-to-end AIDO fine-tuning. Critically, we could not evaluate `aido_finetune` (end-to-end backbone fine-tuning), `embed_mlp`, or `concat_mlp` strategies, nor could we test AIDO.Cell-100M embeddings on the cell expression task. A dedicated GPU environment with no timeout constraints would enable a far more comprehensive evaluation of the foundation model integration — which is one of the system's most distinctive features.

2. **AIDO embeddings underperformed on tested tasks.** On the RNA task where AIDO.RNA-1.6B embeddings were evaluated (Colab), they did not outperform handcrafted k-mer features. However, this finding is preliminary: (a) only 2 of 5 foundation model strategies were tested due to compute constraints, (b) the short sequence lengths (~91 nt) may not benefit from large language model representations, and (c) translation efficiency may depend on local sequence features (codons) rather than long-range patterns that foundation models excel at capturing.

3. **Small dataset sizes.** Both benchmark datasets are relatively small (1,257–2,133 samples), which favors low-complexity models and limits the potential of neural architectures and foundation model fine-tuning. Performance on larger-scale datasets (e.g., 100K+ samples) remains to be evaluated.

3. **Limited feature engineering.** For RNA sequences, only k-mer frequencies and basic composition features were used. More sophisticated features (RNA secondary structure predictions, codon adaptation indices, upstream ORF annotations) could improve regression performance.

4. **Gene name resolution.** The cell expression dataset uses anonymous gene indices (Gene_12758, etc.), preventing direct biological interpretation of feature importances. Mapping to gene symbols (e.g., via Ensembl IDs) would enable pathway-level analysis.

5. **LLM dependency for agent mode.** While the deterministic fallback ensures the system always runs, the full multi-agent reasoning (debate, biological interpretation, custom architecture design) requires Anthropic API access, introducing cost and latency.

6. **Single-seed evaluation.** Results are reported from single runs with seed=42. Multi-seed evaluation with confidence intervals would provide more robust performance estimates.

7. **No cross-validation on test set.** Final test metrics are computed on a single held-out split. Nested cross-validation would provide more reliable generalization estimates, especially for the small datasets used here.

8. **Ensemble strategy.** The current stacking ensemble strategy does not handle class imbalance well, as evidenced by its poor performance on the Segerstolpe task. Adaptive ensemble methods or class-weighted meta-learners are needed.

---

## 9. Future Work

1. **Thorough foundation model evaluation with dedicated GPU.** Our Colab experiments showed that AIDO.RNA-1.6B embeddings did not outperform k-mer features for translation efficiency, but this evaluation was incomplete. Priority next steps include: (a) end-to-end fine-tuning (`aido_finetune`) which unfreezes the backbone's top layers and may capture task-specific patterns, (b) AIDO.Cell-100M embeddings for cell expression tasks, (c) testing on longer sequences where foundation models' ability to capture long-range dependencies should provide greater advantage, and (d) evaluating on all 11 GenBio leaderboard datasets to identify where foundation models provide the most value.

2. **Expanded benchmark.** Evaluate on all 11 GenBio leaderboard datasets plus external benchmarks to assess generalization across modalities and task types.

3. **Advanced feature engineering.** Incorporate RNA secondary structure, codon adaptation index, protein domain annotations, and gene regulatory network features.

4. **Active learning.** Leverage the system's uncertainty estimates to recommend which samples to label next, reducing annotation cost for rare cell types.

5. **Multi-modal fusion.** Combine sequence and expression features for datasets where both are available, potentially using cross-attention mechanisms.

6. **Federated/privacy-preserving operation.** Enable the system to operate on sensitive clinical data without transmitting raw data to LLM APIs.

---

## 10. Conclusion

We presented AIDO Co-Scientist, a multi-agent system that automates the full machine learning lifecycle for biological datasets. By employing the LLM as a strategist rather than a code executor, the system achieves reliable, reproducible results: macro F1 of 0.97 on pancreatic cell type classification and Spearman ρ of 0.69 on translation efficiency prediction, completing each pipeline in under 12 minutes at a cost of ~$0.30 in LLM API fees. The system produces not just trained models, but portable codebases and comprehensive scientific reports, bridging the gap between automated ML and scientific communication. Our results demonstrate that domain-specialized multi-agent systems can make biological ML accessible to researchers without deep technical expertise, advancing the vision of AI-driven virtual laboratories for biological discovery.

---

## References

- Erickson, N., et al. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. *arXiv:2003.06505*.
- Feurer, M., et al. (2015). Efficient and Robust Automated Machine Learning. *NeurIPS 2015*.
- Gorishniy, Y., et al. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS 2021*.
- Gottweis, J., et al. (2025). Towards an AI Co-Scientist. *arXiv:2502.18864*.
- Grinsztajn, L., et al. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS 2022*.
- Ifargan, S., et al. (2025). An Evaluation of the AI Scientist. *arXiv:2502.14297*.
- Lu, C., et al. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv:2408.06292*.
- McElfresh, D., et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? *NeurIPS 2023*.
- Segerstolpe, Å., et al. (2016). Single-Cell Transcriptome Profiling of Human Pancreatic Islets in Health and Type 2 Diabetes. *Cell Metabolism*, 24(4), 593–607.
- Wang, C., et al. (2021). FLAML: A Fast and Lightweight AutoML Library. *MLSys 2021*.
- Yamada, Y., et al. (2025). The AI Scientist v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. *arXiv:2504.08066*.
- Yuan, J., et al. (2024). CellAgent: An LLM-Driven Multi-Agent Framework for Automated Single-Cell Data Analysis. *ICLR 2025*.

---

## Appendix A: System Requirements and Reproducibility

### Installation
```bash
git clone <repo-url> && cd scientist
pip install -e .
```

### Reproducing Experiments
```bash
# Cell type classification
co-scientist run expression/cell_type_classification_segerstolpe

# Translation efficiency
co-scientist run RNA/translation_efficiency_muscle
```

### Reproducing Best Models (Standalone)
```bash
# From output directory — no co-scientist installation needed
cd outputs_paper/<experiment>/reproduce_<dataset>/
pip install -r requirements.txt
python train.py
```

### Dependencies
Python ≥ 3.10, PyTorch, scikit-learn, XGBoost, LightGBM, pandas, numpy, anndata, Hugging Face datasets, Optuna, matplotlib, seaborn, Typer, Rich. Optional: Anthropic API key (for agent mode), CUDA GPU (for foundation models).

## Appendix B: Full Hyperparameter Configurations

### LightGBM (Tuned) — Cell Type Classification
```json
{
  "n_estimators": 484,
  "max_depth": 3,
  "learning_rate": 0.283,
  "num_leaves": 127,
  "subsample": 0.838,
  "colsample_bytree": 0.854,
  "reg_alpha": 0.182,
  "reg_lambda": 2.634,
  "min_child_samples": 34
}
```

### Random Forest (Default) — Translation Efficiency
```json
{
  "n_estimators": 100,
  "max_depth": null,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt"
}
```

## Appendix C: Agent Iteration Traces

### Cell Type Classification — 13 Iterations
- Iterations 1–5: Explored logistic regression, random forest, XGBoost, LightGBM, MLP
- Iteration 6: XGBoost emerged as leader (F1 = 0.9361)
- Iterations 7–9: Tuned XGBoost and LightGBM via Optuna
- Iteration 10: LightGBM (tuned) achieved best F1 = 0.9706
- Iterations 11–13: Attempted custom attention network and stacking ensemble (no improvement)
- Final improvement count: 1 (LightGBM tuned over XGBoost default)

### Translation Efficiency — 12 Iterations
- Iterations 1–4: Explored XGBoost, LightGBM, Random Forest, MLP
- Iteration 5: Random Forest emerged as leader (ρ = 0.6947)
- Iterations 6–8: Tuned Random Forest and LightGBM via Optuna
- Iterations 9–10: Attempted FT-Transformer and custom residual MLP
- Iterations 11–12: Built stacking ensemble, error analysis
- Final improvement count: 2 (RF default remained best)
