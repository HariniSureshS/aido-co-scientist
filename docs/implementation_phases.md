# AIDO Co-Scientist — Implementation Phases

## Phase A: Minimal End-to-End Pipeline (Steps 0-6) ✅ COMPLETE

| Step | Description | Key Files |
|------|-------------|-----------|
| **Step 0** | Project Skeleton — pyproject.toml, CLI entry point, config | `cli.py`, `config.py` |
| **Step 1** | Data Loading + Profiling — HuggingFace loader, DatasetProfile, modality detection | `data/{loader.py, profile.py, types.py}` |
| **Step 2** | Preprocessing + Splitting — k-mer features for RNA, log1p+HVG for expression, 70/15/15 split | `data/{preprocess.py, split.py}` |
| **Step 3** | Baselines + Evaluation — trivial → simple → standard models, metrics, comparison table | `modeling/{registry.py, trainer.py, types.py}`, `evaluation/{metrics.py, auto_config.py}` |
| **Step 4** | Model Export — pickle best model, generate standalone train.py, predict.py, requirements.txt | `export/{exporter.py, code_gen.py, templates/}` |
| **Step 5** | Visualization — figures at each pipeline stage saved to figures/ | `viz/{profiling.py, preprocessing.py, training.py, final.py}` |
| **Step 6** | Report Generation — Markdown report from Jinja template | `report/{generator.py, template.md.jinja}` |

**Milestone:** Produces complete output (model, code, figures, report) for both datasets.

---

## Phase B: Quality Improvements (Steps 7-12) ✅ COMPLETE

| Step | Description | Key Files | Doc |
|------|-------------|-----------|-----|
| **Step 7** | Predefined Defaults YAML — curated configs per modality/task/dataset | `defaults.yaml`, `defaults.py` | `docs/step7_defaults.md` |
| **Step 8** | HP Search (Optuna) — Bayesian optimization after baselines | `modeling/hp_search.py` | `docs/step8_hp_search.md` |
| **Step 9** | Advanced Models (MLP) — PyTorch MLP with sklearn-compatible interface | `modeling/mlp.py` | `docs/step9_mlp.md` |
| **Step 10** | Experiment Log + Checkpointing — resume interrupted runs | `experiment_log.py`, `checkpoint.py` | `docs/step10_experiment_log.md` |
| **Step 11** | Guardrails — 5 categories of scientific discipline checks | `guardrails.py` | `docs/step11_guardrails.md` |
| **Step 11b** | Adaptive Complexity — Google-inspired 0-10 scoring, scales HP/budget | `complexity.py` | `docs/step11b_adaptive_complexity.md` |
| **Step 11c** | Error Recovery — CellAgent-inspired structured error capture + fallbacks | `resilience.py` | `docs/step11c_error_recovery.md` |
| **Step 12** | Rich Dashboard — styled panels, model table, step progress, summary | `dashboard.py` | `docs/step12_rich_dashboard.md` |

**Milestone:** Polished CLI experience with guardrails, adaptive complexity, and resilient execution.

---

## Phase C: Expanded Model Set & Ensembles (Steps 13-16) ⬅️ CURRENT

| Step | Description | Key Files | Status |
|------|-------------|-----------|--------|
| **Step 13** | Tree Model Diversity — Random Forest, LightGBM, Elastic Net | `modeling/registry.py`, `defaults.yaml` | ✅ Done |
| **Step 14** | Multi-Scale BioCNN — custom 1D CNN with parallel kernel branches [3,5,7,9] for biological sequences | `modeling/bio_cnn.py` | ✅ Done |
| **Step 15** | Residual MLP — skip connections + BatchNorm for tabular data | `modeling/residual_mlp.py` | Deferred (basic MLP sufficient for now) |
| **Step 16** | Stacking Ensemble — meta-learner on out-of-fold base model predictions | `modeling/ensemble.py` | ✅ Done |

### Step 13: Tree Model Diversity
- Add **Random Forest** (sklearn) — different inductive bias from XGBoost, less prone to overfitting
- Add **LightGBM** — faster training, histogram-based splitting, different regularization
- Add **Elastic Net** (sklearn) — L1+L2 regularization, handles multicollinearity in k-mer features
- Update `defaults.yaml` with configs for both classification and regression
- Add HP search spaces for all new models
- Quick wins — all sklearn-compatible, no custom code needed

### Step 14: Multi-Scale BioCNN
- Custom 1D convolutional network for biological sequences
- **Parallel conv branches** with kernel sizes [3, 5, 7, 9] — captures codons, motifs, binding sites, regulatory elements simultaneously
- Global average + max pooling for length invariance
- Residual FC layers with dropout
- Inspired by Inception networks, adapted for 1D bio sequences
- Only activated for sequence modalities (RNA, DNA, protein)
- sklearn-compatible interface (fit/predict/predict_proba)

### Step 15: Residual MLP
- Enhanced MLP with **skip connections** + **BatchNorm**
- Addresses the known MLP-vs-trees gap on tabular data
- Replaces basic MLP as the advanced neural option for expression/tabular datasets
- Deeper without vanishing gradient issues

### Step 16: Stacking Ensemble
- Trains all base models → generates **out-of-fold predictions** via cross-validation
- Trains a **meta-learner** (Ridge/Logistic regression) on stacked predictions
- This IS "building a new model" — the ensemble is a novel model combining all base model strengths
- Typically outperforms any individual model by 2-5%
- Falls back gracefully with as few as 2 base models
- Exports as single pickle containing all base models + meta-learner

### Step 16b: AIDO Foundation Model Embeddings (GPU-gated)
- GPU-gated embedding extraction using GenBio AIDO foundation models
- Maps modalities to AIDO models: RNA → `aido_rna_1b600M`, DNA → `aido_dna_300m`, Protein → `aido_protein_16B`, Cell → `aido_cell_100m`
- Foundation tier models: `embed_xgboost`, `embed_mlp` — trained on frozen AIDO embeddings
- Graceful degradation: no GPU → tier skipped, `modelgenerator` not installed → embeddings skipped, extraction fails → warning + continue
- AIDO-aware export templates: standalone `train.py`/`predict.py` include embedding extraction code
- Key files: `modeling/foundation.py`, `data/preprocess.py`, `data/split.py`, `modeling/registry.py`, `modeling/trainer.py`, `export/exporter.py`

### Step 16c: AIDO End-to-End Fine-Tuning (GPU-gated)
- Unfreezes last N layers of AIDO backbone + lightweight task head (Linear→ReLU→Dropout→Linear)
- Mixed precision (fp16) for memory efficiency on H100, AdamW with differential learning rates (backbone LR vs head LR × 10)
- Gradient clipping, early stopping on validation loss, best checkpoint restoration
- `aido_finetune` is a sequence model (takes raw sequences like `bio_cnn`), NOT an embedding model
- Sklearn-compatible interface: `AIDOFinetuneClassifier`/`AIDOFinetuneRegressor` with fit/predict/predict_proba
- HP search space: unfreeze_layers, head_hidden, head_dropout, learning_rate, weight_decay, batch_size
- Export: state_dict + config (not pickle); standalone train.py/predict.py with inline AIDO fine-tuning code
- Key files: `modeling/aido_finetune.py`, `modeling/registry.py`, `defaults.yaml`, `export/exporter.py`

**Milestone:** 7-8 diverse models + stacking ensemble + foundation model embeddings. Custom architectures designed for biology.

---

## Phase D: Agent Layer (Steps 17-23)

| Step | Description | Key Files |
|------|-------------|-----------|
| **Step 17** | Agent Framework (rule-based) — refactor pipeline into agent pattern | `agents/{base.py, coordinator.py}` |
| **Step 18** | LLM Client (Claude API) — API wrapper, cost tracking, structured output | `llm/{client.py, prompts.py}` |
| **Step 19** | Data Analyst + ML Engineer agents — LLM-driven decisions | `agents/{data_analyst.py, ml_engineer.py}` |
| **Step 20** | Search Layer — web search + Semantic Scholar | `search/{web.py, papers.py}` |
| **Step 21** | Biology Specialist agent — biological context and interpretation | `agents/biology.py` |
| **Step 22** | Active Learning Analysis — class-level needs, uncertainty, feature gaps | `evaluation/active_learning.py` |
| **Step 23** | Full Iteration Loop + Interactive Mode — multi-step agent loop, conversational prompts at all decision points including ReAct loop. User can ask questions, give instructions, redirect strategy mid-loop. | `agents/loop.py`, `agents/interactive.py`, `agents/react.py`, `cli.py` |

**Milestone:** Full multi-agent pipeline. LLM makes strategy decisions. Interactive mode works.

---

## Phase E: ReAct Agent (Step 24) ✅ COMPLETE

| Step | Description | Key Files |
|------|-------------|-----------|
| **Step 24** | ReAct Agent Architecture — replaces Steps 3-5 with a single Thought → Action → Observation loop when LLM is available | `agents/{tools.py, react.py}`, `llm/prompts.py`, `cli.py` |

- **10 tools** wrap existing infrastructure (train_model, tune_hyperparameters, get_model_scores, analyze_errors, build_ensemble, inspect_features, get_rankings, design_model, finish, backtrack)
- **Scratchpad** tracks reasoning trace, compressed every 8 steps
- **Deterministic fallback** unchanged — runs when no API key or ReAct fails
- **Report** shows full Thought/Action/Observation trace (§4.9)

**Milestone:** True reasoning agent drives modeling. Full reasoning trace in reports.

---

## Phase F: Advanced Features (Step 25+)

| Feature | Source | Description | Key Files | Status |
|---------|--------|-------------|-----------|--------|
| Tournament Ranking | Google AI Co-Scientist | Elo-style ranking of model *approaches*, not just scores | `modeling/tournament.py` | ✅ Done |
| Agent Debate | Google AI Co-Scientist | DA and ML Engineer present competing arguments at 4 decision points (incl. preprocessing), budget reservation, fallback logging | `agents/debate.py` | ✅ Done |
| Experiment Tree Search | Sakana AI Scientist v2 | Branch/backtrack experiment paths instead of linear iteration | | |
| Biology Performance Context | Novel | Biology Specialist assessment + literature score ranges in report §4.4 | | |
| Pipeline Timeout | Production engineering | Global deadline + per-step budgets, graceful skip-to-export | | |
| Cross-Run Memory | Novel | Learn from previous datasets to improve future runs | | |
| AIDO FM Embeddings | GenBio AIDO | Foundation model embeddings as features — **requires GPU** | `modeling/foundation.py` | ✅ Done |
| Batch Processing | — | `co-scientist batch --datasets D1 D2 D3 --parallel 2` | | |
| Extended Models | — | SVM, KNN, FT-Transformer added to model registry | | |
| Custom Model Design | Novel | `design_model` tool — LLM generates custom PyTorch architectures at runtime based on dataset | | |
| Custom Model Export | Novel | State dict + source code export for dynamically generated PyTorch models | | |
| Validation Agent (Step 32) | Novel | Step-level validate + auto-fix after every pipeline step. Detects NaN, empty splits, syntax errors; fixes deterministically or via LLM. Foundation model routing for embedding/sequence models. Dashboard panel. | `validation.py` | ✅ Done | `docs/step32_validation_agent.md` |
| Dataset Resilience | Novel | HuggingFace config fuzzy-matching, split normalization, OpenMP segfault prevention, LLM recovery | `data/loader.py`, `__init__.py`, `resilience.py` | ✅ Done |
| Export Script Testing | Novel | Validation agent runs train.py and predict.py in subprocess to verify they work | `validation.py` | ✅ Done |
| QC Test Framework | — | End-to-end test script runs all datasets, deep-inspects outputs, generates report | `test_all_datasets.py` | ✅ Done |

**Milestone:** State-of-the-art multi-agent system with all architecture features implemented.

---

## Open Questions

1. **AIDO Foundation Models:** Is GPU access available for the homework? Are AIDO embeddings expected? (Email sent to recruiter for clarification)
2. **Hidden Datasets:** Will the test environment have GPU? What modalities might appear?
3. **Interactive Mode:** Currently in Phase D — should it be prioritized earlier since it's a homework requirement?
