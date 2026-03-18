# AIDO Co-Scientist

A CLI-based multi-agent system for automated machine learning on biological datasets (RNA, protein, genomics, expression).

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/HariniSureshS/aido-co-scientist.git && cd aido-co-scientist
pip install -e .

# 2. (Optional) Set API key — without it, pipeline runs fully deterministic
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run on a dataset
co-scientist run RNA/translation_efficiency_muscle
```

That's it. The pipeline will:
- Download the dataset from HuggingFace
- Profile the data (modality, task type, statistics)
- Train 8-12 models (tree models, neural nets, custom architectures)
- Tune the best performers
- Export the best model + standalone scripts
- Generate a full report

Output appears in `outputs/RNA__translation_efficiency_muscle_<timestamp>/`.

**Requirements:** Python >= 3.10, pip

---

## What You Get

Each run produces a self-contained output directory:

```
outputs/RNA__translation_efficiency_muscle_<timestamp>/
├── report.md                                   # Full analysis report
├── summary.pdf                                 # One-page visual summary
│
├── reproduce_translation_efficiency_muscle/     # Retrain from scratch
│   ├── train.py                                #   Fully standalone — downloads data, trains, evaluates
│   ├── evaluate.py                             #   Score predictions against ground truth
│   └── requirements.txt
│
├── inference_translation_efficiency_muscle/     # Run inference on new data
│   ├── predict.py                              #   Load model → predict
│   ├── model/
│   │   ├── best_model.pkl                      #   Trained model weights
│   │   ├── model_config.json                   #   Hyperparameters + metadata
│   │   └── label_encoder.pkl                   #   (classification only)
│   └── requirements.txt
│
├── logs/experiment_log.jsonl                   # Full experiment log
└── figures/                                    # All visualizations
```

```bash
# Reproduce training (no co-scientist installation needed)
cd outputs/<run_dir>/reproduce_<dataset>
pip install -r requirements.txt
python train.py

# Run inference on new data
cd outputs/<run_dir>/inference_<dataset>
python predict.py --input new_data.csv --output predictions.csv
```

---

## How It Works

### Pipeline

1. **Load & Profile** — Download from HuggingFace, detect modality/task, compute statistics
2. **Preprocess & Split** — k-mers for sequences, log1p + HVG for expression, 70/15/15 split
3. **Literature Search** — Semantic Scholar + PubMed for methods and benchmarks
4. **Train & Model** — ReAct agent dynamically selects and trains models
5. **HP Search** — Optuna Bayesian optimization on top performers
6. **Test Evaluation** — Final held-out test-set evaluation
7. **Export** — Save model + generate standalone scripts
8. **Report** — Markdown report with methods, results, agent reasoning, biology assessment

### Model Tiers

| Tier | Models | Requires |
|------|--------|----------|
| Trivial | Majority class / Mean predictor | CPU |
| Simple | Logistic/Ridge regression, Elastic Net, SVM, KNN | CPU |
| Standard | XGBoost, LightGBM, Random Forest | CPU |
| Advanced | MLP, FT-Transformer, LLM-designed custom models | CPU |
| Ensemble | Stacking ensemble (meta-learner over base models) | CPU |
| Foundation | AIDO embeddings + downstream classifiers, end-to-end fine-tuning | GPU |

Foundation tier activates automatically when a CUDA GPU is detected. On CPU, it's skipped.

### Multi-Agent System

When an Anthropic API key is provided, specialized agents orchestrate the pipeline:

- **ML Engineer** — Model selection, hyperparameter strategy, iteration decisions
- **Data Analyst** — Data quality assessment, feature engineering recommendations
- **Biology Specialist** — Domain-specific guidance, plausibility assessment
- **Research Agent** — Literature search, benchmark comparison
- **ReAct Agent** — Thought → Action → Observation loop that drives modeling
- **Validation Agent** — Auto-detects and fixes data issues, script errors, missing artifacts

Agents **debate** at key decision points before the pipeline acts. Without an API key, the pipeline runs fully deterministic.

---

## Available Datasets

| Path | Task | Modality |
|------|------|----------|
| `RNA/translation_efficiency_muscle` | Regression (Spearman) | RNA sequences |
| `RNA/translation_efficiency_hek` | Regression (Spearman) | RNA sequences |
| `RNA/translation_efficiency_pc3` | Regression (Spearman) | RNA sequences |
| `RNA/splice_site_prediction` | Classification | RNA sequences |
| `RNA/ncrna_family_classification` | Classification | RNA sequences |
| `RNA/expression_muscle` | Regression | RNA sequences |
| `RNA/expression_hek` | Regression | RNA sequences |
| `RNA/expression_pc3` | Regression | RNA sequences |
| `RNA/mean_ribosome_load` | Regression | RNA sequences |
| `expression/cell_type_classification_segerstolpe` | 13-class classification (Macro F1) | Gene expression (h5ad) |
| `expression/cell_type_classification_zheng` | Classification | Gene expression (h5ad) |

Or pass any HuggingFace dataset path directly: `co-scientist run genbio-ai/rna-downstream-tasks:mean_ribosome_load`

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — quick start and overview |
| [GUIDE.md](GUIDE.md) | Docker setup, GPU/Colab instructions, CLI reference, environment variables, testing |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Full map of every file and folder in the repo |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Detailed system design — agents, pipeline, data layer, modeling, evaluation, resilience (~80KB) |
| [research_paper.md](research_paper.md) | Research paper draft with methods, experimental results, and analysis |
| [docs/implementation_phases.md](docs/implementation_phases.md) | Roadmap of all 32 implementation steps |
| [docs/step0...step32](docs/) | Step-by-step implementation docs — one file per feature/module built |
| [docs/gpu_docs/](docs/gpu_docs/) | Foundation model integration: frozen embeddings, fine-tuning, hybrid features |
| [docs/challenges_and_learnings.md](docs/challenges_and_learnings.md) | Design tradeoffs, what worked, what didn't |
