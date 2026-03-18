# AIDO Co-Scientist

A CLI-based multi-agent system for automated machine learning on biological datasets (RNA, protein, genomics, expression).

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd scientist
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

### Available Datasets

| Path | Task | Modality |
|------|------|----------|
| `RNA/translation_efficiency_muscle` | Regression (Spearman) | RNA sequences |
| `RNA/translation_efficiency_hek` | Regression (Spearman) | RNA sequences |
| `RNA/splice_site_prediction` | Classification | RNA sequences |
| `RNA/ncrna_family_classification` | Classification | RNA sequences |
| `expression/cell_type_classification_segerstolpe` | 13-class classification (Macro F1) | Gene expression (h5ad) |
| `expression/cell_type_classification_zheng` | Classification | Gene expression (h5ad) |
| `RNA/translation_efficiency_pc3` | Regression (Spearman) | RNA sequences |
| `RNA/expression_muscle` | Regression | RNA sequences |
| `RNA/expression_hek` | Regression | RNA sequences |
| `RNA/expression_pc3` | Regression | RNA sequences |
| `RNA/mean_ribosome_load` | Regression | RNA sequences |

Or pass any HuggingFace dataset path directly: `co-scientist run genbio-ai/rna-downstream-tasks:mean_ribosome_load`

---

## What You Get

Each run produces a self-contained output directory:

```
outputs/RNA__translation_efficiency_muscle_20260316_143022/
├── report.md                                   # Full analysis report
├── summary.pdf                                 # One-page visual summary
│
├── reproduce_translation_efficiency_muscle/     # Retrain from scratch
│   ├── train.py                                #   Downloads data, trains, evaluates
│   ├── evaluate.py                             #   Score predictions against ground truth
│   └── requirements.txt                        #   All dependencies
│
├── inference_translation_efficiency_muscle/     # Run inference on new data
│   ├── predict.py                              #   Load model → predict
│   ├── model/
│   │   ├── best_model.pkl                      #   Trained model weights
│   │   ├── model_config.json                   #   Hyperparameters + metadata
│   │   └── label_encoder.pkl                   #   (classification only)
│   └── requirements.txt                        #   Minimal deps for inference
│
├── logs/experiment_log.jsonl                   # Full experiment log
└── figures/                                    # All visualizations
```

### Reproduce Training

```bash
cd outputs/<run_dir>/reproduce_<dataset>
pip install -r requirements.txt
python train.py
```

`train.py` is **fully standalone** — downloads the dataset from HuggingFace, preprocesses (k-mers for sequences, log1p + HVG for expression), trains the exact model with the same hyperparameters, and evaluates on val + test sets. No co-scientist installation needed.

### Run Inference on New Data

```bash
cd outputs/<run_dir>/inference_<dataset>
pip install -r requirements.txt
python predict.py --input new_data.csv --output predictions.csv
```

The input CSV should contain preprocessed numeric features matching the model's expected shape (see `model/model_config.json` for details).

### Evaluate Predictions

```bash
cd outputs/<run_dir>/reproduce_<dataset>
python evaluate.py --predictions predictions.csv --ground-truth labels.csv
```

---

## Docker

### Prerequisites

```bash
# 1. Install Docker (if not already installed)
# macOS:
brew install --cask docker
# or download from https://docs.docker.com/get-docker/

# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER  # run docker without sudo (logout/login after)

# Verify Docker is running:
docker --version
```

### CPU (works everywhere)

```bash
# Build
docker build -t co-scientist .

# Run deterministic (no LLM cost)
docker run -v $(pwd)/outputs:/app/outputs \
    co-scientist run RNA/translation_efficiency_muscle --max-cost 0

# Run with LLM agents
docker run -v $(pwd)/outputs:/app/outputs \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    co-scientist run RNA/translation_efficiency_muscle

# Run all datasets (QC test)
docker run -v $(pwd)/outputs:/app/outputs \
    --entrypoint python co-scientist test_all_datasets.py --quick
```

### GPU (NVIDIA Docker — enables AIDO foundation models)

```bash
# 1. Install NVIDIA Container Toolkit (one-time setup, Linux only)
# This lets Docker access your GPU. Skip if already installed.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible from Docker:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 2. Build the GPU image
docker build -f Dockerfile.gpu -t co-scientist-gpu .

# 3. Run with GPU + LLM agents (full pipeline)
docker run --gpus all -v $(pwd)/outputs:/app/outputs \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    co-scientist-gpu run RNA/translation_efficiency_muscle --budget 10

# Run with GPU, no LLM (deterministic + foundation models)
docker run --gpus all -v $(pwd)/outputs:/app/outputs \
    co-scientist-gpu run RNA/translation_efficiency_muscle --max-cost 0

# Run all datasets on GPU
docker run --gpus all -v $(pwd)/outputs:/app/outputs \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    --entrypoint python co-scientist-gpu test_all_datasets.py --quick
```

**Note:** GPU Docker requires Linux with NVIDIA drivers + NVIDIA Container Toolkit. On macOS/Windows, use the CPU image or Google Colab for GPU.

---

## CLI Reference

### `co-scientist run`

| Option | Default | Description |
|--------|---------|-------------|
| `--mode`, `-m` | `auto` | `auto` (fully automated) or `interactive` (approve decisions) |
| `--budget`, `-b` | `10` | Max iteration steps for model improvement |
| `--max-cost` | `5.0` | Max LLM spend in USD |
| `--tree-search` | off | Enable MCTS-inspired tree search |
| `--no-search` | off | Disable literature search |
| `--resume` | off | Resume an interrupted run from checkpoint |
| `--seed` | `42` | Random seed for reproducibility |
| `--api-key` | env var | Anthropic API key (or use config.yaml / ANTHROPIC_API_KEY) |
| `--timeout` | `1800` | Global pipeline deadline in seconds (default 30 min) |
| `--output-dir`, `-o` | `outputs` | Output directory |

### `co-scientist batch`

```bash
co-scientist batch DATASET1 DATASET2 ... [--parallel N] [--budget B] [--max-cost C]
```

### Examples

```bash
# Fully automated (default)
co-scientist run RNA/translation_efficiency_muscle

# Interactive mode (approve decisions mid-pipeline)
co-scientist run expression/cell_type_classification_segerstolpe --mode interactive

# Tree search (explores multiple strategies via branching)
co-scientist run RNA/translation_efficiency_muscle --tree-search

# Deterministic only (no LLM cost)
co-scientist run RNA/translation_efficiency_muscle --max-cost 0

# Batch multiple datasets
co-scientist batch RNA/translation_efficiency_muscle expression/cell_type_classification_segerstolpe --parallel 2
```

---

## How It Works

### Pipeline Steps

1. **Load & Profile** — Download from HuggingFace, detect modality/task, compute statistics
2. **Preprocess & Split** — k-mers for sequences, log1p + HVG for expression, 70/15/15 split
3. **Literature Search** — Semantic Scholar + PubMed for methods and benchmarks
4. **Train & Model** — ReAct agent dynamically selects and trains models based on data analysis
5. **HP Search** — Optuna Bayesian optimization on top performers
6. **Test Evaluation** — Final held-out test-set evaluation of best model
7. **Export** — Save model + generate standalone train/predict/evaluate scripts
8. **Report** — Markdown report with methods, results, agent reasoning trace, biology assessment

Each step is followed by automatic validation and auto-repair via the Validation Agent. If issues are detected (bad data, broken scripts, missing artifacts), the agent attempts a fix before proceeding.

### Model Tiers

| Tier | Models | Requires |
|------|--------|----------|
| Trivial | Majority class / Mean predictor | CPU |
| Simple | Logistic/Ridge regression, Elastic Net, SVM, KNN | CPU |
| Standard | XGBoost, LightGBM, Random Forest | CPU |
| Advanced | MLP, FT-Transformer, LLM-designed custom models | CPU |
| Ensemble | Stacking ensemble (meta-learner over base models) | CPU |
| Foundation | embed_xgboost, embed_mlp (AIDO embeddings), concat_xgboost, concat_mlp (handcrafted + embeddings), aido_finetune (end-to-end fine-tuning) | GPU |

Foundation tier activates automatically when a CUDA GPU is detected. On CPU, it's skipped — the pipeline runs identically to before.

### Multi-Agent System

When an Anthropic API key is provided, specialized agents orchestrate the pipeline:

- **ML Engineer** — Model selection, hyperparameter strategy, iteration decisions
- **Data Analyst** — Data quality assessment, feature engineering recommendations
- **Biology Specialist** — Domain-specific guidance, plausibility assessment, literature score ranges
- **Research Agent** — Literature search, benchmark comparison
- **ReAct Agent** — Thought → Action → Observation loop that drives modeling. Can consult the Biology Specialist and Data Analyst as tools mid-loop.
- **Validation Agent** — Runs after every pipeline step. Detects and auto-fixes data issues (NaN, empty splits, wrong modality), script errors (syntax, imports), and verifies exported scripts actually execute. Visible in live dashboard.

Agents can **debate** at 4 decision points (preprocessing, modeling strategy, model selection, HP search) before the pipeline acts. A fraction of the iteration budget is reserved to guarantee post-debate exploration.

**Without an API key**, the pipeline runs fully deterministic — same models, same HP search, same results. The LLM layer is additive, not required.

### Live Dashboard

The terminal dashboard shows real-time agent thoughts, actions, model scores, tournament rankings (Elo), and agent conversation history.

### Tree Search vs Linear ReAct

- **Linear ReAct** (default): Sequential reasoning loop. The agent tries one strategy at a time.
- **Tree Search** (`--tree-search`): MCTS-inspired branching. Explores multiple strategies and backtracks to try alternatives. Higher LLM cost, broader exploration.

### Timeout & Robustness

A global pipeline deadline (`--timeout`, default 1800s = 30 min) ensures the pipeline always finishes. If time runs out, remaining steps are skipped and the best model found so far is exported. Per-step timeouts, LLM retries with exponential backoff, and worker-thread tool execution prevent any single step from hanging.

**Dataset resilience:**
- Auto-discovers HuggingFace dataset configs via fuzzy matching
- Normalizes arbitrary split structures (species-specific test splits, etc.)
- Prevents OpenMP segfaults from library conflicts

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | No | Enables LLM-driven agents. Without it, pipeline is fully deterministic. |
| `TAVILY_API_KEY` | No | Enables general web search (in addition to Semantic Scholar + PubMed). |
| `NCBI_API_KEY` | No | Higher PubMed rate limits (10 req/sec vs 3 req/sec). |
| `S2_API_KEY` | No | Higher Semantic Scholar rate limits. |

---

## GPU: Foundation Model Pipeline

When a CUDA-capable GPU is available, the pipeline automatically activates **AIDO foundation models** from GenBio — pre-trained biological language models that produce rich embeddings from sequences and expression data.

### What GPU Enables

| Strategy | Models | How it works |
|----------|--------|-------------|
| Frozen embeddings | `embed_xgboost`, `embed_mlp` | Extract AIDO embeddings once, train XGBoost/MLP on them |
| Hybrid features | `concat_xgboost`, `concat_mlp` | Concatenate handcrafted features (k-mers) + AIDO embeddings — often strongest |
| End-to-end fine-tuning | `aido_finetune` | Unfreeze last N layers of AIDO backbone + task head — best on larger datasets |

These 5 models compete alongside the ~10 CPU models on the same validation set. Best model wins regardless of tier.

### AIDO Models Used

| Modality | AIDO Model | Config Key | Activated for |
|----------|-----------|------------|---------------|
| RNA | AIDO.RNA-1.6B | `aido_rna_1b600m` | RNA sequence datasets |
| DNA | AIDO.DNA-300M | `aido_dna_300m` | DNA sequence datasets |
| Protein | AIDO.Protein-16B | `aido_protein_16b` | Protein sequence datasets |
| Cell expression | AIDO.Cell-100M | `aido_cell_100m` | scRNA-seq / expression datasets |

Models are loaded via `modelgenerator.tasks.Embed` with the API: `Embed.from_config({"model.backbone": "<config_key>"})`.

### Running on Google Colab (GPU)

**1. Setup** (use GPU runtime: Runtime → Change runtime type → T4 or better):

```python
# Cell 1: Install co-scientist + AIDO foundation models
# modelgenerator requires numpy<2, so we install it first to let pip resolve dependencies
!pip install modelgenerator -q
!pip install git+https://github.com/genbio-ai/openfold.git@c4aa2fd0d920c06d3fd80b177284a22573528442 -q
```

After the cell above finishes, **restart the runtime** (Runtime → Restart runtime) to pick up the numpy downgrade. Then run:

```python
# Cell 2: Install co-scientist (run AFTER runtime restart)
!git clone <your-repo-url> /content/scientist
%cd /content/scientist
!pip install -e . -q

# Verify everything works
import torch
print(f"GPU: {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
from modelgenerator.tasks import Embed
print("modelgenerator: OK")
```

**2. Run with full agent + GPU pipeline:**

```python
# Cell 3: Full pipeline — LLM agents + GPU foundation models
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # your key

!co-scientist run RNA/translation_efficiency_muscle --budget 10 --max-cost 2.0
```

This gives you: GPU foundation models + ReAct agent reasoning + agent debates + biology specialist + custom model design.

**3. Run deterministic (no LLM cost, GPU still active):**

```python
# Cell 3b: Deterministic — GPU foundation models, no LLM
!co-scientist run RNA/translation_efficiency_muscle --budget 5 --max-cost 0
```

**4. Run full QC test suite:**

```python
# Cell 4: All 11 datasets
!python test_all_datasets.py --quick
```

**5. Inspect results:**

```python
# Cell 5: View results
import glob, json
from IPython.display import Markdown, display

out = sorted(glob.glob("outputs/RNA__translation_efficiency_muscle_*"))[-1]
config = json.load(open(f"{out}/models/model_config.json"))
print(f"Best model: {config['model_name']} ({config['tier']})")
print(f"Type: {config['model_type']}")
print(f"Metric: {config['evaluation']['primary_metric']} = {config['evaluation']['primary_value']:.4f}")

display(Markdown(open(f"{out}/report.md").read()))
```

**Known Colab issue:** `modelgenerator` requires `numpy<2` but Colab ships numpy 2.x. Installing modelgenerator downgrades numpy, which breaks pre-compiled packages until the runtime is restarted. Always restart the runtime after installing modelgenerator.

### Graceful Degradation

| Scenario | Behavior |
|----------|----------|
| No GPU | Foundation tier skipped, CPU pipeline runs unchanged |
| GPU but `modelgenerator` not installed | Embedding extraction fails gracefully, foundation models skipped |
| GPU + `modelgenerator` works | All 5 foundation models compete alongside CPU models |

### GPU vs CPU Performance

Foundation models typically improve the primary metric by 10-25% over handcrafted features alone. The concat models (handcrafted + embeddings) often outperform pure embedding models because they combine domain-engineered signal with learned representations.

See `docs/gpu_docs/` for detailed technical documentation of the three implementation phases.

---

## Testing on Hidden Datasets

```bash
co-scientist run <your_dataset_path>
```

The pipeline handles any HuggingFace-hosted biological dataset:
- **Sequence data** (RNA, DNA, protein) → k-mer feature extraction + sequence-aware models
- **Expression data** (h5ad) → log1p normalization + HVG selection + standard ML
- **Tabular data** → Standard numeric preprocessing
- **Classification or regression** — auto-detected from the label column

No configuration needed — modality, task type, metrics, and model selection are all inferred automatically.

**Metrics coverage:** Classification computes 12 metrics (accuracy, balanced_accuracy, F1 macro/weighted, precision, recall, AUROC, MCC, etc.) and regression computes 10 metrics (Spearman, Pearson, MSE, RMSE, MAE, R², etc.), ensuring any evaluation metric is already computed.

### QC Testing

```bash
# ── Quick test (no LLM, no API key needed) ──
# Runs all 11 datasets deterministic-only. Verifies pipeline works end-to-end.
# ~3-5 min per dataset on CPU.
python test_all_datasets.py --quick

# ── Full test with agents (requires API key) ──
# Runs all 11 datasets with ReAct agent, debates, biology specialist, custom models.
# ~$2 per dataset in LLM costs. ~10-15 min per dataset.
export ANTHROPIC_API_KEY=sk-ant-...
python test_all_datasets.py

# Single dataset with agents (for quick verification)
co-scientist run RNA/translation_efficiency_muscle --budget 10

# Inspect existing outputs without re-running
python test_all_datasets.py --inspect-only

# Full report at outputs/qc_report.md
```

| Command | Agents | LLM Cost | Foundation Models (GPU) |
|---------|--------|----------|------------------------|
| `test_all_datasets.py --quick` | No | $0 | Auto: Yes on GPU, skipped on CPU |
| `test_all_datasets.py` | Yes | ~$2/dataset | Auto: Yes on GPU, skipped on CPU |
| `co-scientist run DATASET` | Yes (if API key set) | Up to $5 | Auto: Yes on GPU, skipped on CPU |
| `co-scientist run DATASET --max-cost 0` | No | $0 | Auto: Yes on GPU, skipped on CPU |

**GPU is automatic** — the pipeline checks `torch.cuda.is_available()` at startup. On a machine with a CUDA GPU (Colab T4, H100 server, etc.), foundation models activate with no flags needed. On CPU (your Mac), they're silently skipped and the pipeline runs the standard models only.

The QC script runs every dataset, then deep-inspects outputs: model loading, script execution, metric consistency, artifact completeness.
