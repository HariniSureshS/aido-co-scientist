# User Guide

Detailed instructions for Docker, GPU/Colab, CLI options, environment setup, and testing.

For quick start, see [README.md](README.md). For system design, see [ARCHITECTURE.md](ARCHITECTURE.md).

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

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | No | Enables LLM-driven agents. Without it, pipeline is fully deterministic. |
| `TAVILY_API_KEY` | No | Enables general web search (in addition to Semantic Scholar + PubMed). |
| `NCBI_API_KEY` | No | Higher PubMed rate limits (10 req/sec vs 3 req/sec). |
| `S2_API_KEY` | No | Higher Semantic Scholar rate limits. |

---

## Docker

### Prerequisites

```bash
# macOS:
brew install --cask docker

# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER  # run docker without sudo (logout/login after)

# Verify:
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
```

**Note:** GPU Docker requires Linux with NVIDIA drivers + NVIDIA Container Toolkit. On macOS/Windows, use the CPU image or Google Colab for GPU.

---

## GPU: Foundation Model Pipeline

When a CUDA-capable GPU is available, the pipeline automatically activates **AIDO foundation models** from GenBio — pre-trained biological language models that produce rich embeddings from sequences and expression data.

### What GPU Enables

| Strategy | Models | How it works |
|----------|--------|-------------|
| Frozen embeddings | `embed_xgboost`, `embed_mlp` | Extract AIDO embeddings once, train XGBoost/MLP on them |
| Hybrid features | `concat_xgboost`, `concat_mlp` | Concatenate handcrafted features (k-mers) + AIDO embeddings — often strongest |
| End-to-end fine-tuning | `aido_finetune` | Unfreeze last N layers of AIDO backbone + task head — best on larger datasets |

### AIDO Models Used

| Modality | AIDO Model | Config Key | Activated for |
|----------|-----------|------------|---------------|
| RNA | AIDO.RNA-1.6B | `aido_rna_1b600m` | RNA sequence datasets |
| DNA | AIDO.DNA-300M | `aido_dna_300m` | DNA sequence datasets |
| Protein | AIDO.Protein-16B | `aido_protein_16b` | Protein sequence datasets |
| Cell expression | AIDO.Cell-100M | `aido_cell_100m` | scRNA-seq / expression datasets |

### Graceful Degradation

| Scenario | Behavior |
|----------|----------|
| No GPU | Foundation tier skipped, CPU pipeline runs unchanged |
| GPU but `modelgenerator` not installed | Embedding extraction fails gracefully, foundation models skipped |
| GPU + `modelgenerator` works | All 5 foundation models compete alongside CPU models |

See `docs/gpu_docs/` for detailed technical documentation.

---

## Google Colab (GPU)

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
!git clone https://github.com/HariniSureshS/aido-co-scientist.git /content/scientist
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

**3. Run deterministic (no LLM cost, GPU still active):**

```python
!co-scientist run RNA/translation_efficiency_muscle --budget 5 --max-cost 0
```

**4. Inspect results:**

```python
import glob, json
from IPython.display import Markdown, display

out = sorted(glob.glob("outputs/RNA__translation_efficiency_muscle_*"))[-1]
config = json.load(open(f"{out}/models/model_config.json"))
print(f"Best model: {config['model_name']} ({config['tier']})")
print(f"Metric: {config['evaluation']['primary_metric']} = {config['evaluation']['primary_value']:.4f}")

display(Markdown(open(f"{out}/report.md").read()))
```

**Known Colab issue:** `modelgenerator` requires `numpy<2` but Colab ships numpy 2.x. Always restart the runtime after installing modelgenerator.

---

## Testing

### QC Testing

```bash
# Quick test (no LLM, no API key needed)
# Runs all 11 datasets deterministic-only. ~3-5 min per dataset on CPU.
python test_all_datasets.py --quick

# Full test with agents (requires API key)
# ~$2 per dataset in LLM costs. ~10-15 min per dataset.
export ANTHROPIC_API_KEY=sk-ant-...
python test_all_datasets.py

# Single dataset with agents
co-scientist run RNA/translation_efficiency_muscle --budget 10

# Inspect existing outputs without re-running
python test_all_datasets.py --inspect-only
```

| Command | Agents | LLM Cost | Foundation Models (GPU) |
|---------|--------|----------|------------------------|
| `test_all_datasets.py --quick` | No | $0 | Auto: Yes on GPU, skipped on CPU |
| `test_all_datasets.py` | Yes | ~$2/dataset | Auto: Yes on GPU, skipped on CPU |
| `co-scientist run DATASET` | Yes (if API key set) | Up to $5 | Auto: Yes on GPU, skipped on CPU |
| `co-scientist run DATASET --max-cost 0` | No | $0 | Auto: Yes on GPU, skipped on CPU |

### Testing on Hidden Datasets

```bash
co-scientist run <your_dataset_path>
```

The pipeline handles any HuggingFace-hosted biological dataset:
- **Sequence data** (RNA, DNA, protein) → k-mer feature extraction + sequence-aware models
- **Expression data** (h5ad) → log1p normalization + HVG selection + standard ML
- **Tabular data** → Standard numeric preprocessing
- **Classification or regression** — auto-detected from the label column

No configuration needed — modality, task type, metrics, and model selection are all inferred automatically.
