# Step 30: Cross-Run Memory

## Overview

Learn from previous pipeline runs to inform future decisions. The system stores model performance history and hyperparameter priors, then injects this knowledge into agent prompts for subsequent runs.

**Problem:** Every pipeline run starts from scratch. If XGBoost consistently outperforms LightGBM on RNA data, the agent has to rediscover this each time. If specific hyperparameters work well for a modality, that knowledge is lost between runs.

**Solution:** A persistent memory system stored in `outputs/.memory/` that accumulates:
- Model performance history (which model types work best for which data)
- HP priors (best-known hyperparameters per model type + modality)
- Natural language summaries injected into agent prompts

**Graceful degradation:** If no `.memory/` exists (first run), all memory queries return empty. No special casing needed.

---

## Architecture

### Memory Flow

```
Run 1: RNA dataset
  ├── Train xgboost (score=0.63)  ──┐
  ├── Train lightgbm (score=0.65)    ├──→ record_performance()
  ├── Train rf (score=0.69)         ─┤    update_hp_priors()
  └── Best: rf_tuned (score=0.71)  ──┘
                                      │
                                      ▼
                            outputs/.memory/
                            ├── model_performance.jsonl  (append-only)
                            └── hp_priors.json           (file-locked)

Run 2: Another RNA dataset
  ├── Load memory ──→ format_for_prompt()
  │   "From 3 past experiments:
  │    Best model types for rna/regression: random_forest, lightgbm, xgboost
  │    Best HP for random_forest: n_estimators=500, max_depth=10 (score=0.71)"
  │
  ├── Memory injected into agent prompts
  ├── Agent prioritizes RF based on history
  └── Faster convergence to good results
```

### Storage Files

| File | Format | Write mode | Contents |
|------|--------|-----------|----------|
| `model_performance.jsonl` | JSONL (one JSON per line) | Append | Every model trained, with type/modality/task/score/HP |
| `hp_priors.json` | JSON | File-locked write | Best HP per model_type + modality |

**Why JSONL for performance?** Append-only is safe for parallel writes (batch mode). No locking needed.

**Why JSON with file locking for HP priors?** HP priors are updated (overwritten) when a better config is found. File locking via `fcntl` prevents corruption during parallel batch runs.

---

## Key Data Structures

### ModelPerformanceEntry

```python
@dataclass
class ModelPerformanceEntry:
    model_type: str           # "random_forest"
    modality: str             # "rna"
    task_type: str            # "regression"
    primary_metric: str       # "spearman"
    score: float              # 0.7103
    hyperparameters: dict     # {"n_estimators": 500, ...}
    dataset_name: str         # "RNA/translation_efficiency_muscle"
    timestamp: str            # ISO 8601
```

### HPPrior

```python
@dataclass
class HPPrior:
    model_type: str
    modality: str
    best_hyperparameters: dict
    score: float
```

### RunMemory

```python
class RunMemory:
    def __init__(self, output_dir: Path):  # stores in outputs/.memory/

    def load_performance_history() -> list[ModelPerformanceEntry]
    def load_hp_priors() -> dict[str, HPPrior]
    def record_performance(entry) -> None         # append to JSONL
    def update_hp_priors(model_type, modality, hp, score) -> None  # file-locked
    def get_recommendations(modality, task_type) -> dict
    def format_for_prompt(modality, task_type) -> str  # natural language
```

---

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/memory.py` | `ModelPerformanceEntry`, `HPPrior`, `RunMemory` |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/agents/types.py` | Added `memory_context: str = ""` to `PipelineContext` |
| `co_scientist/agents/base.py` | Injects `memory_context` into agent user messages |
| `co_scientist/agents/analysis.py` | `build_pipeline_context()` accepts `memory_context` param |
| `co_scientist/agents/coordinator.py` | Added `memory: RunMemory \| None` attribute |
| `co_scientist/cli.py` | Initializes `RunMemory` at start. Injects memory context at profiling. Updates memory at pipeline end with all model results. |

---

## Memory Lifecycle

### At Pipeline Start

```python
# cli.py
run_memory = RunMemory(config.output_dir)
coordinator.memory = run_memory
```

### At Decision Points (via PipelineContext)

```python
memory_ctx = run_memory.format_for_prompt(
    state.profile.modality.value, state.profile.task_type.value,
)
context = build_pipeline_context(..., memory_context=memory_ctx)
```

### In Agent Prompts

The `BaseAgent._build_user_message()` appends memory context:

```python
if context.memory_context:
    parts.append(f"\nCross-run memory:\n{context.memory_context}")
```

This means every agent (ML Engineer, Data Analyst, Biology Specialist) sees the memory when making decisions.

### At Pipeline End

```python
# Record every model's performance
for res, trn in zip(state.results, state.trained_models):
    run_memory.record_performance(ModelPerformanceEntry(
        model_type=trn.config.model_type,
        modality=state.profile.modality.value,
        task_type=state.profile.task_type.value,
        primary_metric=state.eval_config.primary_metric,
        score=res.primary_metric_value,
        hyperparameters=trn.config.hyperparameters,
        dataset_name=config.dataset_path,
    ))

# Update HP priors with best model
run_memory.update_hp_priors(
    model_type=state.best_trained.config.model_type,
    modality=state.profile.modality.value,
    hp=state.best_trained.config.hyperparameters,
    score=state.best_result.primary_metric_value,
)
```

---

## Memory Format in Prompts

```
Cross-run memory:
From 12 past experiment(s):
  Best model types for rna/regression: random_forest, lightgbm, xgboost
  Average scores: random_forest: 0.7103, lightgbm: 0.6455, xgboost: 0.6279
  Best HP for random_forest: n_estimators=500, max_depth=10, min_samples_leaf=2 (score=0.7103)
```

If no past runs exist for this modality/task, the memory context is an empty string and nothing is added to the prompt.

---

## get_recommendations()

Returns a structured dict for programmatic access:

```python
{
    "best_models": ["random_forest", "lightgbm", "xgboost"],
    "hp_priors": {
        "random_forest": {
            "hyperparameters": {"n_estimators": 500, "max_depth": 10},
            "score": 0.7103,
        }
    },
    "avg_scores": {
        "random_forest": 0.7103,
        "lightgbm": 0.6455,
        "xgboost": 0.6279,
    }
}
```

---

## Interaction with Batch Processing (Step 29)

Batch mode is the primary beneficiary of cross-run memory:

```bash
co-scientist batch D1 D2 D3 --parallel 2
```

- D1 runs first → writes performance to `.memory/`
- D2 and D3 start → read memory from D1 → informed decisions
- D2 and D3 complete → append their performance to `.memory/`
- Next batch run → all 3 datasets' history available

**Parallel safety:**
- `model_performance.jsonl`: Append-only, safe for concurrent writes
- `hp_priors.json`: `fcntl.LOCK_EX` ensures exclusive write access

---

## Verification

```bash
# First run:
co-scientist run RNA/translation_efficiency_muscle --budget 5

# Check memory:
ls outputs/.memory/
cat outputs/.memory/model_performance.jsonl
cat outputs/.memory/hp_priors.json

# Second run (should see memory in agent context):
co-scientist run RNA/translation_efficiency_muscle --budget 5
# Agent prompts now include "Cross-run memory: ..."

# Batch (memory accumulates across datasets):
co-scientist batch D1 D2 D3
wc -l outputs/.memory/model_performance.jsonl
```

---

## Design Decisions

### Why file-based, not a database?

Simplicity. The memory is small (hundreds of entries at most), append-only JSONL handles concurrent writes safely, and the files are human-readable. A database would add a dependency and complexity for no real benefit at this scale.

### Why per-output-dir, not global?

Memory is scoped to the output directory (`outputs/.memory/`). This means different projects (different output dirs) have separate memories. Users who want global memory can use the same `--output-dir` across runs.

### Why only record at pipeline end, not after each model?

Recording at the end ensures we only store results from complete, successful runs. If the pipeline crashes mid-run, we don't pollute memory with partial results.

### Why update_hp_priors only replaces if score is better?

This prevents regression — if a later run happens to produce worse HP for a model type (due to different data or random seed), we keep the known-good prior. Over time, the priors converge to the best-known configuration.
