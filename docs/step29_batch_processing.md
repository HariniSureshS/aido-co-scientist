# Step 29: Batch Processing

## Overview

Run the full co-scientist pipeline on multiple datasets with a single command, optionally in parallel.

**Problem:** Running `co-scientist run` on each dataset one at a time is tedious for benchmarking or multi-dataset studies. No way to get a summary table across datasets.

**Solution:** A `batch` CLI command that accepts multiple dataset paths, runs the pipeline on each, and produces a summary table.

```bash
co-scientist batch RNA/translation_efficiency_muscle expression/cell_type_classification_segerstolpe --parallel 2
```

---

## Architecture

### Batch Pipeline

```
co-scientist batch D1 D2 D3 --parallel 2
       │
       ▼
 ┌─────────────────┐
 │  BatchRunner     │
 │                  │
 │  parallel=2      │
 │  ┌──────────┐    │
 │  │ Worker 1 │────│──→ run_pipeline(D1) → DatasetRunResult
 │  └──────────┘    │
 │  ┌──────────┐    │
 │  │ Worker 2 │────│──→ run_pipeline(D2) → DatasetRunResult
 │  └──────────┘    │
 │       ...        │
 │  ┌──────────┐    │
 │  │ Worker 1 │────│──→ run_pipeline(D3) → DatasetRunResult
 │  └──────────┘    │
 └─────────────────┘
       │
       ▼
 Summary Table
 ┌──────────┬────────┬─────────────┬───────┬───────┐
 │ Dataset  │ Status │ Best Model  │ Score │ Time  │
 ├──────────┼────────┼─────────────┼───────┼───────┤
 │ D1       │ OK     │ rf_tuned    │ 0.71  │ 45.2s │
 │ D2       │ OK     │ lgbm_tuned  │ 0.89  │ 32.1s │
 │ D3       │ FAIL   │ -           │ -     │ 12.3s │
 └──────────┴────────┴─────────────┴───────┴───────┘
```

### Prerequisite Refactor

The body of `cli.py:run()` was extracted into a callable `run_pipeline()` function in `batch.py`. The `run()` CLI command remains the primary entry point with full dashboard output. `run_pipeline()` is the programmatic version used by batch processing.

---

## Key Data Structures

### DatasetRunResult

```python
@dataclass
class DatasetRunResult:
    dataset_path: str
    success: bool
    best_model: str | None = None
    best_score: float | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
```

### BatchRunner

```python
class BatchRunner:
    def __init__(self, datasets, parallel=1, output_dir="outputs", common_options={}):
    def run(self) -> list[DatasetRunResult]
    def _run_sequential(self) -> list[DatasetRunResult]
    def _run_parallel(self) -> list[DatasetRunResult]  # ProcessPoolExecutor
    @staticmethod
    def print_summary(results) -> None                  # Rich table
```

### run_pipeline()

```python
def run_pipeline(
    dataset_path: str,
    output_dir: str = "outputs",
    options: dict[str, Any] | None = None,
) -> DatasetRunResult:
    """Full pipeline logic, callable programmatically."""
```

This is a self-contained version of the pipeline that:
- Creates its own `RunConfig`, `CostTracker`, `Coordinator`
- Runs all 7 steps (profile → preprocess → train → HP → iterate → export → report)
- Returns a `DatasetRunResult` instead of printing to console
- Can be called from `ProcessPoolExecutor` for parallel execution

---

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/batch.py` | `DatasetRunResult`, `BatchRunner`, `run_pipeline()` |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/cli.py` | Added `batch` command with `--datasets`, `--parallel`, and shared options |

---

## CLI Interface

### batch Command

```bash
co-scientist batch DATASET1 DATASET2 ... [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--parallel`, `-p` | 1 | Number of parallel workers |
| `--budget`, `-b` | 10 | Max iteration steps per dataset |
| `--max-cost` | 5.0 | Max LLM cost per dataset |
| `--output-dir`, `-o` | "outputs" | Base output directory |
| `--no-search` | False | Disable paper search |
| `--seed` | 42 | Random seed |

### Examples

```bash
# Sequential (default):
co-scientist batch RNA/translation_efficiency_muscle expression/cell_type_classification_segerstolpe

# Parallel with 2 workers:
co-scientist batch RNA/translation_efficiency_muscle expression/cell_type_classification_segerstolpe --parallel 2

# With options:
co-scientist batch D1 D2 D3 --parallel 3 --budget 5 --no-search --seed 123
```

---

## Parallel Execution

### ProcessPoolExecutor

Parallel mode uses `ProcessPoolExecutor` to run datasets in separate processes:

```python
with ProcessPoolExecutor(max_workers=self.parallel) as executor:
    for dataset in self.datasets:
        future = executor.submit(_run_single, dataset, output_dir, options)
        futures[future] = dataset

    for future in as_completed(futures):
        result = future.result()
        results.append(result)
```

**Why ProcessPoolExecutor, not ThreadPoolExecutor?** ML model training (sklearn, XGBoost, LightGBM) releases the GIL during computation, but Python-level preprocessing does not. Processes ensure true parallelism.

### Output Isolation

Each dataset writes to its own subdirectory:
```
outputs/
├── RNA__translation_efficiency_muscle/
│   ├── report.md
│   ├── logs/
│   └── ...
├── expression__cell_type_classification_segerstolpe/
│   ├── report.md
│   ├── logs/
│   └── ...
└── .memory/       ← shared cross-run memory (append-safe)
```

The `.memory/` directory is shared across all datasets. The `model_performance.jsonl` file uses append mode (safe for parallel writes). The `hp_priors.json` file uses `fcntl` file locking.

---

## Summary Output

```
┌─────────────────────────────────────────────────────┐
│                Batch Results Summary                 │
├──────────────────┬────────┬──────────┬───────┬──────┤
│ Dataset          │ Status │ Best     │ Score │ Time │
├──────────────────┼────────┼──────────┼───────┼──────┤
│ RNA/trans...      │ OK     │ rf_tuned │ 0.710 │ 45s  │
│ expression/...   │ OK     │ lgbm     │ 0.891 │ 32s  │
└──────────────────┴────────┴──────────┴───────┴──────┘

2/2 datasets completed successfully.
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Dataset load fails | `DatasetRunResult(success=False, error=...)` |
| Pipeline crashes | Caught by `_run_single()`, error recorded |
| One dataset fails in parallel | Other datasets continue; failure shown in summary |
| All datasets fail | Exit code 1 |
| Worker process dies | `ProcessPoolExecutor` handles cleanup |
| Per-dataset timeout (3600s) | `future.result(timeout=per_dataset_timeout)` raises `TimeoutError` → recorded as failure, other datasets continue |

### Per-Dataset Timeout

Each dataset in parallel mode has a `per_dataset_timeout=3600.0` (1 hour) limit. This is passed to both `as_completed()` and `future.result()`. If a dataset's pipeline hangs (e.g., model training on a very large dataset), the timeout ensures the batch run eventually completes.

In sequential mode, each dataset also has a wall-clock check within the ReactAgent (600s), but no outer per-dataset timeout — the inner guardrails handle runaway behavior.

---

## Verification

```bash
# Batch sequential:
co-scientist batch RNA/translation_efficiency_muscle expression/cell_type_classification_segerstolpe

# Batch parallel:
co-scientist batch RNA/translation_efficiency_muscle expression/cell_type_classification_segerstolpe --parallel 2

# Check outputs:
ls outputs/RNA__translation_efficiency_muscle/report.md
ls outputs/expression__cell_type_classification_segerstolpe/report.md
```

---

## Design Decisions

### Why extract run_pipeline() instead of calling cli.run()?

`cli.run()` uses Typer decorators, Rich console output, and `typer.Exit()` for flow control — all designed for interactive use. `run_pipeline()` is a clean function that returns a result object, suitable for programmatic invocation and parallel execution.

### Why not shared LLM budget across datasets?

Each dataset gets its own `CostTracker(max_cost=...)`. This is simpler and more predictable — the user can reason about per-dataset cost. A shared budget would require coordination between workers and could lead to one dataset starving others.

### Why ProcessPoolExecutor instead of multiprocessing.Pool?

`ProcessPoolExecutor` provides a cleaner API with `as_completed()` for incremental result reporting, and better exception handling via `Future.result()`.
