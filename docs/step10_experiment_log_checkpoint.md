# Step 10: Experiment Log + Checkpointing — Detailed Walkthrough

## Overview

Step 10 answers two questions:
1. **"What exactly happened during this run?"** → Experiment log
2. **"Can I resume an interrupted run?"** → Checkpointing

---

## Experiment Log

### What it records

Every pipeline event is appended to `logs/experiment_log.jsonl` — one JSON object per line. Events include:

| Event | When | What it captures |
|-------|------|-----------------|
| `pipeline_start` | Run begins | Dataset, mode, budget, seed, version |
| `step_start` | Each step begins | Step name |
| `step_end` | Each step ends | Step name + summary (samples, features, models, etc.) |
| `model_trained` | After each model.fit() | Name, tier, model type, hyperparameters, train time |
| `evaluation` | After each evaluate() | Model name, all metrics, primary metric value, split |
| `hp_search` | After Optuna completes | Trials, best params, improved vs baseline |
| `error` | On failure | Step, error message |
| `pipeline_complete` | Run finishes | Best model, best metric, total elapsed time |

### Why JSONL, not JSON?

JSONL (one JSON object per line) is append-only — each event is written immediately, so the log survives crashes. A regular JSON array would need the closing `]` to be valid, meaning a crash mid-pipeline would produce invalid JSON. With JSONL, you always have all events up to the crash point.

### Example log entry

```json
{"timestamp": "2026-03-16T03:55:32.78", "elapsed_seconds": 15.23, "event": "evaluation",
 "data": {"model_name": "xgboost_default", "split": "val", "primary_metric": "spearman",
          "primary_value": 0.627896, "all_metrics": {"mse": 1.1035, "rmse": 1.0505, ...}}}
```

Every entry has a UTC timestamp and elapsed seconds from pipeline start, so you can reconstruct the exact timeline of the run.

### What the log enables

- **Post-hoc analysis:** Read the JSONL to compare runs, identify which hyperparameters worked best across datasets
- **Phase C integration:** LLM agents can read the log to understand what was tried and reason about next steps
- **Debugging:** If a run produces unexpected results, the log shows exactly what happened and when
- **Report enrichment:** The report generator can use the log to populate the "iteration history" section

---

## Checkpointing

### How it works

After each major pipeline step, the full `PipelineState` is serialized to `logs/checkpoint.pkl`. This state contains:
- All completed step names
- All intermediate results (dataset, profile, splits, trained models, metrics, figures)
- The current best model

When you run with `--resume`:
1. The pipeline loads the checkpoint
2. For each step, it checks `state.is_complete(step)`
3. Completed steps are skipped (prints "resumed")
4. The first incomplete step runs normally, using the restored state
5. New checkpoints are saved after each completed step

### Pipeline steps in order

```
load_profile     → Step 1: load dataset + profile + profiling figures
preprocess_split → Step 2: preprocess + split + preprocessing figures
baselines        → Step 3: train all baselines + evaluate + training figures
hp_search        → Step 3b: Optuna HP search
export           → Step 4: export best model
report           → Step 6: generate report
```

### Checkpoint files

```
logs/
├── checkpoint.pkl       ← full pipeline state (pickle)
├── checkpoint_meta.json ← human-readable status summary
└── experiment_log.jsonl ← append-only event log
```

The `checkpoint_meta.json` is a quick summary you can inspect without loading the pickle:

```json
{
  "completed_steps": ["load_profile", "preprocess_split", "baselines", "hp_search", "export", "report"],
  "last_step": "report",
  "n_models_trained": 5,
  "best_model": "xgboost_tuned"
}
```

### Resume scenarios

| Scenario | What happens |
|----------|-------------|
| `--resume` on completed run | All steps skipped, prints "resumed", finishes instantly |
| `--resume` after crash in Step 3 | Steps 1-2 skipped, Step 3 re-runs from scratch |
| `--resume` after HP search timeout | Steps 1-3 skipped, HP search re-runs |
| No `--resume` flag | Always starts fresh (ignores checkpoint) |
| No checkpoint exists + `--resume` | Starts fresh (no error) |
| Corrupt checkpoint + `--resume` | Prints warning, starts fresh |

### Why pickle for checkpoints?

The pipeline state contains numpy arrays, sklearn models, PyTorch models, and custom objects. Pickle handles all of these natively. JSON can't serialize numpy arrays or trained models. The tradeoff: pickle is Python-version-specific and not human-readable, but for checkpointing (which is consumed only by the same pipeline), this is fine.

---

## CLI Refactoring

The `cli.py` was refactored from a linear script to a step-based architecture:

**Before:**
```python
# Step 1
dataset = load_dataset(...)
profile = profile_dataset(...)
# Step 2
preprocessed = preprocess(...)
split = split_dataset(...)
# Step 3
trained_models = train_baselines(...)
```

**After:**
```python
if not state.is_complete("load_profile"):
    state.dataset = load_dataset(...)
    state.profile = profile_dataset(...)
    state.mark_complete("load_profile")
    save_checkpoint(state, output_dir)
else:
    console.print("Step 1... (resumed)")
```

Each step:
1. Checks if already complete
2. Logs step_start
3. Runs the step, storing results in `state`
4. Marks complete + saves checkpoint
5. Logs step_end with summary

This is also the foundation for Phase C's agent architecture — each step becomes an agent action that can be monitored, retried, or replaced with an LLM-driven version.

---

## Output Directory (Updated)

```
outputs/DATASET/
├── report.md
├── requirements.txt
├── models/
│   ├── best_model.pkl
│   ├── model_config.json
│   └── label_encoder.pkl
├── code/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── figures/
│   ├── 01_profiling/
│   ├── 02_preprocessing/
│   └── 03_training/
└── logs/                     ← NEW
    ├── experiment_log.jsonl  ← every event, append-only
    ├── checkpoint.pkl        ← full pipeline state
    └── checkpoint_meta.json  ← human-readable status
```

This matches the ARCHITECTURE.md Section 12.1 output structure.
