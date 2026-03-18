# Step 11c: Error Recovery & Graceful Degradation — Detailed Walkthrough

## Overview

The system **always produces output**, even when components fail. This is the core production engineering principle from Architecture Section 10.2:

```
Full agent → LLM without search → Rule-based only → Baselines only → Partial report
```

Step 11c implements the deterministic foundation for this chain: structured error capture, fallback configurations, and graceful degradation for non-critical steps.

---

## Source: CellAgent Error-Feedback Self-Correction

**Architecture Section 2.3:** *"When training fails or produces implausible results, CellAgent feeds the exact error message and context back to the executor agent for intelligent recovery. We adopt this: instead of just 'retry with fallback config,' the ML Engineer receives the full error trace + the Data Analyst's diagnosis of what likely went wrong, enabling targeted fixes (e.g., 'OOM → reduce batch size from 64 to 16' rather than 'OOM → try a different model')."*

### Deterministic vs. LLM-Driven Recovery

**What we build now (Phase B):**
- Catch training errors with structured `ErrorContext`
- Log the exact error type, message, model config, and what recovery action was taken
- Apply deterministic fallback configs (reduced capacity for OOM, increased regularization for numerical issues)
- Skip models that fail even with fallback

**What Phase C adds:**
- ML Engineer agent reads the `ErrorContext` and reasons about *why* it failed
- Data Analyst diagnoses root cause (e.g., "OOM because 19K features → suggest dimensionality reduction")
- Intelligent recovery: targeted fixes instead of generic fallbacks

The key insight: by logging structured error context *now*, we give Phase C agents everything they need to make intelligent decisions later.

---

## ErrorContext — Structured Error Capture

Every failure is captured as an `ErrorContext`:

```python
@dataclass
class ErrorContext:
    step: str              # which pipeline step
    error_type: str        # "MemoryError", "ValueError", etc.
    error_message: str     # the exception message
    traceback: str         # full traceback for debugging
    model_name: str        # which model (if applicable)
    attempted_config: dict # hyperparameters that were tried
    recovery_action: str   # "retry_fallback", "skip", "halt"
```

This is logged as an `error_recovery` event in the experiment log:

```json
{"event": "error_recovery", "data": {
  "step": "baselines",
  "error_type": "MemoryError",
  "error_message": "...",
  "model_name": "mlp",
  "attempted_config": {"hidden_dims": [256, 128], ...},
  "recovery_action": "retry_fallback"
}}
```

---

## Fallback Configurations

When a model fails, we retry with a reduced configuration based on the error type:

| Error Type | XGBoost Fallback | MLP Fallback |
|-----------|-----------------|-------------|
| MemoryError | n_estimators: 50, max_depth: 4 | hidden_dims: [64, 32], batch_size: 128, max_epochs: 20 |
| ValueError | reg_alpha: 1.0, reg_lambda: 5.0 | dropout: 0.5, weight_decay: 0.01 |
| Default | n_estimators: 50, max_depth: 3 | hidden_dims: [64], max_epochs: 10 |

Fallback configs are merged on top of the original hyperparameters, overriding only the relevant settings.

### Recovery chain for model training

```
1. Try original config
   → Success: return TrainedModel
   → Failure: log ErrorContext, continue to step 2

2. Try fallback config (reduced capacity/increased regularization)
   → Success: return TrainedModel (with "_fallback" name suffix)
   → Failure: log ErrorContext, continue to step 3

3. Skip this model, return None
   → Pipeline continues with remaining models
```

---

## Graceful Degradation for Pipeline Steps

Non-critical steps (visualization, report) are wrapped in `run_step_resilient()`:

| Step | Critical? | On Failure |
|------|-----------|-----------|
| Load dataset | Yes | Pipeline halts — can't do anything without data |
| Profile | Yes | Pipeline halts — need profile for all downstream decisions |
| Preprocess + Split | Yes | Pipeline halts — can't train without split |
| Train baselines | Per-model | Individual models can fail; pipeline continues with survivors |
| Training figures | No | Skip figures, continue with model selection |
| HP search | No | Keep baseline results if HP search fails |
| Export | Yes | Pipeline halts — the model is the primary deliverable |
| Report | No | Skip report if generation fails — model still usable |

### The "always produce output" guarantee

Even in the worst case:
1. All advanced models fail → trivial + simple baselines still produce results
2. Figures fail → report generates without figures
3. HP search crashes → best baseline is exported
4. Report fails → model + code are still exported

The only hard failures are: can't load data, can't profile, can't split, or literally every model crashes.

---

## Integration Points

### Resilient baseline training

`train_baselines_resilient()` replaces the direct `train_baselines()` call:

```python
# Before (Step 3 in cli.py)
state.trained_models = train_baselines(configs, split)

# After
state.trained_models = train_baselines_resilient(configs, split, exp_log)
```

If all models fail, the pipeline halts with a clear error message.

### Resilient step execution

Non-critical steps use `run_step_resilient()`:

```python
state.training_figs = run_step_resilient(
    "training_figures", generate_figs_fn, exp_log
) or []
```

If the step fails, it logs the error and returns `None` instead of crashing.

---

## Connection to Architecture Failure Handling (Section 10.1)

| Failure | Recovery (Deterministic) | Recovery (Phase C, LLM-driven) |
|---------|------------------------|-------------------------------|
| OOM | Reduce capacity → retry | ML Engineer: "reduce batch size specifically" |
| Training crash | Fallback config → skip | ML Engineer reads error + DA diagnosis → targeted fix |
| Timeout | Kill step, log partial | Coordinator: adjust budget, try lighter model |
| Search failure | N/A (no search yet) | Skip search, use predefined defaults |
| LLM parse error | N/A (no LLM yet) | Retry with stricter prompt (3x), then defaults |
| LLM API down | N/A (no LLM yet) | Exponential backoff, then rule-based pipeline |

The rightmost column is built in Phase C on top of the structured error context we capture now.

---

## File Structure

```
co_scientist/
└── resilience.py    ← ErrorContext, fallback configs, recovery wrappers
```

The module exports:
- `ErrorContext` — structured error information dataclass
- `get_fallback_hp(error_type, model_type)` — lookup fallback hyperparameters
- `run_with_recovery(fn, step, exp_log, ...)` — execute with retry + fallback
- `train_model_resilient(config, split, exp_log)` — single model with recovery
- `train_baselines_resilient(configs, split, exp_log)` — all baselines with per-model recovery
- `run_step_resilient(step_name, fn, exp_log, critical)` — pipeline step wrapper
