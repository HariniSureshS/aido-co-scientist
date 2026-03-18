# Step 23: Full Iteration Loop

## Overview

After baselines and initial HP search, the pipeline enters an agent-driven iteration loop where the ML Engineer proposes improvement strategies, the pipeline executes them, and results are evaluated. The loop continues until budget exhaustion, patience exceeded, or the agent recommends stopping.

## Architecture

```
Baselines → HP Search → Iteration Loop → Export → Report
                              │
                    ┌─────────┴─────────┐
                    │  For each iteration: │
                    │  1. Consult ML Engineer │
                    │  2. Execute strategy   │
                    │  3. Evaluate result    │
                    │  4. Check stopping     │
                    └───────────────────────┘
```

## Pipeline Steps (Updated: 7 total)

| Step | Name | Description |
|------|------|-------------|
| 1 | Load & Profile | Load dataset, detect modality/task |
| 2 | Preprocess & Split | Feature engineering, train/val/test split |
| 3 | Train Baselines | Trivial → simple → standard → advanced models |
| 4 | HP Search | Bayesian optimization of best baseline |
| **5** | **Iteration Loop** | **Agent-driven improvement cycle (NEW)** |
| 6 | Export | Save best model + standalone code |
| 7 | Report | Generate markdown report |

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/iteration.py` | Iteration loop: strategy execution, stopping criteria, logging |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/cli.py` | Integrated iteration loop as Step 5/7, updated all step numbers |
| `co_scientist/checkpoint.py` | Added "iteration" step to STEP_ORDER, `iteration_log` to PipelineState |
| `co_scientist/agents/ml_engineer.py` | Rewrote `_next_iteration()` with smarter strategy selection; added `_extract_model_type()` and `_suggest_untrained_models()` helpers |

## Iteration Strategies

The ML Engineer proposes strategies in priority order:

| Priority | Strategy | Action | When |
|----------|----------|--------|------|
| 1 | `hp_tune` | HP-tune best model | Best model not yet tuned |
| 2 | `hp_tune` | HP-tune runner-up | Different type from best, not yet tuned |
| 3 | `try_model` | Train untrained model | Model types exist that haven't been tried |
| 4 | `try_ensemble` | Rebuild stacking ensemble | ≥3 base models, ensemble not tried in loop |
| 5 | `stop` | Stop iterating | All strategies exhausted |

Non-tunable model types (stacking, mean_predictor, majority_class) are automatically excluded from HP-tune suggestions.

## Stopping Criteria

The loop stops when any of these conditions is met:

| Condition | Description |
|-----------|-------------|
| Budget exhausted | `iteration >= config.budget` |
| Patience exceeded | No improvement for N consecutive iterations (N = max(2, budget/3)) |
| Agent says stop | ML Engineer returns `action="stop"` |
| Cost budget depleted | `cost_tracker.can_afford()` returns False |
| Stagnation detected | 3 consecutive non-stop strategies in `decisions_so_far` |

## Strategy Execution

Each strategy is executed by a dedicated function:

### `hp_tune`
- Finds the base config for the target model
- Skips if already HP-tuned in this loop iteration
- Runs Optuna HP search with configured trials/timeout
- Appends result to trained_models/results lists

### `try_model`
- Builds a model config from the suggested type
- Skips if model type already trained
- Uses `build_model()` from registry, `train_model()` from trainer
- Names models with `_iter{N}` suffix

### `try_ensemble`
- Calls `build_stacking_ensemble()` with all current base models
- Useful after new models have been added to the pool

## Iteration Log

```python
@dataclass
class IterationLog:
    iterations: list[IterationResult]
    total_iterations: int
    improvements: int
    stop_reason: str
    best_score_before: float
    best_score_after: float
```

Logged to experiment log and checkpoint for resumability.

## Interactive Mode

In interactive mode (`--mode interactive`), each iteration's strategy decision is presented to the user for approval or override before execution. Users can also type free-form questions (e.g., "why are you choosing this strategy?") or instructions (e.g., "try ensembling instead") at each iteration's decision point. The LLM answers with full pipeline context, and instructions automatically revise the recommendation before the user accepts.

## Example Output

```
── Iteration 1/3 ──
╭─── Ml Engineer — iteration_1 ───╮
│ Action: hp_tune                  │
│   target_model: random_forest    │
│   n_trials: 15                   │
│ Reasoning: HP-tune best model    │
╰──────────────────────────────────╯
  HP-tuning random_forest (15 trials, 120s timeout)
  Improved! 0.6941 → 0.6985

── Iteration 2/3 ──
  HP-tune runner-up (lightgbm, 0.6604)
  No improvement (0.6590 vs best 0.6985)

  Stopping: no improvement for 2 consecutive iterations
  Iteration loop: 2 step(s), 1 improvement(s). Score: 0.6941 → 0.6985
```

## Budget Semantics

The `--budget N` flag controls the iteration loop:
- `--budget 0`: Skip iteration loop entirely
- `--budget 1`: One iteration attempt
- `--budget 10` (default): Up to 10 improvement attempts

The budget does NOT affect baselines or initial HP search — those always run.
