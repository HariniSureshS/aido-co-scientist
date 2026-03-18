# Step 26: Tournament Ranking

## Overview

Adds **Elo-style tournament ranking** of model approaches. Instead of just comparing raw metric values, models are ranked via pairwise matchups using a margin-based Elo system. This gives a more nuanced ranking that accounts for relative performance differences.

**Problem:** Raw metric comparisons (sort by score) don't capture how consistently one model type outperforms another. A model that's 2nd-best by 0.001 is very different from one that's 2nd-best by 0.1, but both get the same rank.

**Solution:** Elo ratings via pairwise matchups with margin-based outcomes. Every pair of models plays a "match" after each training/tuning action, and their Elo ratings update based on the score difference.

---

## Architecture

### Elo System

```
Model A: score=0.72    Model B: score=0.68
         │                      │
         └──── Matchup ─────────┘
              diff = 0.04
              actual_A = sigmoid(diff / margin_scale) ≈ 0.60
              expected_A = f(Elo_A, Elo_B) = 0.50
              Δ = K * (actual - expected) = 32 * 0.10 = 3.2

              Elo_A: 1500 → 1503.2
              Elo_B: 1500 → 1496.8
```

### Margin-Based Outcomes

Unlike standard Elo (binary win/loss), we use a **sigmoid of the score difference** as the actual outcome. This means:

- A close win (0.72 vs 0.71) gives partial credit to the loser
- A dominant win (0.72 vs 0.50) gives nearly full credit to the winner
- Ties (same score) result in no rating change

---

## Key Data Structures

### Player

```python
@dataclass
class Player:
    name: str           # "xgboost_default"
    approach: str       # "xgboost"
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0
```

### EloRanker

```python
class EloRanker:
    players: dict[str, Player]
    k_factor: float = 32.0        # How fast ratings change
    margin_scale: float = 0.1     # Sensitivity of score differences
    match_history: list[dict]

    def register_model(name, approach)
    def record_matchup(model_a, model_b, score_a, score_b, lower_is_better)
    def update_from_results(results: list[ModelResult], lower_is_better)  # all-vs-all
    def get_rankings() -> list[Player]           # sorted by Elo
    def get_approach_rankings() -> list[dict]     # avg Elo per model type
    def format_table() -> str                     # for ReAct tool
    def to_dict() -> dict                         # for report
```

---

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/evaluation/ranking.py` | `Player`, `EloRanker` with margin-based Elo |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/agents/react.py` | Added `elo_ranker` field to `ReactState`. After train/tune actions, calls `update_from_results()`. Added `elo_rankings` to `ReactResult`. Elo shown in state summary. |
| `co_scientist/agents/tools.py` | Added `GetRankingsTool` (read-only, returns Elo table). Registered in default registry. |
| `co_scientist/agents/coordinator.py` | Creates `EloRanker` in `run_react_modeling()`, attaches to state |
| `co_scientist/report/generator.py` | Accepts `elo_rankings` param, passes to template |
| `co_scientist/report/template.md.jinja` | Added §4.11 "Tournament Rankings" section |
| `co_scientist/checkpoint.py` | Added `elo_rankings: dict \| None` to `PipelineState` |
| `co_scientist/cli.py` | Extracts `elo_rankings` from `react_result`, passes to report generator |

---

## Integration with ReAct Agent

### Automatic Updates

After every successful `train_model` or `tune_hyperparameters` action, the Elo ranker automatically runs pairwise matchups across all models:

```python
# In ReactAgent.run(), after each tool execution:
if result.success and result.score is not None and state.elo_ranker:
    state.elo_ranker.update_from_results(state.results, lower_is_better)
```

### GetRankingsTool

The agent can query rankings explicitly:

```
Thought: I've trained 4 models. Let me check the Elo rankings to see which approach is strongest.
Action: get_rankings({})
```

Returns:
```
Model                          Elo   Matches   Wins
--------------------------------------------------------
random_forest_tuned           1548.2        6      4
lightgbm_default              1520.1        6      3
xgboost_default               1495.3        6      2
logistic_regression_default   1436.4        6      1
```

### State Summary

Elo rankings are included in the state summary shown to the agent each step:

```
Elo Rankings:
  random_forest_tuned: Elo=1548
  lightgbm_default: Elo=1520
  xgboost_default: Elo=1495
```

---

## Report

### §4.11 Tournament Rankings

```markdown
### 4.11 Tournament Rankings

Models were ranked via Elo-style pairwise matchups (12 total matchups):

| Model | Elo | Matches | Wins |
|-------|----:|--------:|-----:|
| random_forest_tuned | 1548.2 | 6 | 4 |
| lightgbm_default | 1520.1 | 6 | 3 |
| xgboost_default | 1495.3 | 6 | 2 |
| logistic_regression_default | 1436.4 | 6 | 1 |

**By approach:** random_forest (1548), lightgbm (1520), xgboost (1495), logistic_regression (1436)
```

---

## Verification

```bash
# Run pipeline with LLM (Elo is automatic):
co-scientist run RNA/translation_efficiency_muscle --budget 5

# Check rankings in report:
grep -A 10 "Tournament Rankings" outputs/RNA__translation_efficiency_muscle/report.md
```

---

## Design Decisions

### Why Elo over simple sorting?

Elo captures *relative strength* — how consistently one model beats another — not just who has the highest single score. This is especially useful when the ReAct agent trains multiple variants (default, tuned, react_1) of the same model type. The approach-level rankings average Elo across variants, showing which *type* of model works best for this data.

### Why margin-based instead of binary win/loss?

Binary Elo would treat a 0.001 victory the same as a 0.1 victory. Margin-based Elo scales the outcome by the score difference via sigmoid. This means a close match barely changes ratings, while a dominant win has a large effect. The `margin_scale` parameter (default 0.1) controls sensitivity.

### Why K=32?

K=32 is the standard Elo K-factor for players with <30 games. In our setting, models play few matches (one per training step), so ratings need to converge quickly. K=32 is aggressive enough for 5-10 matchups to produce meaningful differentiation.
