# Step 17: Agent Framework — Detailed Walkthrough

## Overview

The agent framework introduces the **multi-agent pattern** from Architecture Section 4. Instead of a monolithic pipeline making hardcoded decisions, specialized agents advise on key decisions (which models to train, how to preprocess, when to stop). Deterministic code still executes — agents are strategists, not executors.

This is the **"LLM-as-strategist"** pattern: agents propose, the pipeline disposes.

---

## Why Agents?

The pipeline before Step 17 made all decisions via hardcoded rules:
- Model selection: always train everything in defaults.yaml
- Preprocessing: always use modality-specific defaults
- HP search: always tune the best model
- Stopping: always run all steps

This works, but it can't:
- Adapt strategy mid-run based on results
- Diagnose failures intelligently
- Explain *why* it chose a particular approach
- Learn from what's working for this specific dataset

Agents add a decision layer that can reason about the pipeline state.

---

## Architecture

### Dual-Path Decision Making

Every agent has two decision paths:

```
PipelineContext (current state)
        │
        ▼
   BaseAgent.decide()
        │
        ├── LLM available? ──── YES ──→ Call Claude API
        │                                    │
        │                               Parse JSON response
        │                                    │
        │                               Return Decision
        │
        └── NO (or LLM failed) ──→ decide_deterministic()
                                         │
                                    Rule-based logic
                                         │
                                    Return Decision
```

This ensures **graceful degradation**:
1. **Full LLM mode**: Claude reasons about dataset + results → intelligent decisions
2. **Deterministic mode**: Rule-based fallback replicates pre-agent behavior exactly
3. **YAML defaults**: If even deterministic logic fails, pipeline uses defaults.yaml

### Key Types

```python
class PipelineContext:
    """Read-only snapshot of pipeline state, given to agents."""
    dataset_path: str        # e.g. "RNA/translation_efficiency_muscle"
    modality: str            # "rna", "cell_expression", etc.
    task_type: str           # "regression", "multiclass_classification"
    num_samples: int
    num_features: int
    stage: str               # "model_selection", "hp_search", "iteration"
    model_scores: dict       # {"random_forest": 0.6941, "xgboost": 0.6279}
    best_model_name: str
    best_score: float
    remaining_budget: int
    remaining_cost: float    # LLM $ remaining
    complexity_level: str    # "simple", "moderate", "complex"

class Decision:
    """What an agent recommends the pipeline should do."""
    action: str              # e.g. "select_models", "hp_tune", "stop"
    parameters: dict         # action-specific payload
    reasoning: str           # why (for logging and reports)
    confidence: float        # 0-1
    fallback: str            # what to do if this fails
```

### Agent Roles

| Agent | Role | Decides On |
|-------|------|-----------|
| **Data Analyst** | Data expert | Preprocessing, features, data quality, split strategy |
| **ML Engineer** | Model expert | Model selection, HP tuning, iteration strategy, failure diagnosis |
| **Biology Specialist** | Domain expert | Biological context, plausibility, domain features, interpretation |
| **Coordinator** | Orchestrator | Which agents to consult, conflict resolution, stopping |

### Coordinator

The Coordinator is the central hub:

```python
coordinator = Coordinator(cost_tracker=tracker)

# At a decision point:
decision = coordinator.consult("ml_engineer", context, stage="model_selection")
# decision.action = "select_models"
# decision.parameters = {"models": ["xgboost", "lgbm", "rf", ...], "priority": "bio_cnn"}

# Or consult all active agents:
decisions = coordinator.consult_all_active(context, stage="iteration")
resolved = coordinator.resolve_conflict(decisions)
```

The Coordinator also:
- **Activates agents** based on complexity (simple datasets don't need Biology Specialist)
- **Logs all messages** for audit trail
- **Enforces budget** — refuses LLM calls when cost limit reached

---

## Agent Activation by Complexity

| Complexity | Active Agents |
|-----------|--------------|
| Simple (0-2) | Data Analyst, ML Engineer |
| Moderate (3-5) | Data Analyst, ML Engineer |
| Complex (6-8) | Data Analyst, ML Engineer, Biology Specialist |
| Very Complex (9-10) | All (including Research, when implemented) |

---

## Deterministic Fallbacks

Each agent's `decide_deterministic()` replicates the pre-agent pipeline behavior:

### Data Analyst
```
RNA/DNA/Protein → k-mer features (k=4 for RNA, k=3 for DNA) + standard scaling
                  + recommend CNN if >500 samples
Cell Expression → log1p + HVG selection + standard scaling
Tabular         → standard scaling
```

### ML Engineer
```
Model Selection → always include tree models (XGBoost, LightGBM, RF)
                  + linear models (ridge/elastic net)
                  + MLP if >300 samples
                  + BioCNN if sequence data + >200 samples
HP Search       → tune if best score < 0.95 (classification)
                  always tune for regression
Iteration       → try ensemble if not tried
                  tune best model if early iteration
                  stop after 3 HP-tune attempts with no gain
```

### Biology Specialist
```
Plausibility → suspicious if score > 0.99 (possible leakage)
               implausible if score < 0.01
Context      → modality-specific biological background
Features     → codon_usage_bias, gc_content, etc. for RNA
               pathway_scores, cell_cycle_markers for expression
```

---

## Pipeline Integration

The Coordinator is consulted at two decision points in `cli.py`:

### 1. Model Selection (before training baselines)
```python
pipe_ctx = PipelineContext(
    dataset_path=config.dataset_path,
    modality=state.profile.modality.value,
    task_type=state.profile.task_type.value,
    num_samples=state.profile.num_samples,
    ...
)
ml_decision = coordinator.consult("ml_engineer", pipe_ctx, stage="model_selection")
# Logged to experiment_log.jsonl
```

### 2. HP Search (before tuning)
```python
hp_decision = coordinator.consult("ml_engineer", hp_ctx, stage="hp_search")
# decision.action = "hp_tune" or "skip_hp_search"
```

### Non-Invasive Integration

The agent consultation is **additive** — it logs the decision but doesn't change pipeline flow yet. The existing deterministic logic still executes. This allows:
- Validating agent decisions against actual pipeline behavior
- Gradually transferring control from hardcoded logic to agent decisions
- Always having a working fallback

---

## Message Logging

Every agent interaction is logged:

```json
{
  "timestamp": "2026-03-16T06:15:35.488467+00:00",
  "event": "agent_decision",
  "data": {
    "agent": "ml_engineer",
    "stage": "model_selection",
    "action": "select_models",
    "confidence": 0.9,
    "reasoning": "Standard model set for rna/regression with 1257 samples"
  }
}
```

This creates an audit trail of why the pipeline made each decision.

---

## File Structure

```
co_scientist/
├── agents/
│   ├── __init__.py          ← exports Coordinator, AgentRole, Decision, PipelineContext
│   ├── types.py             ← AgentRole, MessageType, AgentMessage, Decision, PipelineContext
│   ├── base.py              ← BaseAgent ABC with decide() + decide_deterministic()
│   ├── coordinator.py       ← Coordinator: agent routing, conflict resolution, budget
│   ├── data_analyst.py      ← DataAnalystAgent: preprocessing, features, data quality
│   ├── ml_engineer.py       ← MLEngineerAgent: model selection, HP, iteration, diagnostics
│   └── biology.py           ← BiologySpecialistAgent: context, plausibility, domain features
├── cli.py                   ← Coordinator integration at decision points
```

---

## Design Decisions

### Why LLM-as-Strategist?

Sakana AI Scientist (v1) had the LLM write and execute arbitrary code, leading to a 42% experiment failure rate. Our agents only make **decisions** (which model, which preprocessing) — execution is handled by deterministic, tested code. This is more reliable and reproducible.

### Why Structured JSON Output?

Agents return JSON, not free text. This means:
- Decisions are machine-parseable and executable
- The pipeline doesn't need NLP to understand agent output
- Logging is structured and queryable
- Fallback on parse failure is clean (use deterministic path)

### Why Per-Agent System Prompts?

Each agent has a focused system prompt with:
- Clear role definition
- Expected JSON output format
- Domain-specific rules and constraints
- Budget awareness

This is more effective than a single "do everything" prompt.
