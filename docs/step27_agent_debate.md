# Step 27: Agent Debate

## Overview

Before high-stakes decisions (model selection, HP search), two agents present competing proposals, write rebuttals, then a judge picks the winner. This provides more robust decisions through **adversarial reasoning**.

**Problem:** Single-agent consultation produces one viewpoint. The ML Engineer might recommend XGBoost because it's usually best for tabular data, while the Data Analyst might notice the data is highly sparse and recommend a different approach. With single-agent consultation, only one perspective is heard.

**Solution:** A structured debate protocol:
1. **Phase 1:** Each agent proposes independently
2. **Phase 2:** Each agent rebuts the other's proposal
3. **Phase 3:** A judge evaluates all proposals + rebuttals and picks the winner

**Cost awareness:** Each debate costs ~5 LLM calls. Only triggered at model_selection and hp_search stages. Checks `can_afford()` before initiating; falls back to simple `consult()` if budget is tight.

---

## Architecture

### Debate Protocol

```
                    ┌──────────────┐
                    │   Topic:     │
                    │ "model       │
                    │  selection"  │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    Phase 1: Propose              Phase 1: Propose
    ┌─────────────────┐       ┌─────────────────┐
    │  ML Engineer    │       │  Data Analyst    │
    │  "Use XGBoost   │       │  "Use LightGBM   │
    │   + RF"         │       │   + Ridge"       │
    └────────┬────────┘       └────────┬────────┘
             │                         │
    Phase 2: Rebuttal           Phase 2: Rebuttal
    ┌─────────────────┐       ┌─────────────────┐
    │  "Data Analyst's │      │  "ML Engineer's  │
    │   Ridge won't    │      │   XGBoost may    │
    │   capture non-   │      │   overfit with   │
    │   linear..."     │      │   only 500..."   │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             └────────────┬────────────┘
                          │
                  Phase 3: Judge
              ┌───────────────────┐
              │  Evaluates all    │
              │  proposals +      │
              │  rebuttals        │
              │                   │
              │  Winner:          │
              │  Data Analyst     │
              └───────────────────┘
```

### LLM Cost

| Phase | Calls | Typical tokens |
|-------|------:|---------------|
| Proposals | 2 | ~500 each |
| Rebuttals | 2 | ~300 each |
| Judge | 1 | ~400 |
| **Total** | **5** | **~2,100** |

At ~$0.01 per call = ~$0.05 per debate. Budget allows ~100 debates within $5 limit.

---

## Key Data Structures

### DebateTranscript

```python
@dataclass
class DebateTranscript:
    topic: str                        # "model selection strategy"
    proposals: dict[str, Decision]    # agent_name -> proposal
    rebuttals: dict[str, str]         # agent_name -> rebuttal text
    judge_reasoning: str
    winner: Decision
    winning_agent: str
```

### Debater

```python
class Debater:
    client: ClaudeClient

    def debate(topic, context, agents: dict[str, BaseAgent]) -> DebateTranscript | None
        # Phase 1: Each agent calls decide(context) → proposals
        # Phase 2: Each sees other's proposal, writes rebuttal via ask_text()
        # Phase 3: Judge prompt sees all proposals+rebuttals, picks winner
        # Returns None if LLM unavailable
```

---

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/agents/debate.py` | `DebateTranscript`, `Debater` with three-phase debate protocol |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/agents/coordinator.py` | Added `debate()` method. Added `debate_transcripts` list attribute. Falls back to `consult()` when LLM unavailable or budget tight. |
| `co_scientist/llm/prompts.py` | Added `DEBATE_REBUTTAL_SYSTEM` and `DEBATE_JUDGE_SYSTEM` prompts |
| `co_scientist/cli.py` | At model_selection and hp_search decision points, uses `debate()` instead of `consult()` |
| `co_scientist/report/generator.py` | Accepts `debate_transcripts` param |
| `co_scientist/report/template.md.jinja` | Added §4.12 "Agent Debates" section |
| `co_scientist/checkpoint.py` | Added `debate_transcripts: list[dict] \| None` to `PipelineState` |

---

## Coordinator Integration

### debate() Method

```python
def debate(self, topic, context, agent_names=None) -> Decision:
    """Run a debate between agents. Falls back to consult() on failure."""

    # Guard: need LLM + enough budget for 5 calls
    if not self.llm_available or not self.cost_tracker.can_afford(min_calls=5):
        return self.consult(agent_names[0], context)

    # Run debate
    debater = Debater(self.client)
    transcript = debater.debate(topic, context, agents)

    # Store transcript for report
    self.debate_transcripts.append(transcript)

    return transcript.winner
```

### Decision Points

Debate is used at high-stakes stages:

| Stage | Debaters | Topic |
|-------|----------|-------|
| Model Selection | ML Engineer vs Data Analyst | "model selection strategy" |
| HP Search | ML Engineer vs Data Analyst | "hyperparameter search strategy" |
| **Pre-ReAct** | **ML Engineer vs Data Analyst** | **"modeling strategy" (before ReAct loop starts)** |

The pre-ReAct debate runs in `cli.py` before `run_react_modeling()`. Previously, debates only fired in the deterministic path and were dead code when ReAct was active. The debate proposals, rebuttals, and judge verdict appear in the Agent Conversations dashboard panel.

Other stages (preprocessing, iteration) continue to use simple `consult()` since they're lower-stakes or more frequent.

### Foundation Model Awareness in Debates

When a GPU is available, agents participating in debates are now aware of foundation model options. The ML Engineer's system prompt includes guidance to recommend `embed_xgboost`, `embed_mlp`, `concat_xgboost`, `concat_mlp`, and `aido_finetune` when appropriate.

This means debates about model selection can now include arguments like:
- **ML Engineer**: "With AIDO embeddings available, concat_xgboost (handcrafted + embeddings) is likely to outperform pure k-mer features on this RNA dataset"
- **Data Analyst**: "The dataset only has 300 samples — fine-tuning risks overfitting. embed_xgboost with frozen embeddings is safer"

The debate protocol itself is unchanged — the foundation model awareness comes from the updated system prompts, not from any debate-specific code.

---

## System Prompts

### DEBATE_REBUTTAL_SYSTEM

Instructs the agent to defend its proposal and critique the alternative. Emphasizes being specific and data-driven, referencing dataset characteristics.

### DEBATE_JUDGE_SYSTEM

Instructs the judge to evaluate both proposals and rebuttals, consider dataset constraints, and pick the winner. Emphasizes practicality over theory.

---

## Report

### §4.12 Agent Debates

```markdown
### 4.12 Agent Debates

**Debate: model selection strategy**

| Agent | Proposal | Confidence |
|-------|----------|-----------|
| Ml Engineer | select_models: XGBoost and RF are standard for tabular... | 85% |
| Data Analyst | select_models: High sparsity suggests LightGBM with... | 78% |

**Winner:** Data Analyst — The dataset's 95% sparsity favors LightGBM's native sparse handling...
```

---

## Graceful Degradation

| Scenario | Behavior |
|----------|----------|
| No API key | `debate()` → `consult()` → `decide_deterministic()` |
| Budget < 5 calls | `debate()` → `consult()` |
| LLM call fails in debate | `debate()` → `consult()` |
| Only 1 agent available | `debate()` → `consult()` |
| Rebuttal call fails | Empty rebuttal, judge still sees proposals |
| Judge call fails | First agent's proposal wins by default |

---

## Verification

```bash
# Run with LLM (debate at model_selection and hp_search):
co-scientist run RNA/translation_efficiency_muscle --budget 5

# Check debate transcripts in report:
grep -A 15 "Agent Debate" outputs/RNA__translation_efficiency_muscle/report.md

# Without API key (debate falls back to consult — unchanged behavior):
unset ANTHROPIC_API_KEY
co-scientist run RNA/translation_efficiency_muscle --budget 5
```

---

## Design Decisions

### Why only 2 debaters?

More debaters = more LLM calls. With 3 debaters, each rebuttal round triples. Two debaters keeps the cost at 5 calls while still providing adversarial tension. The ML Engineer and Data Analyst naturally have different perspectives on model selection.

### Why not debate at every stage?

Debates cost ~5x a simple consultation. Model selection and HP search are high-impact decisions where a wrong choice wastes many pipeline steps. Preprocessing and iteration decisions are lower-stakes or made frequently, so the extra cost isn't justified.

### Why a separate judge instead of confidence-weighted voting?

A judge can weigh the rebuttals — an agent might have high confidence but a weak rebuttal. The judge prompt sees the full argument and can evaluate argument quality, not just confidence scores.
