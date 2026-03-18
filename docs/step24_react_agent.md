# Step 24: ReAct Agent Architecture

## Overview

The ReAct (Reasoning + Acting) agent replaces the separate baselines → HP search → iteration loop (Steps 3-5) with a single **Thought → Action → Observation** cycle driven by the LLM. The agent reasons about observations, selects tools, and adapts its strategy — all within one unified loop.

**Problem:** The previous agent layer was a "fake agent" — the LLM picked from a hardcoded priority list and the pipeline executed. There was no reasoning, no observation feedback, no scratchpad.

**Solution:** A proper ReAct loop where the LLM drives the entire modeling phase through Thought → Action → Observation cycles, with 7 tools wrapping existing infrastructure.

**Constraint:** Must still work without an API key (deterministic fallback unchanged).

---

## Architecture

### Dual-Path Branching

```
                    ┌─ API key set? ──────────────────────────┐
                    │                                          │
               YES  │                                     NO   │
                    ▼                                          ▼
         ┌──────────────────┐                    ┌──────────────────┐
         │  ReAct Agent     │                    │  Deterministic   │
         │  (Steps 3-5)     │                    │  Path (unchanged)│
         │                  │                    │  Step 3: Baselines│
         │  Thought → Act   │                    │  Step 4: HP Search│
         │    → Observe     │                    │  Step 5: Iteration│
         │    → Thought ... │                    │                  │
         └────────┬─────────┘                    └────────┬─────────┘
                  │                                       │
                  └───────────┬───────────────────────────┘
                              ▼
                     Step 6: Export
                     Step 7: Report (with reasoning trace)
```

One ReAct loop replaces three pipeline steps. When the LLM is available, the agent has full control over which models to train, when to tune hyperparameters, when to build ensembles, and when to stop. When the LLM is not available (or the agent fails), the existing deterministic path runs unchanged.

**Pre-ReAct debate:** Before the ReAct loop starts, ML Engineer and Data Analyst debate modeling strategy (in `cli.py`, before `run_react_modeling()`). Previously, debates only ran in the deterministic path and were dead code when ReAct was active. The debate proposals, rebuttals, and judge verdict appear in the Agent Conversations dashboard panel, giving the ReAct agent an informed starting context.

### ReAct Loop

```
┌──────────────────────────────────────────────────┐
│                  ReAct Agent Loop                 │
│                                                   │
│  for step in 1..max_steps:                       │
│    1. Build user message with state + scratchpad  │
│    2. Call LLM (ask_text) → get response         │
│    3. Parse: Thought + Action from response      │
│    4. Execute tool → get ToolResult              │
│    5. Update best model if improved              │
│    6. Append to scratchpad                       │
│    7. Check stopping (budget, patience, finish)  │
│    8. Compress scratchpad every 8 steps          │
└──────────────────────────────────────────────────┘
```

---

## Pipeline Steps (Updated)

| Step | Name | Description |
|------|------|-------------|
| 1 | Load & Profile | Load dataset, detect modality/task |
| 2 | Preprocess & Split | Feature engineering, train/val/test split |
| **3-5** | **ReAct Agent** | **LLM-driven modeling loop (NEW, replaces Steps 3-5 when LLM available)** |
| 3 | Train Baselines | Trivial → simple → standard → advanced models (deterministic fallback) |
| 4 | HP Search | Bayesian optimization of best baseline (deterministic fallback) |
| 5 | Iteration Loop | Agent-driven improvement cycle (deterministic fallback) |
| 6 | Export | Save best model + standalone code |
| 7 | Report | Generate markdown report (with reasoning trace if ReAct was used) |

---

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/agents/tools.py` | Tool + ToolResult + ToolRegistry + 13 tool classes |
| `co_scientist/agents/react.py` | ReactAgent + ReactState + ScratchpadEntry + ReactResult |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/llm/prompts.py` | Added `REACT_AGENT_SYSTEM` prompt constant |
| `co_scientist/agents/coordinator.py` | Added `run_react_modeling()` method |
| `co_scientist/cli.py` | Added ReAct branching after Step 2, before deterministic Steps 3-5 |
| `co_scientist/checkpoint.py` | Added `react_scratchpad: list[dict] \| None` to `PipelineState` |
| `co_scientist/report/generator.py` | Added `react_scratchpad` parameter to `generate_report()` |
| `co_scientist/report/template.md.jinja` | Added §4.9 "Agent Reasoning Trace" section |

---

## Tool Registry

The agent has 13 tools, each wrapping existing infrastructure:

| Tool | Wraps | What it does |
|------|-------|-------------|
| `summarize_data` | `DatasetProfile` + statistical analysis | Return detailed dataset statistics (feature sparsity, correlations, variance, target distribution, class balance, sequence lengths, **AIDO embedding availability**). **Called first** before model selection. |
| `train_model` | `registry.build_model` + `model.fit` + `metrics.evaluate_model` | Train a model type (CPU: xgboost, lightgbm, etc.; GPU: embed_xgboost, concat_xgboost, aido_finetune), return val score |
| `tune_hyperparameters` | `hp_search.run_hp_search` | Optuna HP search on a model, return before/after |
| `get_model_scores` | read-only on results list | Return table of all trained models + scores |
| `analyze_errors` | confusion_matrix / residual stats | Error analysis for a model |
| `build_ensemble` | `ensemble.build_stacking_ensemble` | Stack trained models, return ensemble score |
| `inspect_features` | sklearn `feature_importances_` / `coef_` | Top-N feature importances from a model |
| `get_rankings` | `EloRanker.format_table` | Show Elo tournament rankings of all models |
| `design_model` | LLM code generation + `custom_model.load_custom_model` | Design a custom PyTorch architecture, validate, train, and evaluate |
| `consult_biology` | `BiologySpecialist` agent | Consult the Biology Specialist for validation of biological plausibility, metric appropriateness, or feature interpretation during the ReAct loop |
| `diagnose_data` | `DataAnalyst` agent | Consult the Data Analyst for data diagnosis (quality issues, distribution concerns, feature recommendations) during the ReAct loop |
| `finish` | — | Signal agent wants to stop, with reason |
| `backtrack` | — | (Tree search only) Backtrack to a previous state |

### Key Structures

```python
@dataclass
class ToolResult:
    success: bool
    observation: str      # Human-readable for scratchpad
    data: dict[str, Any]  # Structured for programmatic use
    model_name: str = ""
    score: float | None = None

class Tool(ABC):
    name: str
    description: str
    parameters_schema: dict
    def execute(self, params: dict, state: ReactState) -> ToolResult

class ToolRegistry:
    tools: dict[str, Tool]
    def describe_all(self) -> str  # Injected into system prompt
    def tool_names(self) -> list[str]
```

### Tool Details

#### `summarize_data`
- **Parameters:** none
- **Returns:** Feature sparsity (fraction of zeros per feature), pairwise feature correlations, per-feature variance, target distribution summary, class balance (classification) or target quantiles (regression), and sequence lengths (if applicable)
- Read-only — does not modify state
- The agent is instructed to call this tool **first**, before choosing any models, so that model selection is driven by actual data characteristics

#### `train_model`
- **Parameters:** `model_type` (required), `hyperparameters` (optional)
- **CPU types:** xgboost, lightgbm, random_forest, svm, knn, logistic_regression, ridge_regression, mlp, bio_cnn, ft_transformer
- **GPU types (foundation):** embed_xgboost, embed_mlp (AIDO embeddings), concat_xgboost, concat_mlp (handcrafted + embeddings), aido_finetune (end-to-end fine-tuning)
- Foundation models require AIDO embeddings (`X_embed_train`) — the tool checks availability and returns an error if not present
- Names models with `_default` suffix (first) or `_react_N` suffix (subsequent)
- Automatically injects `random_state` and LightGBM `verbose=-1`
- Appends to `state.trained_models` and `state.results`

#### `tune_hyperparameters`
- **Parameters:** `model_type` (required), `n_trials` (optional, default 15), timeout 120s (previously 180s/20 trials)
- Model must already be trained (checked)
- Uses existing Optuna search spaces from defaults.yaml (available for all model types including svm, knn, ft_transformer)
- Returns before/after comparison

#### `design_model`
- **Parameters:** `architecture_hint` (required), `hyperparameters` (optional)
- Sends dataset characteristics + architecture hint to the LLM
- LLM generates a full PyTorch model class
- Code is AST-validated (blocks disallowed imports: os, subprocess, sys, etc.)
- Model is dynamically loaded, trained, and evaluated
- Costs one extra LLM call — agent is guided to use it only after standard models have been tried
- Cannot be HP-tuned via `tune_hyperparameters` (agent calls `design_model` again with revised hints instead)

#### `get_model_scores`
- **Parameters:** none
- Returns a formatted table sorted by primary metric
- Read-only — does not modify state

#### `analyze_errors`
- **Parameters:** `model_name` (optional, defaults to best)
- Classification: returns `classification_report` (precision, recall, F1 per class)
- Regression: returns mean/std/max residual statistics

#### `build_ensemble`
- **Parameters:** none
- Requires ≥2 non-trivial base models
- Uses existing `build_stacking_ensemble()` with cross-validated meta-learner

#### `inspect_features`
- **Parameters:** `model_name` (optional), `top_n` (optional, default 10)
- Works with tree-based (`feature_importances_`) and linear (`coef_`) models
- Falls back to generic feature names if names unavailable

#### `consult_biology`
- **Parameters:** `question` (required)
- Invokes the Biology Specialist agent within the ReAct loop
- Use for biological plausibility checks, metric appropriateness validation, or feature interpretation
- Read-only — does not modify state
- Returns the specialist's assessment as the observation

#### `diagnose_data`
- **Parameters:** `question` (required)
- Invokes the Data Analyst agent within the ReAct loop
- Use for data quality diagnosis, distribution concerns, or feature engineering recommendations
- Read-only — does not modify state
- Returns the analyst's diagnosis as the observation

#### `finish`
- **Parameters:** `reason` (required)
- Terminates the ReAct loop with the given reason

---

## ReAct Agent

### ReactState

Mutable state shared between the agent and its tools:

```python
@dataclass
class ReactState:
    profile: DatasetProfile     # Read-only
    split: SplitData            # Read-only
    eval_config: EvalConfig     # Read-only
    seed: int = 42

    trained_models: list[TrainedModel] = []  # Mutated by tools
    results: list[ModelResult] = []           # Mutated by tools
    best_result: ModelResult | None = None    # Updated each step
    best_trained: TrainedModel | None = None  # Updated each step
```

### ScratchpadEntry

```python
@dataclass
class ScratchpadEntry:
    step: int
    thought: str
    action: str
    action_params: dict
    observation: str
    score_after: float | None  # best score after this step
```

### ReactResult

```python
@dataclass
class ReactResult:
    trained_models: list[TrainedModel]
    results: list[ModelResult]
    best_result: ModelResult
    best_trained: TrainedModel
    scratchpad: list[ScratchpadEntry]
    stop_reason: str
    total_steps: int
    improvements: int
```

### Response Parsing

The LLM must produce:
```
Thought: <reasoning about current state and what to do next>
Action: tool_name({"param": "value"})
```

Parsed via regex:
- `Thought:` block extracted by `_THOUGHT_RE`
- `Action: name({json})` extracted by `_ACTION_RE`
- Also supports no-params: `Action: tool_name()`

On parse failure: increment counter. After 3 consecutive failures → return `None` → deterministic fallback.

### Scratchpad Compression

Every 8 steps, older entries are compressed:

**Before compression (12 entries):**
```
Step 1: Thought: ... Action: train_model(...) Observation: ...
Step 2: Thought: ... Action: train_model(...) Observation: ...
...
Step 12: Thought: ... Action: ... Observation: ...
```

**After compression:**
```
[Summary of steps 1-8]
Step 1: train_model(model_type=xgboost), score=0.6279; Step 2: train_model(model_type=lightgbm), score=0.6455; ...

Step 9: <full entry>
Step 10: <full entry>
Step 11: <full entry>
Step 12: <full entry>
```

Last 4 entries are always kept verbatim. No LLM call for compression — pure string formatting.

### Stopping Conditions

| Condition | Description |
|-----------|-------------|
| Max steps reached | `step >= max_steps` (default 25) |
| Patience exceeded | No improvement for 8 consecutive **scoring** steps (non-scoring actions such as `summarize_data`, `get_model_scores`, `analyze_errors`, `inspect_features`, `get_rankings`, `consult_biology`, and `diagnose_data` do not count against the patience budget) |
| Cost budget depleted | `cost_tracker.can_afford()` returns False |
| Agent calls `finish` | Agent explicitly signals it's done |
| Parse failures | 3 consecutive LLM response parse failures → return None |
| LLM unavailable | `ask_text()` returns None → exit loop |
| Wall-clock timeout | `max_wall_seconds=min(900, 60% of remaining pipeline time)` — dynamic, checked at each step, hard stop if exceeded. Capped at 900s. Previously a fixed 1800s. |
| Repeated actions (hard stop) | `max_repeated_actions=4` — after 4 identical tool calls, the loop force-stops |
| Consecutive tool failures | `max_consecutive_tool_failures=5` — if 5 tools fail in a row, the loop exits |
| User stops (interactive) | In `--mode interactive`, user types "exit"/"stop" at the post-step pause |

### Interactive Mode in ReAct Loop

When `--mode interactive` is active, the ReAct loop pauses after every Thought/Action/Observation step. The user can:
- **Press Enter** — continue to next step
- **Type "exit"/"stop"** — halt the ReAct loop immediately (pipeline continues to export/report with best model so far)
- **Type feedback** — injected into the agent's next LLM call as: `"IMPORTANT — The user has provided feedback: '...' Take this into account."`
- **Ask a question** — answered by the LLM using current pipeline context (model scores, data stats, etc.)

This enables real-time human-in-the-loop steering of the modeling process. The user can:
- Redirect the agent ("try foundation models", "focus on regularization")
- Ask why the agent made a choice ("why did you pick random forest?")
- Stop early when satisfied ("the score is good enough, stop")

In auto mode, no pauses occur — the loop runs uninterrupted as before.

**Implementation:** `ReactAgent._interactive_pause()` in `agents/react.py`. User feedback is stored in `self._user_feedback` and appended to the next `user_msg` sent to the LLM.

### Repeated Action Detection

If the agent calls the same tool with the same parameters 3 times in a row, a hint is injected into the observation:
```
HINT: You've called this exact action 3 times. Try something different.
```

After 4 identical calls (`max_repeated_actions`), the loop **hard-stops** rather than continuing to hint.

### Tool Execution

Tools run synchronously with a default timeout of 120s (previously 300s). The ReAct loop's wall-clock timeout handles overall time limits. During long tool executions, heartbeat progress messages are printed every 30s so the user knows the pipeline is still active.

### LLM Request Timeout

The Anthropic SDK client is initialized with `request_timeout=60.0` seconds (previously 120s). If any LLM API call takes longer than 60 seconds, it raises a timeout exception that is caught and treated as a failed call.

### LLM Retry with Exponential Backoff

LLM requests in `client.py` use exponential backoff: up to 3 attempts with delays of 2s, 4s, and 8s between retries. This handles transient API errors and rate limits without immediately falling back to the deterministic path.

---

## System Prompt

The `REACT_AGENT_SYSTEM` prompt in `co_scientist/llm/prompts.py` includes:

1. **Role description:** "You are an ML Engineer driving an automated pipeline for biological datasets."
2. **Tool descriptions:** Injected at runtime from `ToolRegistry.describe_all()`
3. **Output format:** Strict `Thought: ...\nAction: tool_name({"params": ...})` format with examples
4. **Strategy guidance (10 steps, 0-9):**
   - **Step 0:** Start with `summarize_data()` to understand the dataset before choosing any models
   - **Step 1:** Dynamically select models based on data analysis, not from a fixed list
   - **Step 2:** Try diverse model types including ft_transformer
   - **Step 3:** Explain **why** each model is chosen in the Thought (e.g., "FT-Transformer can capture feature interactions via attention", "1257 samples is borderline for attention networks"). Reference dataset characteristics (sparsity, sample size, feature count) and biological context in reasoning
   - **Step 4:** Consult specialists — use `consult_biology` to validate biological plausibility or metric choices, and `diagnose_data` to get data quality assessments or feature recommendations
   - **Step 5:** Tune the best after 2-3 models trained
   - **Step 6:** Use `analyze_errors` to understand failures
   - **Step 7:** Build ensemble when 3+ diverse models available
   - **Step 8:** Try `design_model` if standard models have plateaued
   - **Step 9:** Call `finish` when no further improvement is likely
5. **Rules:**
   - Always call `summarize_data()` first
   - Always train 2-3 model types before tuning
   - Don't tune untrained models
   - Don't repeat the same action
   - Be efficient

The prompt uses `{tool_descriptions}` and `{tool_names}` format placeholders, filled at runtime.

---

## Pipeline Integration

### cli.py Branching

After Step 2 (Preprocess & Split), before the existing Step 3:

```python
# ReAct path: when LLM is available
if coordinator.llm_available and not state.is_complete("baselines"):
    react_result = coordinator.run_react_modeling(
        profile=state.profile,
        split=state.split,
        eval_config=state.eval_config,
        seed=config.seed,
        exp_log=exp_log,
    )
    if react_result is not None:
        # Populate state from react_result
        state.trained_models = react_result.trained_models
        state.results = react_result.results
        state.best_result = react_result.best_result
        state.best_trained = react_result.best_trained
        state.react_scratchpad = [...]  # Serialized scratchpad entries

        # Mark Steps 3-5 complete → skip to Step 6
        state.mark_complete("baselines")
        state.mark_complete("hp_search")
        state.mark_complete("iteration")

# Deterministic fallback: runs ONLY when LLM unavailable or ReAct returns None
if not state.is_complete("baselines"):
    # ... unchanged Steps 3, 4, 5 ...
```

### Coordinator.run_react_modeling()

New method on the `Coordinator` class:

```python
def run_react_modeling(self, profile, split, eval_config, seed, exp_log, max_steps, patience):
    if not self.llm_available:
        return None
    registry = build_default_registry()
    agent = ReactAgent(client=self.client, tool_registry=registry, ...)
    state = ReactState(profile=profile, split=split, eval_config=eval_config, seed=seed)
    return agent.run(state, exp_log)  # ReactResult or None
```

Wrapped in a try/except — any unhandled exception returns `None` (deterministic fallback).

---

## Checkpoint & Resume

The `PipelineState` now includes:

```python
self.react_scratchpad: list[dict] | None = None
```

When the ReAct agent completes, the scratchpad is serialized as a list of dicts:
```python
{
    "step": 1,
    "thought": "RNA data with 1257 samples...",
    "action": "train_model",
    "action_params": {"model_type": "xgboost"},
    "observation": "Trained xgboost_default: spearman=0.6279 (0.3s)",
    "score_after": 0.6279
}
```

This is saved to the checkpoint pickle and used for the report.

---

## Report Generation

### §4.9 Agent Reasoning Trace

When `react_scratchpad` is present, the report shows the full reasoning trace:

```markdown
### 4.9 Agent Reasoning Trace

The ReAct agent drove the modeling phase through Thought → Action → Observation cycles:

**Step 1**
- **Thought:** RNA data with 1257 samples. Tree models are a good starting point.
- **Action:** `train_model({"model_type": "xgboost"})`
- **Observation:** Trained xgboost_default: spearman=0.6279 (0.3s)
- **Best score after:** 0.6279

**Step 2**
- **Thought:** XGBoost scored 0.63. Let me try LightGBM for comparison.
- **Action:** `train_model({"model_type": "lightgbm"})`
- **Observation:** Trained lightgbm_default: spearman=0.6455 (0.2s)
- **Best score after:** 0.6455
```

When `react_scratchpad` is not present (deterministic path), the template falls back to the existing flat Agent Decision Log table.

This is handled in `template.md.jinja` with:
```jinja
{% if react_scratchpad %}
  {# Show reasoning trace #}
{% elif agent_reasoning %}
  {# Show flat decision table (unchanged) #}
{% endif %}
```

The `generate_report()` function now accepts a `react_scratchpad` parameter and passes it to the template.

---

## Safety & Fallback

| Scenario | Behavior |
|----------|----------|
| No API key | `coordinator.llm_available` is False → skip ReAct → deterministic path |
| ReAct agent fails (exception) | Caught by try/except → returns None → deterministic path |
| ReAct returns no models | Returns None → deterministic path |
| LLM parse failures (3x) | Exit loop → return None → deterministic path |
| Tool execution error | Returns `ToolResult(success=False, observation="Error: ...")` → agent sees error and adapts |
| Repeated actions (3x same) | Hint injected: "Try something different" |
| Repeated actions (4x same) | **Hard stop** — loop exits immediately |
| Consecutive tool failures (5x) | Loop exits — likely systemic issue |
| Tool hangs | Tool timeout default 120s; heartbeat messages print every 30s during long executions |
| LLM request hangs (>60s) | Anthropic SDK raises timeout, retried up to 3 times with exponential backoff (2s/4s/8s) |
| Wall-clock timeout (>min(900s, 60% remaining)) | Hard stop — dynamic budget from PipelineDeadline, capped at 900s |
| Cost overrun | `cost_tracker.can_afford()` checked before each LLM call |
| Max steps reached | Loop exits naturally after 25 steps |
| No improvement (patience) | Loop exits after 8 steps without improvement |

The deterministic path is **completely unchanged** — it runs exactly as before for users without an API key.

---

## Example Output

### With API Key (ReAct Agent)

```
  Agent framework: LLM-powered decisions enabled

  ── ReAct Agent: Automated Modeling ──
  Primary metric: spearman
  ReAct agent starting...

  Step 1  [ML Engineer]
    Thought: First, I need to understand the dataset before choosing models.
    Action: summarize_data()
    Observation: 1257 samples, 1024 k-mer features, sparsity=0.38, low pairwise
      correlations (mean=0.05), target variance=0.42, continuous target (regression).

  Step 2  [ML Engineer]
    Thought: 1024 sparse k-mer features with low correlations — tree models handle
      sparse tabular data well. Starting with XGBoost as a strong baseline.
    Action: train_model({"model_type": "xgboost"})
    Observation: Trained xgboost_default: spearman=0.6279 (0.3s)

  Step 3  [ML Engineer]
    Thought: XGBoost scored 0.63. Let me try LightGBM — it handles sparse features
      well and is often competitive with XGBoost.
    Action: train_model({"model_type": "lightgbm"})
    Observation: Trained lightgbm_default: spearman=0.6455 (0.2s)

  Step 4  [ML Engineer]
    Thought: LightGBM is slightly ahead. Let me also try Random Forest — it's robust
      to sparse features and won't overfit with this sample size.
    Action: train_model({"model_type": "random_forest"})
    Observation: Trained random_forest_default: spearman=0.6941 (1.2s)

  Step 5  [ML Engineer]
    Thought: Random Forest is best at 0.69. Let me tune its hyperparameters.
    Action: tune_hyperparameters({"model_type": "random_forest"})
    Observation: Tuned random_forest: spearman 0.6941 → 0.7103 (improved)

  Step 6  [Data Analyst]
    Thought: Good improvement from tuning. With 3 diverse models, let me try an ensemble.
    Action: build_ensemble({})
    Observation: Stacking ensemble: spearman=0.7050

  Step 7  [Evaluator]
    Thought: Ensemble didn't beat the tuned RF. I've trained 3 diverse model types
      chosen based on the data characteristics, tuned the best, and tried ensembling.
      Time to stop.
    Action: finish({"reason": "Best model tuned, ensemble tried, diminishing returns"})
    Agent finished: Best model tuned, ensemble tried, diminishing returns

  ReAct complete: 7 steps, 2 improvement(s), best=random_forest_tuned (spearman=0.7103)
```

### Without API Key (Deterministic — unchanged)

```
  Agent framework: deterministic mode (no API key)

  ── Step 3/7: baselines ──
  Training majority_class (trivial)...
  Training logistic_regression (simple)...
  Training xgboost (standard)...
  ...
```

---

## Dashboard Integration

The live dashboard now displays which agent persona is active during each ReAct step. The three agent roles are:

| Agent | Role |
|-------|------|
| **ML Engineer** | Model selection, training, hyperparameter tuning |
| **Data Analyst** | Data summarization, error analysis, feature inspection, ensemble building |
| **Evaluator** | Score comparison, ranking review, final evaluation, finish decision |

The active agent label (e.g., `[ML Engineer]`) appears next to each step in the dashboard output, giving real-time visibility into which reasoning persona is driving the current action. This is a display-level annotation — all three roles are played by the same underlying ReAct agent and share the same scratchpad and state.

---

## Design Decisions

### Why ReAct over Multi-Agent Consultation?

The previous architecture (Steps 17-23) used multiple agents consulted at fixed decision points:
- ML Engineer for model selection
- ML Engineer for HP search decision
- ML Engineer for each iteration strategy

This had several limitations:
1. **No memory between consultations** — each agent call was independent
2. **Fixed execution order** — baselines always before HP search, always before iteration
3. **No observation-driven adaptation** — the agent couldn't react to training results mid-stream
4. **Reasoning not captured** — agent decisions were opaque JSON outputs

ReAct fixes all of these:
1. **Scratchpad provides memory** — the agent sees all past Thought/Action/Observation entries
2. **Flexible execution order** — the agent decides what to do next based on observations
3. **Observation-driven** — training results directly inform the next action
4. **Full reasoning trace** — every thought is recorded and shown in the report

### Why Tools as Thin Wrappers?

Each tool wraps existing, tested infrastructure (`registry.build_model`, `hp_search.run_hp_search`, `ensemble.build_stacking_ensemble`). This means:
- No new ML logic to test
- Tool failures are handled by existing error paths
- The agent's job is purely strategic — deciding *what* to do, not *how*

### Why Scratchpad Compression?

LLM context windows are finite. With 25 max steps, the scratchpad can grow large. Compression every 8 steps keeps the prompt manageable:
- Old entries reduced to one-line summaries
- Last 4 entries kept verbatim (recent context matters most)
- No LLM call needed — pure string formatting

### Why 3 Parse Failures → Deterministic Fallback?

If the LLM consistently fails to produce valid `Thought: ... Action: ...` output, something is fundamentally wrong (model issue, prompt too long, etc.). Rather than looping forever, we bail out after 3 consecutive failures and let the deterministic path handle it. This ensures the pipeline always produces output.

---

## Verification

```bash
# Without API key (deterministic — should be identical to current behavior):
co-scientist run RNA/translation_efficiency_muscle --budget 2

# With API key (ReAct agent drives modeling):
export ANTHROPIC_API_KEY="sk-ant-..."
co-scientist run RNA/translation_efficiency_muscle --budget 2

# Check report has reasoning trace:
cat outputs/RNA__translation_efficiency_muscle/report.md | grep "Thought:"
```

Both paths should produce a valid report with a trained model. The ReAct path's report should have §4.9 with Thought/Action/Observation traces instead of the flat decision table.

---

## File Structure

```
co_scientist/
├── agents/
│   ├── __init__.py
│   ├── types.py
│   ├── base.py
│   ├── coordinator.py       ← Added run_react_modeling()
│   ├── data_analyst.py
│   ├── ml_engineer.py
│   ├── biology.py
│   ├── research.py
│   ├── tools.py             ← NEW: 11 tools + ToolResult + ToolRegistry
│   ├── react.py             ← NEW: ReactAgent + ReactState + ScratchpadEntry
│   ├── analysis.py
│   └── interactive.py
├── llm/
│   ├── client.py
│   ├── prompts.py           ← Added REACT_AGENT_SYSTEM
│   ├── parser.py
│   └── cost.py
├── report/
│   ├── generator.py         ← Added react_scratchpad parameter
│   └── template.md.jinja    ← Added §4.9 Agent Reasoning Trace
├── checkpoint.py             ← Added react_scratchpad to PipelineState
├── cli.py                    ← Added ReAct branching after Step 2
└── ...
```
