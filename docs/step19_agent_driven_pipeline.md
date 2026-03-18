# Step 19: Agent-Driven Pipeline + Interactive Mode — Detailed Walkthrough

## Overview

Step 19 transitions agents from passive observers to active decision-makers. In Steps 17-18, agents were consulted and their decisions logged, but the pipeline always followed its hardcoded path. Now, agent decisions **actually drive** pipeline behavior: model priority, HP search execution, and report content.

This also implements the two required CLI modes from the homework spec:
- **Auto mode** (`--mode auto`): agents make all decisions autonomously
- **Interactive mode** (`--mode interactive`): conversational interface at every decision point — user can approve, override, ask questions in natural language, give instructions, or exit. The LLM uses full pipeline context (splits, class distribution, scores) to answer. Typing "exit"/"stop"/"quit" halts the pipeline immediately. **Crucially, interactive mode now also pauses after every ReAct agent step** — the user can provide feedback that gets injected into the agent's next reasoning cycle, redirect strategy mid-loop, or stop the modeling phase at any point.

---

## What Changed

### Before (Steps 17-18)
```
Agent decision → logged to experiment_log.jsonl
Pipeline         → always does the same thing regardless of agent advice
```

### After (Step 19)
```
Agent decision → presented to user (interactive) or applied directly (auto)
Pipeline         → adapts behavior based on agent recommendation
```

---

## Agent Decision Points

The pipeline now consults agents at **5 decision points**:

| # | Stage | Agent | What It Controls |
|---|-------|-------|-----------------|
| 1 | Preprocessing | Data Analyst | Preprocessing strategy advice |
| 2 | Model Selection | ML Engineer | Model training priority order |
| 3 | Post-Training | ML Engineer + Data Analyst + Biology | Analysis and next-step recommendation |
| 4 | HP Search | ML Engineer | Whether to tune, how many trials |
| 5 | Report | Biology Specialist | Biological interpretation section |
| 6 | **ReAct Loop** | **Data Analyst + Biology Specialist** | **On-demand consultation via `diagnose_data` and `consult_biology` tools** |

**Note:** The Data Analyst and Biology Specialist now participate not only at pre- and post-training decision points (stages 1, 3, 5) but also **during the ReAct loop itself**. The ReAct agent can invoke `diagnose_data` to consult the Data Analyst for data quality diagnosis or feature recommendations, and `consult_biology` to consult the Biology Specialist for biological validation, at any point during the modeling loop. This enables continuous specialist feedback rather than one-shot consultations.

**Foundation model awareness (GPU):** When a GPU is detected, the ML Engineer's model selection includes foundation models (`embed_xgboost`, `embed_mlp`, `concat_xgboost`, `concat_mlp`, `aido_finetune`). The `SummarizeDataTool` reports embedding availability at runtime, allowing the ReAct agent to decide whether to use foundation models based on actual data characteristics. All agent prompts have been updated to mention foundation models as options when AIDO embeddings are available.

### Decision Detail: Selection Reasons and Explanations

Agent decisions now include structured detail beyond just the action and confidence:

- **`selection_reasons`**: A list of per-model rationale strings explaining why each model was chosen or excluded. The `_print_decision` function in `interactive.py` renders these as **"Model-by-model rationale:"** bullet points in the terminal, giving users visibility into the agent's reasoning for every model.
- **Preprocessing explanations**: Preprocessing decisions include human-readable explanations for parameter choices (e.g., *"Using k=4 for RNA because 4-mers capture codon context"*), making it clear why specific feature engineering settings were selected.
- **HP tuning explanations**: HP tuning decisions explain why tuning is or is not worthwhile for the current situation (e.g., *"Tuning recommended: best model at 0.68 with gap to next — likely room for improvement"* or *"Skipping HP search: model already at 0.97, near ceiling"*).

### 1. Preprocessing (Data Analyst)

The Data Analyst recommends a preprocessing strategy based on modality:

```
╭──────────────── Data Analyst — preprocessing ────────────────╮
│ Action: set_preprocessing                                     │
│   steps: ['kmer_frequency_k4', 'standard_scale', 'recommend_cnn'] │
│   kmer_k: 4                                                  │
│ Reasoning: Sequence modality (rna): k-mer features with k=4  │
│ Confidence: 90%                                               │
╰──────────────────────────────────────────────────────────────╯
```

Currently advisory (preprocessing follows defaults.yaml). Future steps can wire this to actually change preprocessing parameters.

### 2. Model Selection (ML Engineer)

The ML Engineer recommends models and their priority order. The pipeline **reorders baseline configs** based on this recommendation:

```python
# Agent recommends: bio_cnn first (for RNA data)
recommended = ["bio_cnn", "xgboost", "lightgbm", "random_forest", ...]

# Pipeline reorders: recommended models trained first
baseline_configs = sorted(
    baseline_configs,
    key=lambda c: priority_map.get(c.model_type, fallback_idx),
)
```

This matters because:
- Prioritized models appear first in comparison tables
- If pipeline is interrupted, the most important models are already trained
- In future iteration loops, priority affects resource allocation

### 3. Post-Training Analysis (All Active Agents)

After baselines are evaluated, all active agents analyze results:

```
Agent Analysis:
  Ml Engineer: hp_tune — Early iteration — tune random_forest for better generalization (70%)
  Data Analyst: data_quality_assessment — Data quality looks good (80%)
```

In interactive mode, users can ask questions about the results before choosing an action. Then they pick: `continue`, `tune`, `stop`, or `skip`.

### 4. HP Search (ML Engineer)

The ML Engineer decides whether to run HP search and with how many trials. **This decision is binding** — if the agent says `skip_hp_search`, the pipeline skips it:

```python
hp_decision = agent_hp_decision(coordinator, hp_ctx)
skip_hp = hp_decision.action == "skip_hp_search"

if skip_hp:
    console.print("HP search skipped: {reason}")
else:
    n_trials = hp_decision.parameters.get("n_trials", default)
    run_hp_search(n_trials_override=n_trials, ...)
```

The agent also controls `n_trials` — adapting search intensity to dataset characteristics.

### 5. Report (Biology Specialist)

The Biology Specialist provides a biological interpretation that appears in the report:

```markdown
### 4.4 Biological Interpretation

RNA sequence features can capture codon usage bias, UTR regulatory
elements, and sequence composition effects on gene expression.
Translation efficiency is influenced by codon optimality, mRNA
structure, and UTR elements.
```

With an API key, this becomes an LLM-generated interpretation specific to the results. Without it, the deterministic fallback provides modality-specific context.

---

## Interactive Mode

### How It Works

```bash
co-scientist run RNA/translation_efficiency_muscle --mode interactive
```

At each decision point, the pipeline:
1. Displays the agent's recommendation in a styled panel
2. Lets the user **accept**, **override**, or **ask questions / give instructions**
3. Applies the user's choice (possibly revised by conversation)

### Decision Presentation

```
╭──────────────── Ml Engineer — model_selection ──────────────╮
│ Action: select_models                                        │
│   models: ['xgboost', 'lightgbm', 'random_forest', ...]    │
│   priority: bio_cnn                                          │
│ Reasoning: Standard model set for rna/regression             │
│ Confidence: 90%                                              │
╰─────────────────────────────────────────────────────────────╯

  Options: y=accept, n=override, or type a question/instruction
  Accept Ml Engineer's recommendation?
```

### Conversational Chat

At every decision point, users can type **free-form questions or instructions** instead of just y/n:

```
  Accept Ml Engineer's recommendation? why not try a neural network?

  ╭─ Co-Scientist ───────────────────────────────────────╮
  │ With only 1257 samples, neural networks are likely to │
  │ overfit. Tree models are more sample-efficient for    │
  │ this dataset size.                                    │
  ╰───────────────────────────────────────────────────────╯

  Accept Ml Engineer's recommendation? use random forest as priority

  Revised recommendation based on your input:
  ╭─ Ml Engineer — model_selection ──────────────────────╮
  │ Action: select_models                                 │
  │   models: ['random_forest', 'xgboost', 'lightgbm']   │
  │   priority: random_forest                             │
  ╰───────────────────────────────────────────────────────╯

  Accept Ml Engineer's recommendation? y
```

**How it works:**
- Questions (e.g., "what does spearman measure?") are answered by the LLM with full pipeline context, then the user is re-prompted
- Instructions (e.g., "use random forest", "skip linear models") automatically trigger a **revised recommendation** from the LLM
- The chat loops — users can ask multiple questions before accepting
- If LLM is unavailable, falls back to y/n only

### User Override

If the user types `n`, they get stage-specific override prompts:

```
  Models to train (comma-separated, or 'all'): xgboost,random_forest
```

For HP search, they can simply say yes/no:

```
  Run HP search? [Y/n]: n
```

### Confirmation Points

Interactive mode also confirms before proceeding with major steps. Users can ask questions here too:

```
  Ask a question, or press Enter to continue: what modality was detected?

  ╭─ Co-Scientist ──────────────────────────────────────╮
  │ The dataset was detected as RNA modality with a      │
  │ regression task (translation efficiency prediction). │
  ╰──────────────────────────────────────────────────────╯

  Ask a question, or press Enter to continue:
  Proceed with data profiling? (Detected rna / regression) [Y/n]:
```

---

## Report Enhancements

### Biological Interpretation Section (4.4)

Added to the report template when `biological_interpretation` is provided:

```markdown
### 4.4 Biological Interpretation

{agent-provided biological context specific to dataset and results}
```

### Agent Decision Log (4.5)

A table of all agent decisions made during the run:

```markdown
### 4.5 Agent Decision Log

| Agent | Decision | Confidence |
|-------|----------|-----------|
| Data Analyst | set_preprocessing | 90% |
| Ml Engineer | select_models | 90% |
| Ml Engineer | hp_tune | 70% |
| Data Analyst | data_quality_assessment | 80% |
| Ml Engineer | hp_tune | 80% |
```

### Footer

The report footer now reflects whether agents were used:
- With agents: *"Report generated by AIDO Co-Scientist with multi-agent decision support."*
- Without: *"Report generated automatically by AIDO Co-Scientist in deterministic mode."*

---

## Centralized Context Building

The new `agents/analysis.py` module provides `build_pipeline_context()` — a single function that builds a `PipelineContext` from the current pipeline state:

```python
ctx = build_pipeline_context(
    config=config,
    profile=state.profile,
    eval_config=state.eval_config,
    results=state.results,
    best_result=state.best_result,
    stage="hp_search",
    complexity_budget=state.complexity_budget,
    cost_remaining=cost_tracker.budget_remaining,
)
```

This replaces the manual `PipelineContext(...)` construction that was scattered across `cli.py`, ensuring consistency across all decision points.

---

## Results

### RNA/translation_efficiency_muscle

With agent-driven HP search (20 trials instead of 10):

| Model | Spearman |
|-------|----------|
| random_forest_tuned | **0.6985** |
| random_forest | 0.6941 |
| stacking_ensemble | 0.6941 |
| bio_cnn | 0.6655 |

The ML Engineer's recommendation of 20 trials led to a tuned model that **improved** over the baseline (0.6941 → 0.6985), whereas the previous 10-trial search did not find an improvement.

---

## File Structure

```
co_scientist/
├── agents/
│   ├── interactive.py    ← NEW: present_decision, confirm_step, user override, conversational chat
│   ├── analysis.py       ← NEW: build_pipeline_context, agent_hp_decision,
│   │                        agent_post_training_analysis, agent_biology_interpretation
│   ├── coordinator.py    ← Updated: complexity-based agent activation
│   ├── data_analyst.py   ← Added: assess_data_quality() method
│   └── ml_engineer.py    ← Added: diagnose_failure() method
├── report/
│   ├── generator.py      ← Added: biological_interpretation, agent_reasoning params
│   └── template.md.jinja ← Added: §4.4 Bio Interpretation, §4.5 Agent Decision Log
├── cli.py                ← Major: agent decisions drive model order, HP search, report
```

---

## Design Decisions

### Why Reorder, Not Filter?

The ML Engineer's model recommendation **reorders** but never **removes** models from the training list. This prevents agent errors from accidentally dropping useful models. The guardrail system handles actual filtering (e.g., blocking BioCNN on non-sequence data).

### Why Is HP Search Skip Binding?

Unlike model selection (advisory reorder), the HP search decision is binding because:
- HP search is expensive (tens of seconds)
- A near-perfect classifier (>0.95) genuinely doesn't benefit from tuning
- The agent has full context (scores, metric type) to make this call
- Interactive mode lets the user override if they disagree

### Why Not Wire Preprocessing Agent Decisions Yet?

The preprocessing pipeline is modality-specific and tightly coupled to defaults.yaml. Wiring agent decisions into actual preprocessing parameters requires careful validation (wrong params could crash preprocessing). Step 19 keeps preprocessing advice **advisory** — the agent's recommendation is logged and displayed but preprocessing follows the proven defaults. This is a deliberate conservative choice; future steps can incrementally wire in agent-driven preprocessing.
