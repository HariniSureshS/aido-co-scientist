# Step 12: Rich Dashboard — Detailed Walkthrough

## Overview

The Rich dashboard transforms the CLI output from plain text into a styled terminal interface with panels, rules, tables, and color-coded severity indicators. This implements Architecture Section 11.2.

**Phase B (current):** Static styled output — panels, step progress dividers, enhanced model table, summary panel.
**Phase C upgrade:** Rich Live display for real-time updates during the agent iteration loop (Step 4 in the pipeline will show models being trained/evaluated live).

---

## Dashboard Components

### 1. Header Panel

Displays at pipeline start as a bordered panel:

```
╭──────────────────────────── Co-Scientist v0.1.0 ─────────────────────────────╮
│  Dataset:    RNA/translation_efficiency_muscle                               │
│  Mode:       auto                                                            │
│  Budget:     10 steps                                                        │
│  Max cost:   $5.00                                                           │
│  Output:     outputs/RNA__translation_efficiency_muscle                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### 2. Step Progress Rules

Each pipeline step gets a styled horizontal rule with step number and label:

```
─────────────────────────── Step 1/6: Load & Profile ───────────────────────────
```

Resumed steps show as dimmed single-line entries:
```
  Step 1/6: Load & Profile (resumed)
```

### 3. Model Comparison Table

Enhanced table with blue borders and a status column for highlighting the best model:

```
                                Model Comparison
┏━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃    ┃ Model          ┃ Tier     ┃ spearman ┃ pearson ┃    mse ┃   rmse ┃ Time ┃
┡━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│  * │ xgboost_tuned  │ tuned    │   0.7044 │  0.7234 │ 0.8923 │ 0.9446 │ 0.4s │
│    │ xgboost_defau… │ standard │   0.6279 │  0.6816 │ 1.1035 │ 1.0505 │ 0.3s │
│    │ mlp            │ advanced │   0.6229 │  0.6800 │ 1.1534 │ 1.0740 │ 1.3s │
└────┴────────────────┴──────────┴──────────┴─────────┴────────┴────────┴──────┘
```

The `*` marker and green highlighting indicates the best model.

### 4. Guardrail Alerts

Color-coded severity indicators throughout the pipeline:
- `✓ Pre-training checks: all checks passed` (green)
- `WARNING mlp: estimated 117,505 parameters > 1,000 training samples` (yellow)
- `ERROR constant_target: Training target has zero variance` (bold red)

### 5. Final Summary Panel

A bordered panel at pipeline end with key results:

```
╭───────────────────────────── Pipeline Complete ──────────────────────────────╮
│  Best model:  xgboost_tuned                                                  │
│  Metric:      spearman = 0.6807                                              │
│  Models tried: 5                                                             │
│  Total time:  14.1s                                                          │
│  Output:      outputs/RNA__translation_efficiency_muscle                     │
│  Warnings:    3 (see experiment log)                                         │
╰───────────────────── RNA/translation_efficiency_muscle ──────────────────────╯
```

---

## Pipeline Step Mapping

The dashboard maps internal step keys to human-readable labels:

| Step Key | Label | Number |
|----------|-------|--------|
| load_profile | Load & Profile | 1/6 |
| preprocess_split | Preprocess & Split | 2/6 |
| baselines | Train Baselines | 3/6 |
| hp_search | HP Search | 4/6 |
| export | Export Model | 5/6 |
| report | Generate Report | 6/6 |

---

## Complexity Color Coding

The complexity level is color-coded in the header and throughout:

| Level | Color | Meaning |
|-------|-------|---------|
| simple | Green | Lightweight pipeline, minimal resources needed |
| moderate | Yellow | Standard pipeline |
| complex | Red | Deep exploration needed |
| very_complex | Bold red | Maximum resource allocation |

---

## Resume Display

When resuming from a checkpoint, the dashboard shows completed steps as dimmed one-liners and only active steps get the full rule header:

```
  Step 1/6: Load & Profile (resumed)
  Step 2/6: Preprocess & Split (resumed)
  Step 3/6: Train Baselines (resumed)

───────────────────────────── Step 4/6: HP Search ──────────────────────────────
  Optuna HP search: 10 trials...
```

---

## Phase C Upgrade Path

In Phase C (agent iteration loop), the dashboard will upgrade to use `rich.live.Live` for real-time updates:

```python
with Live(layout, refresh_per_second=4) as live:
    while not done:
        # Agent proposes model
        layout["model_table"].update(render_model_table(results))
        layout["status"].update(f"Agent: '{current_action}'")
        layout["header"].update(render_header(step, budget_used))
```

This enables:
- Live model table that updates as each model trains
- Real-time agent status messages
- Progress bar for HP search trials
- Cost tracking that updates with each LLM call

### Active Agent Tracking

The live dashboard now shows which agent is active during each ReAct step via a new **"Agent:"** line in the agent reasoning panel. The `set_agent_name()` method sets the active agent name, and actions are mapped to agents as follows:

| Action | Agent |
|--------|-------|
| `train` / `tune` / `design_model` | ML Engineer |
| `analyze_features` / `diagnose_data` | Data Analyst |
| `evaluate_test_set` | Evaluator |
| `consult_biology` | Biology Specialist |

This makes it easy to see at a glance which specialist is driving each step of the ReAct loop.

### Agent Conversations Panel

The live dashboard includes an **Agent Conversations** panel that shows the last 8 agent messages in real time. `LiveDashboard` maintains an `_agent_log` list and provides `add_agent_message(agent, stage, message, msg_type)` to record interactions.

Each entry displays:
- **Agent name** (color-coded: Biology Specialist=green, Data Analyst=cyan, ML Engineer=magenta, Research Agent=blue, Coordinator=yellow)
- **Stage** (e.g., consult, debate, react)
- **Elapsed time** since pipeline start
- **Message** content (truncated to fit the panel)

Icons indicate message type: `💬` consult, `⚔️` debate, `🔧` react_tool.

```
─── Agent Conversations ───────────────────────────────────────────────
💬 Biology Specialist  │ consult   │ +2m 14s │ RNA k-mer features recommended...
⚔️ ML Engineer         │ debate    │ +3m 01s │ XGBoost preferred over MLP for...
🔧 Data Analyst        │ react     │ +4m 33s │ Class imbalance detected, SMOTE...
```

Messages are pushed automatically:
- **Coordinator** pushes all `consult()` and `debate()` results
- **ReAct tools** (`consult_biology`, `diagnose_data`) push their results
- **ReAct loop** logs each action with the responsible agent name

### Tournament Rankings (Elo) Panel

The dashboard includes a **Tournament Rankings** panel with a bright magenta border. It displays the top 8 models ranked by Elo rating in a table with four columns:

| Column | Description |
|--------|-------------|
| # | Rank (1–8) |
| Model | Model name |
| Elo | Current Elo rating |
| W/M | Wins / total matches |

Updated via `set_elo_rankings(rankings)`, which is called after each pairwise Elo update in both the linear ReAct loop and tree search ReAct loop. The rankings list contains dicts with keys `name`, `elo`, `wins`, `matches`.

### Tree Search Panel

When `--tree-search` is active, a bright-yellow-bordered **Tree Search** panel shows live MCTS state:

- **Mode** — search algorithm (MCTS)
- **Total nodes** — number of nodes in the search tree
- **Current branch** — branch ID, depth, and score of the active branch

Updated via `set_tree_search(active, branch_id, depth, score, total_nodes)`, called on each branching event. Tree search steps also update the dashboard's thought, action, and step indicators. Model results from tree search branches appear in the model leaderboard alongside linear results.

### Thought Display

Agent thoughts are displayed truncated to **500 characters** in the dashboard to keep the output readable. The full thought text is preserved in the scratchpad and report.

The current static dashboard components (`print_model_table`, `print_header`, etc.) will be reused as render functions inside the Live context.

---

## File Structure

```
co_scientist/
└── dashboard.py     ← header, step progress, model table, final summary
```

The module exports:
- `print_header(version, dataset_path, mode, budget, ...)` — styled header panel
- `print_step_start(step_key, step_num, total)` — step rule divider
- `print_step_resumed(step_key, step_num, total)` — dimmed resumed indicator
- `print_model_table(results, eval_config, best_name)` — enhanced model comparison
- `print_final_summary(...)` — bordered results panel
