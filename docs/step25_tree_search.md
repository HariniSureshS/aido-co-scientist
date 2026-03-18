# Step 25: Experiment Tree Search

## Overview

Replaces the linear ReAct loop with an **MCTS-inspired tree search**. The agent can branch (try multiple strategies from the same state) and backtrack (restore a previous state when a line of experimentation stalls).

**Problem:** The linear ReAct loop commits to a single exploration path. If the agent makes a bad early decision (e.g., tuning the wrong model), it wastes remaining steps trying to recover. There's no way to go back and try a fundamentally different approach.

**Solution:** A tree of experiment states with UCB1 selection and backpropagation. Each node stores a snapshot of the pipeline state. The agent can explore multiple branches from the same starting point and the system tracks the globally best result across all branches.

**Constraint:** Three-tier fallback: tree search → linear ReAct → deterministic. Must still work without an API key.

---

## Architecture

### Tree Search vs Linear ReAct

```
Linear ReAct (Step 24):
  State₀ → Step 1 → Step 2 → Step 3 → ... → Step N → Result

Tree Search (Step 25):
  State₀ ──┬── Branch A: Step 1 → Step 2 → Step 3 (score=0.72)
            │
            ├── Branch B: Step 1 → Step 2 (backtrack)
            │                        └── Branch B2: Step 1 → Step 2 (score=0.68)
            │
            └── Branch C: Step 1 → Step 2 → Step 3 (score=0.75) ← best
```

### Tree Search Loop

```
1. Create root node from initial state
2. While nodes < max_nodes (20) and cost budget OK:
   a. Select node to expand via UCB1
   b. Restore state from that node (shallow copy)
   c. Run 3-5 linear ReAct steps from that state
   d. If agent calls backtrack, or no improvement: create branch point
   e. Add child node to tree
   f. Backpropagate scores to ancestors
3. Return best result across all nodes
```

### Three-Tier Fallback

```
┌─────────────┐     fail     ┌──────────────┐     fail     ┌──────────────┐
│ Tree Search │ ──────────── │ Linear ReAct │ ──────────── │ Deterministic│
│  (--tree-   │              │  (default    │              │  (no API key)│
│   search)   │              │   LLM path)  │              │              │
└─────────────┘              └──────────────┘              └──────────────┘
```

---

## Key Data Structures

### TreeNode

```python
@dataclass
class TreeNode:
    id: int
    state_snapshot: ReactState    # shallow copy
    scratchpad: list[ScratchpadEntry]
    parent: TreeNode | None
    children: list[TreeNode]
    score: float                  # best metric at this node
    visits: int
    depth: int
    action_taken: str             # description of what led to this branch
```

### ExperimentTree

```python
class ExperimentTree:
    root: TreeNode
    nodes: list[TreeNode]
    global_best_score: float
    global_best_node: TreeNode | None
    max_depth: int = 4
    max_nodes: int = 20

    def snapshot_state(state) -> ReactState   # shallow list copy
    def restore_state(node) -> ReactState     # restore from snapshot
    def ucb1_score(node, exploration=1.41)    # UCB1 for selection
    def select_node_to_expand() -> TreeNode   # walk tree via UCB1
    def add_child(parent, state, ...) -> TreeNode
    def backpropagate(node, score)            # update ancestors
    def best_path() -> list[TreeNode]         # root → best node
    def summary() -> dict                     # for reporting
```

### State Copying Design

**Key insight:** `TrainedModel` objects are immutable after `fit()`. We only shallow-copy the `trained_models` and `results` lists — the model objects themselves are shared references. `profile`, `split`, and `eval_config` are read-only shared references.

This means branching is cheap: no model re-training, no data duplication.

---

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/agents/tree_search.py` | `TreeNode`, `ExperimentTree` with UCB1 selection, backpropagation, snapshot/restore |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/agents/react.py` | Added `run_tree_search()` method to `ReactAgent` |
| `co_scientist/agents/tools.py` | Added `BacktrackTool`, `build_tree_search_registry()` |
| `co_scientist/agents/coordinator.py` | `run_react_modeling()` accepts `tree_search: bool` param with three-tier fallback |
| `co_scientist/cli.py` | Added `--tree-search` CLI flag, passed through to coordinator |
| `co_scientist/llm/prompts.py` | Added `REACT_TREE_SEARCH_SYSTEM` prompt with backtrack guidance |
| `co_scientist/checkpoint.py` | Added `tree_search_log: dict \| None` to `PipelineState` |
| `co_scientist/report/template.md.jinja` | Added §4.10 "Tree Search Summary" section |

---

## BacktrackTool

The `backtrack` tool is only available in tree search mode (not in linear ReAct). When the agent calls it:

1. The tool returns `ToolResult(data={"backtrack": True})`
2. The tree search loop intercepts this signal
3. The current branch ends
4. The loop selects a new node to expand via UCB1

```python
class BacktrackTool(Tool):
    name = "backtrack"
    description = "Backtrack to a previous state. Use when current approach is not improving."
    parameters_schema = {
        "reason": {"type": "string", "required": True, "description": "Why you want to backtrack"},
    }
```

---

## UCB1 Selection

The tree uses **Upper Confidence Bound 1** to balance exploitation (expand high-scoring nodes) and exploration (try under-visited branches):

```
UCB1(node) = score + C * sqrt(ln(parent_visits) / node_visits)
```

Where `C = 1.41` (sqrt(2), standard exploration weight).

For lower-is-better metrics (MSE, RMSE, MAE), the score is negated so that UCB1 prefers lower values.

---

## System Prompt

`REACT_TREE_SEARCH_SYSTEM` extends the regular ReAct prompt with:

1. **Backtrack tool description** — when and why to use it
2. **Branching guidance** — "Branch when stuck", "Explore diverse strategies across branches (svm, knn, ft_transformer)"
3. **Strategy for tree search** — don't repeat approaches across branches
4. **Custom model design** — "In a dedicated branch, use `design_model` to create a novel architecture"

---

## Pipeline Integration

### CLI Flag

```bash
co-scientist run RNA/translation_efficiency_muscle --tree-search --budget 5
```

### Coordinator

```python
def run_react_modeling(self, ..., tree_search: bool = False):
    if tree_search:
        registry = build_tree_search_registry()  # includes backtrack
        result = agent.run_tree_search(state, exp_log)
        if result is None:
            # Fall back to linear ReAct
            result = agent.run(state_linear, exp_log)
    else:
        registry = build_default_registry()
        result = agent.run(state, exp_log)
```

---

## Report

### §4.10 Tree Search Summary

When tree search is used, the report includes:

```markdown
### 4.10 Tree Search Summary

The experiment tree search explored **8** node(s) across **3** branch(es).

| Property | Value |
|----------|-------|
| Total nodes | 8 |
| Max depth reached | 3 |
| Best path length | 4 |
| Global best score | 0.7103 |

**Best path:** root → branch_1 → branch_3 → branch_5
```

---

## Verification

```bash
# Tree search mode:
co-scientist run RNA/translation_efficiency_muscle --tree-search --budget 5

# Check tree search log in report:
grep -A 10 "Tree Search Summary" outputs/RNA__translation_efficiency_muscle/report.md

# Without --tree-search (linear ReAct — unchanged):
co-scientist run RNA/translation_efficiency_muscle --budget 5
```

---

## Design Decisions

### Why MCTS-inspired, not full MCTS?

Full MCTS requires many rollouts per node. In our setting, each "rollout" involves training models (expensive). We use MCTS *concepts* (tree structure, UCB1, backpropagation) but run only a few steps per branch rather than full rollouts.

### Why shallow copy, not deep copy?

TrainedModel objects hold fitted sklearn/XGBoost/LightGBM models. Deep copying these would be expensive and fragile (some models don't pickle well mid-computation). Since models are immutable after `fit()`, sharing references is safe.

### Why max_depth=4, max_nodes=20?

These are tuned for the LLM cost budget. Each branch costs 3-5 LLM calls. With a $5 budget and ~$0.01 per call, 20 nodes × 4 calls ≈ $0.80 for tree search alone. This leaves plenty of budget for other agent calls.

---

## Guardrails

Tree search inherits all ReactAgent guardrails (see Step 24), which apply to each branch:

| Guardrail | Value | Description |
|-----------|-------|-------------|
| Wall-clock timeout | min(900s, 60% remaining) | Dynamic budget from PipelineDeadline, checked at top of each step across all branches. Hard stop if exceeded. |
| Tool execution timeout | 300s (5 min) | Per-tool timeout. Heartbeat messages print every 30s. |
| LLM request timeout | 60s (1 min) | Per-request limit via Anthropic SDK. Retried up to 3 times with exponential backoff (2s/4s/8s). |
| Repeated actions (hard stop) | 4 identical calls | Force-stops the current branch. |
| Consecutive tool failures | 5 in a row | Exits the current branch. |
| Max nodes | 20 | Limits total tree expansion. |
| Max depth | 4 | Limits branch depth. |
| Cost budget | `can_afford()` | Checked before each LLM call. |

The wall-clock timeout is shared across all branches — if tree search takes 10 minutes total, it stops regardless of how many branches remain unexplored.
