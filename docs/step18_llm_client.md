# Step 18: LLM Client (Claude API) — Detailed Walkthrough

## Overview

The LLM client layer provides a thin, cost-aware wrapper around the Anthropic Claude API. It handles API communication, cost tracking, budget enforcement, and response parsing — so agent code only needs to send prompts and receive structured decisions.

This implements Architecture Section 4.3 (agent implementation) and supports the graceful degradation principle: if the API is unavailable, the pipeline continues in deterministic mode.

---

## Components

### 1. ClaudeClient (`llm/client.py`)

The main API wrapper. Two methods:

| Method | Returns | Use Case |
|--------|---------|----------|
| `ask()` | `dict` (parsed JSON) or `None` | Agent decisions — structured output |
| `ask_text()` | `str` or `None` | Report sections, interpretations — free text |

```python
client = ClaudeClient(cost_tracker=tracker)

# Structured decision
response = client.ask(
    system_prompt="You are the ML Engineer...",
    user_message="Dataset: RNA, 1257 samples, spearman=0.69...",
    agent_name="ml_engineer",
)
# response = {"action": "hp_tune", "parameters": {...}, "confidence": 0.8}

# Free text
text = client.ask_text(
    system_prompt="You are a biology expert...",
    user_message="Interpret these RNA translation results...",
    agent_name="biology_specialist",
)
# text = "The model captures codon optimization patterns..."
```

#### Safety Features

- **Lazy initialization**: Anthropic client only created on first use
- **Missing API key**: Returns `None` gracefully, never crashes
- **Missing package**: If `anthropic` not installed, logs warning and returns `None`
- **Budget check**: Refuses calls when budget exhausted
- **Error handling**: All API errors caught, logged, return `None`
- **Parse failure**: Returns `{"raw_text": "...", "_parse_failed": True}` so callers can decide

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `claude-sonnet-4-20250514` | Claude model to use |
| `max_tokens` | 1024 (ask), 2048 (ask_text) | Max response tokens |
| `temperature` | 0.2 (ask), 0.3 (ask_text) | Lower for decisions, slightly higher for text |

Sonnet is the default for cost-efficiency. The agents don't need Opus-level reasoning for structured decisions.

---

### 2. CostTracker (`llm/cost.py`)

Tracks cumulative LLM costs and enforces the `--max-cost` budget from the CLI.

```python
tracker = CostTracker(max_cost=5.0)

# After each API call (done automatically by ClaudeClient):
tracker.record(
    model="claude-sonnet-4-20250514",
    agent="ml_engineer",
    input_tokens=1500,
    output_tokens=300,
    latency_seconds=1.2,
)

print(tracker.total_cost)         # $0.0090
print(tracker.budget_remaining)   # $4.9910
print(tracker.can_afford())       # True
print(tracker.per_agent_costs())  # {"ml_engineer": 0.009}
```

#### Pricing

Pricing per 1M tokens (updated for current Claude models):

| Model | Input | Output |
|-------|-------|--------|
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $0.80 | $4.00 |

For a typical pipeline run with ~10 agent calls:
- ~15K input tokens + ~3K output tokens per call
- Estimated cost: ~$0.50-$1.00 total
- Well within the default $5.00 budget

#### Properties

| Property | Description |
|----------|-------------|
| `total_cost` | Sum of all call costs |
| `budget_remaining` | `max_cost - total_cost` |
| `num_calls` | Total API calls made |
| `total_input_tokens` | Sum of input tokens |
| `total_output_tokens` | Sum of output tokens |
| `can_afford(est)` | Whether budget allows another call |
| `per_agent_costs()` | Cost breakdown by agent name |
| `summary()` | Dict for logging |

---

### 3. JSON Parser (`llm/parser.py`)

LLMs don't always return clean JSON. The parser handles common patterns:

```python
from co_scientist.llm.parser import extract_json, parse_decision

# Pure JSON
extract_json('{"action": "stop"}')
# → {"action": "stop"}

# JSON in code block
extract_json('Here is my answer:\n```json\n{"action": "stop"}\n```')
# → {"action": "stop"}

# JSON embedded in text
extract_json('I recommend {"action": "hp_tune"} because...')
# → {"action": "hp_tune"}

# Unparseable → safe fallback
parse_decision('I think we should tune the model')
# → {"action": "no_op", "confidence": 0.0, "reasoning": "Failed to parse..."}
```

#### Functions

| Function | Input | Output | Use |
|----------|-------|--------|-----|
| `extract_json(text)` | Raw LLM text | `dict` or `None` | General JSON extraction |
| `parse_decision(text)` | Raw LLM text | Decision-compatible `dict` | Always returns valid dict |
| `parse_list(text)` | Raw LLM text | `list[str]` | Extract lists from various formats |

#### Parse Strategy (in order)

1. Direct `json.loads()` on stripped text
2. Extract from `` ```json ... ``` `` code block
3. Find first `{` to last `}` and parse
4. Give up → return `None` (or fallback for `parse_decision`)

---

### 4. System Prompts (`llm/prompts.py`)

Each agent has a tailored system prompt that defines:
- **Identity**: "You are the ML Engineer agent..."
- **Capabilities**: What decisions this agent can make
- **Output format**: Expected JSON structure with examples
- **Rules**: Constraints and best practices

```
COORDINATOR_SYSTEM   → Orchestration, budget management, stopping criteria
DATA_ANALYST_SYSTEM  → Preprocessing, features, data quality
ML_ENGINEER_SYSTEM   → Model selection, HP tuning, iteration strategy
BIOLOGY_SPECIALIST_SYSTEM → Biological context, plausibility, interpretation
```

Each prompt explicitly shows the expected JSON format so the LLM produces parseable output consistently.

---

## Graceful Degradation Chain

```
1. ANTHROPIC_API_KEY set + budget remaining
   → ClaudeClient.ask() → Claude API → JSON response → Decision

2. API key set but budget exhausted
   → ClaudeClient.ask() returns None
   → BaseAgent.decide() falls through to decide_deterministic()

3. API key not set
   → ClaudeClient.available = False
   → BaseAgent.decide() skips LLM, calls decide_deterministic()

4. anthropic package not installed
   → ClaudeClient._get_client() returns None
   → Same as #3

5. API call fails (network, rate limit, etc.)
   → Caught, logged, returns None
   → Falls through to decide_deterministic()

6. Response can't be parsed as JSON
   → Returns {"_parse_failed": True}
   → BaseAgent.decide() falls through to decide_deterministic()
```

At every level, the pipeline continues. No agent failure ever crashes the pipeline.

---

## Cost Logging

At pipeline completion, the cost summary is logged:

```json
{
  "event": "llm_costs",
  "data": {
    "total_cost": 0.4523,
    "budget_remaining": 4.5477,
    "num_calls": 8,
    "total_input_tokens": 12400,
    "total_output_tokens": 2800
  }
}
```

In deterministic mode (no API key), the summary shows zero calls — confirming no LLM costs.

---

## Environment Setup

To enable LLM-powered agent decisions:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
co-scientist run RNA/translation_efficiency_muscle --max-cost 5.0
```

Without the key, the pipeline runs identically to before — all decisions are rule-based.

---

## File Structure

```
co_scientist/
├── llm/
│   ├── __init__.py     ← exports ClaudeClient, CostTracker
│   ├── client.py       ← ClaudeClient: API wrapper with cost tracking
│   ├── cost.py         ← CostTracker: budget enforcement, per-agent costs
│   ├── parser.py       ← extract_json, parse_decision, parse_list
│   └── prompts.py      ← System prompts for all agent roles
├── pyproject.toml      ← added anthropic>=0.39.0 dependency
```

---

## Design Decisions

### Why Sonnet, Not Opus?

Agent decisions are structured choices (which model, which preprocessing), not open-ended reasoning. Sonnet is 5x cheaper and fast enough. The temperature is kept low (0.2) to produce consistent, deterministic-like decisions.

### Why JSON, Not Tool Use?

Claude's tool use API could enforce structured output, but JSON in the response body is:
- Simpler to implement and debug
- Easier to log and review
- More portable (works with any LLM provider)
- Sufficient for our use case (small, well-defined schemas)

### Why Lazy Client Initialization?

The `anthropic` package is only imported when the first LLM call is made. This means:
- Pipeline startup is fast (no SDK init)
- Deterministic mode doesn't load unnecessary dependencies
- The `anthropic` dependency is effectively optional at runtime
