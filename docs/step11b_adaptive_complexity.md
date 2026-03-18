# Step 11b: Adaptive Complexity Scoring — Detailed Walkthrough

## Overview

Adaptive complexity scoring computes a **0-10 difficulty score** from the dataset profile and uses it to scale pipeline resource allocation. Simple datasets get a lightweight search; complex datasets get deeper exploration.

**Source:** Google AI Co-Scientist (Architecture Section 2.1): *"Test-time compute scaling — Our adaptive complexity system scales agent activation and iteration depth based on dataset difficulty (same principle, applied to ML pipelines instead of hypothesis generation)"*

The key insight from Google's paper: spending more compute on harder problems (and less on easy ones) improves overall quality. Their system uses this for hypothesis generation depth; we use it for HP search budget, iteration steps, and (in Phase C) agent activation.

---

## Complexity Score Computation

The score is a weighted combination of five factors, each rated 0-3:

### Factor 1: Sample Size (weight: 15%)

| Samples | Score | Reasoning |
|---------|-------|-----------|
| < 100 | 3.0 | Very small — careful regularization and validation needed |
| 100-500 | 2.0 | Small — limited capacity for complex models |
| 500-2000 | 1.0 | Moderate — standard approaches work |
| 2000-10000 | 0.5 | Comfortable — sample size is not a concern |
| > 10000 | 0.0 | Large — can afford complex models |

### Factor 2: Dimensionality (weight: 20%)

Measured as the ratio of features to samples:

| Feature/Sample Ratio | Score | Reasoning |
|---------------------|-------|-----------|
| > 10 | 3.0 | Extreme — features vastly outnumber samples |
| 1-10 | 2.0 | High-dimensional — regularization essential |
| 0.5-1 | 1.0 | Moderate — some overfitting risk |
| < 0.5 | 0.0 | Low — dimensionality is not a concern |

### Factor 3: Class Complexity (weight: 20%)

Only applies to classification tasks (0 for regression).

| Condition | Score contribution |
|-----------|-------------------|
| > 50 classes | +2.0 |
| 20-50 classes | +1.5 |
| 5-20 classes | +0.5 |
| Smallest class < 1% | +2.0 |
| Smallest class 1-5% | +1.0 |
| Smallest class 5-10% | +0.5 |

Capped at 3.0 total for this factor.

### Factor 4: Modality (weight: 25%)

| Modality | Score | Reasoning |
|----------|-------|-----------|
| Tabular | 0.0 | Standard ML, well-understood |
| RNA / DNA | 1.0 | Sequence features need domain knowledge |
| Protein | 1.5 | Complex structure-function relationship |
| Cell expression | 2.0 | High-dimensional, sparse, complex biology |
| Spatial | 2.5 | Spatial + expression, graph structure |
| Multimodal | 3.0 | Multiple input types need fusion |
| Unknown | 1.5 | Uncertainty adds complexity |

This is the highest-weighted factor because modality fundamentally determines what approaches are viable.

### Factor 5: Data Quality (weight: 20%)

| Condition | Score contribution |
|-----------|-------------------|
| Missing values > 10% | +1.0 |
| Missing values 1-10% | +0.5 |
| Sparsity > 90% | +0.5 |
| Per serious profiler issue | +0.5 (max 1.5) |

Capped at 2.0 total.

### Final Score

```
weighted_sum = Σ(factor_score × weight)     # range 0-3
complexity = weighted_sum × (10/3)           # scale to 0-10
```

---

## Budget Mapping (from Architecture Section 4.2)

| Level | Score | HP Trials | HP Timeout | Iteration Steps | Search (Phase C) | Agents (Phase C) |
|-------|-------|-----------|------------|-----------------|-----------------|-----------------|
| Simple | 0-2 | 10 | 90s | 4 | 0 web | Coordinator + DA + ML |
| Moderate | 3-5 | 20 | 120s | 6 | 3 web | + Research (lite) |
| Complex | 6-8 | 30 | 180s | 10 | 6 web + 3 paper | All five agents |
| Very Complex | 9-10 | 50 | 300s | 15 | 10 web + 6 paper | All five (deep) |

---

## Real-World Results

### RNA translation efficiency (1257 samples, RNA sequence)

```
Complexity: 1.3/10 (simple)  →  HP trials: 10, timeout: 90s
```

Factors: small-moderate samples (1.0), low dimensionality after k-mer features (0.0), no class complexity (0.0, regression), RNA modality (1.0), good data quality (0.0). Weighted sum = 0.40, scaled = 1.3.

**Effect:** HP search runs 10 trials in 5.8s instead of 30 trials in 27s. Finds spearman=0.68 (vs 0.70 with 30 trials — marginal difference, big time savings).

### Cell type classification (2133 samples, cell expression, 13 classes)

```
Complexity: 5.3/10 (complex)  →  HP trials: 30, timeout: 180s
```

Factors: moderate samples (1.0), extreme dimensionality ratio (3.0 — 19K features / 2K samples), moderate class count + severe imbalance (2.0), cell expression modality (2.0), moderate data quality issues (0.5). Weighted sum = 1.60, scaled = 5.3.

**Effect:** Gets the full 30 trials and 300s timeout — this dataset needs the deeper search.

---

## Dynamic Escalation (Phase C)

Architecture Section 4.2: *"If results are unexpectedly poor (model barely beats baseline), the Coordinator can escalate mid-run — activating dormant agents without restarting."*

The deterministic version stores the complexity budget on the pipeline state. In Phase C, the Coordinator agent can check if the best model's performance is close to the trivial baseline and escalate:

```python
if best_metric < trivial_metric * 1.1:  # barely beating trivial
    budget.level = escalate(budget.level)  # simple → moderate → complex
```

This is not yet implemented — it requires the agent loop (Phase C, Step 13+).

---

## Integration

The complexity score is:
1. **Computed** after profiling in Step 1
2. **Logged** as a `complexity` event in the experiment log
3. **Stored** on `PipelineState.complexity_budget` (survives checkpointing)
4. **Used** to set `n_trials` and `timeout` for HP search in Step 3b
5. **Available** for Phase C agents to read and adapt behavior

---

## File Structure

```
co_scientist/
└── complexity.py     ← scoring factors, budget mapping, display
```

The module exports:
- `compute_complexity(profile)` → `ComplexityBudget`
- `print_complexity(budget)` → rich console output
- `ComplexityBudget` dataclass with score, level, hp_trials, hp_timeout, iteration_steps, search/agent budgets
