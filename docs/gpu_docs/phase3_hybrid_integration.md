# Phase 3: Hybrid Features + Full Pipeline Integration

## Idea

Phase 1 and 2 added the foundation models themselves. Phase 3 addresses two remaining gaps:

1. **Hybrid concat models** — combine handcrafted features with AIDO embeddings for the best of both worlds
2. **Full pipeline integration** — make every component of the pipeline aware of foundation models: agents, HP search, validation, export, reporting

Without Phase 3, foundation models exist as isolated experiments. With Phase 3, they are first-class citizens in the agent-driven pipeline.

## Part A: Concat Models (concat_xgboost, concat_mlp)

### Why Concatenation Works

Consider an RNA translation efficiency prediction task:
- **k-mer features** (handcrafted): Capture codon usage frequency, GC content — directly relevant statistics
- **AIDO embeddings** (learned): Capture UTR structure, long-range codon context, evolutionary patterns

These are **complementary signals**:
- k-mer features know that "this sequence has high GC3 content" (a known translation efficiency correlate)
- AIDO embeddings know that "this specific UTR configuration resembles highly translated mRNAs in the training corpus"

Neither alone captures the full picture. By concatenating them:
```python
X_combined = np.hstack([X_handcrafted,  # e.g., 320 features
                         X_embed])        # e.g., 768 features
# → 1088 total features for XGBoost/MLP
```

XGBoost can learn that "codon bias features are most predictive for some samples, while embedding dimensions 200-300 are most predictive for others." This adaptive feature selection is why concat models often outperform either feature set alone by 10-20%.

### Implementation

Concat models reuse existing model builders (XGBoost, MLP) — the only difference is the input features. The trainer concatenates:

```python
# In trainer.py
if config.model_type in _CONCAT_MODELS:
    X_combined = np.hstack([split.X_train, split.X_embed_train])
    model.fit(X_combined, y_train)
```

The evaluator does the same for validation/test data.

## Part B: Full Pipeline Integration

### Agent Awareness

**Problem**: Without integration, the LLM-based agents don't know foundation models exist. The ReAct agent would never say "let me try embed_xgboost" because it's not in its vocabulary.

**Solution**: Updated across all agent touchpoints:

| Component | Change |
|-----------|--------|
| `llm/prompts.py` — ML_ENGINEER_SYSTEM | Added rule: "When GPU available, recommend embed_*, concat_*, aido_finetune" |
| `llm/prompts.py` — REACT_AGENT_SYSTEM | Added strategy guideline 3b about foundation models |
| `llm/prompts.py` — REACT_TREE_SEARCH | Added foundation models to branch suggestions |
| `agents/tools.py` — TrainModelTool | Added all 5 foundation models to description and routing |
| `agents/tools.py` — AnalyzeErrorsTool | Routes embedding models to X_embed_val |
| `agents/tools.py` — SummarizeDataTool | Reports embedding availability/dimensions at runtime |
| `agents/ml_engineer.py` — _select_models | Adds foundation models when `gpu_available()` |
| `agents/ml_engineer.py` — _suggest_untrained_models | Includes foundation models in suggestions |

The key design choice: **the static prompts mention foundation models as an option, and the runtime `summarize_data()` tool tells the agent whether they're actually available.** This way the agent learns about GPUs from the data, not from hardcoded assumptions.

### HP Search Integration

Foundation models now have full HP search support:

```yaml
# In defaults.yaml
embed_xgboost:
  n_estimators: { type: int, low: 50, high: 500 }
  max_depth:    { type: int, low: 3, high: 10 }
  learning_rate: { type: float, low: 0.01, high: 0.3, log: true }
  ...

concat_xgboost:  # same space as embed_xgboost
  ...

embed_mlp:
  dropout: { type: float, low: 0.1, high: 0.5 }
  learning_rate: { type: float, low: 0.0001, high: 0.01, log: true }
  ...
```

The HP search module (`hp_search.py`) routes data correctly:
- Embedding models → `split.X_embed_train`
- Concat models → `np.hstack([split.X_train, split.X_embed_train])`
- Sequence models → `sequences=split.seqs_train`

### Validation Integration

The validation agent (`validation.py`) now tests foundation models with the correct data:
```python
if trained.needs_embeddings:
    X_check = split.X_embed_val[:n_check]
elif trained.needs_sequences:
    X_check = split.X_val[:n_check]
    seqs_check = split.seqs_val[:n_check]
    y_pred = trained.model.predict(X_check, sequences=seqs_check)
else:
    X_check = split.X_val[:n_check]
```

Without this fix, the validator would feed wrong-shaped features to embedding models and flag them as broken.

### Stacking Ensemble

Foundation models are **excluded** from the stacking ensemble's out-of-fold cross-validation:
```python
_SKIP_TYPES = {"stacking", "custom", "embed_xgboost", "embed_mlp",
               "aido_finetune", "concat_xgboost", "concat_mlp"}
```

Reason: OOF CV re-trains each base model on each fold. For embedding models, this would require re-extracting AIDO embeddings per fold (expensive). For fine-tuning, each fold would take 10+ minutes. The cost-benefit isn't worth it — the standard models provide enough diversity for the stacking ensemble.

Foundation models still compete as individual models. If `concat_xgboost` scores highest, it wins — no ensemble needed.

### Report Generation

The report generator now includes explanations for all foundation model types:

```python
model_explanations = {
    ...
    "embed_xgboost": "XGBoost trained on AIDO foundation model embeddings...",
    "concat_xgboost": "XGBoost trained on concatenation of handcrafted + AIDO embeddings...",
    "aido_finetune": "End-to-end fine-tuning of AIDO backbone with task head...",
    ...
}
```

### Active Learning

The CLI now routes the best model to the correct feature set for active learning analysis:
```python
if best_trained.needs_embeddings:
    al_X = split.X_embed_test
elif best_trained.config.model_type in ('concat_xgboost', 'concat_mlp'):
    al_X = np.hstack([split.X_test, split.X_embed_test])
else:
    al_X = split.X_test
```

### Export

Foundation-tier models get specialized export templates:
- **Embedding models** (`embed_xgboost`, `embed_mlp`): `train.py` includes AIDO loading + embedding extraction inline
- **Fine-tune models** (`aido_finetune`): `train.py` includes full fine-tuning loop; model saved as state_dict
- **Concat models**: Use the same embedding export template (extract embeddings, then train downstream model on combined features)

All templates include:
- `modelgenerator` and `transformers` in `requirements.txt`
- GPU check with clear error message
- Inline embedding extraction (no co-scientist dependency)

## End-to-End: What Happens on an H100

When the pipeline runs on a machine with an H100 GPU:

1. **Preprocessing**: k-mer/HVG features extracted (CPU, ~30s) + AIDO embeddings extracted (GPU, ~5-10 min, cached for reuse)
2. **Splitting**: Both X and X_embed are split using same indices
3. **Baselines**: ~12 CPU models trained (trivial → simple → standard → advanced) + 5 foundation models trained
4. **ReAct agent**: Calls `summarize_data()`, sees "AIDO embeddings available: 768 dims", tries foundation models alongside standard ones
5. **HP search**: Tunes top 2 models (may include foundation models)
6. **Ensemble**: Stacking built from CPU models only (foundation models compete individually)
7. **Evaluation**: All ~17 models evaluated on same val set. Best wins regardless of tier.
8. **Export**: Winner gets a standalone train.py — if it's a foundation model, the script includes GPU code.

**Total wall clock**: ~20-30 min (vs ~5-8 min CPU-only). The extra time is spent on AIDO embedding extraction and fine-tuning — but the performance gain is typically 10-25% on the primary metric.

## Files Modified in Phase 3

| File | Change |
|------|--------|
| `co_scientist/modeling/registry.py` | Added `concat_xgboost`, `concat_mlp` builders |
| `co_scientist/modeling/trainer.py` | Added `_CONCAT_MODELS` routing |
| `co_scientist/modeling/types.py` | Added `_CONCAT_MODELS` set |
| `co_scientist/modeling/ensemble.py` | Exclude foundation models from OOF CV |
| `co_scientist/modeling/hp_search.py` | Added embedding/concat routing + model sets |
| `co_scientist/evaluation/metrics.py` | Concat model routing in `evaluate_model` |
| `co_scientist/llm/prompts.py` | Foundation model awareness in all agent prompts |
| `co_scientist/agents/tools.py` | Full routing in TrainModel, AnalyzeErrors, SummarizeData |
| `co_scientist/agents/ml_engineer.py` | Foundation models in selection + suggestions |
| `co_scientist/validation.py` | Correct data routing for validation checks |
| `co_scientist/report/generator.py` | Explanations for all foundation model types |
| `co_scientist/cli.py` | Active learning routing for foundation models |
| `co_scientist/defaults.yaml` | Concat model configs + HP search spaces |
