# Step 21: Enhanced Biology Specialist Agent

## Overview

The Biology Specialist agent was enhanced from a minimal rule-based system into a rich biological knowledge base that provides task-specific context, feature importance interpretation, metric validation, plausibility assessment, and detailed report contributions.

## What Changed

### Before (Step 17)
- Basic context strings per modality (4 modalities)
- Simple feature suggestion lists
- Binary plausibility check (suspicious if >0.99, implausible if <0.01)

### After (Step 21)
- **Rich knowledge base** with task-specific entries (RNA translation, cell type classification, RNA stability, protein function)
- **Feature importance interpretation** mapping ML features to biological mechanisms (k-mers → codons, genes → cell types)
- **Metric appropriateness validation** (e.g., flags "accuracy" for regression tasks)
- **Task-specific plausibility ranges** from literature (e.g., spearman 0.3–0.85 for RNA TE)
- **Marker gene checking** for expression datasets
- **Detailed report generation** with multi-paragraph biological interpretation
- **Always active** (previously only for "complex" datasets)

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/agents/biology.py` | Complete rewrite with knowledge base, feature interpretation, metric validation |
| `co_scientist/agents/analysis.py` | Enhanced `agent_biology_interpretation()` with feature_names/research_papers params; added `agent_feature_interpretation()` |
| `co_scientist/agents/coordinator.py` | Biology Specialist now always active (was complex-only) |
| `co_scientist/llm/prompts.py` | Enhanced BIOLOGY_SPECIALIST_SYSTEM prompt with more specific instructions |
| `co_scientist/cli.py` | Extracts feature importances from best model, passes to biology agent and report |
| `co_scientist/report/generator.py` | Added `feature_interpretation` parameter |
| `co_scientist/report/template.md.jinja` | Added §4.6 Top Feature Interpretation table; renumbered Agent Decision Log to §4.7 |

## Biological Knowledge Base

### Task-Specific Knowledge (`_TASK_KNOWLEDGE`)

Each entry keyed by `"modality:task_hint"` contains:

| Field | Description |
|-------|-------------|
| `context` | Multi-sentence biological context paragraph |
| `key_features` | Dict mapping feature names → biological meaning |
| `marker_genes` | Dict mapping cell types → known marker genes (expression only) |
| `expected_metric_range` | Dict mapping metric → (low, high) expected range from literature |
| `appropriate_metrics` | Metrics that make biological sense for this task |
| `inappropriate_metrics` | Metrics that don't make sense (flagged to user) |
| `biological_signals` | What to look for in model results |

Current entries:
- `rna:translation` — RNA translation efficiency
- `cell_expression:cell_type` — Cell type classification (Segerstolpe pancreas)
- `rna:stability` — mRNA stability prediction
- `protein:function` — Protein function prediction

### Modality Fallback (`_MODALITY_KNOWLEDGE`)

When no task-specific entry matches, falls back to modality-level knowledge for: `rna`, `dna`, `protein`, `cell_expression`.

## Feature Importance Interpretation

The `interpret_features()` method maps ML feature names to biological meaning:

1. **Key features** — matched against task-specific knowledge base
2. **Marker genes** — matched against known cell type markers
3. **k-mer features** — codon interpretation for trinucleotides, GC/AT richness for tetranucleotides
4. **Amino acid features** — physicochemical properties per amino acid
5. **Generic patterns** — gc_content, seq_length, molecular_weight, etc.

Example output for RNA translation efficiency:
```
kmer_ATG  → Start codon — frequency reflects upstream ORF density
kmer_CGA  → Rare arginine codon — associated with low translation efficiency
gc_content → GC-rich regions form more stable secondary structures
```

## Plausibility Assessment

Uses task-specific expected ranges from literature:

| Task | Metric | Expected Range | Source |
|------|--------|---------------|--------|
| RNA TE | spearman | 0.30 – 0.85 | Literature |
| Cell type | f1_macro | 0.65 – 0.95 | Literature |
| RNA stability | spearman | 0.25 – 0.75 | Literature |
| Protein function | accuracy | 0.60 – 0.95 | Literature |

Score assessment includes position within range (excellent/good/moderate/low).

## Report Sections

### §4.5 Biological Interpretation
Multi-paragraph interpretation combining:
- Task-specific biological context
- Score assessment with expected range
- Metric appropriateness note
- Key biological signals
- Research paper context (when available)

### §4.6 Top Feature Interpretation (NEW)
Table mapping top 10 features to biological meaning:
```
| Feature | Importance | Biological Meaning |
|---------|-----------|-------------------|
| kmer_ATG | 0.1500 | Start codon — frequency reflects upstream ORF density |
```

## Agent Activation

Biology Specialist is now **always active** regardless of complexity level. It's lightweight (no LLM calls in deterministic mode) and provides value for all biological datasets.

### Consultation During ReAct Loop

The Biology Specialist is now consultable during the ReAct loop via the `consult_biology` tool. The ReAct agent can invoke this tool at any point to request biological validation — for example, checking whether a chosen metric is appropriate for the task, assessing the biological plausibility of a model's score, or interpreting top features in biological context. This allows the agent to incorporate domain expertise mid-loop rather than only at pre- or post-training decision points.

## LLM Enhancement

When LLM is available, the biology interpretation prompt is enriched with:
- Knowledge base context (pre-populated biological background)
- Marker gene findings (which known markers appear in top features)
- More specific instructions (3-5 sentences covering predictability, expected performance, biological signals)
