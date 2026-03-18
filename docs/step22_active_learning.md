# Step 22: Active Learning Analysis

## Overview

Adds post-training analysis that identifies what additional data would improve model performance the most. Three analysis types run automatically after model training:

1. **Class-level performance analysis** (classification) — per-class F1, bottleneck classes, confusion pairs
2. **Uncertainty-based sample prioritization** (classification) — entropy of predictions, most uncertain samples
3. **Residual analysis** (regression) — prediction error by target value range, tail behavior

Plus **feature gap suggestions** from domain knowledge — what new types of biological data (not just more samples) would help.

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/evaluation/active_learning.py` | Active learning analysis module |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/cli.py` | Runs active learning analysis before report generation |
| `co_scientist/report/generator.py` | Accepts `active_learning_report` parameter |
| `co_scientist/report/template.md.jinja` | New §5 "Recommendations for Further Data Collection" with per-class, uncertainty, and feature gap sections; renumbered Reproducibility to §6, Appendix to §7 |

## Analysis Types

### 1. Class-Level Performance (Classification)

For each class:
- F1, precision, recall, support (sample count)
- Most confused class (from confusion matrix off-diagonal)
- Bottleneck identification: classes with F1 < 0.5, or bottom 3 if all are above

Output: `ClassAnalysis` dataclass per class, sorted by F1 ascending (worst first).

### 2. Uncertainty-Based Prioritization (Classification)

- Computes prediction entropy: `H = -Σ p·log(p)` over class probabilities
- Ranks samples by entropy (highest = most uncertain)
- Reports top N (default 20) with predicted label, true label, and top probability
- High entropy + misclassification = most informative for active learning

Output: `UncertainSample` dataclass per sample.

### 3. Residual Analysis (Regression)

- Bins test samples by target value quantiles (5 bins)
- Computes mean absolute error per bin
- Identifies which target range has highest prediction error
- Detects if errors are higher at distribution tails (suggests tail data scarcity)

Output: `ResidualBin` dataclass per bin, sorted by MAE descending.

### 4. Feature Gap Suggestions

Domain-specific suggestions for additional data types, based on modality:

| Modality | Suggestions |
|----------|------------|
| RNA | Ribo-seq, RNA structure probing (DMS-seq, SHAPE), polysome fractionation |
| DNA | ATAC-seq, ChIP-seq, conservation scores (PhyloP) |
| Protein | AlphaFold structures, MSA evolutionary profiles |
| Cell expression | FACS-sorted bottleneck populations, CITE-seq, spatial transcriptomics |

For cell expression, bottleneck class names are included in the FACS suggestion.

## Report Section

New **§5. Recommendations for Further Data Collection** contains:

- **Summary paragraph** — natural language description of findings
- **§5.1 Per-Class Performance** table (classification) or **Prediction Error by Target Range** table (regression)
- **§5.2 Most Uncertain Samples** table (classification only, top 10)
- **§5.3 Suggested Additional Data Types** — biology-informed suggestions

The section is only rendered when there are actionable findings.

## Data Types

```python
@dataclass
class ActiveLearningReport:
    class_analyses: list[ClassAnalysis]       # per-class metrics
    bottleneck_classes: list[str]              # worst-performing classes
    uncertain_samples: list[UncertainSample]   # highest-entropy samples
    residual_bins: list[ResidualBin]           # error by target range
    worst_predicted_range: str                 # hardest range
    feature_gap_suggestions: list[str]         # biology-informed suggestions
    summary: str                               # natural language summary
```

## Integration

The analysis runs as a non-critical step between model export and report generation. Failures are caught and logged — they never block the pipeline.

```
Export → Active Learning Analysis → Report Generation
```

## Example Output (Classification)

```
Active learning analysis:
  Bottleneck classes: gamma, delta
  Top 20 uncertain samples identified
```

Report §5:
```markdown
| Class | F1 | Most Confused With |
|-------|---:|-------------------|
| gamma | 0.450 | delta (5) |
| delta | 0.600 | gamma (4) |
| alpha | 0.900 | beta (2) |
```
