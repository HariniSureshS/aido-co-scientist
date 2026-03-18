# Step 28: Automated Report Review

## Status: REMOVED

The automated report review feature was removed from the pipeline. The LLM-based reviewer was producing false positives — flagging "inconsistencies" between the report and experiment log that were actually caused by the ReAct agent logging differently than the deterministic path (e.g., `n_models_evaluated: 0` in the log while the report correctly showed 9 models). These false issues were confusing and misleading to readers.

### What replaced it

Instead of a separate review step, report quality is now ensured by:

1. **Biology Specialist performance context** — The report's §4.4 "Performance Context" section includes the Biology Specialist's plausibility assessment with expected score ranges from literature, giving readers immediate context for interpreting results.
2. **Guardrail system (§10)** — Scientific validation checks run during the pipeline (not after report generation), catching data quality issues, overfitting, and leakage at the source.
3. **Structured template rendering** — Report values come directly from pipeline state variables, minimizing the chance of numerical mismatches.

### Files affected

| File | Change |
|------|--------|
| `co_scientist/cli.py` | Removed `review_report()` call and report re-generation with review |
| `co_scientist/report/template.md.jinja` | Removed §8 "Report Review" section |
| `co_scientist/report/reviewer.py` | Still exists (not deleted) but is no longer called by the pipeline |
| `co_scientist/llm/prompts.py` | `REPORT_REVIEWER_SYSTEM` prompt still exists but is unused |
