# Step 20: Search Layer (Semantic Scholar + PubMed + Tavily)

## Overview

The Search Layer gives the Co-Scientist access to scientific literature during pipeline execution. It searches Semantic Scholar and PubMed for relevant papers, extracts methods and benchmarks, and synthesizes findings into a `ResearchReport` that feeds into the final report.

Web search via Tavily is optional — users must provide a `TAVILY_API_KEY` environment variable to enable it.

## Architecture

```
SearchOrchestrator
├── SemanticScholarClient  (free, no key required)
├── PubMedClient           (free, no key required)
└── TavilyClient           (optional, requires TAVILY_API_KEY)
        │
        ▼
  ResearchAgent (BaseAgent)
        │
        ▼
  ResearchReport → Report §4.4 Literature Context
```

## Files Created

| File | Purpose |
|------|---------|
| `co_scientist/search/__init__.py` | Package init |
| `co_scientist/search/types.py` | `SearchResult` and `ResearchReport` dataclasses |
| `co_scientist/search/semantic_scholar.py` | Semantic Scholar Graph API client |
| `co_scientist/search/pubmed.py` | PubMed NCBI E-utilities client |
| `co_scientist/search/tavily.py` | Optional Tavily web search client |
| `co_scientist/search/orchestrator.py` | Coordinates sources, deduplicates, ranks |
| `co_scientist/agents/research.py` | Research Agent (BaseAgent subclass) |

## Files Modified

| File | Change |
|------|--------|
| `co_scientist/agents/coordinator.py` | Creates SearchOrchestrator, passes to ResearchAgent, adds `no_search` flag |
| `co_scientist/agents/analysis.py` | Added `agent_research()` function |
| `co_scientist/checkpoint.py` | Added `research_results: dict` to PipelineState |
| `co_scientist/cli.py` | Calls research after profiling, passes results to report |
| `co_scientist/report/generator.py` | Accepts `research_report` parameter |
| `co_scientist/report/template.md.jinja` | §4.4 Literature Context section |

## API Details

### Semantic Scholar (Graph API v1)
- **Base URL:** `https://api.semanticscholar.org/graph/v1`
- **Rate limit:** 100 requests / 5 minutes (unauthenticated)
- **Auth:** Optional `S2_API_KEY` env var for higher limits
- **Fields requested:** title, abstract, authors, year, citationCount, url, externalIds
- **Retry:** Exponential backoff (1s/2s/4s) on 429 responses

### PubMed (NCBI E-utilities)
- **ESearch:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`
- **EFetch:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi`
- **Rate limit:** 3 requests/second (unauthenticated), 10/sec with key
- **Auth:** Optional `NCBI_API_KEY` env var
- **XML parsing:** `xml.etree.ElementTree` for article metadata
- **Delay:** 0.35s between requests per NCBI policy

### Tavily (Optional)
- **URL:** `https://api.tavily.com/search`
- **Auth:** Required `TAVILY_API_KEY` env var (Bearer token)
- **Enabled:** Only when key is present; gracefully skipped otherwise

## SearchOrchestrator Logic

### Query Construction
Queries are built from dataset metadata and vary by pipeline stage:
- **profiling:** `"{modality_term} {task} benchmark dataset"` — finds benchmark papers
- **planning:** `"{modality_term} {task} state-of-the-art methods"` — finds SOTA approaches
- **post_training:** `"{modality_term} {task} model comparison"` — finds comparison studies

### Deduplication
Papers from multiple sources are deduplicated by:
1. DOI match (exact)
2. Fuzzy title match (lowercased, stripped punctuation, first 50 chars)

### Ranking
Papers are scored by: `log1p(citation_count) * recency_factor + abstract_boost`
- Recency: 1.0 (≤2yr), 0.8 (3-4yr), 0.5 (5-7yr), 0.3 (older)
- Abstract boost: +0.5 if abstract is present

### Extraction
- **Methods:** Keyword matching against a set of known ML methods (random forest, xgboost, cnn, transformer, etc.)
- **Benchmarks:** Regex for patterns like "achieved X score" or "obtained X accuracy"

## ResearchAgent

The Research Agent wraps the SearchOrchestrator as a `BaseAgent` subclass:
- `research(context, stage)` → `ResearchReport`
- If LLM is available, synthesizes paper findings via Claude (`_llm_synthesize`)
- Falls back to orchestrator's rule-based synthesis
- Integrated into pipeline via `agent_research()` in `analysis.py`

## Report Integration

The report template (§4.4) renders:
- Top 5 papers with title, year, citation count, truncated abstract, source link
- Synthesis paragraph (LLM-generated or rule-based)
- Methods found in literature
- Published benchmarks found

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `S2_API_KEY` | No | Higher Semantic Scholar rate limits |
| `NCBI_API_KEY` | No | Higher PubMed rate limits (10 req/sec) |
| `TAVILY_API_KEY` | No | Enables web search via Tavily |

## Graceful Degradation

1. All three sources available → full search
2. Semantic Scholar rate-limited → PubMed + Tavily results only
3. No API keys, all sources fail → empty ResearchReport, pipeline continues
4. `--no-search` flag → search layer skipped entirely
