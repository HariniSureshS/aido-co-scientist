"""Types for the search layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result from any source."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    citation_count: int = 0
    source: str = ""  # "semantic_scholar", "pubmed", "tavily"
    url: str = ""
    doi: str = ""
    relevance_score: float = 0.0


class ResearchReport(BaseModel):
    """Aggregated research findings from all search sources."""

    query_used: str = ""
    papers: list[SearchResult] = Field(default_factory=list)
    benchmarks_found: list[str] = Field(default_factory=list)
    methods_found: list[str] = Field(default_factory=list)
    synthesis: str = ""  # LLM-generated or rule-based summary
    citations: list[str] = Field(default_factory=list)  # formatted citation strings

    def to_dict(self) -> dict:
        return self.model_dump()
