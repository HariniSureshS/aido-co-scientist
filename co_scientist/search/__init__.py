"""Search & Research Layer — Semantic Scholar, PubMed, and optional web search."""

from co_scientist.search.orchestrator import SearchOrchestrator
from co_scientist.search.types import ResearchReport, SearchResult

__all__ = ["SearchOrchestrator", "ResearchReport", "SearchResult"]
