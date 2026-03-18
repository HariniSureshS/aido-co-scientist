"""Tavily web search client (optional — requires TAVILY_API_KEY).

Tavily provides AI-optimized web search results.
Docs: https://docs.tavily.com/
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error

from co_scientist.search.types import SearchResult

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.tavily.com/search"


class TavilyClient:
    """Optional web search via Tavily. Requires TAVILY_API_KEY env var."""

    def __init__(self):
        self._api_key = os.environ.get("TAVILY_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def search_web(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search the web via Tavily. Returns empty list if no API key or on failure."""
        if not self.available:
            return []

        if not query.strip():
            return []

        body = json.dumps({
            "query": query,
            "search_depth": "basic",
            "max_results": min(max_results, 10),
            "include_answer": False,
        }).encode()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            req = urllib.request.Request(_SEARCH_URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning("Tavily search failed: %s", e)
            return []

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                abstract=item.get("content", "")[:1000],
                url=item.get("url", ""),
                source="tavily",
                relevance_score=item.get("score", 0.0),
            ))

        return results
