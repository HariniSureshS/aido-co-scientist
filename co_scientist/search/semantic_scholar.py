"""Semantic Scholar Academic Graph API client.

Free, no API key required (100 requests / 5 min).
Optional: set S2_API_KEY env var for higher rate limits.
Docs: https://api.semanticscholar.org/
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any

from co_scientist.search.types import SearchResult

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,authors,year,citationCount,url,externalIds"
_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]  # seconds


class SemanticScholarClient:
    """Search Semantic Scholar for academic papers."""

    def __init__(self):
        self._api_key = os.environ.get("S2_API_KEY", "")

    @property
    def available(self) -> bool:
        return True  # always available (free API)

    def search_papers(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search for papers matching a query string.

        Returns up to `limit` results. Returns empty list on any failure.
        """
        if not query.strip():
            return []

        params = urllib.parse.urlencode({
            "query": query,
            "limit": min(limit, 20),
            "fields": _FIELDS,
        })
        url = f"{_BASE_URL}/paper/search?{params}"

        data = self._get(url)
        if data is None:
            return []

        papers = data.get("data", [])
        results = []
        for paper in papers:
            try:
                results.append(self._parse_paper(paper))
            except Exception as e:
                logger.debug("Failed to parse S2 paper: %s", e)

        return results

    def get_paper(self, paper_id: str) -> SearchResult | None:
        """Get a specific paper by Semantic Scholar ID or DOI."""
        url = f"{_BASE_URL}/paper/{paper_id}?fields={_FIELDS}"
        data = self._get(url)
        if data is None:
            return None
        try:
            return self._parse_paper(data)
        except Exception:
            return None

    def _parse_paper(self, paper: dict[str, Any]) -> SearchResult:
        """Parse a Semantic Scholar paper dict into a SearchResult."""
        authors = []
        for author in (paper.get("authors") or []):
            name = author.get("name", "")
            if name:
                authors.append(name)

        external_ids = paper.get("externalIds") or {}
        doi = external_ids.get("DOI", "")

        url = paper.get("url", "")
        if not url and doi:
            url = f"https://doi.org/{doi}"

        return SearchResult(
            title=paper.get("title", ""),
            authors=authors[:5],  # cap at 5
            abstract=(paper.get("abstract") or "")[:1000],
            year=paper.get("year"),
            citation_count=paper.get("citationCount", 0) or 0,
            source="semantic_scholar",
            url=url,
            doi=doi,
        )

    def _get(self, url: str) -> dict | None:
        """Make a GET request with retry and rate limit handling."""
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429:  # rate limited
                    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                    logger.info("S2 rate limited, retrying in %ds", delay)
                    time.sleep(delay)
                    continue
                logger.warning("S2 HTTP error %d for %s", e.code, url[:100])
                return None
            except Exception as e:
                logger.warning("S2 request failed (attempt %d): %s", attempt + 1, e)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAYS[attempt])

        return None
