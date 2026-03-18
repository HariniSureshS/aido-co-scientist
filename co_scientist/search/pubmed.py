"""PubMed / NCBI E-utilities client.

Free, no API key required (3 req/sec).
Optional: set NCBI_API_KEY env var for 10 req/sec.
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET

from co_scientist.search.types import SearchResult

logger = logging.getLogger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

_TOOL = "aido_coscientist"
_EMAIL = "noreply@example.com"
_MAX_RETRIES = 2
_REQUEST_DELAY = 0.35  # seconds between requests (stay under 3/sec limit)


class PubMedClient:
    """Search PubMed for biomedical papers."""

    def __init__(self):
        self._api_key = os.environ.get("NCBI_API_KEY", "")

    @property
    def available(self) -> bool:
        return True  # always available (free API)

    def search_papers(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search PubMed and return results with abstracts.

        Two-step process: ESearch for PMIDs, then EFetch for abstracts.
        Returns empty list on any failure.
        """
        if not query.strip():
            return []

        # Step 1: ESearch — get PMIDs
        pmids = self._esearch(query, limit)
        if not pmids:
            return []

        # Step 2: EFetch — get paper details + abstracts
        time.sleep(_REQUEST_DELAY)
        results = self._efetch(pmids)

        return results

    def _esearch(self, query: str, limit: int) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(limit, 20),
            "retmode": "json",
            "tool": _TOOL,
            "email": _EMAIL,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        url = f"{_ESEARCH_URL}?{urllib.parse.urlencode(params)}"
        data = self._get_json(url)
        if data is None:
            return []

        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: list[str]) -> list[SearchResult]:
        """Fetch paper details and abstracts for a list of PMIDs."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
            "tool": _TOOL,
            "email": _EMAIL,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        url = f"{_EFETCH_URL}?{urllib.parse.urlencode(params)}"
        xml_text = self._get_text(url)
        if not xml_text:
            return []

        return self._parse_xml(xml_text)

    def _parse_xml(self, xml_text: str) -> list[SearchResult]:
        """Parse PubMed XML response into SearchResults."""
        results = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning("Failed to parse PubMed XML: %s", e)
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                results.append(self._parse_article(article))
            except Exception as e:
                logger.debug("Failed to parse PubMed article: %s", e)

        return results

    def _parse_article(self, article: ET.Element) -> SearchResult:
        """Parse a single PubmedArticle XML element."""
        medline = article.find(".//MedlineCitation")
        if medline is None:
            medline = article

        # Title
        title_el = medline.find(".//ArticleTitle")
        title = (title_el.text or "") if title_el is not None else ""

        # Abstract
        abstract_parts = []
        for abs_text in medline.findall(".//AbstractText"):
            if abs_text.text:
                abstract_parts.append(abs_text.text)
        abstract = " ".join(abstract_parts)[:1000]

        # Authors
        authors = []
        for author in medline.findall(".//Author"):
            last = author.findtext("LastName", "")
            first = author.findtext("Initials", "")
            if last:
                authors.append(f"{last} {first}".strip())
        authors = authors[:5]

        # Year
        year = None
        year_el = medline.find(".//PubDate/Year")
        if year_el is not None and year_el.text:
            try:
                year = int(year_el.text)
            except ValueError:
                pass

        # PMID
        pmid_el = medline.find(".//PMID")
        pmid = (pmid_el.text or "") if pmid_el is not None else ""

        # DOI
        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi" and id_el.text:
                doi = id_el.text
                break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        return SearchResult(
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            citation_count=0,  # PubMed doesn't provide citation counts
            source="pubmed",
            url=url,
            doi=doi,
        )

    def _get_json(self, url: str) -> dict | None:
        """Make a GET request and return parsed JSON."""
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode())
            except Exception as e:
                logger.warning("PubMed request failed (attempt %d): %s", attempt + 1, e)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(1)
        return None

    def _get_text(self, url: str) -> str | None:
        """Make a GET request and return raw text."""
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.read().decode()
            except Exception as e:
                logger.warning("PubMed request failed (attempt %d): %s", attempt + 1, e)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(1)
        return None
