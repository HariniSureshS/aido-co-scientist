"""Search orchestrator — coordinates Semantic Scholar, PubMed, and Tavily.

Builds search queries from pipeline context, runs searches, deduplicates,
ranks by citation count and recency, and returns a ResearchReport.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from co_scientist.search.pubmed import PubMedClient
from co_scientist.search.semantic_scholar import SemanticScholarClient
from co_scientist.search.tavily import TavilyClient
from co_scientist.search.types import ResearchReport, SearchResult

logger = logging.getLogger(__name__)

# Known ML method names to extract from paper titles/abstracts
_ML_METHODS = {
    "random forest", "xgboost", "gradient boosting", "lightgbm",
    "neural network", "deep learning", "cnn", "convolutional",
    "transformer", "attention", "bert", "gpt", "lstm", "rnn",
    "svm", "support vector", "logistic regression", "elastic net",
    "lasso", "ridge", "ensemble", "stacking", "bagging",
    "automl", "hyperparameter", "feature selection",
}


class SearchOrchestrator:
    """Coordinates searches across multiple APIs and produces a ResearchReport."""

    def __init__(self, no_search: bool = False):
        self.no_search = no_search
        self.semantic_scholar = SemanticScholarClient()
        self.pubmed = PubMedClient()
        self.tavily = TavilyClient()

    @property
    def available(self) -> bool:
        return not self.no_search

    def search(
        self,
        dataset_path: str,
        modality: str,
        task_type: str,
        stage: str = "profiling",
        top_k: int = 8,
    ) -> ResearchReport:
        """Run a coordinated search and return aggregated results.

        Args:
            dataset_path: e.g. "RNA/translation_efficiency_muscle"
            modality: e.g. "rna", "cell_expression"
            task_type: e.g. "regression", "multiclass_classification"
            stage: search context — "profiling", "planning", "post_baselines", "stuck"
            top_k: max results to return
        """
        if not self.available:
            return ResearchReport()

        query = self._build_query(dataset_path, modality, task_type, stage)
        logger.info("Search query: %s", query)

        # Collect results from all available sources
        all_results: list[SearchResult] = []

        # Semantic Scholar (free, academic papers)
        try:
            s2_results = self.semantic_scholar.search_papers(query, limit=10)
            all_results.extend(s2_results)
            logger.info("Semantic Scholar: %d results", len(s2_results))
        except Exception as e:
            logger.warning("Semantic Scholar search failed: %s", e)

        # PubMed (free, biomedical papers)
        pubmed_query = self._build_pubmed_query(dataset_path, modality, task_type, stage)
        try:
            pm_results = self.pubmed.search_papers(pubmed_query, limit=10)
            all_results.extend(pm_results)
            logger.info("PubMed: %d results", len(pm_results))
        except Exception as e:
            logger.warning("PubMed search failed: %s", e)

        # Tavily (optional, web search)
        if self.tavily.available:
            try:
                web_results = self.tavily.search_web(query, max_results=5)
                all_results.extend(web_results)
                logger.info("Tavily: %d results", len(web_results))
            except Exception as e:
                logger.warning("Tavily search failed: %s", e)

        if not all_results:
            return ResearchReport(query_used=query)

        # Deduplicate, rank, select top
        deduped = self._deduplicate(all_results)
        ranked = self._rank(deduped)
        top = ranked[:top_k]

        # Extract benchmarks and methods from results
        benchmarks = self._extract_benchmarks(top)
        methods = self._extract_methods(top)

        # Build citations
        citations = [self._format_citation(r) for r in top]

        # Rule-based synthesis (LLM synthesis done by Research Agent)
        synthesis = self._rule_synthesis(top, modality, task_type)

        return ResearchReport(
            query_used=query,
            papers=top,
            benchmarks_found=benchmarks,
            methods_found=methods,
            synthesis=synthesis,
            citations=citations,
        )

    def _build_query(self, dataset_path: str, modality: str, task_type: str, stage: str) -> str:
        """Build a search query appropriate for the stage."""
        # Extract meaningful terms from dataset path
        task_terms = self._extract_task_terms(dataset_path)
        modality_term = _MODALITY_TERMS.get(modality, modality)
        task_term = task_type.replace("_", " ")

        if stage == "profiling":
            return f"{modality_term} {task_terms} benchmark machine learning"
        elif stage == "planning":
            return f"{modality_term} {task_terms} {task_term} state of the art methods"
        elif stage == "post_baselines":
            return f"{modality_term} {task_terms} improve prediction performance"
        elif stage == "stuck":
            return f"{modality_term} {task_terms} machine learning challenges"
        else:
            return f"{modality_term} {task_terms} {task_term}"

    def _build_pubmed_query(self, dataset_path: str, modality: str, task_type: str, stage: str) -> str:
        """Build a PubMed-optimized query (MeSH terms work better)."""
        task_terms = self._extract_task_terms(dataset_path)
        modality_term = _MODALITY_TERMS.get(modality, modality)
        return f"{modality_term} {task_terms} prediction machine learning"

    def _extract_task_terms(self, dataset_path: str) -> str:
        """Extract readable terms from dataset path.

        'RNA/translation_efficiency_muscle' → 'translation efficiency muscle'
        'expression/cell_type_classification_segerstolpe' → 'cell type classification'
        """
        parts = dataset_path.split("/")
        if len(parts) > 1:
            task = parts[-1]
        else:
            task = parts[0]

        # Remove common suffixes that are dataset names, not task descriptors
        task = re.sub(r"_(segerstolpe|muscle|brain|liver|kidney|heart)$", "", task)
        return task.replace("_", " ")

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicates by DOI or title similarity."""
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        deduped = []

        for r in results:
            # Exact DOI match
            if r.doi and r.doi in seen_dois:
                continue

            # Fuzzy title match (lowercase, strip punctuation)
            title_key = re.sub(r"[^a-z0-9 ]", "", r.title.lower()).strip()
            if title_key and title_key in seen_titles:
                continue

            if r.doi:
                seen_dois.add(r.doi)
            if title_key:
                seen_titles.add(title_key)
            deduped.append(r)

        return deduped

    def _rank(self, results: list[SearchResult]) -> list[SearchResult]:
        """Rank results by citation count weighted by recency."""
        current_year = datetime.now().year

        for r in results:
            recency = 1.0
            if r.year:
                age = current_year - r.year
                if age <= 2:
                    recency = 1.0
                elif age <= 4:
                    recency = 0.8
                elif age <= 7:
                    recency = 0.5
                else:
                    recency = 0.3

            # Score: log-scaled citations * recency
            import math
            citation_score = math.log1p(r.citation_count)
            r.relevance_score = citation_score * recency

            # Boost papers that have abstracts (more useful)
            if r.abstract:
                r.relevance_score += 0.5

        return sorted(results, key=lambda r: r.relevance_score, reverse=True)

    def _extract_benchmarks(self, results: list[SearchResult]) -> list[str]:
        """Extract benchmark scores from abstracts (simple pattern matching)."""
        benchmarks = []
        # Look for patterns like "achieved 0.85 AUC" or "accuracy of 95%"
        pattern = re.compile(
            r"(?:achiev|obtain|reach|report)\w*\s+"
            r"(?:an?\s+)?(?:(?:test|val\w*|cross.val\w*)\s+)?"
            r"(\w[\w\s]*?)\s+(?:of\s+)?([\d.]+)(?:\s*%)?",
            re.IGNORECASE,
        )
        for r in results:
            text = f"{r.title} {r.abstract}"
            for match in pattern.finditer(text):
                metric = match.group(1).strip()
                value = match.group(2)
                if metric and value:
                    benchmarks.append(f"{metric}: {value} ({r.title[:60]})")
                    if len(benchmarks) >= 5:
                        return benchmarks
        return benchmarks

    def _extract_methods(self, results: list[SearchResult]) -> list[str]:
        """Extract ML method names mentioned in results."""
        found = set()
        for r in results:
            text = f"{r.title} {r.abstract}".lower()
            for method in _ML_METHODS:
                if method in text:
                    found.add(method)
        return sorted(found)

    def _format_citation(self, r: SearchResult) -> str:
        """Format a SearchResult as a citation string."""
        authors = ", ".join(r.authors[:3])
        if len(r.authors) > 3:
            authors += " et al."
        year = f" ({r.year})" if r.year else ""
        return f"{authors}{year}. {r.title}. {r.url}"

    def _rule_synthesis(self, results: list[SearchResult], modality: str, task_type: str) -> str:
        """Generate a simple rule-based synthesis of search results."""
        if not results:
            return ""

        n = len(results)
        methods = self._extract_methods(results)
        methods_str = ", ".join(methods[:5]) if methods else "various approaches"

        modality_desc = _MODALITY_TERMS.get(modality, modality)
        task_desc = task_type.replace("_", " ")

        synthesis = (
            f"Found {n} relevant paper(s) on {modality_desc} {task_desc}. "
            f"Methods used in the literature include {methods_str}."
        )

        # Note highly cited papers
        top_cited = [r for r in results if r.citation_count > 50]
        if top_cited:
            best = top_cited[0]
            synthesis += (
                f" The most cited work is \"{best.title}\" "
                f"({best.citation_count} citations, {best.year or 'n/a'})."
            )

        return synthesis


# Mapping modality codes to human-readable terms for better search queries
_MODALITY_TERMS = {
    "rna": "RNA sequence",
    "dna": "DNA sequence",
    "protein": "protein sequence",
    "cell_expression": "single-cell gene expression",
    "spatial": "spatial transcriptomics",
    "tabular": "tabular biological data",
}
