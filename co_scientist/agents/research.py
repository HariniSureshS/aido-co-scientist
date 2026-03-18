"""Research Agent — searches academic literature for biological context and benchmarks."""

from __future__ import annotations

import logging

from co_scientist.agents.base import BaseAgent
from co_scientist.agents.types import AgentRole, Decision, PipelineContext
from co_scientist.search.orchestrator import SearchOrchestrator
from co_scientist.search.types import ResearchReport

logger = logging.getLogger(__name__)

RESEARCH_AGENT_SYSTEM = """\
You are the Research Agent in an automated ML pipeline for biological datasets.

Your role:
- Synthesize search results from academic papers into actionable insights
- Identify published benchmarks and state-of-the-art methods for the dataset/task
- Recommend methods or techniques found in literature
- Provide citations for the report bibliography

You will receive search results (paper titles, abstracts, citations). Synthesize them.

Output format:
```json
{
    "action": "research_synthesis",
    "parameters": {
        "benchmarks": [{"method": "...", "score": "...", "metric": "...", "source": "..."}],
        "recommended_methods": ["...", "..."],
        "key_findings": ["...", "..."],
        "synthesis": "2-3 sentence summary of what the literature suggests for this task"
    },
    "reasoning": "...",
    "confidence": 0.0-1.0
}
```

Rules:
- Focus on recent papers (2022+) unless older ones are highly cited
- Flag if published benchmarks exist for this exact dataset
- Note if transformer/foundation model approaches dominate (suggests our traditional ML may have a ceiling)
- Be honest about what the search found — don't invent findings
"""


class ResearchAgent(BaseAgent):
    """Searches literature and synthesizes findings for the pipeline.

    Unlike other agents that primarily return Decisions, the Research Agent's
    main output is a ResearchReport (via the `research()` method). The standard
    `decide()` method is also available for the agent protocol.
    """

    role = AgentRole.RESEARCH

    def __init__(self, client=None, orchestrator: SearchOrchestrator | None = None):
        super().__init__(client=client)
        self.orchestrator = orchestrator or SearchOrchestrator()

    def system_prompt(self) -> str:
        return RESEARCH_AGENT_SYSTEM

    def decide_deterministic(self, context: PipelineContext) -> Decision:
        """Run search without LLM and return findings as a Decision."""
        report = self._run_search(context)
        return Decision(
            action="research_findings",
            parameters={
                "num_papers": len(report.papers),
                "methods_found": report.methods_found,
                "benchmarks_found": report.benchmarks_found,
                "synthesis": report.synthesis,
            },
            reasoning=f"Found {len(report.papers)} papers via Semantic Scholar + PubMed",
            confidence=0.6 if report.papers else 0.2,
        )

    def research(self, context: PipelineContext, stage: str = "profiling") -> ResearchReport:
        """Full research flow: search + optional LLM synthesis.

        This is the primary entry point (richer than decide()).
        """
        report = self._run_search(context, stage=stage)

        if not report.papers:
            return report

        # Try LLM synthesis for richer interpretation
        if self.client and self.client.available and report.papers:
            llm_synthesis = self._llm_synthesize(context, report)
            if llm_synthesis:
                report.synthesis = llm_synthesis

        return report

    def _run_search(self, context: PipelineContext, stage: str = "profiling") -> ResearchReport:
        """Execute the search via the orchestrator."""
        if not self.orchestrator.available:
            return ResearchReport()

        try:
            return self.orchestrator.search(
                dataset_path=context.dataset_path,
                modality=context.modality,
                task_type=context.task_type,
                stage=stage,
            )
        except Exception as e:
            logger.warning("Research search failed: %s", e)
            return ResearchReport()

    def _llm_synthesize(self, context: PipelineContext, report: ResearchReport) -> str | None:
        """Use LLM to synthesize search results into actionable insights."""
        if not self.client:
            return None

        # Build a concise summary of papers for the LLM
        paper_summaries = []
        for i, paper in enumerate(report.papers[:8], 1):
            abstract_snippet = paper.abstract[:300] if paper.abstract else "No abstract"
            paper_summaries.append(
                f"{i}. \"{paper.title}\" ({paper.year or 'n/a'}, {paper.citation_count} citations)\n"
                f"   {abstract_snippet}"
            )

        papers_text = "\n".join(paper_summaries)

        user_message = (
            f"Dataset: {context.dataset_path}\n"
            f"Modality: {context.modality}\n"
            f"Task: {context.task_type}\n"
            f"Current best model: {context.best_model_name} ({context.primary_metric}={context.best_score:.4f})\n\n"
            f"Search results ({len(report.papers)} papers):\n\n"
            f"{papers_text}\n\n"
            f"Synthesize these findings into 2-3 actionable sentences. "
            f"What methods work best for this type of task? "
            f"Are there published benchmarks we should compare against?"
        )

        result = self.client.ask_text(
            system_prompt=self.system_prompt(),
            user_message=user_message,
            agent_name="research",
            max_tokens=512,
        )
        return result
