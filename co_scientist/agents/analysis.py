"""Agent-driven analysis — agents interpret results and advise on next steps.

This module provides functions that consult agents at specific pipeline stages
and translate their decisions into concrete pipeline actions.
"""

from __future__ import annotations

from typing import Any

from co_scientist.agents.coordinator import Coordinator
from co_scientist.agents.types import Decision, PipelineContext
from co_scientist.data.types import DatasetProfile
from co_scientist.evaluation.types import EvalConfig, ModelResult
from co_scientist.modeling.types import ModelConfig
from co_scientist.search.types import ResearchReport


def build_pipeline_context(
    config: Any,  # RunConfig
    profile: DatasetProfile,
    eval_config: EvalConfig | None = None,
    results: list[ModelResult] | None = None,
    best_result: ModelResult | None = None,
    stage: str = "",
    iteration: int = 0,
    complexity_budget: Any = None,
    cost_remaining: float = 5.0,
    memory_context: str = "",
) -> PipelineContext:
    """Build a PipelineContext from current pipeline state.

    Centralizes context building so it's consistent across all decision points.
    """
    model_scores = {}
    trained_names = []
    if results:
        for r in results:
            model_scores[r.model_name] = r.primary_metric_value
            trained_names.append(r.model_name)

    return PipelineContext(
        dataset_path=config.dataset_path,
        modality=profile.modality.value,
        task_type=profile.task_type.value,
        num_samples=profile.num_samples,
        num_features=profile.num_features,
        num_classes=profile.num_classes,
        stage=stage,
        trained_model_names=trained_names,
        model_scores=model_scores,
        best_model_name=best_result.model_name if best_result else "",
        best_score=best_result.primary_metric_value if best_result else 0.0,
        primary_metric=eval_config.primary_metric if eval_config else "",
        iteration=iteration,
        remaining_budget=config.budget,
        remaining_cost=cost_remaining,
        complexity_level=complexity_budget.level if complexity_budget else "moderate",
        complexity_score=complexity_budget.score if complexity_budget else 5.0,
        gpu_available=_check_gpu(),
        memory_context=memory_context,
    )


def _check_gpu() -> bool:
    """Check GPU availability (cached)."""
    try:
        from co_scientist.modeling.foundation import gpu_available
        return gpu_available()
    except Exception:
        return False


def agent_model_selection(
    coordinator: Coordinator,
    context: PipelineContext,
    baseline_configs: list[ModelConfig],
) -> list[ModelConfig]:
    """Let the ML Engineer prioritize model order.

    The agent's recommended model list is used to reorder (not filter) the
    baseline configs. Models recommended first are trained first. Models not
    in the agent's list are still trained (just later), ensuring we never
    lose models due to agent error.
    """
    decision = coordinator.consult("ml_engineer", context, stage="model_selection")

    if decision.action != "select_models" or not decision.parameters.get("models"):
        return baseline_configs  # no change

    recommended = decision.parameters["models"]
    priority = decision.parameters.get("priority", "")

    # Build priority map: recommended models get lower sort keys
    priority_map = {name: i for i, name in enumerate(recommended)}
    fallback_idx = len(recommended)  # non-recommended models go last

    reordered = sorted(
        baseline_configs,
        key=lambda c: priority_map.get(c.model_type, priority_map.get(c.name, fallback_idx)),
    )

    return reordered, decision


def agent_hp_decision(
    coordinator: Coordinator,
    context: PipelineContext,
) -> Decision:
    """Let the ML Engineer decide whether to run HP search and with what config."""
    return coordinator.consult("ml_engineer", context, stage="hp_search")


def agent_post_training_analysis(
    coordinator: Coordinator,
    context: PipelineContext,
) -> dict[str, Decision]:
    """Consult all active agents for post-training analysis.

    Returns a dict of agent_name → Decision with each agent's assessment.
    """
    decisions = {}

    # ML Engineer: what to try next
    ml_dec = coordinator.consult("ml_engineer", context, stage="iteration")
    decisions["ml_engineer"] = ml_dec

    # Data Analyst: data quality assessment
    da_agent = coordinator.agents.get("data_analyst")
    if da_agent and "data_analyst" in coordinator.active_agents:
        da_dec = da_agent.assess_data_quality(context)
        decisions["data_analyst"] = da_dec

    # Biology Specialist: plausibility
    if "biology_specialist" in coordinator.active_agents:
        bio_dec = coordinator.consult("biology_specialist", context, stage="post_training")
        decisions["biology_specialist"] = bio_dec

    return decisions


def agent_biology_interpretation(
    coordinator: Coordinator,
    context: PipelineContext,
    feature_names: list[str] | None = None,
    research_papers: list | None = None,
) -> str:
    """Get biological interpretation from the Biology Specialist.

    Returns a text string for the report. Falls back to rich deterministic interpretation.
    """
    bio_agent = coordinator.agents.get("biology_specialist")
    if bio_agent is None:
        return ""

    # Try LLM for rich interpretation
    if coordinator.client and coordinator.client.available:
        # Build richer prompt with knowledge base context
        knowledge_ctx = ""
        det_interp = bio_agent.generate_interpretation(context, feature_names, research_papers)
        if det_interp:
            knowledge_ctx = f"\n\nBiological context from knowledge base:\n{det_interp}\n"

        marker_info = ""
        if feature_names:
            markers = bio_agent.check_marker_genes(context, feature_names)
            if markers:
                marker_lines = [f"  - {ct}: {', '.join(genes)}" for ct, genes in markers.items()]
                marker_info = f"\n\nKnown marker genes found in features:\n" + "\n".join(marker_lines) + "\n"

        text = coordinator.client.ask_text(
            system_prompt=bio_agent.system_prompt(),
            user_message=(
                f"Provide a biological interpretation of these ML results.\n\n"
                f"Dataset: {context.dataset_path}\n"
                f"Modality: {context.modality}\n"
                f"Task: {context.task_type}\n"
                f"Best model: {context.best_model_name} ({context.primary_metric}={context.best_score:.4f})\n"
                f"All scores: {context.model_scores}\n"
                f"{knowledge_ctx}{marker_info}\n"
                f"Write 3-5 sentences interpreting what the results suggest biologically. "
                f"Cover: (1) what the model performance tells us about predictability of the biological "
                f"phenotype from sequence/expression data, (2) whether the performance level is expected "
                f"given known biology, and (3) what biological signals the model is likely capturing."
            ),
            agent_name="biology_specialist",
        )
        if text:
            return text

    # Rich deterministic fallback using knowledge base
    return bio_agent.generate_interpretation(context, feature_names, research_papers)


def agent_feature_interpretation(
    coordinator: Coordinator,
    context: PipelineContext,
    top_features: list[tuple[str, float]],
) -> list[dict[str, str]]:
    """Get biological interpretation of top features from the Biology Specialist.

    Args:
        top_features: List of (feature_name, importance_score) sorted desc.

    Returns:
        List of dicts with feature, importance, biological_meaning.
    """
    bio_agent = coordinator.agents.get("biology_specialist")
    if bio_agent is None:
        return []

    return bio_agent.interpret_features(context, top_features)


def agent_research(
    coordinator: Coordinator,
    context: PipelineContext,
    stage: str = "profiling",
) -> ResearchReport:
    """Run the Research Agent to find relevant papers and benchmarks.

    Returns a ResearchReport (may be empty if search is disabled or fails).
    """
    research_agent = coordinator.agents.get("research")
    if research_agent is None:
        return ResearchReport()

    if "research" not in coordinator.active_agents:
        return ResearchReport()

    try:
        report = research_agent.research(context, stage=stage)

        # Push to live dashboard
        if coordinator.live_dashboard and report.papers:
            methods = ", ".join(report.methods_found[:3]) if report.methods_found else "none"
            coordinator.live_dashboard.add_agent_message(
                agent="Research Agent",
                stage=stage,
                message=f"Found {len(report.papers)} paper(s). Methods: {methods}. Query: {report.query_used}",
                msg_type="consult",
            )
        elif coordinator.live_dashboard:
            coordinator.live_dashboard.add_agent_message(
                agent="Research Agent",
                stage=stage,
                message="No relevant papers found (search may be rate-limited)",
                msg_type="consult",
            )

        return report
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Research agent failed: %s", e)
        return ResearchReport()
