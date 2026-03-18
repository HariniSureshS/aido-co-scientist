"""Coordinator agent — orchestrates the pipeline and routes between agents.

The Coordinator is the "supervisor" from the Architecture (Section 4.1).
It manages agent activation, conflict resolution, and budget enforcement.

Integration with cli.py:
    The Coordinator is consulted at key decision points. If it's not available
    (no API key, budget exhausted), the pipeline falls through to existing
    deterministic logic seamlessly.
"""

from __future__ import annotations

import logging
from typing import Any

from co_scientist.agents.base import BaseAgent
from co_scientist.agents.biology import BiologySpecialistAgent
from co_scientist.agents.data_analyst import DataAnalystAgent
from co_scientist.agents.ml_engineer import MLEngineerAgent
from co_scientist.agents.research import ResearchAgent
from co_scientist.search.orchestrator import SearchOrchestrator
from co_scientist.agents.types import (
    AgentMessage,
    AgentRole,
    Decision,
    MessageType,
    PipelineContext,
)
from co_scientist.llm.client import ClaudeClient
from co_scientist.llm.cost import CostTracker

logger = logging.getLogger(__name__)

# Mapping from internal agent keys to display names for the dashboard
_AGENT_DISPLAY_NAMES = {
    "data_analyst": "Data Analyst",
    "ml_engineer": "ML Engineer",
    "biology_specialist": "Biology Specialist",
    "research": "Research Agent",
}


class Coordinator:
    """Orchestrates the multi-agent pipeline.

    Usage:
        tracker = CostTracker(max_cost=5.0)
        coord = Coordinator(cost_tracker=tracker)

        # At a decision point in the pipeline:
        context = PipelineContext(...)
        decision = coord.consult("ml_engineer", context, stage="model_selection")

        # The decision.action and decision.parameters tell the pipeline what to do
    """

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        api_key: str | None = None,
        model: str | None = None,
        complexity_level: str = "moderate",
        no_search: bool = False,
    ):
        self.cost_tracker = cost_tracker or CostTracker()
        self.no_search = no_search

        # Create the LLM client (may be None if no API key)
        client_kwargs: dict[str, Any] = {"cost_tracker": self.cost_tracker}
        if api_key:
            client_kwargs["api_key"] = api_key
        if model:
            client_kwargs["model"] = model
        self.client = ClaudeClient(**client_kwargs)

        # Search orchestrator (free APIs — Semantic Scholar + PubMed)
        self.search_orchestrator = SearchOrchestrator(no_search=no_search)

        # Initialize agents
        self.agents: dict[str, BaseAgent] = {
            "data_analyst": DataAnalystAgent(client=self.client),
            "ml_engineer": MLEngineerAgent(client=self.client),
            "biology_specialist": BiologySpecialistAgent(client=self.client),
            "research": ResearchAgent(client=self.client, orchestrator=self.search_orchestrator),
        }

        # Which agents are active (based on complexity)
        self.complexity_level = complexity_level
        self.active_agents = self._activate_agents(complexity_level)

        # Message log for audit trail
        self.message_log: list[AgentMessage] = []

        # Debate transcripts for reporting
        self.debate_transcripts: list = []

        # Cross-run memory (set by CLI)
        self.memory: Any = None

        # Live dashboard (set by CLI)
        self.live_dashboard: Any = None

    @property
    def llm_available(self) -> bool:
        """Check if LLM calls are possible."""
        return self.client.available and self.cost_tracker.can_afford()

    def _activate_agents(self, complexity: str) -> set[str]:
        """Determine which agents to activate based on complexity."""
        # Always active — biology specialist is lightweight and always valuable
        active = {"data_analyst", "ml_engineer", "biology_specialist"}

        # Research agent: active for moderate+ complexity when search is enabled
        if not self.no_search and complexity in ("moderate", "complex", "very_complex"):
            active.add("research")

        return active

    def consult(
        self,
        agent_name: str,
        context: PipelineContext,
        stage: str | None = None,
    ) -> Decision:
        """Consult an agent for a decision.

        If the agent is not active or not available, returns a deterministic decision.
        """
        if stage:
            context = context.model_copy(update={"stage": stage})

        agent = self.agents.get(agent_name)
        if agent is None:
            logger.warning("Unknown agent: %s", agent_name)
            return Decision(
                action="no_op",
                parameters={},
                reasoning=f"Agent {agent_name} not found",
                confidence=0.0,
            )

        # Get the decision (LLM → deterministic fallback is handled by BaseAgent)
        decision = agent.decide(context)

        # Log the interaction
        self._log_message(
            from_agent=AgentRole.COORDINATOR,
            to_agent=agent.role,
            message_type=MessageType.DECISION,
            summary=f"{agent_name}: {decision.action}",
            data={"action": decision.action, "parameters": decision.parameters},
            confidence=decision.confidence,
        )

        # Push to live dashboard
        if self.live_dashboard:
            display_name = _AGENT_DISPLAY_NAMES.get(agent_name, agent_name)
            msg = decision.reasoning or decision.action
            self.live_dashboard.add_agent_message(
                agent=display_name,
                stage=stage or context.stage,
                message=msg,
                msg_type="consult",
            )

        return decision

    def consult_all_active(self, context: PipelineContext, stage: str | None = None) -> dict[str, Decision]:
        """Consult all active agents and return their decisions."""
        decisions = {}
        for name in self.active_agents:
            decisions[name] = self.consult(name, context, stage=stage)
        return decisions

    def resolve_conflict(self, decisions: dict[str, Decision]) -> Decision:
        """Resolve conflicting decisions from multiple agents.

        Strategy: weighted by confidence, with ML Engineer having priority
        for model-related decisions and Data Analyst for data-related ones.
        """
        if not decisions:
            return Decision(action="no_op", parameters={}, reasoning="No decisions to resolve")

        # If only one decision, return it
        if len(decisions) == 1:
            return next(iter(decisions.values()))

        # Priority: highest confidence wins, with tie-breaking by role
        priority_order = ["ml_engineer", "data_analyst", "biology_specialist"]
        best_decision = None
        best_score = -1.0

        for agent_name in priority_order:
            if agent_name in decisions:
                d = decisions[agent_name]
                # Weighted score: confidence + role priority bonus
                role_bonus = 0.1 * (len(priority_order) - priority_order.index(agent_name))
                score = d.confidence + role_bonus
                if score > best_score:
                    best_score = score
                    best_decision = d

        return best_decision or next(iter(decisions.values()))

    def should_stop(self, context: PipelineContext) -> bool:
        """Decide whether the iteration loop should stop."""
        # Hard stops
        if context.remaining_budget <= 0:
            return True
        if not self.cost_tracker.can_afford():
            return True

        # Consult ML Engineer
        decision = self.consult("ml_engineer", context, stage="iteration")
        return decision.action == "stop"

    def debate(
        self,
        topic: str,
        context: PipelineContext,
        agent_names: list[str] | None = None,
    ) -> Decision:
        """Run a debate between agents at a high-stakes decision point.

        Falls back to consult() if LLM is unavailable or debate fails.
        """
        _fallback_agent = agent_names[0] if agent_names else "ml_engineer"

        if not self.llm_available:
            self._log_debate_fallback(topic, "LLM unavailable", _fallback_agent)
            return self.consult(_fallback_agent, context)

        if not self.cost_tracker.can_afford(min_calls=5):
            remaining = f"${self.cost_tracker.budget_remaining:.3f}"
            self._log_debate_fallback(topic, f"budget too low ({remaining} left, need $0.05)", _fallback_agent)
            return self.consult(_fallback_agent, context)

        if agent_names is None:
            agent_names = ["ml_engineer", "data_analyst"]

        agents_for_debate = {name: self.agents[name] for name in agent_names if name in self.agents}
        if len(agents_for_debate) < 2:
            self._log_debate_fallback(topic, "fewer than 2 agents available", _fallback_agent)
            return self.consult(_fallback_agent, context)

        try:
            from co_scientist.agents.debate import Debater
            debater = Debater(self.client)
            transcript = debater.debate(topic, context, agents_for_debate)

            if transcript is None or transcript.winner is None:
                self._log_debate_fallback(topic, "debate returned no winner", agent_names[0])
                return self.consult(agent_names[0], context)

            self.debate_transcripts.append(transcript)

            self._log_message(
                from_agent=AgentRole.COORDINATOR,
                to_agent=AgentRole.COORDINATOR,
                message_type=MessageType.DECISION,
                summary=f"Debate: {topic} — winner: {transcript.winning_agent}",
                data={
                    "topic": topic,
                    "winner": transcript.winning_agent,
                    "action": transcript.winner.action,
                },
            )

            # Push debate to live dashboard
            if self.live_dashboard:
                # Show each agent's proposal
                for agent_name, proposal in transcript.proposals.items():
                    display_name = _AGENT_DISPLAY_NAMES.get(agent_name, agent_name)
                    self.live_dashboard.add_agent_message(
                        agent=display_name,
                        stage=topic,
                        message=f"Proposal: {proposal.reasoning}",
                        msg_type="debate",
                    )
                # Show rebuttals
                for agent_name, rebuttal in transcript.rebuttals.items():
                    display_name = _AGENT_DISPLAY_NAMES.get(agent_name, agent_name)
                    self.live_dashboard.add_agent_message(
                        agent=display_name,
                        stage=topic,
                        message=f"Rebuttal: {rebuttal}",
                        msg_type="debate",
                    )
                # Show winner
                winner_name = _AGENT_DISPLAY_NAMES.get(transcript.winning_agent, transcript.winning_agent)
                self.live_dashboard.add_agent_message(
                    agent="Coordinator",
                    stage=topic,
                    message=f"Winner: {winner_name} — {transcript.judge_reasoning}",
                    msg_type="debate",
                )

            return transcript.winner

        except Exception as e:
            logger.warning("Debate failed (%s), falling back to consult", e)
            self._log_debate_fallback(topic, f"error: {e}", agent_names[0])
            return self.consult(agent_names[0], context)

    def _log_debate_fallback(self, topic: str, reason: str, fallback_agent: str) -> None:
        """Log when a debate falls back to a simple consult — visible in dashboard."""
        msg = f"Debate '{topic}' → consult ({reason})"
        logger.info(msg)

        self._log_message(
            from_agent=AgentRole.COORDINATOR,
            to_agent=AgentRole.COORDINATOR,
            message_type=MessageType.DECISION,
            summary=msg,
            data={"topic": topic, "reason": reason, "fallback_agent": fallback_agent},
        )

        if self.live_dashboard:
            self.live_dashboard.add_agent_message(
                agent="Coordinator",
                stage=topic,
                message=f"Debate skipped ({reason}) → consulting {_AGENT_DISPLAY_NAMES.get(fallback_agent, fallback_agent)} instead",
                msg_type="consult",
            )
            self.live_dashboard.add_warning(f"Debate '{topic}' skipped: {reason}")

    def run_react_modeling(
        self,
        profile: Any,
        split: Any,
        eval_config: Any,
        seed: int = 42,
        exp_log: Any = None,
        max_steps: int = 25,
        patience: int = 8,
        tree_search: bool = False,
        max_wall_seconds: float = 900.0,
        interactive: bool = False,
    ) -> Any:
        """Run the ReAct agent for the modeling phase.

        Returns a ReactResult on success, or None to trigger deterministic fallback.
        Three-tier fallback: tree search → linear ReAct → deterministic.
        """
        if not self.llm_available:
            return None

        from co_scientist.agents.react import ReactAgent, ReactState
        from co_scientist.evaluation.ranking import EloRanker

        if tree_search:
            from co_scientist.agents.tools import build_tree_search_registry
            registry = build_tree_search_registry()
        else:
            from co_scientist.agents.tools import build_default_registry
            registry = build_default_registry()

        agent = ReactAgent(
            client=self.client,
            tool_registry=registry,
            cost_tracker=self.cost_tracker,
            max_steps=max_steps,
            patience=patience,
            max_wall_seconds=max_wall_seconds,
        )

        # Initialize Elo ranker
        elo_ranker = EloRanker()

        state = ReactState(
            profile=profile,
            split=split,
            eval_config=eval_config,
            seed=seed,
            elo_ranker=elo_ranker,
            llm_client=self.client,
            dashboard=self.live_dashboard,
        )

        try:
            if tree_search:
                result = agent.run_tree_search(state, exp_log=exp_log)
                # If tree search fails, fall back to linear
                if result is None:
                    logger.info("Tree search returned None, falling back to linear ReAct")
                    state_linear = ReactState(
                        profile=profile, split=split,
                        eval_config=eval_config, seed=seed,
                        elo_ranker=elo_ranker,
                        llm_client=self.client,
                        dashboard=self.live_dashboard,
                    )
                    result = agent.run(state_linear, exp_log=exp_log,
                                       interactive=interactive, coordinator=self)
            else:
                result = agent.run(state, exp_log=exp_log,
                                   interactive=interactive, coordinator=self)
        except Exception as e:
            logger.warning("ReAct agent failed: %s", e, exc_info=True)
            from rich.console import Console as _C
            _C().print(f"  [bold red]ReAct agent error: {e}[/bold red]")
            return None

        return result

    def get_cost_summary(self) -> dict:
        """Return cost tracking summary."""
        return self.cost_tracker.summary()

    def _log_message(
        self,
        from_agent: AgentRole,
        to_agent: AgentRole,
        message_type: MessageType,
        summary: str,
        data: dict | None = None,
        confidence: float = 1.0,
    ) -> None:
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            summary=summary,
            structured_data=data or {},
            confidence=confidence,
        )
        self.message_log.append(msg)
