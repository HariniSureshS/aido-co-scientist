"""Agent debate — two agents argue competing proposals before a judge decides.

Used at high-stakes decision points (model selection, HP search) to get
more robust decisions through adversarial reasoning.

Each debate costs ~5 LLM calls, so it's only triggered at key stages
and only when the cost budget allows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from co_scientist.agents.base import BaseAgent
from co_scientist.agents.types import Decision, PipelineContext
from co_scientist.llm.client import ClaudeClient
from co_scientist.llm.prompts import DEBATE_REBUTTAL_SYSTEM, DEBATE_JUDGE_SYSTEM

logger = logging.getLogger(__name__)


@dataclass
class DebateTranscript:
    """Full transcript of a debate between agents."""

    topic: str
    proposals: dict[str, Decision]  # agent_name -> proposal
    rebuttals: dict[str, str]  # agent_name -> rebuttal text
    judge_reasoning: str = ""
    winner: Decision | None = None
    winning_agent: str = ""


class Debater:
    """Orchestrates debates between agents.

    Process:
    1. Each agent proposes a decision independently
    2. Each agent sees the other's proposal and writes a rebuttal
    3. A judge sees all proposals + rebuttals and picks the winner
    """

    def __init__(self, client: ClaudeClient):
        self.client = client

    def debate(
        self,
        topic: str,
        context: PipelineContext,
        agents: dict[str, BaseAgent],
    ) -> DebateTranscript | None:
        """Run a debate between agents on a topic.

        Args:
            topic: What the debate is about (e.g. "model selection strategy")
            context: Current pipeline state
            agents: Dict of agent_name -> BaseAgent (should have 2 agents)

        Returns:
            DebateTranscript or None if LLM is unavailable.
        """
        if not self.client or not self.client.available:
            return None

        agent_names = list(agents.keys())
        if len(agent_names) < 2:
            return None

        # Use first two agents as debaters
        name_a, name_b = agent_names[0], agent_names[1]
        agent_a, agent_b = agents[name_a], agents[name_b]

        # Phase 1: Each agent proposes independently
        proposal_a = agent_a.decide(context)
        proposal_b = agent_b.decide(context)

        proposals = {name_a: proposal_a, name_b: proposal_b}

        # Phase 2: Each writes a rebuttal of the other's proposal
        rebuttal_a = self._get_rebuttal(
            name_a, proposal_a, name_b, proposal_b, context,
        )
        rebuttal_b = self._get_rebuttal(
            name_b, proposal_b, name_a, proposal_a, context,
        )

        rebuttals = {name_a: rebuttal_a or "", name_b: rebuttal_b or ""}

        # Phase 3: Judge picks the winner
        judge_reasoning, winning_agent = self._judge(
            topic, context, proposals, rebuttals, agent_names,
        )

        winner = proposals.get(winning_agent, proposal_a)

        return DebateTranscript(
            topic=topic,
            proposals=proposals,
            rebuttals=rebuttals,
            judge_reasoning=judge_reasoning or "",
            winner=winner,
            winning_agent=winning_agent or name_a,
        )

    def _get_rebuttal(
        self,
        my_name: str,
        my_proposal: Decision,
        other_name: str,
        other_proposal: Decision,
        context: PipelineContext,
    ) -> str | None:
        """Have an agent write a rebuttal of the other's proposal."""
        user_msg = (
            f"You are {my_name}. You proposed:\n"
            f"  Action: {my_proposal.action}\n"
            f"  Parameters: {my_proposal.parameters}\n"
            f"  Reasoning: {my_proposal.reasoning}\n\n"
            f"{other_name} proposed instead:\n"
            f"  Action: {other_proposal.action}\n"
            f"  Parameters: {other_proposal.parameters}\n"
            f"  Reasoning: {other_proposal.reasoning}\n\n"
            f"Context:\n"
            f"  Dataset: {context.dataset_path}\n"
            f"  Modality: {context.modality}, Task: {context.task_type}\n"
            f"  Samples: {context.num_samples}, Features: {context.num_features}\n"
            f"  Best model: {context.best_model_name} ({context.best_score:.4f})\n\n"
            f"Write a brief rebuttal (2-3 sentences) explaining why your approach "
            f"is better than {other_name}'s proposal for this specific dataset."
        )

        return self.client.ask_text(
            system_prompt=DEBATE_REBUTTAL_SYSTEM,
            user_message=user_msg,
            agent_name=f"debate_rebuttal_{my_name}",
            max_tokens=512,
            temperature=0.4,
        )

    def _judge(
        self,
        topic: str,
        context: PipelineContext,
        proposals: dict[str, Decision],
        rebuttals: dict[str, str],
        agent_names: list[str],
    ) -> tuple[str | None, str]:
        """Have a judge evaluate all proposals and rebuttals."""
        proposals_text = ""
        for name, prop in proposals.items():
            rebuttal = rebuttals.get(name, "")
            proposals_text += (
                f"\n### {name}\n"
                f"Action: {prop.action}\n"
                f"Parameters: {prop.parameters}\n"
                f"Reasoning: {prop.reasoning}\n"
                f"Confidence: {prop.confidence}\n"
                f"Rebuttal of other's proposal: {rebuttal}\n"
            )

        user_msg = (
            f"Topic: {topic}\n\n"
            f"Context:\n"
            f"  Dataset: {context.dataset_path}\n"
            f"  Modality: {context.modality}, Task: {context.task_type}\n"
            f"  Samples: {context.num_samples}, Features: {context.num_features}\n"
            f"  Current best: {context.best_model_name} ({context.best_score:.4f})\n"
            f"  Budget remaining: {context.remaining_budget} steps, ${context.remaining_cost:.2f}\n\n"
            f"Proposals and rebuttals:{proposals_text}\n\n"
            f"Pick the winning agent. Reply with exactly one of: {', '.join(agent_names)}\n"
            f"Then explain your reasoning in 2-3 sentences."
        )

        response = self.client.ask_text(
            system_prompt=DEBATE_JUDGE_SYSTEM,
            user_message=user_msg,
            agent_name="debate_judge",
            max_tokens=512,
            temperature=0.1,
        )

        if response is None:
            # Default to first agent
            return None, agent_names[0]

        # Parse winner from response
        winning_agent = agent_names[0]  # default
        response_lower = response.lower()
        for name in agent_names:
            if name.lower() in response_lower.split("\n")[0]:
                winning_agent = name
                break

        return response, winning_agent
