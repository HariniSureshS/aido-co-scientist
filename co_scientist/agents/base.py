"""Base agent class — all agents inherit from this.

Design:
- Each agent has two paths: `decide()` (LLM-powered) and `decide_deterministic()` (rule-based)
- The Coordinator calls `decide()` first; if LLM is unavailable or fails, falls back to `decide_deterministic()`
- This ensures graceful degradation: Full LLM → Rule-based → YAML defaults
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from co_scientist.agents.types import AgentRole, Decision, PipelineContext
from co_scientist.llm.client import ClaudeClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    role: AgentRole  # set by subclasses

    def __init__(self, client: ClaudeClient | None = None):
        self.client = client

    @property
    def name(self) -> str:
        return self.role.value

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent's LLM calls."""
        ...

    @abstractmethod
    def decide_deterministic(self, context: PipelineContext) -> Decision:
        """Rule-based fallback when LLM is unavailable.

        Must always return a valid Decision — never raises.
        """
        ...

    def _build_user_message(self, context: PipelineContext) -> str:
        """Format the pipeline context into a user message for the LLM."""
        parts = [
            f"Dataset: {context.dataset_path}",
            f"Modality: {context.modality}",
            f"Task: {context.task_type}",
            f"Samples: {context.num_samples}, Features: {context.num_features}",
            f"Stage: {context.stage}",
        ]
        if context.model_scores:
            parts.append(f"Model scores: {context.model_scores}")
            parts.append(f"Best: {context.best_model_name} ({context.best_score:.4f})")
        if context.errors_encountered:
            parts.append(f"Errors: {context.errors_encountered}")
        parts.append(f"Iteration: {context.iteration}, Budget remaining: {context.remaining_budget}")
        parts.append(f"Complexity: {context.complexity_level} ({context.complexity_score})")
        if context.gpu_available:
            parts.append("GPU: Available — foundation models (embed_xgboost, concat_xgboost, aido_finetune) can be used")
        else:
            parts.append("GPU: NOT available — do NOT recommend foundation models (embed_*, concat_*, aido_finetune)")
        if context.memory_context:
            parts.append(f"\nCross-run memory:\n{context.memory_context}")

        return "\n".join(parts)

    def decide(self, context: PipelineContext) -> Decision:
        """Make a decision — tries LLM first, falls back to deterministic.

        This is the main entry point called by the Coordinator.
        """
        # Try LLM path
        if self.client and self.client.available:
            try:
                response = self.client.ask(
                    system_prompt=self.system_prompt(),
                    user_message=self._build_user_message(context),
                    agent_name=self.name,
                )
                if response and not response.get("_parse_failed"):
                    return Decision(
                        action=response.get("action", "no_op"),
                        parameters=response.get("parameters", {}),
                        reasoning=response.get("reasoning", ""),
                        confidence=float(response.get("confidence", 0.5)),
                        fallback=response.get("fallback", ""),
                    )
                logger.info("%s: LLM response unparseable, using deterministic fallback", self.name)
            except Exception as e:
                logger.warning("%s: LLM call failed (%s), using deterministic fallback", self.name, e)

        # Deterministic fallback
        return self.decide_deterministic(context)
