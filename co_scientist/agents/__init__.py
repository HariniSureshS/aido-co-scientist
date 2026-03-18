"""Multi-agent framework for the AIDO Co-Scientist pipeline."""

from co_scientist.agents.coordinator import Coordinator
from co_scientist.agents.types import AgentRole, Decision, PipelineContext

__all__ = ["Coordinator", "AgentRole", "Decision", "PipelineContext"]
