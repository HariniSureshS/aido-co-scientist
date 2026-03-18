"""Types for the multi-agent framework."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Roles for the specialized agents."""

    COORDINATOR = "coordinator"
    DATA_ANALYST = "data_analyst"
    ML_ENGINEER = "ml_engineer"
    BIOLOGY_SPECIALIST = "biology_specialist"
    RESEARCH = "research"


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    RECOMMENDATION = "recommendation"
    RESULT = "result"
    QUESTION = "question"
    OVERRIDE = "override"
    DECISION = "decision"
    STATUS = "status"


class AgentMessage(BaseModel):
    """Structured message passed between agents via the Coordinator."""

    from_agent: AgentRole
    to_agent: AgentRole
    message_type: MessageType
    summary: str  # one-line natural language
    structured_data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0  # 0-1
    evidence: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    """A concrete decision made by an agent, executed by deterministic code.

    The LLM proposes the decision (what to do); the pipeline code executes it.
    This is the "LLM-as-strategist" pattern.
    """

    action: str  # e.g. "select_model", "set_preprocessing", "tune_hyperparameters"
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 1.0
    fallback: str = ""  # what to do if the action fails


class PipelineContext(BaseModel):
    """Snapshot of the current pipeline state, provided to agents for decision-making.

    Agents receive this context (read-only) and return Decisions.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Dataset info
    dataset_path: str = ""
    modality: str = ""
    task_type: str = ""
    num_samples: int = 0
    num_features: int = 0
    num_classes: int = 0

    # Profile details (populated after profiling)
    target_column: str = ""
    class_distribution: dict[str, int] = Field(default_factory=dict)
    target_stats: dict[str, float] = Field(default_factory=dict)  # mean, std, min, max
    split_info: dict[str, Any] = Field(default_factory=dict)  # type, split sizes
    missing_value_pct: float = 0.0
    feature_sparsity: float = 0.0
    sequence_length_stats: dict[str, float] = Field(default_factory=dict)
    detected_issues: list[str] = Field(default_factory=list)
    preprocessing_steps: list[str] = Field(default_factory=list)

    # Current pipeline stage
    stage: str = ""  # "preprocessing", "model_selection", "hp_search", "iteration"

    # Results so far
    trained_model_names: list[str] = Field(default_factory=list)
    model_scores: dict[str, float] = Field(default_factory=dict)  # name → primary metric
    best_model_name: str = ""
    best_score: float = 0.0
    primary_metric: str = ""

    # History
    iteration: int = 0
    decisions_so_far: list[str] = Field(default_factory=list)
    errors_encountered: list[str] = Field(default_factory=list)

    # Budget
    remaining_budget: int = 0
    remaining_cost: float = 0.0

    # Complexity
    complexity_level: str = ""
    complexity_score: float = 0.0

    # GPU status
    gpu_available: bool = False

    # Cross-run memory context
    memory_context: str = ""
