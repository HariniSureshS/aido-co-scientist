"""LLM cost tracking and budget enforcement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# Pricing per 1M tokens (Claude Sonnet 4, as of 2025)
_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}
# Fallback for unknown models
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


@dataclass
class LLMCall:
    """Record of a single LLM API call."""

    model: str
    agent: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_seconds: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class CostTracker:
    """Tracks cumulative LLM costs and enforces budget limits.

    Usage:
        tracker = CostTracker(max_cost=5.0)
        tracker.record(model, agent, input_tokens, output_tokens, latency)
        if tracker.budget_remaining <= 0:
            # stop making LLM calls
    """

    max_cost: float = 5.0
    calls: list[LLMCall] = field(default_factory=list)
    _reserved: float = 0.0  # budget reserved for future debates

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.max_cost - self.total_cost)

    @property
    def available_budget(self) -> float:
        """Budget available for general use (excluding reservations)."""
        return max(0.0, self.budget_remaining - self._reserved)

    def reserve(self, amount: float) -> None:
        """Reserve budget for future high-priority operations (e.g., debates)."""
        self._reserved = max(0.0, amount)

    def release_reserve(self, amount: float | None = None) -> None:
        """Release reserved budget back to general pool."""
        if amount is None:
            self._reserved = 0.0
        else:
            self._reserved = max(0.0, self._reserved - amount)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def num_calls(self) -> int:
        return len(self.calls)

    def can_afford(self, estimated_cost: float = 0.01, min_calls: int = 1) -> bool:
        """Check if we can afford another call (or multiple calls)."""
        return self.budget_remaining >= estimated_cost * min_calls

    def record(
        self,
        model: str,
        agent: str,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float,
    ) -> LLMCall:
        """Record a completed API call and return the LLMCall object."""
        pricing = _PRICING.get(model, _DEFAULT_PRICING)
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        call = LLMCall(
            model=model,
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_seconds=latency_seconds,
        )
        self.calls.append(call)
        return call

    def summary(self) -> dict:
        """Return a summary dict for logging."""
        return {
            "total_cost": round(self.total_cost, 4),
            "budget_remaining": round(self.budget_remaining, 4),
            "num_calls": self.num_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    def per_agent_costs(self) -> dict[str, float]:
        """Return cost breakdown by agent."""
        costs: dict[str, float] = {}
        for call in self.calls:
            costs[call.agent] = costs.get(call.agent, 0.0) + call.cost
        return costs
