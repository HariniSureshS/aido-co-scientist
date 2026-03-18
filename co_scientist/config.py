"""Global configuration and constants."""

import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MODE = "auto"
DEFAULT_BUDGET = 10  # iteration steps
DEFAULT_MAX_COST = 5.0  # dollars
DEFAULT_TIMEOUT = 1800  # seconds (30 minutes)


def _make_timestamp() -> str:
    """Generate a compact timestamp for directory names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class RunConfig(BaseModel):
    """Configuration for a single co-scientist run."""

    dataset_path: str
    mode: str = Field(default=DEFAULT_MODE, pattern=r"^(auto|interactive)$")
    budget: int = Field(default=DEFAULT_BUDGET, ge=1, le=100)
    max_cost: float = Field(default=DEFAULT_MAX_COST, ge=0)
    timeout: int = Field(default=DEFAULT_TIMEOUT, ge=60, le=7200)
    output_dir: Path = DEFAULT_OUTPUT_DIR
    no_search: bool = False
    resume: bool = False
    seed: int = 42
    run_timestamp: str = Field(default_factory=_make_timestamp)

    @property
    def task_output_dir(self) -> Path:
        """Output directory for this specific run.

        Format: outputs/RNA__translation_efficiency_muscle_20260316_143022/
        Each run gets its own timestamped directory so previous results are preserved.
        """
        sanitized = self.dataset_path.replace("/", "__")
        return self.output_dir / f"{sanitized}_{self.run_timestamp}"


class PipelineDeadline:
    """Global deadline tracker — ensures the pipeline finishes within --timeout.

    Usage:
        deadline = PipelineDeadline(timeout_seconds=1800)
        if deadline.expired():
            skip remaining steps...
        remaining = deadline.remaining()  # seconds left
        deadline.check("step_name")       # raises TimeoutError if expired
    """

    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT):
        self.start_time = time.time()
        self.timeout_seconds = timeout_seconds
        self.deadline = self.start_time + timeout_seconds

    def remaining(self) -> float:
        """Seconds remaining before deadline."""
        return max(0.0, self.deadline - time.time())

    def elapsed(self) -> float:
        """Seconds elapsed since start."""
        return time.time() - self.start_time

    def expired(self) -> bool:
        """True if we've exceeded the deadline."""
        return time.time() >= self.deadline

    def check(self, step_name: str = "") -> None:
        """Raise TimeoutError if deadline has passed."""
        if self.expired():
            raise TimeoutError(
                f"Pipeline deadline exceeded ({self.timeout_seconds}s) "
                f"during {step_name or 'unknown step'}. "
                f"Elapsed: {self.elapsed():.0f}s"
            )

    def budget_for_step(self, step_name: str, fraction: float = 0.5) -> float:
        """Return time budget for a step (capped by remaining time).

        fraction: what fraction of remaining time to allocate to this step.
        """
        return max(30.0, self.remaining() * fraction)
