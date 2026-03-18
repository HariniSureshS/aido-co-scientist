"""Adaptive complexity scoring — scales pipeline effort based on dataset difficulty.

Adopted from Google AI Co-Scientist (Architecture Section 2.1):
  "Test-time compute scaling — Our adaptive complexity system scales agent
   activation and iteration depth based on dataset difficulty."

The complexity score (0-10) is computed entirely from the DatasetProfile
and controls: HP trials, iteration budget, search budget, agent activation.
"""

from __future__ import annotations

from dataclasses import dataclass

from co_scientist.data.types import DatasetProfile, Modality, TaskType


@dataclass
class ComplexityBudget:
    """Pipeline resource budget derived from complexity score."""

    score: float                # 0-10
    level: str                  # "simple", "moderate", "complex", "very_complex"
    hp_trials: int              # Optuna trials
    hp_timeout: int             # seconds
    iteration_steps: int        # Phase 4 iteration budget
    search_web: int             # web search queries (Phase C)
    search_paper: int           # paper search queries (Phase C)
    agents: list[str]           # which agents to activate (Phase C)

    def summary(self) -> dict:
        return {
            "complexity_score": round(self.score, 1),
            "level": self.level,
            "hp_trials": self.hp_trials,
            "hp_timeout": self.hp_timeout,
            "iteration_steps": self.iteration_steps,
        }


# ---------------------------------------------------------------------------
# Scoring factors — each returns a 0-10 contribution
# ---------------------------------------------------------------------------

def _score_sample_size(profile: DatasetProfile) -> float:
    """Small datasets are trickier — need careful regularization and validation."""
    n = profile.num_samples
    if n < 100:
        return 3.0    # very small — high complexity
    if n < 500:
        return 2.0    # small
    if n < 2000:
        return 1.0    # moderate
    if n < 10000:
        return 0.5    # comfortable
    return 0.0         # large — sample size is not a concern


def _score_dimensionality(profile: DatasetProfile) -> float:
    """High dimensionality relative to samples increases complexity."""
    n_feat = profile.num_features
    n_samp = max(profile.num_samples, 1)
    ratio = n_feat / n_samp

    if ratio > 10:
        return 3.0    # extreme — features >> samples
    if ratio > 1:
        return 2.0    # high-dimensional
    if ratio > 0.5:
        return 1.0    # moderate
    return 0.0


def _score_class_complexity(profile: DatasetProfile) -> float:
    """More classes and imbalance increase complexity."""
    if profile.task_type == TaskType.REGRESSION:
        return 0.0  # no class complexity for regression

    score = 0.0

    # Number of classes
    if profile.num_classes > 50:
        score += 2.0
    elif profile.num_classes > 20:
        score += 1.5
    elif profile.num_classes > 5:
        score += 0.5

    # Imbalance
    if profile.class_distribution:
        total = sum(profile.class_distribution.values())
        if total > 0:
            min_pct = min(profile.class_distribution.values()) / total * 100
            if min_pct < 1:
                score += 2.0    # severe imbalance
            elif min_pct < 5:
                score += 1.0    # moderate imbalance
            elif min_pct < 10:
                score += 0.5

    return min(score, 3.0)


def _score_modality(profile: DatasetProfile) -> float:
    """Some modalities are inherently harder to model."""
    modality_scores = {
        Modality.TABULAR: 0.0,
        Modality.RNA: 1.0,
        Modality.DNA: 1.0,
        Modality.PROTEIN: 1.5,
        Modality.CELL_EXPRESSION: 2.0,    # high-dim, sparse, complex biology
        Modality.SPATIAL: 2.5,            # spatial + expression
        Modality.MULTIMODAL: 3.0,         # multiple input types
        Modality.UNKNOWN: 1.5,            # uncertainty adds complexity
    }
    return modality_scores.get(profile.modality, 1.0)


def _score_data_quality(profile: DatasetProfile) -> float:
    """More issues = more complexity (need to handle edge cases)."""
    score = 0.0

    if profile.missing_value_pct > 10:
        score += 1.0
    elif profile.missing_value_pct > 1:
        score += 0.5

    if profile.feature_sparsity > 90:
        score += 0.5

    # Count detected issues (excluding INFO)
    serious_issues = sum(
        1 for issue in profile.detected_issues
        if issue.startswith("CRITICAL") or issue.startswith("WARNING")
    )
    score += min(serious_issues * 0.5, 1.5)

    return min(score, 2.0)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

# Weights for each factor (sum to 1.0)
_WEIGHTS = {
    "sample_size": 0.15,
    "dimensionality": 0.20,
    "class_complexity": 0.20,
    "modality": 0.25,
    "data_quality": 0.20,
}


def compute_complexity(profile: DatasetProfile) -> ComplexityBudget:
    """Compute complexity score (0-10) and derive resource budget.

    Architecture Section 4.2:
      Simple (0-2)       → 10 HP trials, 4 iteration steps
      Moderate (3-5)     → 20 HP trials, 6 iteration steps
      Complex (6-8)      → 30 HP trials, 10 iteration steps
      Very Complex (9-10) → 50 HP trials, 15 iteration steps
    """
    factors = {
        "sample_size": _score_sample_size(profile),
        "dimensionality": _score_dimensionality(profile),
        "class_complexity": _score_class_complexity(profile),
        "modality": _score_modality(profile),
        "data_quality": _score_data_quality(profile),
    }

    # Weighted sum, then scale to 0-10
    # Each factor is 0-3, weighted sum is 0-3, scale to 0-10
    weighted = sum(factors[k] * _WEIGHTS[k] for k in factors)
    score = min(weighted * (10.0 / 3.0), 10.0)

    # Map to level and budget
    if score <= 2:
        level = "simple"
        hp_trials = 10
        hp_timeout = 90
        iteration_steps = 4
        search_web = 0
        search_paper = 0
        agents = ["coordinator", "data_analyst", "ml_engineer"]
    elif score <= 5:
        level = "moderate"
        hp_trials = 15
        hp_timeout = 120
        iteration_steps = 6
        search_web = 3
        search_paper = 0
        agents = ["coordinator", "data_analyst", "ml_engineer", "research"]
    elif score <= 8:
        level = "complex"
        hp_trials = 20
        hp_timeout = 180
        iteration_steps = 10
        search_web = 6
        search_paper = 3
        agents = ["coordinator", "data_analyst", "ml_engineer", "biology", "research"]
    else:
        level = "very_complex"
        hp_trials = 30
        hp_timeout = 300
        iteration_steps = 15
        search_web = 10
        search_paper = 6
        agents = ["coordinator", "data_analyst", "ml_engineer", "biology", "research"]

    return ComplexityBudget(
        score=score,
        level=level,
        hp_trials=hp_trials,
        hp_timeout=hp_timeout,
        iteration_steps=iteration_steps,
        search_web=search_web,
        search_paper=search_paper,
        agents=agents,
    )


def print_complexity(budget: ComplexityBudget) -> None:
    """Print complexity score and budget."""
    from rich.console import Console
    console = Console()

    color = {
        "simple": "green",
        "moderate": "yellow",
        "complex": "red",
        "very_complex": "bold red",
    }.get(budget.level, "white")

    console.print(f"  Complexity: [{color}]{budget.score:.1f}/10 ({budget.level})[/{color}]"
                  f"  →  HP trials: {budget.hp_trials}, timeout: {budget.hp_timeout}s")
