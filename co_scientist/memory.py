"""Cross-run memory — learn from previous runs to inform future decisions.

Stores model performance history and hyperparameter priors across runs.
Memory is stored in outputs/.memory/ as JSONL and JSON files.

Graceful degradation: if no .memory/ exists (first run), all queries return empty.
"""

from __future__ import annotations

import fcntl
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceEntry:
    """Record of a single model's performance on a dataset."""

    model_type: str
    modality: str
    task_type: str
    primary_metric: str
    score: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    dataset_name: str = ""
    timestamp: str = ""


@dataclass
class HPPrior:
    """Best known hyperparameters for a model type + modality combination."""

    model_type: str
    modality: str
    best_hyperparameters: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class RunMemory:
    """Cross-run memory for learning from past experiments.

    Storage:
    - outputs/.memory/model_performance.jsonl (append-only, safe for parallel)
    - outputs/.memory/hp_priors.json (file-locked on write)
    - outputs/.memory/dataset_insights.json (file-locked on write)
    """

    def __init__(self, output_dir: Path | str):
        self.memory_dir = Path(output_dir) / ".memory"
        self.performance_path = self.memory_dir / "model_performance.jsonl"
        self.hp_priors_path = self.memory_dir / "hp_priors.json"
        self.insights_path = self.memory_dir / "dataset_insights.json"

    def _ensure_dir(self) -> None:
        """Create memory directory if it doesn't exist."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def load_performance_history(self) -> list[ModelPerformanceEntry]:
        """Load all past performance records."""
        if not self.performance_path.exists():
            return []

        entries = []
        try:
            with open(self.performance_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entries.append(ModelPerformanceEntry(**data))
        except Exception as e:
            logger.warning("Could not load performance history: %s", e)

        return entries

    def load_hp_priors(self) -> dict[str, HPPrior]:
        """Load best known hyperparameters per model_type+modality.

        Returns dict keyed by "{model_type}_{modality}".
        """
        if not self.hp_priors_path.exists():
            return {}

        try:
            data = json.loads(self.hp_priors_path.read_text())
            return {k: HPPrior(**v) for k, v in data.items()}
        except Exception as e:
            logger.warning("Could not load HP priors: %s", e)
            return {}

    def record_performance(self, entry: ModelPerformanceEntry) -> None:
        """Append a performance record (thread/process-safe via append mode)."""
        self._ensure_dir()

        if not entry.timestamp:
            entry.timestamp = datetime.now(timezone.utc).isoformat()

        try:
            with open(self.performance_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            logger.warning("Could not record performance: %s", e)

    def update_hp_priors(
        self,
        model_type: str,
        modality: str,
        hp: dict[str, Any],
        score: float,
    ) -> None:
        """Update the best-known hyperparameters for a model type + modality.

        Only updates if the new score is better (higher) than the stored one.
        Uses file locking for safety.
        """
        self._ensure_dir()
        key = f"{model_type}_{modality}"

        try:
            priors = {}
            if self.hp_priors_path.exists():
                priors = json.loads(self.hp_priors_path.read_text())

            existing = priors.get(key, {})
            existing_score = existing.get("score", 0.0)

            if score > existing_score:
                priors[key] = {
                    "model_type": model_type,
                    "modality": modality,
                    "best_hyperparameters": hp,
                    "score": score,
                }

                # Write with file locking
                with open(self.hp_priors_path, "w") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(priors, f, indent=2)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            logger.warning("Could not update HP priors: %s", e)

    def get_recommendations(self, modality: str, task_type: str) -> dict[str, Any]:
        """Get recommendations based on past runs for this modality/task.

        Returns:
            Dict with 'best_models', 'hp_priors', and 'avg_scores'.
        """
        history = self.load_performance_history()
        hp_priors = self.load_hp_priors()

        # Filter by modality and task type
        relevant = [e for e in history if e.modality == modality and e.task_type == task_type]

        if not relevant:
            return {"best_models": [], "hp_priors": {}, "avg_scores": {}}

        # Aggregate scores by model type
        scores_by_type: dict[str, list[float]] = {}
        for e in relevant:
            scores_by_type.setdefault(e.model_type, []).append(e.score)

        avg_scores = {mt: sum(s) / len(s) for mt, s in scores_by_type.items()}
        best_models = sorted(avg_scores, key=avg_scores.get, reverse=True)[:5]

        # Get relevant HP priors
        relevant_priors = {}
        for key, prior in hp_priors.items():
            if prior.modality == modality:
                relevant_priors[prior.model_type] = {
                    "hyperparameters": prior.best_hyperparameters,
                    "score": prior.score,
                }

        return {
            "best_models": best_models,
            "hp_priors": relevant_priors,
            "avg_scores": avg_scores,
        }

    def format_for_prompt(self, modality: str, task_type: str) -> str:
        """Format memory as natural language for injection into agent prompts.

        Returns empty string if no relevant memory exists.
        """
        recs = self.get_recommendations(modality, task_type)

        if not recs["best_models"]:
            return ""

        parts = [f"From {len(self.load_performance_history())} past experiment(s):"]

        if recs["best_models"]:
            parts.append(f"  Best model types for {modality}/{task_type}: {', '.join(recs['best_models'])}")

        if recs["avg_scores"]:
            score_parts = [f"{mt}: {s:.4f}" for mt, s in sorted(
                recs["avg_scores"].items(), key=lambda x: x[1], reverse=True
            )[:5]]
            parts.append(f"  Average scores: {', '.join(score_parts)}")

        if recs["hp_priors"]:
            for mt, info in list(recs["hp_priors"].items())[:3]:
                hp_str = ", ".join(f"{k}={v}" for k, v in list(info["hyperparameters"].items())[:5])
                parts.append(f"  Best HP for {mt}: {hp_str} (score={info['score']:.4f})")

        return "\n".join(parts)
