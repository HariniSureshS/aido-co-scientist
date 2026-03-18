"""Experiment logging — append-only JSONL log of pipeline events."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLog:
    """Append-only experiment log that writes to a JSONL file."""

    def __init__(self, output_dir: Path):
        self._log_dir = output_dir / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._log_dir / "experiment_log.jsonl"
        self._start_time = time.time()

    @property
    def path(self) -> Path:
        return self._path

    def log(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Append a single event to the log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - self._start_time, 2),
            "event": event,
        }
        if data:
            entry["data"] = _make_serializable(data)

        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_step_start(self, step: str) -> None:
        self.log("step_start", {"step": step})

    def log_step_end(self, step: str, summary: dict[str, Any] | None = None) -> None:
        self.log("step_end", {"step": step, **(summary or {})})

    def log_model_trained(
        self,
        name: str,
        tier: str,
        model_type: str,
        hyperparameters: dict[str, Any],
        train_time: float,
    ) -> None:
        self.log("model_trained", {
            "name": name,
            "tier": tier,
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "train_time_seconds": round(train_time, 2),
        })

    def log_evaluation(
        self,
        model_name: str,
        metrics: dict[str, float],
        primary_metric: str,
        primary_value: float,
        split: str = "val",
    ) -> None:
        self.log("evaluation", {
            "model_name": model_name,
            "split": split,
            "primary_metric": primary_metric,
            "primary_value": round(primary_value, 6),
            "all_metrics": {k: round(v, 6) for k, v in metrics.items()},
        })

    def log_hp_search(
        self,
        n_trials: int,
        best_trial: int,
        best_value: float,
        best_params: dict[str, Any],
        duration: float,
        improved: bool,
    ) -> None:
        self.log("hp_search", {
            "n_trials": n_trials,
            "best_trial": best_trial,
            "best_value": round(best_value, 6),
            "best_params": best_params,
            "duration_seconds": round(duration, 1),
            "improved_over_baseline": improved,
        })

    def log_error(self, step: str, error: str) -> None:
        self.log("error", {"step": step, "error": error})

    def log_pipeline_complete(self, best_model: str, best_metric: str, best_value: float) -> None:
        self.log("pipeline_complete", {
            "best_model": best_model,
            "best_metric": best_metric,
            "best_value": round(best_value, 6),
            "total_elapsed_seconds": round(time.time() - self._start_time, 1),
        })


def _make_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return obj
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj
