"""Pipeline checkpointing — save/restore state for resumable runs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

CHECKPOINT_FILE = "checkpoint.pkl"
CHECKPOINT_META = "checkpoint_meta.json"

# Steps in pipeline order
STEP_ORDER = [
    "load_profile",     # Step 1: dataset loaded + profiled
    "preprocess_split", # Step 2: preprocessed + split
    "baselines",        # Step 3: baselines trained + evaluated
    "hp_search",        # Step 3b: HP search completed
    "iteration",        # Step 4: iteration loop completed
    "export",           # Step 5: model exported
    "report",           # Step 6: report generated
]


class PipelineState:
    """Container for pipeline state at any checkpoint."""

    def __init__(self):
        self.completed_steps: list[str] = []

        # Step 1 outputs
        self.dataset: Any = None
        self.profile: Any = None
        self.profiling_figs: list[Path] = []

        # Step 2 outputs
        self.preprocessed: Any = None
        self.split: Any = None
        self.preprocessing_figs: list[Path] = []

        # Step 3 outputs
        self.eval_config: Any = None
        self.trained_models: list = []
        self.results: list = []
        self.training_figs: list[Path] = []
        self.best_result: Any = None
        self.best_trained: Any = None

        # Complexity budget
        self.complexity_budget: Any = None

        # Research results (from Search Layer)
        self.research_results: dict = {}  # serialized ResearchReport

        # Step 3b outputs
        self.hp_search_done: bool = False

        # Step 4: iteration loop outputs
        self.iteration_log: dict | None = None

        # ReAct agent scratchpad (when ReAct path is used)
        self.react_scratchpad: list[dict] | None = None

        # Tree search log (when tree search is used)
        self.tree_search_log: dict | None = None

        # Elo tournament rankings
        self.elo_rankings: dict | None = None

        # Debate transcripts
        self.debate_transcripts: list[dict] | None = None

        # Report review result
        self.review_result: dict | None = None

        # Test set evaluation
        self.test_metrics: dict | None = None

        # Step 5 outputs
        self.export_path: Path | None = None

        # Step 6 outputs
        self.report_path: Path | None = None

    def mark_complete(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)

    def is_complete(self, step: str) -> bool:
        return step in self.completed_steps

    def last_completed_step(self) -> str | None:
        if not self.completed_steps:
            return None
        return self.completed_steps[-1]


def save_checkpoint(state: PipelineState, output_dir: Path) -> None:
    """Save pipeline state to disk."""
    checkpoint_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save full state as pickle
    # Custom (LLM-designed) models may not be picklable — skip them gracefully
    try:
        with open(checkpoint_dir / CHECKPOINT_FILE, "wb") as f:
            pickle.dump(state, f)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        import logging
        logging.getLogger(__name__).warning("Checkpoint pickle failed (%s), saving without trained models", e)
        # Save a copy without unpicklable models
        import copy
        safe_state = copy.copy(state)
        safe_models = []
        safe_results = []
        for tm, r in zip(state.trained_models, state.results):
            try:
                pickle.dumps(tm)
                safe_models.append(tm)
                safe_results.append(r)
            except Exception:
                safe_results.append(r)  # keep result even if model isn't picklable
        safe_state.trained_models = safe_models
        # Keep all results for reporting even if some models can't be checkpointed
        safe_state.results = state.results
        with open(checkpoint_dir / CHECKPOINT_FILE, "wb") as f:
            pickle.dump(safe_state, f)

    # Save human-readable metadata
    meta = {
        "completed_steps": state.completed_steps,
        "last_step": state.last_completed_step(),
        "n_models_trained": len(state.trained_models),
        "best_model": state.best_result.model_name if state.best_result else None,
    }
    with open(checkpoint_dir / CHECKPOINT_META, "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(output_dir: Path) -> PipelineState | None:
    """Load pipeline state from disk. Returns None if no checkpoint exists."""
    checkpoint_path = output_dir / "logs" / CHECKPOINT_FILE

    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        console.print(f"  [bold cyan]Resumed from checkpoint:[/bold cyan] "
                      f"completed {state.last_completed_step()}")
        return state
    except Exception as e:
        console.print(f"  [yellow]Warning: Could not load checkpoint: {e}. Starting fresh.[/yellow]")
        return None
