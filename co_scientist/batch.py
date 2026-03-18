"""Batch processing — run the pipeline on multiple datasets.

Usage:
    co-scientist batch D1 D2 D3 --parallel 2

Runs the full pipeline on each dataset, optionally in parallel using
ProcessPoolExecutor. Collects results into a summary table.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class DatasetRunResult:
    """Result of running the pipeline on a single dataset."""

    dataset_path: str
    success: bool
    best_model: str | None = None
    best_score: float | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None


class BatchRunner:
    """Run the pipeline on multiple datasets, optionally in parallel."""

    def __init__(
        self,
        datasets: list[str],
        parallel: int = 1,
        output_dir: str = "outputs",
        common_options: dict[str, Any] | None = None,
        per_dataset_timeout: float = 3600.0,
    ):
        self.datasets = datasets
        self.parallel = max(1, parallel)
        self.output_dir = output_dir
        self.common_options = common_options or {}
        self.per_dataset_timeout = per_dataset_timeout

    def run(self) -> list[DatasetRunResult]:
        """Run all datasets and return results."""
        console.print(f"\n[bold cyan]Batch mode: {len(self.datasets)} dataset(s), "
                      f"parallel={self.parallel}[/bold cyan]\n")

        if self.parallel <= 1:
            return self._run_sequential()
        else:
            return self._run_parallel()

    def _run_sequential(self) -> list[DatasetRunResult]:
        """Run datasets one at a time."""
        results = []
        for i, dataset in enumerate(self.datasets, 1):
            console.print(f"\n[bold]── Dataset {i}/{len(self.datasets)}: {dataset} ──[/bold]")
            result = _run_single(dataset, self.output_dir, self.common_options)
            results.append(result)
            if result.success:
                console.print(f"  [green]Done: {result.best_model} = {result.best_score:.4f} "
                              f"({result.elapsed_seconds:.1f}s)[/green]")
            else:
                console.print(f"  [red]Failed: {result.error}[/red]")
        return results

    def _run_parallel(self) -> list[DatasetRunResult]:
        """Run datasets in parallel using ProcessPoolExecutor."""
        results: list[DatasetRunResult] = []
        futures = {}

        with ProcessPoolExecutor(max_workers=self.parallel) as executor:
            for dataset in self.datasets:
                future = executor.submit(
                    _run_single, dataset, self.output_dir, self.common_options,
                )
                futures[future] = dataset

            for future in as_completed(futures, timeout=self.per_dataset_timeout * len(self.datasets)):
                dataset = futures[future]
                try:
                    result = future.result(timeout=self.per_dataset_timeout)
                except TimeoutError:
                    result = DatasetRunResult(
                        dataset_path=dataset,
                        success=False,
                        error=f"Timed out after {self.per_dataset_timeout:.0f}s",
                    )
                except Exception as e:
                    result = DatasetRunResult(
                        dataset_path=dataset,
                        success=False,
                        error=str(e),
                    )
                results.append(result)
                if result.success:
                    console.print(f"  [green]{dataset}: {result.best_model} = "
                                  f"{result.best_score:.4f}[/green]")
                else:
                    console.print(f"  [red]{dataset}: {result.error}[/red]")

        return results

    @staticmethod
    def print_summary(results: list[DatasetRunResult]) -> None:
        """Print a summary table of all batch results."""
        table = Table(title="Batch Results Summary")
        table.add_column("Dataset", style="bold")
        table.add_column("Status")
        table.add_column("Best Model")
        table.add_column("Score", justify="right")
        table.add_column("Time", justify="right")

        for r in results:
            if r.success:
                table.add_row(
                    r.dataset_path,
                    "[green]OK[/green]",
                    r.best_model or "-",
                    f"{r.best_score:.4f}" if r.best_score is not None else "-",
                    f"{r.elapsed_seconds:.1f}s",
                )
            else:
                table.add_row(
                    r.dataset_path,
                    "[red]FAIL[/red]",
                    "-",
                    "-",
                    f"{r.elapsed_seconds:.1f}s",
                )

        console.print()
        console.print(table)

        n_success = sum(1 for r in results if r.success)
        console.print(f"\n{n_success}/{len(results)} datasets completed successfully.")


def _run_single(
    dataset_path: str,
    output_dir: str,
    common_options: dict[str, Any],
) -> DatasetRunResult:
    """Run the pipeline on a single dataset. Called in subprocess for parallel."""
    start = time.time()

    try:
        result = run_pipeline(dataset_path, output_dir, common_options)
        result.elapsed_seconds = time.time() - start
        return result
    except Exception as e:
        return DatasetRunResult(
            dataset_path=dataset_path,
            success=False,
            elapsed_seconds=time.time() - start,
            error=f"{type(e).__name__}: {e}",
        )


def run_pipeline(
    dataset_path: str,
    output_dir: str = "outputs",
    options: dict[str, Any] | None = None,
) -> DatasetRunResult:
    """Run the full co-scientist pipeline on a single dataset.

    This is the core pipeline logic extracted from cli.py's run() command,
    callable programmatically for batch processing.

    Args:
        dataset_path: Dataset path (e.g. "RNA/translation_efficiency_muscle")
        output_dir: Base output directory
        options: Dict of options (budget, max_cost, seed, no_search, etc.)

    Returns:
        DatasetRunResult with success/failure info.
    """
    import time as _time
    from pathlib import Path as _Path

    from co_scientist.config import RunConfig
    from co_scientist.checkpoint import PipelineState, load_checkpoint, save_checkpoint
    from co_scientist.experiment_log import ExperimentLog
    from co_scientist.agents import Coordinator
    from co_scientist.agents.analysis import (
        build_pipeline_context, agent_hp_decision,
        agent_post_training_analysis, agent_biology_interpretation,
        agent_feature_interpretation, agent_research,
    )
    from co_scientist.llm.cost import CostTracker
    from co_scientist.resilience import train_baselines_resilient, run_step_resilient
    from co_scientist.guardrails import (
        check_pre_training, check_post_training, check_pipeline_summary,
        check_metric_sanity, check_model_data_compatibility, verify_task_type,
        has_blocking_errors,
    )

    opts = options or {}
    config = RunConfig(
        dataset_path=dataset_path,
        mode="auto",
        budget=opts.get("budget", 10),
        max_cost=opts.get("max_cost", 5.0),
        output_dir=_Path(output_dir),
        no_search=opts.get("no_search", False),
        resume=opts.get("resume", False),
        seed=opts.get("seed", 42),
    )

    exp_log = ExperimentLog(config.task_output_dir)
    cost_tracker = CostTracker(max_cost=config.max_cost)
    coordinator = Coordinator(cost_tracker=cost_tracker, no_search=config.no_search)

    state: PipelineState | None = None
    if config.resume:
        state = load_checkpoint(config.task_output_dir)
    if state is None:
        state = PipelineState()

    # Step 1: Load and profile
    if not state.is_complete("load_profile"):
        from co_scientist.data.loader import load_dataset
        from co_scientist.data.profile import profile_dataset

        state.dataset = load_dataset(config.dataset_path)
        state.profile = profile_dataset(state.dataset, config.dataset_path)

        from co_scientist.complexity import compute_complexity
        state.complexity_budget = compute_complexity(state.profile)
        coordinator.complexity_level = state.complexity_budget.level
        coordinator.active_agents = coordinator._activate_agents(state.complexity_budget.level)

        from co_scientist.viz.profiling import generate_profiling_figures
        state.profiling_figs = generate_profiling_figures(
            state.dataset, state.profile, config.task_output_dir,
        )

        if not config.no_search:
            research_ctx = build_pipeline_context(
                config, state.profile, stage="profiling",
                complexity_budget=state.complexity_budget,
                cost_remaining=cost_tracker.budget_remaining,
            )
            research_report = agent_research(coordinator, research_ctx, stage="profiling")
            if research_report.papers:
                state.research_results = research_report.to_dict()

        state.mark_complete("load_profile")
        save_checkpoint(state, config.task_output_dir)

    # Step 2: Preprocess and split
    if not state.is_complete("preprocess_split"):
        from co_scientist.data.preprocess import preprocess
        from co_scientist.data.split import split_dataset

        preprocess_ctx = build_pipeline_context(
            config, state.profile, stage="preprocessing",
            complexity_budget=state.complexity_budget,
            cost_remaining=cost_tracker.budget_remaining,
        )
        coordinator.consult("data_analyst", preprocess_ctx, stage="preprocessing")

        state.preprocessed = preprocess(state.dataset, state.profile)
        state.split = split_dataset(state.dataset, state.preprocessed, state.profile, seed=config.seed)

        from co_scientist.viz.preprocessing import generate_preprocessing_figures
        state.preprocessing_figs = generate_preprocessing_figures(
            state.split, state.profile, config.task_output_dir,
        )

        state.mark_complete("preprocess_split")
        save_checkpoint(state, config.task_output_dir)

    # Pre-training guardrails
    if state.is_complete("preprocess_split") and not state.is_complete("baselines"):
        pre_alerts = check_pre_training(state.profile, state.split)
        if has_blocking_errors(pre_alerts):
            return DatasetRunResult(
                dataset_path=dataset_path, success=False,
                error="Blocking data quality errors",
            )

    # ReAct agent path
    react_used = False
    if coordinator.llm_available and not state.is_complete("baselines"):
        from co_scientist.evaluation.auto_config import auto_eval_config

        if state.eval_config is None:
            state.eval_config = auto_eval_config(state.profile)

        react_result = coordinator.run_react_modeling(
            profile=state.profile, split=state.split,
            eval_config=state.eval_config, seed=config.seed,
            exp_log=exp_log,
            max_steps=min(25, max(config.budget + 10, 15)),
            patience=8,
        )

        if react_result is not None:
            react_used = True
            state.trained_models = react_result.trained_models
            state.results = react_result.results
            state.best_result = react_result.best_result
            state.best_trained = react_result.best_trained
            state.react_scratchpad = [
                {
                    "step": e.step, "thought": e.thought,
                    "action": e.action, "action_params": e.action_params,
                    "observation": e.observation, "score_after": e.score_after,
                }
                for e in react_result.scratchpad
            ]
            state.iteration_log = {
                "total_iterations": react_result.total_steps,
                "improvements": react_result.improvements,
                "stop_reason": react_result.stop_reason,
                "best_before": 0.0,
                "best_after": react_result.best_result.primary_metric_value,
                "react_agent": True,
            }

            state.mark_complete("baselines")
            state.hp_search_done = True
            state.mark_complete("hp_search")
            state.mark_complete("iteration")
            save_checkpoint(state, config.task_output_dir)

    # Deterministic baselines fallback
    if not state.is_complete("baselines"):
        from co_scientist.modeling.registry import get_baseline_configs
        from co_scientist.evaluation.auto_config import auto_eval_config
        from co_scientist.evaluation.metrics import evaluate_model

        state.eval_config = auto_eval_config(state.profile)
        baseline_configs = get_baseline_configs(state.profile, seed=config.seed)
        state.trained_models = train_baselines_resilient(baseline_configs, state.split, exp_log)

        if not state.trained_models:
            return DatasetRunResult(
                dataset_path=dataset_path, success=False,
                error="All models failed to train",
            )

        state.results = []
        for trained in state.trained_models:
            result = evaluate_model(trained, state.split, state.eval_config, use_test=False)
            state.results.append(result)

        lower_is_better = state.eval_config.primary_metric in ("mse", "rmse", "mae")
        ranked = sorted(
            zip(state.results, state.trained_models),
            key=lambda pair: pair[0].primary_metric_value,
            reverse=not lower_is_better,
        )
        state.best_result, state.best_trained = ranked[0]

        from co_scientist.modeling.ensemble import build_stacking_ensemble
        is_clf = state.eval_config.task_type in ("binary_classification", "multiclass_classification")
        ensemble_trained = build_stacking_ensemble(
            trained_models=state.trained_models, split=state.split,
            task_type="classification" if is_clf else "regression", seed=config.seed,
        )
        if ensemble_trained is not None:
            ensemble_eval = evaluate_model(ensemble_trained, state.split, state.eval_config, use_test=False)
            state.trained_models.append(ensemble_trained)
            state.results.append(ensemble_eval)
            ranked = sorted(
                zip(state.results, state.trained_models),
                key=lambda pair: pair[0].primary_metric_value,
                reverse=not lower_is_better,
            )
            state.best_result, state.best_trained = ranked[0]

        state.mark_complete("baselines")
        save_checkpoint(state, config.task_output_dir)

    # HP search
    if not state.is_complete("hp_search"):
        from co_scientist.modeling.hp_search import run_hp_search

        hp_ctx = build_pipeline_context(
            config, state.profile, eval_config=state.eval_config,
            results=state.results, best_result=state.best_result,
            stage="hp_search", complexity_budget=state.complexity_budget,
            cost_remaining=cost_tracker.budget_remaining,
        )
        hp_decision = agent_hp_decision(coordinator, hp_ctx)

        if hp_decision.action != "skip_hp_search":
            hp_result = run_hp_search(
                base_config=state.best_trained.config,
                split=state.split, eval_config=state.eval_config,
                profile=state.profile, seed=config.seed,
            )
            if hp_result is not None:
                tuned_trained, tuned_result = hp_result
                lower_is_better = state.eval_config.primary_metric in ("mse", "rmse", "mae")
                improved = (
                    (tuned_result.primary_metric_value < state.best_result.primary_metric_value)
                    if lower_is_better else
                    (tuned_result.primary_metric_value > state.best_result.primary_metric_value)
                )
                if improved:
                    state.results.append(tuned_result)
                    state.trained_models.append(tuned_trained)
                    state.best_result = tuned_result
                    state.best_trained = tuned_trained

        state.hp_search_done = True
        state.mark_complete("hp_search")
        save_checkpoint(state, config.task_output_dir)

    # Iteration
    if not state.is_complete("iteration"):
        state.iteration_log = {"total_iterations": 0, "stop_reason": "batch_mode"}
        state.mark_complete("iteration")
        save_checkpoint(state, config.task_output_dir)

    # Export
    if not state.is_complete("export"):
        from co_scientist.export.exporter import export_model
        state.export_path = export_model(
            trained=state.best_trained, result=state.best_result,
            profile=state.profile, eval_config=state.eval_config,
            split=state.split, output_dir=config.task_output_dir,
            preprocessing_steps=state.preprocessed.steps_applied,
        )
        state.mark_complete("export")
        save_checkpoint(state, config.task_output_dir)

    # Report
    if not state.is_complete("report"):
        from co_scientist.report.generator import generate_report

        report_ctx = build_pipeline_context(
            config, state.profile, eval_config=state.eval_config,
            results=state.results, best_result=state.best_result,
            stage="report", complexity_budget=state.complexity_budget,
            cost_remaining=cost_tracker.budget_remaining,
        )
        bio_interpretation = agent_biology_interpretation(coordinator, report_ctx)

        state.report_path = generate_report(
            profile=state.profile, split=state.split,
            eval_config=state.eval_config, results=state.results,
            best_result=state.best_result, best_trained=state.best_trained,
            preprocessing_steps=state.preprocessed.steps_applied,
            output_dir=config.task_output_dir,
            profiling_figures=state.profiling_figs,
            preprocessing_figures=state.preprocessing_figs,
            training_figures=getattr(state, 'training_figs', []),
            seed=config.seed,
            biological_interpretation=bio_interpretation,
            react_scratchpad=getattr(state, 'react_scratchpad', None),
            iteration_log=getattr(state, 'iteration_log', None),
        )
        state.mark_complete("report")
        save_checkpoint(state, config.task_output_dir)

    return DatasetRunResult(
        dataset_path=dataset_path,
        success=True,
        best_model=state.best_result.model_name,
        best_score=state.best_result.primary_metric_value,
    )
