"""Iteration loop — multi-step agent-driven improvement cycle.

After baselines + initial HP search, this loop consults the ML Engineer
for improvement strategies and executes them. Stops when:
- Budget exhausted (max iterations reached)
- Patience exceeded (no improvement for N consecutive iterations)
- Agent recommends stopping
- LLM cost budget depleted
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rich.console import Console

from co_scientist.agents.coordinator import Coordinator
from co_scientist.agents.types import Decision, PipelineContext
from co_scientist.data.types import DatasetProfile, SplitData
from co_scientist.evaluation.metrics import evaluate_model
from co_scientist.evaluation.types import EvalConfig, ModelResult
from co_scientist.modeling.types import ModelConfig, TrainedModel

console = Console()
logger = logging.getLogger(__name__)

# ── Strategies that the iteration loop can execute ────────────────────────

EXECUTABLE_STRATEGIES = {
    "hp_tune",          # HP-tune a specific model
    "try_model",        # Train a model not yet tried
    "try_ensemble",     # Build/rebuild ensemble
    "change_features",  # Modify feature engineering (future)
    "stop",             # Stop iterating
    "no_op",            # Do nothing
}


@dataclass
class IterationResult:
    """Result of one iteration step."""

    iteration: int
    strategy: str
    model_name: str = ""
    score: float = 0.0
    improved: bool = False
    details: str = ""
    duration: float = 0.0


@dataclass
class IterationLog:
    """Full log of the iteration loop."""

    iterations: list[IterationResult] = field(default_factory=list)
    total_iterations: int = 0
    improvements: int = 0
    stop_reason: str = ""
    best_score_before: float = 0.0
    best_score_after: float = 0.0


def run_iteration_loop(
    coordinator: Coordinator,
    config: Any,  # RunConfig
    profile: DatasetProfile,
    split: SplitData,
    eval_config: EvalConfig,
    trained_models: list[TrainedModel],
    results: list[ModelResult],
    best_result: ModelResult,
    best_trained: TrainedModel,
    complexity_budget: Any,
    cost_tracker: Any,
    exp_log: Any,
    interactive: bool = False,
    seed: int = 42,
) -> tuple[list[TrainedModel], list[ModelResult], ModelResult, TrainedModel, IterationLog]:
    """Run the iteration loop.

    Returns updated (trained_models, results, best_result, best_trained, iteration_log).
    """
    from co_scientist.agents.interactive import present_decision

    max_iterations = config.budget
    patience = max(2, max_iterations // 3)  # stop after N no-improvement steps

    lower_is_better = eval_config.primary_metric in ("mse", "rmse", "mae")
    log = IterationLog(best_score_before=best_result.primary_metric_value)

    no_improve_count = 0
    decisions_so_far: list[str] = []

    # Track what we've already done
    tried_models = {r.model_name for r in results}
    tried_hp_models = set()  # models we've HP-tuned in the loop

    for iteration in range(1, max_iterations + 1):
        console.print(f"\n  [bold cyan]── Iteration {iteration}/{max_iterations} ──[/bold cyan]")

        # Build context with current state
        context = PipelineContext(
            dataset_path=config.dataset_path,
            modality=profile.modality.value,
            task_type=profile.task_type.value,
            num_samples=profile.num_samples,
            num_features=profile.num_features,
            num_classes=profile.num_classes,
            stage="iteration",
            trained_model_names=[r.model_name for r in results],
            model_scores={r.model_name: r.primary_metric_value for r in results},
            best_model_name=best_result.model_name,
            best_score=best_result.primary_metric_value,
            primary_metric=eval_config.primary_metric,
            iteration=iteration,
            decisions_so_far=decisions_so_far,
            remaining_budget=max_iterations - iteration,
            remaining_cost=cost_tracker.budget_remaining if cost_tracker else 5.0,
            complexity_level=complexity_budget.level if complexity_budget else "moderate",
            complexity_score=complexity_budget.score if complexity_budget else 5.0,
        )

        # Consult all active agents (Architecture Phase 4: multi-agent per iteration)
        # Data Analyst: data quality / preprocessing changes
        if "data_analyst" in coordinator.active_agents:
            da_dec = coordinator.consult("data_analyst", context, stage="iteration")
            if da_dec.parameters.get("issues"):
                console.print(f"  [cyan]Data Analyst:[/cyan] {da_dec.reasoning}")

        # Biology Specialist: plausibility check
        if "biology_specialist" in coordinator.active_agents:
            bio_dec = coordinator.consult("biology_specialist", context, stage="iteration")
            plausibility = bio_dec.parameters.get("plausibility", "")
            if plausibility in ("suspicious", "implausible"):
                console.print(f"  [yellow]Biology Specialist:[/yellow] {bio_dec.parameters.get('plausibility_detail', bio_dec.reasoning)}")

        # ML Engineer: what to try next (primary decision driver)
        decision = coordinator.consult("ml_engineer", context, stage="iteration")
        decision = present_decision("ml_engineer", decision, f"iteration_{iteration}", interactive=interactive, coordinator=coordinator, context=context)

        strategy = decision.action
        decisions_so_far.append(strategy)

        # Check stopping conditions
        if strategy == "stop":
            log.stop_reason = decision.parameters.get("reason", "agent_recommended_stop")
            console.print(f"  [yellow]Stopping: {decision.reasoning}[/yellow]")
            break

        if not cost_tracker.can_afford():
            log.stop_reason = "cost_budget_exhausted"
            console.print("  [yellow]Stopping: LLM cost budget exhausted[/yellow]")
            break

        # Execute the strategy
        t0 = time.time()
        iter_result = _execute_strategy(
            strategy=strategy,
            decision=decision,
            split=split,
            eval_config=eval_config,
            profile=profile,
            trained_models=trained_models,
            results=results,
            best_result=best_result,
            best_trained=best_trained,
            tried_models=tried_models,
            tried_hp_models=tried_hp_models,
            complexity_budget=complexity_budget,
            seed=seed,
            iteration=iteration,
        )
        iter_result.duration = time.time() - t0

        # Check if it improved
        if iter_result.score != 0.0:
            is_better = (
                (iter_result.score < best_result.primary_metric_value)
                if lower_is_better else
                (iter_result.score > best_result.primary_metric_value)
            )
            if is_better:
                iter_result.improved = True
                no_improve_count = 0
                log.improvements += 1
                console.print(
                    f"  [bold green]Improved![/bold green] "
                    f"{best_result.primary_metric_value:.4f} → {iter_result.score:.4f}"
                )
                # Update best
                for r, t in zip(results, trained_models):
                    if r.model_name == iter_result.model_name and r.primary_metric_value == iter_result.score:
                        best_result = r
                        best_trained = t
                        break
            else:
                no_improve_count += 1
                console.print(
                    f"  [dim]No improvement ({iter_result.score:.4f} vs "
                    f"best {best_result.primary_metric_value:.4f})[/dim]"
                )
        else:
            no_improve_count += 1
            console.print(f"  [dim]{iter_result.details}[/dim]")

        log.iterations.append(iter_result)
        log.total_iterations = iteration

        # Log to experiment log
        if exp_log:
            exp_log.log("iteration", {
                "iteration": iteration,
                "strategy": strategy,
                "model": iter_result.model_name,
                "score": iter_result.score,
                "improved": iter_result.improved,
                "details": iter_result.details,
            })

        # Patience check
        if no_improve_count >= patience:
            log.stop_reason = f"patience_exceeded ({patience} iterations without improvement)"
            console.print(f"  [yellow]Stopping: no improvement for {patience} consecutive iterations[/yellow]")
            break
    else:
        log.stop_reason = "budget_exhausted"
        console.print(f"  [yellow]Iteration budget ({max_iterations}) exhausted[/yellow]")

    log.best_score_after = best_result.primary_metric_value

    # Summary
    if log.total_iterations > 0:
        console.print(
            f"\n  Iteration loop: {log.total_iterations} step(s), "
            f"{log.improvements} improvement(s). "
            f"Score: {log.best_score_before:.4f} → {log.best_score_after:.4f} "
            f"({log.stop_reason})"
        )

    return trained_models, results, best_result, best_trained, log


def _execute_strategy(
    strategy: str,
    decision: Decision,
    split: SplitData,
    eval_config: EvalConfig,
    profile: DatasetProfile,
    trained_models: list[TrainedModel],
    results: list[ModelResult],
    best_result: ModelResult,
    best_trained: TrainedModel,
    tried_models: set[str],
    tried_hp_models: set[str],
    complexity_budget: Any,
    seed: int,
    iteration: int,
) -> IterationResult:
    """Execute a single iteration strategy. Returns the result."""
    params = decision.parameters
    ir = IterationResult(iteration=iteration, strategy=strategy)

    try:
        if strategy == "hp_tune":
            return _strategy_hp_tune(
                params, split, eval_config, profile, best_result, best_trained,
                trained_models, results, tried_hp_models, complexity_budget, seed, iteration,
            )

        elif strategy == "try_model":
            return _strategy_try_model(
                params, split, eval_config, profile, trained_models, results,
                tried_models, seed, iteration,
            )

        elif strategy == "try_ensemble":
            return _strategy_try_ensemble(
                split, eval_config, trained_models, results, seed, iteration,
            )

        else:
            ir.details = f"Unknown strategy: {strategy}"
            return ir

    except Exception as e:
        logger.warning("Iteration %d strategy '%s' failed: %s", iteration, strategy, e)
        ir.details = f"Failed: {str(e)[:200]}"
        return ir


def _strategy_hp_tune(
    params: dict,
    split: SplitData,
    eval_config: EvalConfig,
    profile: DatasetProfile,
    best_result: ModelResult,
    best_trained: TrainedModel,
    trained_models: list[TrainedModel],
    results: list[ModelResult],
    tried_hp_models: set[str],
    complexity_budget: Any,
    seed: int,
    iteration: int,
) -> IterationResult:
    """HP-tune a model."""
    from co_scientist.modeling.hp_search import run_hp_search

    ir = IterationResult(iteration=iteration, strategy="hp_tune")

    # Determine which model to tune
    target_model = params.get("target_model", best_result.model_name)

    # Find the base config for this model
    base_config = best_trained.config
    for t in trained_models:
        if t.config.name == target_model or t.config.model_type == target_model:
            base_config = t.config
            break

    # Skip if we already HP-tuned this model in the loop
    if base_config.model_type in tried_hp_models:
        ir.details = f"Already HP-tuned {base_config.model_type} in this loop"
        return ir

    n_trials = params.get("n_trials", 15)
    timeout = complexity_budget.hp_timeout if complexity_budget else 120

    console.print(f"  HP-tuning {base_config.name} ({n_trials} trials, {timeout}s timeout)")

    result = run_hp_search(
        base_config=base_config,
        split=split,
        eval_config=eval_config,
        profile=profile,
        seed=seed + iteration,
        n_trials_override=n_trials,
        timeout_override=timeout,
    )

    tried_hp_models.add(base_config.model_type)

    if result is None:
        ir.details = f"HP search returned no result for {base_config.name}"
        return ir

    tuned_trained, tuned_result = result
    trained_models.append(tuned_trained)
    results.append(tuned_result)

    ir.model_name = tuned_result.model_name
    ir.score = tuned_result.primary_metric_value
    ir.details = f"HP-tuned {base_config.model_type}"
    return ir


def _strategy_try_model(
    params: dict,
    split: SplitData,
    eval_config: EvalConfig,
    profile: DatasetProfile,
    trained_models: list[TrainedModel],
    results: list[ModelResult],
    tried_models: set[str],
    seed: int,
    iteration: int,
) -> IterationResult:
    """Train a model not yet tried."""
    from co_scientist.modeling.registry import build_model
    from co_scientist.modeling.trainer import train_model

    ir = IterationResult(iteration=iteration, strategy="try_model")

    target_type = params.get("target_model", params.get("model_type", ""))
    if not target_type:
        ir.details = "No target model specified"
        return ir

    # Check if already tried
    already_trained = any(
        t.config.model_type == target_type or t.config.name == target_type
        for t in trained_models
    )
    if already_trained:
        ir.details = f"Model {target_type} already trained"
        return ir

    # Build and train
    task = "classification" if "classification" in eval_config.task_type else "regression"
    model_config = ModelConfig(
        name=f"{target_type}_iter{iteration}",
        tier="iteration",
        model_type=target_type,
        task_type=task,
    )

    console.print(f"  Training {model_config.name}...")
    try:
        model = build_model(model_config)
    except Exception as e:
        ir.details = f"Could not build model {target_type}: {e}"
        return ir

    trained = train_model(model, model_config, split, seed=seed + iteration)
    if trained is None:
        ir.details = f"Training failed for {target_type}"
        return ir

    result = evaluate_model(trained, split, eval_config, use_test=False)
    trained_models.append(trained)
    results.append(result)
    tried_models.add(target_type)

    ir.model_name = result.model_name
    ir.score = result.primary_metric_value
    ir.details = f"Trained {target_type}"
    return ir


def _strategy_try_ensemble(
    split: SplitData,
    eval_config: EvalConfig,
    trained_models: list[TrainedModel],
    results: list[ModelResult],
    seed: int,
    iteration: int,
) -> IterationResult:
    """Build or rebuild a stacking ensemble."""
    from co_scientist.modeling.ensemble import build_stacking_ensemble

    ir = IterationResult(iteration=iteration, strategy="try_ensemble")

    is_clf = eval_config.task_type in ("binary_classification", "multiclass_classification")
    ensemble = build_stacking_ensemble(
        trained_models=trained_models,
        split=split,
        task_type="classification" if is_clf else "regression",
        seed=seed + iteration,
    )

    if ensemble is None:
        ir.details = "Ensemble build failed (need ≥2 base models)"
        return ir

    result = evaluate_model(ensemble, split, eval_config, use_test=False)
    trained_models.append(ensemble)
    results.append(result)

    ir.model_name = result.model_name
    ir.score = result.primary_metric_value
    ir.details = f"Stacking ensemble from {len(trained_models) - 1} base models"
    return ir
