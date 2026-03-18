"""CLI entry point using Typer."""

import logging
import os
from typing import Optional

import typer
from rich.console import Console

# ── Prevent OpenMP segfaults on macOS ────────────────────────────────────────
# Multiple libraries (torch, sklearn, xgboost, lightgbm) each ship their own
# libomp.dylib. When loaded together they conflict and cause segfaults.
# KMP_DUPLICATE_LIB_OK=TRUE tells Intel's OpenMP to tolerate duplicates.
# OMP_NUM_THREADS=1 avoids parallel OMP issues entirely (sklearn/xgb use their
# own thread pools anyway).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # suppress HF warning
# ─────────────────────────────────────────────────────────────────────────────

from co_scientist import __version__
from co_scientist.config import RunConfig, PipelineDeadline, DEFAULT_BUDGET, DEFAULT_MAX_COST, DEFAULT_MODE, DEFAULT_TIMEOUT

app = typer.Typer(
    name="co-scientist",
    help="AIDO Co-Scientist — automated ML model building for biological datasets.",
    add_completion=False,
)
console = Console()
logger = logging.getLogger(__name__)


def version_callback(value: bool) -> None:
    if value:
        console.print(f"co-scientist {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


def _resolve_api_key(cli_key: str | None = None) -> str:
    """Resolve the Anthropic API key from CLI flag, config file, or environment."""
    import os
    from pathlib import Path

    # 1. CLI flag takes precedence
    if cli_key:
        return cli_key

    # 2. Config file: config.yaml in project root or ~/.co-scientist/config.yaml
    try:
        import yaml

        config_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path.home() / ".co-scientist" / "config.yaml",
        ]
        for cp in config_paths:
            if cp.exists():
                try:
                    with open(cp) as f:
                        cfg = yaml.safe_load(f) or {}
                    key = cfg.get("anthropic_api_key") or cfg.get("ANTHROPIC_API_KEY", "")
                    if key:
                        return str(key)
                except Exception:
                    pass
    except ImportError:
        pass

    # 3. Environment variable
    return os.environ.get("ANTHROPIC_API_KEY", "")


def _llm_recover_dataset_load(dataset_path: str, error_msg: str, coordinator, console) -> "LoadedDataset":
    """Use LLM to diagnose and fix a dataset loading failure."""
    from co_scientist.data.loader import load_dataset, resolve_dataset_path, _discover_hf_configs

    # Ask the LLM to figure out the correct loading strategy
    prompt = (
        f"A dataset loading attempt failed.\n"
        f"Dataset path: {dataset_path}\n"
        f"Error: {error_msg}\n\n"
        f"Based on the error, suggest the correct way to load this dataset. "
        f"Return ONLY the corrected dataset path in the format 'repo:subset' or 'modality/task'. "
        f"If the error mentions available configs, pick the most likely match."
    )
    try:
        response = coordinator.client.ask_text(prompt, system="You are a helpful data engineering assistant. Respond with just the corrected dataset path, nothing else.")
        suggested_path = response.strip().strip("'\"` ")
        console.print(f"  [green]LLM suggested: '{suggested_path}'[/green]")
        return load_dataset(suggested_path)
    except Exception as e2:
        console.print(f"  [red]LLM recovery also failed: {e2}[/red]")

    # Last resort: if the original path looks like a HF repo, try listing configs
    try:
        repo, subset, fmt, task_name = resolve_dataset_path(dataset_path)
        configs = _discover_hf_configs(repo)
        if configs:
            console.print(f"  [yellow]Available configs for {repo}: {configs}[/yellow]")
            # Try the first config as a last resort
            return load_dataset(f"{repo}:{configs[0]}")
    except Exception:
        pass

    raise ValueError(
        f"Could not load dataset '{dataset_path}' after recovery attempts. "
        f"Original error: {error_msg}"
    )


def _llm_recover_preprocess(dataset, profile, error_msg: str, coordinator, console) -> "PreprocessingResult":
    """Use LLM to diagnose preprocessing failure and fall back to basic preprocessing."""
    from co_scientist.data.preprocess import PreprocessingResult
    import numpy as np

    # Ask LLM for diagnosis
    prompt = (
        f"Preprocessing failed for a {profile.modality.value} dataset.\n"
        f"Shape: {dataset.X.shape if hasattr(dataset.X, 'shape') else 'unknown'}\n"
        f"Columns: {list(dataset.X.columns)[:20] if hasattr(dataset.X, 'columns') else 'N/A'}\n"
        f"Error: {error_msg}\n\n"
        f"What went wrong and how should we handle this data? "
        f"Suggest a simple preprocessing approach."
    )
    try:
        response = coordinator.client.ask_text(prompt, system="You are a data preprocessing expert. Be concise.")
        console.print(f"  [dim]LLM diagnosis: {response[:200]}[/dim]")
    except Exception:
        pass

    # Fallback: basic numeric preprocessing
    console.print("  [yellow]Falling back to basic numeric preprocessing...[/yellow]")
    import pandas as pd
    X = dataset.X
    y = dataset.y

    if isinstance(X, pd.DataFrame):
        # Keep only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Handle sequence columns: if there's a sequence column, do basic k-mer
        seq_cols = [c for c in X.columns if c.lower() in ("sequences", "sequence", "seq")]
        if seq_cols and not numeric_cols:
            # Basic k-mer extraction as fallback
            from co_scientist.data.preprocess import _preprocess_sequence
            try:
                return _preprocess_sequence(dataset, profile)
            except Exception:
                pass

        if numeric_cols:
            X_array = X[numeric_cols].fillna(0).values.astype(np.float32)
            feature_names = numeric_cols
        else:
            # Encode everything as numeric
            X_encoded = pd.get_dummies(X, drop_first=True)
            X_array = X_encoded.fillna(0).values.astype(np.float32)
            feature_names = X_encoded.columns.tolist()
    else:
        X_array = np.array(X, dtype=np.float32)
        X_array = np.nan_to_num(X_array, 0.0)
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    # Encode labels if needed
    from sklearn.preprocessing import LabelEncoder
    label_encoder = None
    y_array = np.array(y)
    if y_array.dtype == object or str(y_array.dtype) == "category":
        label_encoder = LabelEncoder()
        y_array = label_encoder.fit_transform(y_array)

    return PreprocessingResult(
        X=X_array,
        y=y_array,
        feature_names=feature_names,
        steps_applied=["fallback_basic_preprocessing", "nan_fill_zero"],
        label_encoder=label_encoder,
        raw_sequences=None,
    )


@app.command()
def run(
    dataset_path: str = typer.Argument(..., help="Dataset path, e.g. RNA/translation_efficiency_muscle"),
    mode: str = typer.Option(DEFAULT_MODE, "--mode", "-m", help="Run mode: auto or interactive."),
    budget: int = typer.Option(DEFAULT_BUDGET, "--budget", "-b", help="Max iteration steps."),
    max_cost: float = typer.Option(DEFAULT_MAX_COST, "--max-cost", help="Max LLM cost in dollars."),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", help="Output directory."),
    no_search: bool = typer.Option(False, "--no-search", help="Disable web/paper search."),
    resume: bool = typer.Option(False, "--resume", help="Resume an interrupted run."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    timeout: int = typer.Option(DEFAULT_TIMEOUT, "--timeout", "-t", help="Pipeline timeout in seconds (default: 1800 = 30 min)."),
    tree_search: bool = typer.Option(False, "--tree-search", help="Use MCTS-inspired tree search instead of linear ReAct."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key (or set ANTHROPIC_API_KEY env var, or put in config.yaml)."),
) -> None:
    """Run the co-scientist pipeline on a dataset."""
    from pathlib import Path

    import time as _time

    from co_scientist.checkpoint import PipelineState, load_checkpoint, save_checkpoint
    from co_scientist.complexity import compute_complexity, print_complexity
    from co_scientist.dashboard import (
        print_header, print_step_start, print_step_resumed,
        print_model_table, print_final_summary,
    )
    from co_scientist.experiment_log import ExperimentLog
    from co_scientist.guardrails import (
        check_pre_training, check_post_training, check_pipeline_summary,
        check_metric_sanity, check_model_data_compatibility, verify_task_type,
        print_alerts, has_blocking_errors,
    )
    from co_scientist.resilience import train_baselines_resilient, run_step_resilient

    # Agent framework
    from co_scientist.agents import Coordinator
    from co_scientist.agents.analysis import (
        build_pipeline_context, agent_hp_decision,
        agent_post_training_analysis, agent_biology_interpretation,
        agent_feature_interpretation, agent_research,
    )
    from co_scientist.agents.interactive import present_decision, present_results_analysis, confirm_step
    from co_scientist.llm.cost import CostTracker

    pipeline_start_time = _time.time()
    total_warnings = 0
    interactive = mode == "interactive"

    # If resuming, find the latest existing run directory for this dataset
    resume_timestamp = None
    if resume:
        sanitized = dataset_path.replace("/", "__")
        existing = sorted(Path(output_dir).glob(f"{sanitized}_*"), reverse=True)
        if existing:
            # Extract timestamp from the most recent directory
            resume_timestamp = existing[0].name.replace(f"{sanitized}_", "")

    config = RunConfig(
        dataset_path=dataset_path,
        mode=mode,
        budget=budget,
        max_cost=max_cost,
        timeout=timeout,
        output_dir=Path(output_dir),
        no_search=no_search,
        resume=resume,
        seed=seed,
        **({"run_timestamp": resume_timestamp} if resume_timestamp else {}),
    )

    # Global deadline — ensures pipeline finishes within --timeout
    deadline = PipelineDeadline(timeout_seconds=config.timeout)
    console.print(f"  [dim]Pipeline timeout: {config.timeout}s ({config.timeout // 60} min)[/dim]")

    print_header(
        version=__version__,
        dataset_path=config.dataset_path,
        mode=config.mode,
        budget=config.budget,
        max_cost=config.max_cost,
        output_dir=str(config.task_output_dir),
    )

    # Initialize experiment log
    exp_log = ExperimentLog(config.task_output_dir)
    exp_log.log("pipeline_start", {
        "dataset_path": config.dataset_path,
        "mode": config.mode,
        "budget": config.budget,
        "seed": config.seed,
        "version": __version__,
    })

    # Initialize cross-run memory
    from co_scientist.memory import RunMemory
    run_memory = RunMemory(config.output_dir)

    # Initialize live dashboard (auto mode only, not interactive)
    from co_scientist.live_dashboard import LiveDashboard
    live_dash: LiveDashboard | None = None
    if not interactive:
        live_dash = LiveDashboard(
            dataset_path=config.dataset_path,
            mode=config.mode,
            budget=config.budget,
            max_cost=config.max_cost,
            version=__version__,
            output_dir=config.task_output_dir,
        )

    # Initialize agent coordinator (graceful — works without API key)
    import os as _os
    resolved_api_key = _resolve_api_key(api_key)
    if resolved_api_key:
        if api_key:
            _key_src = "--api-key flag"
        elif _os.environ.get("ANTHROPIC_API_KEY"):
            _key_src = "ANTHROPIC_API_KEY env var"
        else:
            _key_src = "config.yaml"
        console.print(f"  [dim]API key: found via {_key_src}[/dim]")
    else:
        console.print("  [yellow]API key: not found — running in deterministic mode[/yellow]")
        console.print("  [dim]Set via: config.yaml | ANTHROPIC_API_KEY env var | --api-key flag[/dim]")
    cost_tracker = CostTracker(max_cost=config.max_cost)
    coordinator = Coordinator(cost_tracker=cost_tracker, no_search=config.no_search, api_key=resolved_api_key or None)
    coordinator.memory = run_memory
    coordinator.live_dashboard = live_dash
    # Wrap print_step_start to also update live dashboard
    _orig_print_step_start = print_step_start
    def print_step_start(step_key, step_num, total):
        _orig_print_step_start(step_key, step_num, total)
        if live_dash:
            live_dash.set_step(step_key, step_num, total)
            live_dash.set_cost(cost_tracker.total_cost)

    def _update_dashboard_cost():
        """Push current LLM cost to the live dashboard."""
        if live_dash:
            live_dash.set_cost(cost_tracker.total_cost)

    if coordinator.llm_available:
        console.print("  [dim]Agent framework: LLM-powered decisions enabled[/dim]")
    else:
        console.print("  [dim]Agent framework: deterministic mode (no API key)[/dim]")
    if interactive:
        console.print("  [bold cyan]Interactive mode: you will be prompted at decision points[/bold cyan]")

    # Try to resume from checkpoint
    state: PipelineState | None = None
    if config.resume:
        state = load_checkpoint(config.task_output_dir)

    if state is None:
        state = PipelineState()

    # Start live dashboard
    if live_dash:
        live_dash.start()

    # ── Step 1: Load and profile ─────────────────────────────────────────
    if not state.is_complete("load_profile"):
        from co_scientist.data.loader import load_dataset
        from co_scientist.data.profile import profile_dataset, print_profile

        exp_log.log_step_start("load_profile")
        print_step_start("load_profile", 1, 7)
        try:
            state.dataset = load_dataset(config.dataset_path)
        except Exception as load_err:
            console.print(f"  [bold red]Dataset loading failed:[/bold red] {load_err}")
            # Try LLM-assisted recovery if available
            if coordinator.llm_available:
                console.print("  [yellow]Attempting LLM-assisted recovery...[/yellow]")
                state.dataset = _llm_recover_dataset_load(
                    config.dataset_path, str(load_err), coordinator, console
                )
            else:
                raise
        console.print(f"  Loaded {state.dataset.info.num_raw_samples:,} samples.")

        try:
            state.profile = profile_dataset(state.dataset, config.dataset_path)
        except Exception as profile_err:
            console.print(f"  [yellow]Profiling error: {profile_err}. Using fallback profiling.[/yellow]")
            from co_scientist.data.profile import fallback_profile
            state.profile = fallback_profile(state.dataset, config.dataset_path)
        print_profile(state.profile)

        # Complexity scoring (from Google AI Co-Scientist, Architecture Section 4.2)
        state.complexity_budget = compute_complexity(state.profile)
        print_complexity(state.complexity_budget)
        exp_log.log("complexity", state.complexity_budget.summary())

        # Update coordinator complexity for agent activation
        coordinator.complexity_level = state.complexity_budget.level
        coordinator.active_agents = coordinator._activate_agents(state.complexity_budget.level)

        # Task type verification (multi-signal, from Architecture Section 10.3)
        task_alerts = verify_task_type(state.profile)
        if task_alerts:
            print_alerts(task_alerts, "Task type verification")
            for alert in task_alerts:
                exp_log.log("guardrail", alert.to_dict())

        # Interactive: confirm task detection
        if interactive:
            from co_scientist.agents.types import PipelineContext
            profile_ctx = PipelineContext(
                dataset_path=state.profile.dataset_path,
                modality=state.profile.modality.value,
                task_type=state.profile.task_type.value,
                num_samples=state.profile.num_samples,
                num_features=state.profile.num_features,
                num_classes=state.profile.num_classes,
                target_column=state.profile.target_column,
                class_distribution=state.profile.class_distribution,
                target_stats=state.profile.target_stats,
                split_info=state.profile.split_info,
                missing_value_pct=state.profile.missing_value_pct,
                feature_sparsity=state.profile.feature_sparsity,
                sequence_length_stats=state.profile.sequence_length_stats,
                detected_issues=state.profile.detected_issues,
                stage="profiling",
            )
            if not confirm_step(
                "data profiling",
                f"Detected {state.profile.modality.value} / {state.profile.task_type.value}",
                interactive=True,
                coordinator=coordinator,
                context=profile_ctx,
            ):
                console.print("[yellow]User cancelled pipeline.[/yellow]")
                raise typer.Exit(code=0)

        # Profiling figures
        from co_scientist.viz.profiling import generate_profiling_figures
        state.profiling_figs = generate_profiling_figures(
            state.dataset, state.profile, config.task_output_dir,
        )
        console.print(f"  Saved {len(state.profiling_figs)} profiling figure(s)")

        # Research: search for relevant papers and benchmarks
        if not config.no_search:
            memory_ctx = run_memory.format_for_prompt(
                state.profile.modality.value, state.profile.task_type.value,
            )
            research_ctx = build_pipeline_context(
                config, state.profile, stage="profiling",
                complexity_budget=state.complexity_budget,
                cost_remaining=cost_tracker.budget_remaining,
                memory_context=memory_ctx,
            )
            research_report = agent_research(coordinator, research_ctx, stage="profiling")
            _update_dashboard_cost()
            if research_report.papers:
                state.research_results = research_report.to_dict()
                console.print(f"  Found [bold]{len(research_report.papers)}[/bold] relevant paper(s) "
                              f"via Semantic Scholar + PubMed")
                if research_report.methods_found:
                    console.print(f"    Methods in literature: {', '.join(research_report.methods_found[:5])}")
                exp_log.log("research", {
                    "num_papers": len(research_report.papers),
                    "methods": research_report.methods_found,
                    "benchmarks": research_report.benchmarks_found,
                    "query": research_report.query_used,
                })
            else:
                console.print("  [dim]No research results found (search may be rate-limited)[/dim]")

        # Biology Specialist: validate metric choice and suggest domain features
        # (Architecture Section 5, Phase 1: UNDERSTAND)
        if "biology_specialist" in coordinator.active_agents:
            bio_profile_ctx = build_pipeline_context(
                config, state.profile, stage="profiling",
                complexity_budget=state.complexity_budget,
                cost_remaining=cost_tracker.budget_remaining,
            )
            bio_decision = coordinator.consult("biology_specialist", bio_profile_ctx, stage="profiling")
            _update_dashboard_cost()

            # Report metric validation
            metric_valid = bio_decision.parameters.get("metric_appropriate", True)
            metric_note = bio_decision.parameters.get("metric_note", "")
            if not metric_valid and metric_note:
                console.print(f"  [yellow]Biology Specialist: {metric_note}[/yellow]")
            elif metric_note:
                console.print(f"  [dim]Biology: {metric_note}[/dim]")

            # Report suggested features
            suggested = bio_decision.parameters.get("suggested_features", [])
            if suggested:
                console.print(f"  [dim]Suggested domain features: {', '.join(suggested[:5])}[/dim]")

            exp_log.log("agent_decision", {
                "agent": "biology_specialist",
                "stage": "profiling",
                "action": bio_decision.action,
                "reasoning": bio_decision.reasoning,
            })

            # Store biology assessment for report
            state.biology_assessment = bio_decision.parameters

        # Validate + auto-fix loaded data and profile
        from co_scientist.validation import validate_and_fix_loaded_data, validate_and_fix_profile
        if live_dash:
            live_dash.set_agent_name("Validation Agent")
            live_dash.set_validation_running("validate_data")
        state.dataset, v_data = validate_and_fix_loaded_data(state.dataset, state.profile)
        if live_dash:
            live_dash.update_validation_result("validate_data", v_data.passed, v_data.issues, v_data.fixes_applied)
            live_dash.set_validation_running("validate_profile")
        state.profile, v_prof = validate_and_fix_profile(state.profile, state.dataset)
        if live_dash:
            live_dash.update_validation_result("validate_profile", v_prof.passed, v_prof.issues, v_prof.fixes_applied)
        if not v_data.passed:
            exp_log.log("validation_failure", {"step": "data_loading", "issues": v_data.issues})
        if not v_prof.passed:
            exp_log.log("validation_failure", {"step": "profiling", "issues": v_prof.issues})
        if v_data.fixes_applied:
            exp_log.log("validation_fix", {"step": "data_loading", "fixes": v_data.fixes_applied})
        if v_prof.fixes_applied:
            exp_log.log("validation_fix", {"step": "profiling", "fixes": v_prof.fixes_applied})

        state.mark_complete("load_profile")
        save_checkpoint(state, config.task_output_dir)
        exp_log.log_step_end("load_profile", {
            "num_samples": state.profile.num_samples,
            "modality": state.profile.modality.value,
            "task_type": state.profile.task_type.value,
        })
    else:
        print_step_resumed("load_profile", 1, 7)

    # ── Step 2: Preprocess and split ─────────────────────────────────────
    if not state.is_complete("preprocess_split"):
        from co_scientist.data.preprocess import preprocess
        from co_scientist.data.split import split_dataset

        exp_log.log_step_start("preprocess_split")
        print_step_start("preprocess_split", 2, 7)

        # Agent debate: preprocessing strategy (high-stakes — wrong preprocessing ruins everything)
        preprocess_ctx = build_pipeline_context(
            config, state.profile, stage="preprocessing",
            complexity_budget=state.complexity_budget,
            cost_remaining=cost_tracker.budget_remaining,
        )
        da_decision = coordinator.debate(
            "preprocessing strategy", preprocess_ctx,
            agent_names=["data_analyst", "ml_engineer"],
        )
        _update_dashboard_cost()
        exp_log.log("agent_decision", {
            "agent": "data_analyst",
            "stage": "preprocessing",
            "action": da_decision.action,
            "reasoning": da_decision.reasoning,
        })
        present_decision("data_analyst", da_decision, "preprocessing", interactive=interactive, coordinator=coordinator, context=preprocess_ctx)

        try:
            state.preprocessed = preprocess(state.dataset, state.profile)
        except Exception as preprocess_err:
            console.print(f"  [bold red]Preprocessing failed:[/bold red] {preprocess_err}")
            if coordinator.llm_available:
                console.print("  [yellow]Attempting LLM-assisted recovery...[/yellow]")
                state.preprocessed = _llm_recover_preprocess(
                    state.dataset, state.profile, str(preprocess_err), coordinator, console
                )
            else:
                raise
        for step in state.preprocessed.steps_applied:
            console.print(f"    - {step}")

        try:
            state.split = split_dataset(state.dataset, state.preprocessed, state.profile, seed=config.seed)
        except Exception as split_err:
            console.print(f"  [bold red]Splitting failed:[/bold red] {split_err}")
            console.print("  [yellow]Falling back to random 70/15/15 split...[/yellow]")
            from co_scientist.data.split import _split_random
            state.split = _split_random(state.preprocessed, state.profile, seed=config.seed)

        # Validate + auto-fix preprocessing and split
        from co_scientist.validation import validate_and_fix_preprocessing, validate_and_fix_split
        if live_dash:
            live_dash.set_agent_name("Validation Agent")
            live_dash.set_validation_running("validate_preprocess")
        state.preprocessed, v_preprocess = validate_and_fix_preprocessing(state.preprocessed, state.profile)
        if live_dash:
            live_dash.update_validation_result("validate_preprocess", v_preprocess.passed, v_preprocess.issues, v_preprocess.fixes_applied)
            live_dash.set_validation_running("validate_split")
        state.split, v_split = validate_and_fix_split(state.split, state.preprocessed, state.profile, seed=config.seed)
        if live_dash:
            live_dash.update_validation_result("validate_split", v_split.passed, v_split.issues, v_split.fixes_applied)
        if not v_preprocess.passed:
            exp_log.log("validation_failure", {"step": "preprocessing", "issues": v_preprocess.issues})
        if not v_split.passed:
            exp_log.log("validation_failure", {"step": "splitting", "issues": v_split.issues})
        if v_preprocess.fixes_applied:
            exp_log.log("validation_fix", {"step": "preprocessing", "fixes": v_preprocess.fixes_applied})
        if v_split.fixes_applied:
            exp_log.log("validation_fix", {"step": "splitting", "fixes": v_split.fixes_applied})

        # Preprocessing figures
        from co_scientist.viz.preprocessing import generate_preprocessing_figures
        state.preprocessing_figs = generate_preprocessing_figures(
            state.split, state.profile, config.task_output_dir,
        )
        console.print(f"  Saved {len(state.preprocessing_figs)} preprocessing figure(s)")

        state.mark_complete("preprocess_split")
        save_checkpoint(state, config.task_output_dir)
        exp_log.log_step_end("preprocess_split", {
            "n_features": state.split.X_train.shape[1],
            "split_sizes": state.split.summary(),
            "preprocessing_steps": state.preprocessed.steps_applied,
        })
    else:
        print_step_resumed("preprocess_split", 2, 7)

    # ── Deadline check after Step 2 ─────────────────────────────────────
    if deadline.expired():
        console.print(f"  [bold yellow]Pipeline deadline reached ({config.timeout}s) — skipping to export/report[/bold yellow]")

    # ── Pre-training guardrails ──────────────────────────────────────────
    if not deadline.expired() and state.is_complete("preprocess_split") and not state.is_complete("baselines"):
        pre_alerts = check_pre_training(state.profile, state.split)
        print_alerts(pre_alerts, "Pre-training checks")
        if has_blocking_errors(pre_alerts):
            exp_log.log_error("pre_training_guardrails", "Blocking errors detected")
            console.print("[bold red]Pipeline halted due to data quality errors.[/bold red]")
            raise typer.Exit(code=1)

    # ── ReAct Agent Path (replaces Steps 3-5 when LLM is available) ────
    react_used = False
    _llm_ok = coordinator.llm_available
    _bl_done = state.is_complete("baselines")
    _it_done = state.is_complete("iteration")
    console.print(f"  [dim]ReAct check: llm={_llm_ok}, baselines_done={_bl_done}, iteration_done={_it_done}[/dim]")
    if deadline.expired():
        console.print("  [bold yellow]Deadline reached — skipping modeling phase[/bold yellow]")
    elif _llm_ok and not _bl_done and not _it_done:
        from co_scientist.evaluation.auto_config import auto_eval_config

        print_step_start("baselines", 3, 7)  # Update dashboard step counter
        console.print("\n  [bold cyan]── ReAct Agent: Dynamic Model Selection & Training ──[/bold cyan]")
        console.print("  [dim]The agent will dynamically choose which models to train, tune, and ensemble[/dim]")
        console.print("  [dim]based on dataset characteristics and observed results.[/dim]")
        if state.eval_config is None:
            state.eval_config = auto_eval_config(state.profile)
            console.print(f"  Primary metric: [bold]{state.eval_config.primary_metric}[/bold]")

        # Pre-ReAct debate: agents discuss modeling strategy before the loop starts
        if coordinator.cost_tracker.can_afford(min_calls=5):
            debate_ctx = build_pipeline_context(
                config, state.profile, eval_config=state.eval_config,
                stage="model_selection",
                complexity_budget=state.complexity_budget,
                cost_remaining=cost_tracker.budget_remaining,
            )
            strategy_decision = coordinator.debate(
                "modeling strategy", debate_ctx,
                agent_names=["ml_engineer", "data_analyst"],
            )
            _update_dashboard_cost()
            console.print(f"  [dim]Strategy debate winner: {strategy_decision.action} — {strategy_decision.reasoning[:120]}[/dim]")
            exp_log.log("agent_decision", {
                "agent": "debate",
                "stage": "pre_react_strategy",
                "action": strategy_decision.action,
                "reasoning": strategy_decision.reasoning,
            })

        # Reserve budget for post-ReAct debates (model_selection + hp_search = ~10 LLM calls)
        # Each debate costs ~5 calls × ~$0.01 = ~$0.05, reserve for 2 debates
        cost_tracker.reserve(0.10)
        console.print(f"  [dim]Reserved $0.10 for post-ReAct debates[/dim]")

        # Budget 60% of remaining time for the ReAct loop (rest for export + report)
        react_wall_seconds = min(deadline.budget_for_step("react", fraction=0.6), 900.0)
        console.print(f"  [dim]ReAct time budget: {react_wall_seconds:.0f}s ({react_wall_seconds / 60:.0f} min)[/dim]")

        react_result = coordinator.run_react_modeling(
            profile=state.profile,
            split=state.split,
            eval_config=state.eval_config,
            seed=config.seed,
            exp_log=exp_log,
            max_steps=min(25, max(config.budget + 10, 15)),
            patience=8,
            tree_search=tree_search,
            max_wall_seconds=react_wall_seconds,
            interactive=interactive,
        )

        # Release debate budget reservation now that ReAct is done
        cost_tracker.release_reserve()

        if react_result is not None:
            react_used = True
            state.trained_models = react_result.trained_models
            state.results = react_result.results
            state.best_result = react_result.best_result
            state.best_trained = react_result.best_trained
            state.react_scratchpad = [
                {
                    "step": e.step,
                    "thought": e.thought,
                    "action": e.action,
                    "action_params": e.action_params,
                    "observation": e.observation,
                    "score_after": e.score_after,
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
            state.tree_search_log = react_result.tree_search_log
            state.elo_rankings = react_result.elo_rankings
            # Use ReAct biology assessment if available (richer, post-training)
            if react_result.biology_assessment:
                state.biology_assessment = react_result.biology_assessment

            # Training figures (non-critical)
            from co_scientist.evaluation.metrics import evaluate_model as _eval_model
            def _gen_training_figs_react():
                from co_scientist.viz.training import generate_training_figures
                return generate_training_figures(
                    state.results, state.trained_models, state.split,
                    state.eval_config, state.profile, config.task_output_dir,
                )
            state.training_figs = run_step_resilient("training_figures", _gen_training_figs_react, exp_log) or []
            if state.training_figs:
                console.print(f"  Saved {len(state.training_figs)} training figure(s)")

            # Mark all modeling steps complete
            state.mark_complete("baselines")
            state.hp_search_done = True
            state.mark_complete("hp_search")
            state.mark_complete("iteration")
            save_checkpoint(state, config.task_output_dir)

            exp_log.log_step_end("baselines", {
                "n_models": len(state.results),
                "best_model": state.best_result.model_name,
                "best_value": state.best_result.primary_metric_value,
                "react_agent": True,
            })
            exp_log.log_step_end("iteration", state.iteration_log)

            from co_scientist.dashboard import print_model_table as _pmt
            _pmt(state.results, state.eval_config)
            console.print(
                f"  Best model: [bold green]{state.best_result.model_name}[/bold green] "
                f"({state.eval_config.primary_metric}={state.best_result.primary_metric_value:.4f})"
            )
        else:
            console.print("  [yellow]ReAct agent returned None — falling back to deterministic path[/yellow]")

    # ── Step 3: Baselines + Evaluation (deterministic fallback) ────────
    if deadline.expired() and not state.is_complete("baselines"):
        console.print("  [bold yellow]Deadline reached — skipping baselines[/bold yellow]")
    if not deadline.expired() and not state.is_complete("baselines"):
        from co_scientist.modeling.registry import get_baseline_configs
        from co_scientist.evaluation.auto_config import auto_eval_config
        from co_scientist.evaluation.metrics import evaluate_model

        exp_log.log_step_start("baselines")
        print_step_start("baselines", 3, 7)
        state.eval_config = auto_eval_config(state.profile)
        console.print(f"  Primary metric: [bold]{state.eval_config.primary_metric}[/bold]")
        if live_dash:
            lower = state.eval_config.primary_metric in ("mse", "rmse", "mae")
            live_dash.set_metric(state.eval_config.primary_metric, lower_is_better=lower)

        # Agent consultation: model selection — drives model priority order
        model_ctx = build_pipeline_context(
            config, state.profile, eval_config=state.eval_config,
            stage="model_selection",
            complexity_budget=state.complexity_budget,
            cost_remaining=cost_tracker.budget_remaining,
        )
        ml_decision = coordinator.debate(
            "model selection strategy", model_ctx,
            agent_names=["ml_engineer", "data_analyst"],
        )
        exp_log.log("agent_decision", {
            "agent": "ml_engineer",
            "stage": "model_selection",
            "action": ml_decision.action,
            "confidence": ml_decision.confidence,
            "reasoning": ml_decision.reasoning,
        })
        ml_decision = present_decision("ml_engineer", ml_decision, "model_selection", interactive=interactive, coordinator=coordinator, context=model_ctx)

        # Metric sanity check (Section 10.3)
        metric_alerts = check_metric_sanity(state.profile, state.eval_config)
        if metric_alerts:
            print_alerts(metric_alerts, "Metric sanity")
            for alert in metric_alerts:
                exp_log.log("guardrail", alert.to_dict())
            if has_blocking_errors(metric_alerts):
                exp_log.log_error("metric_sanity", "Metric-task mismatch")
                console.print("[bold red]Pipeline halted due to metric configuration error.[/bold red]")
                raise typer.Exit(code=1)

        # Get baseline configs and apply agent-recommended prioritization
        baseline_configs = get_baseline_configs(state.profile, seed=config.seed)

        # Agent-driven model prioritization: reorder based on ML Engineer's recommendation
        if ml_decision.action == "select_models" and ml_decision.parameters.get("models"):
            recommended = ml_decision.parameters["models"]
            priority_map = {name: i for i, name in enumerate(recommended)}
            fallback_idx = len(recommended)
            baseline_configs = sorted(
                baseline_configs,
                key=lambda c: priority_map.get(c.model_type, priority_map.get(c.name, fallback_idx)),
            )

        # Model-data compatibility (Section 10.3)
        compat_alerts = []
        valid_configs = []
        for bc in baseline_configs:
            model_alerts = check_model_data_compatibility(bc, state.profile, state.split)
            if model_alerts:
                compat_alerts.extend(model_alerts)
                for alert in model_alerts:
                    exp_log.log("guardrail", alert.to_dict())
            if not has_blocking_errors(model_alerts):
                valid_configs.append(bc)
            else:
                console.print(f"  [yellow]Skipping {bc.name}: blocked by guardrail[/yellow]")

        if compat_alerts:
            print_alerts(compat_alerts, "Model-data compatibility")

        state.trained_models = train_baselines_resilient(valid_configs, state.split, exp_log)

        if not state.trained_models:
            exp_log.log_error("baselines", "All models failed to train")
            console.print("[bold red]All models failed to train. Pipeline halted.[/bold red]")
            raise typer.Exit(code=1)

        console.print("  Evaluating on validation set...")
        state.results = []
        for trained in state.trained_models:
            result = evaluate_model(trained, state.split, state.eval_config, use_test=False)
            state.results.append(result)

            # Update live dashboard with model result
            if live_dash:
                live_dash.add_model(
                    result.model_name, trained.config.tier,
                    result.primary_metric_value, trained.train_time_seconds,
                )
                live_dash.set_cost(cost_tracker.total_cost)

            # Log each model
            exp_log.log_model_trained(
                name=trained.config.name, tier=trained.config.tier,
                model_type=trained.config.model_type,
                hyperparameters=trained.config.hyperparameters,
                train_time=trained.train_time_seconds,
            )
            exp_log.log_evaluation(
                model_name=result.model_name, metrics=result.metrics,
                primary_metric=state.eval_config.primary_metric,
                primary_value=result.primary_metric_value,
            )

        print_model_table(state.results, state.eval_config)

        # Post-training guardrails
        trivial_result = next((r for r in state.results if r.tier == "trivial"), None)
        all_alerts = []
        for res, trn in zip(state.results, state.trained_models):
            model_alerts = check_post_training(res, trn, state.split, state.eval_config, trivial_result)
            all_alerts.extend(model_alerts)
            for alert in model_alerts:
                exp_log.log("guardrail", alert.to_dict())

        summary_alerts = check_pipeline_summary(state.results, state.eval_config)
        all_alerts.extend(summary_alerts)
        for alert in summary_alerts:
            exp_log.log("guardrail", alert.to_dict())

        print_alerts(all_alerts, "Post-training checks")
        total_warnings += sum(1 for a in all_alerts if a.severity.value in ("warning", "error"))

        # Training figures (non-critical — degrade gracefully)
        def _gen_training_figs():
            from co_scientist.viz.training import generate_training_figures
            return generate_training_figures(
                state.results, state.trained_models, state.split,
                state.eval_config, state.profile, config.task_output_dir,
            )
        state.training_figs = run_step_resilient("training_figures", _gen_training_figs, exp_log) or []
        if state.training_figs:
            console.print(f"  Saved {len(state.training_figs)} training figure(s)")

        # Identify best baseline
        lower_is_better = state.eval_config.primary_metric in ("mse", "rmse", "mae")
        ranked = sorted(
            zip(state.results, state.trained_models),
            key=lambda pair: pair[0].primary_metric_value,
            reverse=not lower_is_better,
        )
        state.best_result, state.best_trained = ranked[0]
        console.print(f"  Best baseline: [bold green]{state.best_result.model_name}[/bold green] "
                      f"({state.eval_config.primary_metric}={state.best_result.primary_metric_value:.4f})")

        # Stacking ensemble — combine all base models
        from co_scientist.modeling.ensemble import build_stacking_ensemble
        is_clf = state.eval_config.task_type in ("binary_classification", "multiclass_classification")
        ensemble_result = build_stacking_ensemble(
            trained_models=state.trained_models,
            split=state.split,
            task_type="classification" if is_clf else "regression",
            seed=config.seed,
        )
        if ensemble_result is not None:
            ensemble_trained = ensemble_result
            ensemble_eval = evaluate_model(ensemble_trained, state.split, state.eval_config, use_test=False)
            state.trained_models.append(ensemble_trained)
            state.results.append(ensemble_eval)
            if live_dash:
                live_dash.add_model(
                    ensemble_eval.model_name, ensemble_trained.config.tier,
                    ensemble_eval.primary_metric_value, ensemble_trained.train_time_seconds,
                )
                live_dash.set_cost(cost_tracker.total_cost)

            # Re-rank to see if ensemble is the best
            ranked = sorted(
                zip(state.results, state.trained_models),
                key=lambda pair: pair[0].primary_metric_value,
                reverse=not lower_is_better,
            )
            state.best_result, state.best_trained = ranked[0]

            # Show updated table with ensemble
            print_model_table(state.results, state.eval_config)
            console.print(f"  Best overall: [bold green]{state.best_result.model_name}[/bold green] "
                          f"({state.eval_config.primary_metric}={state.best_result.primary_metric_value:.4f})")

        # Agent post-training analysis
        post_ctx = build_pipeline_context(
            config, state.profile, eval_config=state.eval_config,
            results=state.results, best_result=state.best_result,
            stage="post_training",
            complexity_budget=state.complexity_budget,
            cost_remaining=cost_tracker.budget_remaining,
        )
        agent_analyses = agent_post_training_analysis(coordinator, post_ctx)
        _update_dashboard_cost()
        for agent_name, decision in agent_analyses.items():
            exp_log.log("agent_decision", {
                "agent": agent_name,
                "stage": "post_training",
                "action": decision.action,
                "reasoning": decision.reasoning,
            })
        present_results_analysis(agent_analyses, interactive=interactive, coordinator=coordinator, context=model_ctx)

        state.mark_complete("baselines")
        save_checkpoint(state, config.task_output_dir)
        exp_log.log_step_end("baselines", {
            "n_models": len(state.results),
            "best_model": state.best_result.model_name,
            "best_value": state.best_result.primary_metric_value,
        })
    else:
        print_step_resumed("baselines", 3, 7)

    # ── Step 3b: Hyperparameter search ───────────────────────────────────
    if deadline.expired() and not state.is_complete("hp_search"):
        console.print("  [bold yellow]Deadline reached — skipping HP search[/bold yellow]")
        state.hp_search_done = True
        state.mark_complete("hp_search")
    if not state.is_complete("hp_search"):
        from co_scientist.modeling.hp_search import run_hp_search
        from co_scientist.evaluation.metrics import evaluate_model

        exp_log.log_step_start("hp_search")
        print_step_start("hp_search", 4, 7)

        # Agent-driven HP search decision
        hp_ctx = build_pipeline_context(
            config, state.profile, eval_config=state.eval_config,
            results=state.results, best_result=state.best_result,
            stage="hp_search",
            complexity_budget=state.complexity_budget,
            cost_remaining=cost_tracker.budget_remaining,
        )
        hp_decision = coordinator.debate(
            "hyperparameter search strategy", hp_ctx,
            agent_names=["ml_engineer", "data_analyst"],
        )
        exp_log.log("agent_decision", {
            "agent": "ml_engineer",
            "stage": "hp_search",
            "action": hp_decision.action,
            "reasoning": hp_decision.reasoning,
        })
        hp_decision = present_decision("ml_engineer", hp_decision, "hp_search", interactive=interactive, coordinator=coordinator, context=hp_ctx)

        # Agent decision actually drives whether HP search runs
        skip_hp = hp_decision.action == "skip_hp_search"

        lower_is_better = state.eval_config.primary_metric in ("mse", "rmse", "mae")
        improved = False
        hp_result = None

        if skip_hp:
            console.print(f"  [yellow]HP search skipped: {hp_decision.reasoning}[/yellow]")
        else:
            # Use agent-recommended n_trials if available
            n_trials = hp_decision.parameters.get("n_trials")
            if n_trials is None:
                n_trials = state.complexity_budget.hp_trials if state.complexity_budget else None

            hp_result = run_hp_search(
                base_config=state.best_trained.config,
                split=state.split,
                eval_config=state.eval_config,
                profile=state.profile,
                seed=config.seed,
                n_trials_override=n_trials,
                timeout_override=state.complexity_budget.hp_timeout if state.complexity_budget else None,
            )

        if hp_result is not None:
            tuned_trained, tuned_result = hp_result
            improved = (
                (tuned_result.primary_metric_value < state.best_result.primary_metric_value)
                if lower_is_better else
                (tuned_result.primary_metric_value > state.best_result.primary_metric_value)
            )
            if improved:
                console.print(f"  [bold green]Tuned model improves over baseline![/bold green] "
                              f"({state.best_result.primary_metric_value:.4f} → {tuned_result.primary_metric_value:.4f})")
                state.results.append(tuned_result)
                state.trained_models.append(tuned_trained)
                state.best_result = tuned_result
                if live_dash:
                    live_dash.add_model(
                        tuned_result.model_name, tuned_trained.config.tier,
                        tuned_result.primary_metric_value, tuned_trained.train_time_seconds,
                    )
                    live_dash.set_cost(cost_tracker.total_cost)
                state.best_trained = tuned_trained
            else:
                console.print(f"  [yellow]Tuned model did not improve ({tuned_result.primary_metric_value:.4f} "
                              f"vs baseline {state.best_result.primary_metric_value:.4f}). Keeping baseline.[/yellow]")

            exp_log.log_hp_search(
                n_trials=30,  # from config
                best_trial=0,
                best_value=tuned_result.primary_metric_value,
                best_params=tuned_trained.config.hyperparameters,
                duration=0,
                improved=improved,
            )

        console.print(f"  Final best: [bold green]{state.best_result.model_name}[/bold green] "
                      f"({state.eval_config.primary_metric}={state.best_result.primary_metric_value:.4f})")

        state.hp_search_done = True
        state.mark_complete("hp_search")
        save_checkpoint(state, config.task_output_dir)
        exp_log.log_step_end("hp_search", {
            "improved": improved,
            "skipped": skip_hp,
            "best_model": state.best_result.model_name,
            "best_value": state.best_result.primary_metric_value,
        })
    else:
        print_step_resumed("hp_search", 4, 7)

    # ── Step 5: Iteration Loop ───────────────────────────────────────────
    if deadline.expired() and not state.is_complete("iteration"):
        console.print("  [bold yellow]Deadline reached — skipping iteration loop[/bold yellow]")
        state.iteration_log = {"total_iterations": 0, "stop_reason": "deadline_reached"}
        state.mark_complete("iteration")
    if not state.is_complete("iteration"):
        from co_scientist.iteration import run_iteration_loop

        exp_log.log_step_start("iteration")
        print_step_start("iteration", 5, 7)

        if config.budget > 0:
            (
                state.trained_models,
                state.results,
                state.best_result,
                state.best_trained,
                iter_log,
            ) = run_iteration_loop(
                coordinator=coordinator,
                config=config,
                profile=state.profile,
                split=state.split,
                eval_config=state.eval_config,
                trained_models=state.trained_models,
                results=state.results,
                best_result=state.best_result,
                best_trained=state.best_trained,
                complexity_budget=state.complexity_budget,
                cost_tracker=cost_tracker,
                exp_log=exp_log,
                interactive=interactive,
                seed=config.seed,
            )

            state.iteration_log = {
                "total_iterations": iter_log.total_iterations,
                "improvements": iter_log.improvements,
                "stop_reason": iter_log.stop_reason,
                "best_before": iter_log.best_score_before,
                "best_after": iter_log.best_score_after,
            }

            # Show updated model table if anything changed
            if iter_log.total_iterations > 0:
                print_model_table(state.results, state.eval_config)
                console.print(f"  Best overall: [bold green]{state.best_result.model_name}[/bold green] "
                              f"({state.eval_config.primary_metric}={state.best_result.primary_metric_value:.4f})")
        else:
            console.print("  [dim]Budget=0, skipping iteration loop[/dim]")
            state.iteration_log = {"total_iterations": 0, "stop_reason": "zero_budget"}

        state.mark_complete("iteration")
        save_checkpoint(state, config.task_output_dir)
        exp_log.log_step_end("iteration", state.iteration_log or {})
    else:
        print_step_resumed("iteration", 5, 7)

    # ── Final Evaluation on Test Set ──────────────────────────────────────
    if state.best_trained and state.split.X_test is not None:
        from co_scientist.evaluation.metrics import evaluate_model as _eval_final
        console.print("\n  [bold cyan]── Final Evaluation (held-out test set) ──[/bold cyan]")
        try:
            test_result = _eval_final(
                state.best_trained, state.split, state.eval_config, use_test=True,
            )
            console.print(f"  Model: [bold]{state.best_result.model_name}[/bold]")
            for metric_name, metric_val in test_result.metrics.items():
                console.print(f"    {metric_name}: [bold green]{metric_val:.4f}[/bold green]")
            state.test_metrics = test_result.metrics
            exp_log.log("test_evaluation", {
                "model_name": state.best_result.model_name,
                "metrics": {k: round(v, 4) for k, v in test_result.metrics.items()},
            })
            if live_dash:
                live_dash.set_agent_name("Evaluator")
                live_dash.set_agent_thought(
                    f"Final test-set evaluation: {state.eval_config.primary_metric}="
                    f"{test_result.metrics.get(state.eval_config.primary_metric, 0):.4f}"
                )
                live_dash.set_agent_action("evaluate_test_set", {"model": state.best_result.model_name})
        except Exception as e:
            logger.warning("Test-set evaluation failed: %s", e)
            console.print(f"  [yellow]Test evaluation skipped: {e}[/yellow]")

    # ── Step 6: Model Export ─────────────────────────────────────────────
    if not state.is_complete("export") and state.best_trained is None:
        console.print("  [yellow]No model trained — skipping export and report[/yellow]")
        state.mark_complete("export")
        state.mark_complete("report")
    if not state.is_complete("export"):
        from co_scientist.export.exporter import export_model

        exp_log.log_step_start("export")
        print_step_start("export", 6, 7)
        state.export_path = export_model(
            trained=state.best_trained,
            result=state.best_result,
            profile=state.profile,
            eval_config=state.eval_config,
            split=state.split,
            output_dir=config.task_output_dir,
            preprocessing_steps=state.preprocessed.steps_applied,
        )
        console.print(f"  Exported to: [bold]{state.export_path}[/bold]")

        # Validate + auto-fix exported scripts (syntax, imports, execution)
        from co_scientist.validation import validate_and_fix_export
        llm_client = coordinator.client if coordinator.llm_available else None
        if live_dash:
            live_dash.set_agent_name("Validation Agent")
            live_dash.set_validation_running("validate_export")
        v_export = validate_and_fix_export(config.task_output_dir, state.profile, client=llm_client)
        if live_dash:
            live_dash.update_validation_result("validate_export", v_export.passed, v_export.issues, v_export.fixes_applied)
        if not v_export.passed:
            exp_log.log("validation_failure", {"step": "export", "issues": v_export.issues})
        if v_export.fixes_applied:
            exp_log.log("validation_fix", {"step": "export", "fixes": v_export.fixes_applied})

        state.mark_complete("export")
        save_checkpoint(state, config.task_output_dir)
        exp_log.log_step_end("export", {"path": str(state.export_path)})
    else:
        print_step_resumed("export", 6, 7)

    # ── Active Learning Analysis (non-critical) ─────────────────────────
    al_report = None
    try:
        from co_scientist.evaluation.active_learning import (
            run_active_learning_analysis, get_feature_gap_suggestions,
        )
        # Route to correct feature set for foundation models
        if state.best_trained.needs_embeddings and state.split.X_embed_test is not None:
            al_X = state.split.X_embed_test
        elif hasattr(state.best_trained.config, 'model_type') and state.best_trained.config.model_type in ('concat_xgboost', 'concat_mlp'):
            import numpy as _np
            al_X = _np.hstack([state.split.X_test, state.split.X_embed_test]) if state.split.X_embed_test is not None else state.split.X_test
        else:
            al_X = state.split.X_test
        al_report = run_active_learning_analysis(
            model=state.best_trained,
            X=al_X,
            y_true=state.split.y_test,
            task_type=state.eval_config.task_type,
            label_encoder=getattr(state.split, 'label_encoder', None),
            sequences=getattr(state.split, 'seqs_test', None),
        )
        # Add feature gap suggestions from biology knowledge
        al_report.feature_gap_suggestions = get_feature_gap_suggestions(
            modality=state.profile.modality.value,
            task_type=state.profile.task_type.value,
            bottleneck_classes=al_report.bottleneck_classes,
            worst_range=al_report.worst_predicted_range,
        )
        # Rebuild summary with suggestions
        from co_scientist.evaluation.active_learning import _build_summary
        is_clf = "classification" in state.eval_config.task_type
        al_report.summary = _build_summary(al_report, is_clf)

        if al_report.summary and al_report.summary != "No significant data needs identified.":
            console.print("  [bold cyan]Active learning analysis:[/bold cyan]")
            if al_report.bottleneck_classes:
                console.print(f"    Bottleneck classes: {', '.join(al_report.bottleneck_classes)}")
            if al_report.worst_predicted_range:
                console.print(f"    Hardest target range: {al_report.worst_predicted_range}")
            if al_report.uncertain_samples:
                console.print(f"    Top {len(al_report.uncertain_samples)} uncertain samples identified")

        exp_log.log("active_learning", {
            "bottleneck_classes": al_report.bottleneck_classes,
            "worst_range": al_report.worst_predicted_range,
            "n_uncertain": len(al_report.uncertain_samples),
            "feature_gap_suggestions": al_report.feature_gap_suggestions,
        })
    except Exception as e:
        logger.warning("Active learning analysis failed: %s", e)

    # Collect debate transcripts
    if coordinator.debate_transcripts:
        state.debate_transcripts = [
            {
                "topic": dt.topic,
                "proposals": {name: {"action": d.action, "parameters": d.parameters,
                                     "reasoning": d.reasoning, "confidence": d.confidence}
                              for name, d in dt.proposals.items()},
                "rebuttals": dt.rebuttals,
                "judge_reasoning": dt.judge_reasoning,
                "winning_agent": dt.winning_agent,
            }
            for dt in coordinator.debate_transcripts
        ]

    # ── Step 6: Report Generation ────────────────────────────────────────
    if not state.is_complete("report"):
        from co_scientist.report.generator import generate_report

        exp_log.log_step_start("report")
        print_step_start("report", 7, 7)

        # Get biological interpretation from agent
        report_ctx = build_pipeline_context(
            config, state.profile, eval_config=state.eval_config,
            results=state.results, best_result=state.best_result,
            stage="report",
            complexity_budget=state.complexity_budget,
            cost_remaining=cost_tracker.budget_remaining,
        )
        # Extract feature names for biological interpretation
        report_feature_names = getattr(state.split, 'feature_names', None) or []
        research_papers = (getattr(state, 'research_results', {}) or {}).get("papers", [])

        bio_interpretation = agent_biology_interpretation(
            coordinator, report_ctx,
            feature_names=report_feature_names,
            research_papers=research_papers,
        )

        # Feature importance interpretation (if best model has feature importances)
        feature_interp = []
        if report_feature_names and state.best_trained:
            top_features = _extract_top_features(state.best_trained, report_feature_names)
            if top_features:
                feature_interp = agent_feature_interpretation(coordinator, report_ctx, top_features)
                if feature_interp:
                    exp_log.log("feature_interpretation", feature_interp[:10])

        # Get agent reasoning summary for report
        agent_reasoning = _collect_agent_reasoning(coordinator, exp_log)

        # Generate architecture diagrams
        arch_diagram_rel = None
        flow_diagram_rel = None
        try:
            from co_scientist.viz.architecture import generate_architecture_diagram, generate_agent_flow_diagram
            arch_path = generate_architecture_diagram(config.task_output_dir)
            arch_diagram_rel = str(arch_path.relative_to(config.task_output_dir))
            flow_path = generate_agent_flow_diagram(
                config.task_output_dir, agent_reasoning,
                debate_transcripts=getattr(coordinator, 'debate_transcripts', None),
            )
            flow_diagram_rel = str(flow_path.relative_to(config.task_output_dir))
        except Exception as e:
            console.print(f"  [yellow]Architecture diagram skipped: {e}[/yellow]")

        # Collect guardrail warnings for report
        guardrail_alerts = _collect_guardrails(exp_log)

        # Get iteration log for report
        iter_log_data = getattr(state, 'iteration_log', None) or {}

        state.report_path = generate_report(
            profile=state.profile,
            split=state.split,
            eval_config=state.eval_config,
            results=state.results,
            best_result=state.best_result,
            best_trained=state.best_trained,
            preprocessing_steps=state.preprocessed.steps_applied,
            output_dir=config.task_output_dir,
            profiling_figures=state.profiling_figs,
            preprocessing_figures=state.preprocessing_figs,
            training_figures=state.training_figs,
            seed=config.seed,
            biological_interpretation=bio_interpretation,
            agent_reasoning=agent_reasoning,
            research_report=getattr(state, 'research_results', {}),
            feature_interpretation=feature_interp,
            active_learning_report=al_report,
            guardrail_alerts=guardrail_alerts,
            iteration_log=iter_log_data,
            react_scratchpad=getattr(state, 'react_scratchpad', None),
            elo_rankings=getattr(state, 'elo_rankings', None),
            debate_transcripts=getattr(state, 'debate_transcripts', None),
            tree_search_log=getattr(state, 'tree_search_log', None),
            architecture_diagram=arch_diagram_rel,
            agent_flow_diagram=flow_diagram_rel,
            test_metrics=getattr(state, 'test_metrics', None),
            biology_assessment=getattr(state, 'biology_assessment', None),
        )
        console.print(f"  Report saved to: [bold]{state.report_path}[/bold]")

        # Generate summary PDF
        try:
            from co_scientist.report.summary_pdf import generate_summary_pdf
            from co_scientist.report.generator import _build_benchmark_comparison
            bench_comp = _build_benchmark_comparison(
                state.best_result, state.eval_config,
                getattr(state, 'research_results', {}) or {},
            )
            pdf_path = generate_summary_pdf(
                profile=state.profile,
                eval_config=state.eval_config,
                best_result=state.best_result,
                results=state.results,
                output_dir=config.task_output_dir,
                benchmark_comparison=bench_comp,
                research_report=getattr(state, 'research_results', None),
                react_scratchpad=getattr(state, 'react_scratchpad', None),
            )
            console.print(f"  Summary PDF saved to: [bold]{pdf_path}[/bold]")
        except Exception as e:
            console.print(f"  [yellow]Summary PDF generation skipped: {e}[/yellow]")

        state.mark_complete("report")
        save_checkpoint(state, config.task_output_dir)
        exp_log.log_step_end("report", {"path": str(state.report_path)})
    else:
        print_step_resumed("report", 7, 7)

    # ── Update cross-run memory ──────────────────────────────────────────
    try:
        from co_scientist.memory import ModelPerformanceEntry
        for res, trn in zip(state.results, state.trained_models):
            run_memory.record_performance(ModelPerformanceEntry(
                model_type=trn.config.model_type,
                modality=state.profile.modality.value,
                task_type=state.profile.task_type.value,
                primary_metric=state.eval_config.primary_metric,
                score=res.primary_metric_value,
                hyperparameters=trn.config.hyperparameters,
                dataset_name=config.dataset_path,
            ))
        # Update HP priors with best model
        if state.best_trained and state.best_result:
            run_memory.update_hp_priors(
                model_type=state.best_trained.config.model_type,
                modality=state.profile.modality.value,
                hp=state.best_trained.config.hyperparameters,
                score=state.best_result.primary_metric_value,
            )
        console.print("  [dim]Cross-run memory updated[/dim]")
    except Exception as e:
        logger.warning("Failed to update cross-run memory: %s", e)

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = _time.time() - pipeline_start_time

    # Log agent cost summary
    cost_summary = coordinator.get_cost_summary()
    if cost_summary["num_calls"] > 0:
        exp_log.log("llm_costs", cost_summary)
        console.print(f"  [dim]LLM calls: {cost_summary['num_calls']}, "
                      f"cost: ${cost_summary['total_cost']:.4f}[/dim]")

    if state.best_result:
        exp_log.log_pipeline_complete(
            best_model=state.best_result.model_name,
            best_metric=state.eval_config.primary_metric,
            best_value=state.best_result.primary_metric_value,
        )

        print_final_summary(
            dataset_path=config.dataset_path,
            best_model=state.best_result.model_name,
            best_metric=state.eval_config.primary_metric,
            best_value=state.best_result.primary_metric_value,
            total_models=len(state.results),
            elapsed_seconds=elapsed,
            output_dir=str(config.task_output_dir),
            warnings=total_warnings,
            complexity_level=state.complexity_budget.level if state.complexity_budget else "",
        )
    else:
        console.print(f"\n  [bold yellow]Pipeline completed in {elapsed:.0f}s but no models were trained.[/bold yellow]")
        console.print("  [dim]Try increasing --timeout or --budget[/dim]")

    # Stop live dashboard
    if live_dash:
        live_dash.stop()


def _extract_top_features(best_trained, feature_names: list[str], top_n: int = 15) -> list[tuple[str, float]]:
    """Extract top feature importances from the best trained model."""
    import numpy as np

    model = best_trained.model
    importances = None

    # Try tree-based feature_importances_
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Try linear model coef_
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return []

    # Sort by importance descending
    indices = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in indices]


def _collect_agent_reasoning(coordinator, exp_log) -> list[dict]:
    """Collect agent decisions from the coordinator message log for the report."""
    reasoning = []
    for msg in coordinator.message_log:
        data = msg.structured_data or {}
        reasoning.append({
            "agent": msg.to_agent.value,
            "stage": msg.summary.split(": ", 1)[0] if ": " in msg.summary else msg.summary,
            "action": data.get("action", ""),
            "reasoning": _summarize_params(data.get("parameters", {})),
            "confidence": msg.confidence,
        })
    return reasoning


def _summarize_params(params: dict) -> str:
    """Build a readable one-line summary from decision parameters."""
    # Priority keys to surface as narrative
    skip_keys = {"plausibility_detail", "biological_context", "biological_signals", "metric_note"}
    narrative_parts = []

    # Handle known action types with natural language
    if "models" in params:
        models = params["models"]
        if isinstance(models, list):
            model_str = ", ".join(str(m) for m in models[:4])
            if len(models) > 4:
                model_str += f" (+{len(models) - 4} more)"
            narrative_parts.append(f"Selected models: {model_str}")
    if "target_model" in params:
        narrative_parts.append(f"Target: {params['target_model']}")
    if "n_trials" in params:
        narrative_parts.append(f"{params['n_trials']} HP trials")
    if "steps" in params and isinstance(params["steps"], list):
        steps = params["steps"]
        step_str = ", ".join(str(s) for s in steps[:3])
        if len(steps) > 3:
            step_str += f" (+{len(steps) - 3} more)"
        narrative_parts.append(f"Steps: {step_str}")
    if "scaling" in params:
        narrative_parts.append(f"Scaling: {params['scaling']}")
    if "type" in params:
        narrative_parts.append(f"Type: {params['type']}")
    if "reason" in params:
        narrative_parts.append(str(params["reason"]).replace("_", " "))
    if "priority" in params:
        narrative_parts.append(f"Priority: {params['priority']}")
    if "plausibility" in params:
        narrative_parts.append(f"Assessment: {params['plausibility']}")
    if "plausibility_detail" in params:
        narrative_parts.append(str(params["plausibility_detail"]))

    if narrative_parts:
        return ". ".join(narrative_parts)

    # Fallback for unknown params
    parts = []
    for key, val in params.items():
        if key in skip_keys:
            continue
        if isinstance(val, list) and len(val) > 3:
            parts.append(f"{key}: {', '.join(str(v) for v in val[:3])} (+{len(val)-3} more)")
        elif isinstance(val, list):
            parts.append(f"{key}: {', '.join(str(v) for v in val)}")
        elif isinstance(val, dict):
            continue
        else:
            parts.append(f"{key}: {val}")
    return "; ".join(parts) if parts else ""


@app.command()
def batch(
    datasets: list[str] = typer.Argument(..., help="Dataset paths to process"),
    parallel: int = typer.Option(1, "--parallel", "-p", help="Number of parallel workers."),
    budget: int = typer.Option(DEFAULT_BUDGET, "--budget", "-b", help="Max iteration steps per dataset."),
    max_cost: float = typer.Option(DEFAULT_MAX_COST, "--max-cost", help="Max LLM cost per dataset."),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", help="Output directory."),
    no_search: bool = typer.Option(False, "--no-search", help="Disable web/paper search."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """Run the co-scientist pipeline on multiple datasets."""
    from co_scientist.batch import BatchRunner

    common_options = {
        "budget": budget,
        "max_cost": max_cost,
        "no_search": no_search,
        "seed": seed,
    }

    runner = BatchRunner(
        datasets=datasets,
        parallel=parallel,
        output_dir=output_dir,
        common_options=common_options,
    )

    results = runner.run()
    BatchRunner.print_summary(results)

    # Check for failures
    n_failed = sum(1 for r in results if not r.success)
    if n_failed > 0:
        raise typer.Exit(code=1)


def _collect_guardrails(exp_log) -> list[dict]:
    """Collect guardrail alerts from the experiment log for the report."""
    import json
    guardrails = []
    log_path = exp_log.path
    if not log_path or not log_path.exists():
        return guardrails
    try:
        with open(log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("event") == "guardrail":
                    d = entry.get("data", {})
                    severity = d.get("severity", "info")
                    if severity in ("warning", "error"):
                        guardrails.append({
                            "severity": severity,
                            "check": d.get("check_name", ""),
                            "message": d.get("message", ""),
                        })
    except Exception:
        pass
    # Deduplicate by message
    seen = set()
    unique = []
    for g in guardrails:
        if g["message"] not in seen:
            seen.add(g["message"])
            unique.append(g)
    return unique
