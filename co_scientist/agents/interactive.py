"""Interactive mode — presents agent decisions to user for approval.

In interactive mode, the pipeline pauses at key decision points and shows
the agent's recommendation. The user can accept, modify, or override.
Users can also ask free-form questions or give instructions at any decision point.

In auto mode, agent decisions are applied directly without prompting.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from co_scientist.agents.types import Decision, PipelineContext

logger = logging.getLogger(__name__)

console = Console()


def present_decision(
    agent_name: str,
    decision: Decision,
    stage: str,
    interactive: bool = False,
    coordinator: Any = None,
    context: PipelineContext | None = None,
) -> Decision:
    """Present an agent decision and optionally get user approval.

    In auto mode: prints the decision and returns it unchanged.
    In interactive mode: shows the decision and lets user accept, override, or chat.
    """
    _print_decision(agent_name, decision, stage)

    if not interactive:
        return decision

    return _interactive_loop(agent_name, decision, stage, coordinator, context)


def present_results_analysis(
    agent_decisions: dict[str, Decision],
    interactive: bool = False,
    coordinator: Any = None,
    context: PipelineContext | None = None,
) -> str | None:
    """Present multi-agent analysis of results. Returns user's chosen action or None."""
    if not agent_decisions:
        return None

    console.print()
    console.print("[bold]Agent Analysis:[/bold]")

    for agent_name, decision in agent_decisions.items():
        _print_compact_decision(agent_name, decision)

    if not interactive:
        return None

    while True:
        console.print()
        console.print(
            "  [dim]Options: [bold]continue[/bold] (train more), [bold]tune[/bold] (HP search), "
            "[bold]stop[/bold] (finish), or type a question/instruction[/dim]"
        )
        user_input = Prompt.ask("  [bold]What next?[/bold]", default="continue")
        stripped = user_input.strip().lower()

        if stripped in ("continue", "tune", "stop", "skip"):
            return stripped

        # Natural language → chat with LLM, then ask again
        response = chat_with_agent(user_input, coordinator, context, "post-training analysis")
        if response:
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Co-Scientist[/bold green]",
                border_style="green",
                padding=(0, 1),
            ))
        else:
            console.print("  [dim]LLM not available — choose: continue, tune, stop, skip[/dim]")


def confirm_step(
    step_name: str,
    description: str,
    interactive: bool = False,
    coordinator: Any = None,
    context: PipelineContext | None = None,
) -> bool:
    """In interactive mode, ask user to confirm before proceeding with a step.

    Supports conversational input: user can type questions, instructions,
    or 'y'/'n' to accept/reject. Works like Claude Code — natural language first.
    """
    if not interactive:
        return True

    while True:
        console.print()
        console.print(
            f"  [bold]Proceed with {step_name}?[/bold] ({description})"
        )
        console.print(
            "  [dim]Enter [bold]y[/bold]=yes, [bold]n[/bold]=no, [bold]exit[/bold]=stop pipeline, "
            "or type a question/instruction[/dim]"
        )
        user_input = Prompt.ask("  ", default="y")
        stripped = user_input.strip().lower()

        if stripped in ("y", "yes", ""):
            return True

        if stripped in ("n", "no", "skip"):
            return False

        if stripped in ("exit", "quit", "stop", "cancel", "abort"):
            console.print("  [yellow]Stopping pipeline.[/yellow]")
            raise SystemExit(0)

        # Anything else → treat as a question or instruction, chat with LLM
        response = chat_with_agent(user_input, coordinator, context, step_name)
        if response:
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Co-Scientist[/bold green]",
                border_style="green",
                padding=(0, 1),
            ))
        else:
            console.print("  [dim]LLM not available — type 'y' to proceed or 'n' to skip[/dim]")
        # Loop back and ask again


def _print_decision(agent_name: str, decision: Decision, stage: str) -> None:
    """Print a styled decision panel with clear reasoning."""
    lines = []
    lines.append(f"[bold]Action:[/bold] {decision.action}")

    # Show parameters, but skip internal/verbose keys
    skip_keys = {"selection_reasons"}
    if decision.parameters:
        for key, val in decision.parameters.items():
            if key in skip_keys:
                continue
            if isinstance(val, list) and len(val) > 5:
                val = val[:5] + [f"... +{len(val)-5} more"]
            lines.append(f"  {key}: {val}")

    # Show reasoning prominently
    lines.append("")
    lines.append(f"[bold]Why:[/bold] {decision.reasoning}")

    # Show detailed selection reasons if available (e.g. why each model was chosen)
    selection_reasons = decision.parameters.get("selection_reasons", [])
    if selection_reasons:
        lines.append("")
        lines.append("[bold]Model-by-model rationale:[/bold]")
        for reason in selection_reasons:
            lines.append(f"  • {reason}")

    lines.append("")
    lines.append(f"[dim]Confidence: {decision.confidence:.0%}[/dim]")

    agent_display = agent_name.replace("_", " ").title()
    panel = Panel(
        "\n".join(lines),
        title=f"[bold cyan]{agent_display}[/bold cyan] — {stage}",
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)


def _print_compact_decision(agent_name: str, decision: Decision) -> None:
    """Print a one-line summary of a decision."""
    agent_display = agent_name.replace("_", " ").title()
    conf = f"{decision.confidence:.0%}"
    console.print(
        f"  [{_agent_color(agent_name)}]{agent_display}[/{_agent_color(agent_name)}]: "
        f"{decision.action} — {decision.reasoning} [dim]({conf})[/dim]"
    )


def _get_user_override(agent_name: str, original: Decision, stage: str) -> Decision:
    """Get a user override for a decision."""
    console.print("  [yellow]Enter your override:[/yellow]")

    if stage == "model_selection":
        models_str = Prompt.ask(
            "  Models to train (comma-separated, or 'all')",
            default="all",
        )
        if models_str.strip().lower() == "all":
            return original  # keep original
        models = [m.strip() for m in models_str.split(",")]
        return Decision(
            action="select_models",
            parameters={"models": models, "priority": models[0] if models else ""},
            reasoning="User override",
            confidence=1.0,
        )

    elif stage == "hp_search":
        should_tune = Confirm.ask("  Run HP search?", default=True)
        return Decision(
            action="hp_tune" if should_tune else "skip_hp_search",
            parameters=original.parameters,
            reasoning="User override",
            confidence=1.0,
        )

    else:
        # Generic: just ask for action name
        action = Prompt.ask("  Action", default=original.action)
        return Decision(
            action=action,
            parameters=original.parameters,
            reasoning="User override",
            confidence=1.0,
        )


def _agent_color(agent_name: str) -> str:
    return {
        "data_analyst": "blue",
        "ml_engineer": "green",
        "biology_specialist": "magenta",
        "coordinator": "yellow",
    }.get(agent_name, "white")


# ── Conversational chat ──────────────────────────────────────────────────────

INTERACTIVE_CHAT_SYSTEM = """\
You are AIDO Co-Scientist, an AI assistant for automated ML on biological datasets.

The user is running the pipeline in interactive mode and has a question or instruction
at the current decision point.

Current pipeline state:
{pipeline_state}

Current stage: {stage}

Your role is to be a knowledgeable collaborator — like a senior ML engineer pair-programming
with the user. You should:

1. **Answer questions** clearly with specifics from the pipeline state (data stats, scores, etc.)
2. **Explain decisions** — why the pipeline chose this approach for this data
3. **Push back when the user is wrong** — if they ask for something that would hurt performance
   or doesn't make sense for this dataset, explain WHY politely but clearly. For example:
   - "Do single classification" on a 13-class dataset → explain that the data has 13 cell types,
     binary classification would discard information, and multiclass is the correct approach
   - "Skip preprocessing" → explain that raw expression counts need log1p normalization
   - "Use CNN" on tabular data → explain that CNNs need sequence input, not expression matrices
4. **Suggest alternatives** — if the user's idea has merit but the execution is wrong,
   suggest the right way to achieve their goal
5. **Acknowledge good ideas** — if the user suggests something smart, say so and explain
   how the pipeline can incorporate it

Be direct and honest. Don't just say "ok" to bad ideas. The user trusts you to guide them.
Keep answers to 3-6 sentences unless more detail is needed.
"""


def _build_pipeline_state_text(context: PipelineContext | None) -> str:
    """Build a human-readable summary of current pipeline state for chat.

    Includes all available info so the LLM can answer any question about the data,
    models, splits, preprocessing, etc.
    """
    if context is None:
        return "No pipeline context available yet."

    lines = []

    # Dataset basics
    if context.dataset_path:
        lines.append(f"Dataset: {context.dataset_path}")
    if context.modality:
        lines.append(f"Modality: {context.modality}")
    if context.task_type:
        lines.append(f"Task: {context.task_type}")
    if context.num_samples:
        lines.append(f"Total samples: {context.num_samples:,}")
    if context.num_features:
        lines.append(f"Features: {context.num_features:,}")
    if context.target_column:
        lines.append(f"Target column: {context.target_column}")

    # Class/target info
    if context.num_classes:
        lines.append(f"Number of classes: {context.num_classes}")
    if context.class_distribution:
        lines.append("Class distribution:")
        for cls, count in sorted(context.class_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls}: {count} samples")
    if context.target_stats:
        stats = context.target_stats
        lines.append(f"Target stats: mean={stats.get('mean', 0):.4f}, std={stats.get('std', 0):.4f}, "
                      f"min={stats.get('min', 0):.4f}, max={stats.get('max', 0):.4f}")

    # Split info
    if context.split_info:
        lines.append(f"Split strategy: {context.split_info}")

    # Data quality
    if context.missing_value_pct > 0:
        lines.append(f"Missing values: {context.missing_value_pct:.1f}%")
    if context.feature_sparsity > 0:
        lines.append(f"Feature sparsity: {context.feature_sparsity:.1f}% zeros")
    if context.sequence_length_stats:
        lines.append(f"Sequence lengths: {context.sequence_length_stats}")
    if context.detected_issues:
        lines.append(f"Detected issues: {', '.join(context.detected_issues)}")

    # Preprocessing
    if context.preprocessing_steps:
        lines.append(f"Preprocessing applied: {', '.join(context.preprocessing_steps)}")

    # Metrics and primary metric
    if context.primary_metric:
        lines.append(f"Primary metric: {context.primary_metric}")

    # Model results
    if context.model_scores:
        lines.append("Model scores:")
        for name, score in sorted(context.model_scores.items(), key=lambda x: -x[1]):
            marker = " ← best" if name == context.best_model_name else ""
            lines.append(f"  {name}: {score:.4f}{marker}")
    if context.best_model_name:
        lines.append(f"Best model: {context.best_model_name} ({context.best_score:.4f})")

    # GPU status
    from co_scientist.modeling.foundation import gpu_available
    if gpu_available():
        lines.append("GPU: Available — foundation models (embed_xgboost, concat_xgboost, aido_finetune) are active")
    else:
        lines.append("GPU: Not available — foundation models are NOT available on this machine. Do NOT suggest embed_*, concat_*, or aido_finetune models.")

    # Budget
    if context.remaining_budget:
        lines.append(f"Budget remaining: {context.remaining_budget} steps, ${context.remaining_cost:.2f}")
    if context.complexity_level:
        lines.append(f"Complexity: {context.complexity_score:.1f}/10 ({context.complexity_level})")
    if context.decisions_so_far:
        lines.append(f"Decisions so far: {', '.join(context.decisions_so_far[-5:])}")
    if context.memory_context:
        lines.append(f"Cross-run memory: {context.memory_context[:200]}")

    return "\n".join(lines) if lines else "Pipeline just started, no results yet."


def chat_with_agent(
    user_message: str,
    coordinator: Any,
    context: PipelineContext | None = None,
    stage: str = "",
) -> str | None:
    """Send a free-form user question to the LLM and return the response.

    Returns None if LLM is unavailable.
    """
    if coordinator is None or not coordinator.llm_available:
        return None

    state_text = _build_pipeline_state_text(context)
    system_prompt = INTERACTIVE_CHAT_SYSTEM.format(
        pipeline_state=state_text,
        stage=stage,
    )

    response = coordinator.client.ask_text(
        system_prompt=system_prompt,
        user_message=user_message,
        agent_name="interactive_chat",
        max_tokens=1024,
        temperature=0.3,
    )
    return response


def _offer_chat(
    coordinator: Any,
    context: PipelineContext | None,
    stage: str,
) -> None:
    """Offer the user a chance to ask questions before proceeding."""
    if coordinator is None:
        return

    while True:
        console.print()
        user_input = Prompt.ask(
            "  [bold]Ask a question, or press Enter to continue[/bold]",
            default="",
        )

        if not user_input.strip():
            return

        response = chat_with_agent(user_input, coordinator, context, stage)
        if response:
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Co-Scientist[/bold green]",
                border_style="green",
                padding=(0, 1),
            ))
        else:
            console.print("  [dim]LLM not available — cannot answer questions[/dim]")
            return


def _interactive_loop(
    agent_name: str,
    decision: Decision,
    stage: str,
    coordinator: Any,
    context: PipelineContext | None,
) -> Decision:
    """Full interactive loop: chat, accept, or override."""
    while True:
        console.print()
        console.print(
            "  [dim]Options: [bold]y[/bold]=accept, [bold]n[/bold]=override, "
            "[bold]exit[/bold]=stop pipeline, or type a question/instruction[/dim]"
        )
        user_input = Prompt.ask(
            f"  [bold]Accept {agent_name}'s recommendation?[/bold]",
            default="y",
        )

        stripped = user_input.strip().lower()

        if stripped in ("y", "yes", ""):
            # Check if the current decision is a stop action
            if decision.action in ("stop_pipeline", "stop", "exit"):
                console.print("  [yellow]Stopping pipeline as requested.[/yellow]")
                raise SystemExit(0)
            return decision

        if stripped in ("n", "no"):
            return _get_user_override(agent_name, decision, stage)

        if stripped in ("exit", "quit", "stop", "cancel", "abort"):
            console.print("  [yellow]Stopping pipeline.[/yellow]")
            raise SystemExit(0)

        # Free-form question or instruction — send to LLM
        response = chat_with_agent(user_input, coordinator, context, stage)
        if response:
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Co-Scientist[/bold green]",
                border_style="green",
                padding=(0, 1),
            ))
        else:
            console.print("  [dim]LLM not available — type 'y' to accept or 'n' to override[/dim]")

        # Check if user gave an instruction that should revise the decision
        if coordinator is not None and coordinator.llm_available and _looks_like_instruction(user_input):
            revised = _try_revise_decision(
                user_input, agent_name, decision, stage, coordinator, context,
            )
            if revised is not None and revised.action != decision.action:
                console.print()
                console.print("  [yellow]Revised recommendation based on your input:[/yellow]")
                _print_decision(agent_name, revised, stage)
                decision = revised

        # Loop back to ask again


def _looks_like_instruction(text: str) -> bool:
    """Heuristic: does the user input look like an instruction vs a question?"""
    text_lower = text.strip().lower()
    # Instructions typically start with verbs or contain directive words
    instruction_markers = [
        "use ", "try ", "don't ", "do not ", "skip ", "add ", "remove ",
        "change ", "switch ", "prefer ", "focus ", "include ", "exclude ",
        "train ", "tune ", "stop ", "instead", "rather ",
    ]
    return any(text_lower.startswith(m) or f" {m}" in f" {text_lower}" for m in instruction_markers)


def _try_revise_decision(
    instruction: str,
    agent_name: str,
    current_decision: Decision,
    stage: str,
    coordinator: Any,
    context: PipelineContext | None,
) -> Decision | None:
    """Ask the LLM to revise the current decision based on user instruction."""
    if context is None:
        return None

    revision_prompt = f"""\
The user has given this instruction at the {stage} stage:
"{instruction}"

The current recommendation is:
  Action: {current_decision.action}
  Parameters: {current_decision.parameters}
  Reasoning: {current_decision.reasoning}

Revise the recommendation to incorporate the user's instruction.
Return a JSON object with: action, parameters, reasoning, confidence.
"""

    result = coordinator.client.ask(
        system_prompt=f"You are revising an ML pipeline decision. Current stage: {stage}.",
        user_message=revision_prompt,
        agent_name=f"{agent_name}_revision",
        max_tokens=512,
        temperature=0.2,
    )

    if result is None or result.get("_parse_failed"):
        return None

    try:
        return Decision(
            action=result.get("action", current_decision.action),
            parameters=result.get("parameters", current_decision.parameters),
            reasoning=result.get("reasoning", "Revised based on user instruction"),
            confidence=result.get("confidence", 0.9),
        )
    except Exception:
        return None
