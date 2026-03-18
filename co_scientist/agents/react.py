"""ReAct (Reasoning + Acting) agent for the modeling phase.

Replaces the separate baselines → HP search → iteration loop with a single
Thought → Action → Observation cycle driven by the LLM.

Falls back to None (triggering the deterministic path) when:
- 3 consecutive parse failures
- LLM budget exhausted
- Max steps reached
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from co_scientist.agents.tools import ToolRegistry, ToolResult
from co_scientist.data.types import DatasetProfile, SplitData
from co_scientist.evaluation.types import EvalConfig, ModelResult
from co_scientist.llm.client import ClaudeClient
from co_scientist.llm.cost import CostTracker
from co_scientist.llm.prompts import REACT_AGENT_SYSTEM, REACT_TREE_SEARCH_SYSTEM
from co_scientist.modeling.types import TrainedModel

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class ScratchpadEntry:
    """One step in the agent's reasoning trace."""

    step: int
    thought: str
    action: str
    action_params: dict[str, Any]
    observation: str
    score_after: float | None = None  # best score after this step


@dataclass
class ReactState:
    """Mutable state shared between the agent and its tools."""

    # Pipeline references (set once)
    profile: DatasetProfile
    split: SplitData
    eval_config: EvalConfig
    seed: int = 42

    # Mutated by tools
    trained_models: list[TrainedModel] = field(default_factory=list)
    results: list[ModelResult] = field(default_factory=list)
    best_result: ModelResult | None = None
    best_trained: TrainedModel | None = None

    # Optional: Elo ranker (set by coordinator when tournament ranking is active)
    elo_ranker: Any = None

    # LLM client for tools that need to call the LLM (e.g., design_model)
    llm_client: Any = None

    # Live dashboard (set by coordinator/CLI when live display is active)
    dashboard: Any = None


@dataclass
class ReactResult:
    """Final result of the ReAct agent run."""

    trained_models: list[TrainedModel]
    results: list[ModelResult]
    best_result: ModelResult
    best_trained: TrainedModel
    scratchpad: list[ScratchpadEntry]
    stop_reason: str
    total_steps: int
    improvements: int
    tree_search_log: dict[str, Any] | None = None
    elo_rankings: dict[str, Any] | None = None
    biology_assessment: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)\((\{.*?\})\)", re.DOTALL)
_ACTION_NO_PARAMS_RE = re.compile(r"Action:\s*(\w+)\(\s*\)")


def _parse_response(text: str) -> tuple[str, str, dict[str, Any]] | None:
    """Parse Thought + Action from LLM response.

    Returns (thought, action_name, action_params) or None on failure.
    """
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = _ACTION_RE.search(text)
    if action_match:
        action_name = action_match.group(1)
        try:
            action_params = json.loads(action_match.group(2))
        except json.JSONDecodeError:
            return None
        return thought, action_name, action_params

    # Try no-params variant: tool_name({}) or tool_name()
    no_params_match = _ACTION_NO_PARAMS_RE.search(text)
    if no_params_match:
        return thought, no_params_match.group(1), {}

    return None


# ---------------------------------------------------------------------------
# Scratchpad compression
# ---------------------------------------------------------------------------

def _compress_scratchpad(entries: list[ScratchpadEntry], keep_last: int = 4) -> str:
    """Compress older scratchpad entries into a summary, keep last N verbatim."""
    if len(entries) <= keep_last:
        return _format_entries(entries)

    old = entries[:-keep_last]
    recent = entries[-keep_last:]

    # Summarize old entries
    summary_parts = []
    for e in old:
        score_str = f", score={e.score_after:.4f}" if e.score_after is not None else ""
        summary_parts.append(f"Step {e.step}: {e.action}({_brief_params(e.action_params)}){score_str}")

    compressed = f"[Summary of steps 1-{old[-1].step}]\n" + "; ".join(summary_parts) + "\n\n"
    compressed += _format_entries(recent)
    return compressed


def _format_entries(entries: list[ScratchpadEntry]) -> str:
    """Format scratchpad entries for the prompt."""
    parts = []
    for e in entries:
        parts.append(
            f"Step {e.step}:\n"
            f"  Thought: {e.thought}\n"
            f"  Action: {e.action}({json.dumps(e.action_params)})\n"
            f"  Observation: {e.observation}"
        )
    return "\n\n".join(parts)


def _brief_params(params: dict) -> str:
    """One-line summary of params."""
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 30:
            v = v[:27] + "..."
        parts.append(f"{k}={v}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# State summary for prompt
# ---------------------------------------------------------------------------

def _build_state_summary(state: ReactState) -> str:
    """Build a summary of the current pipeline state for the prompt."""
    parts = [
        f"Dataset: {state.profile.dataset_name}",
        f"Modality: {state.profile.modality.value}",
        f"Task: {state.profile.task_type.value}",
        f"Samples: {state.profile.num_samples}, Features: {state.split.X_train.shape[1]}",
        f"Primary metric: {state.eval_config.primary_metric}",
    ]
    if state.profile.num_classes > 0:
        parts.append(f"Classes: {state.profile.num_classes}")

    if state.results:
        parts.append(f"\nModels trained: {len(state.results)}")
        lower_is_better = state.eval_config.primary_metric in ("mse", "rmse", "mae")
        sorted_results = sorted(
            state.results,
            key=lambda r: r.primary_metric_value,
            reverse=not lower_is_better,
        )
        for r in sorted_results[:5]:
            parts.append(f"  {r.model_name}: {r.primary_metric_value:.4f}")
        if state.best_result:
            parts.append(f"Current best: {state.best_result.model_name} = {state.best_result.primary_metric_value:.4f}")

    # Include Elo rankings if available
    if state.elo_ranker and state.elo_ranker.players:
        rankings = state.elo_ranker.get_rankings()[:5]
        parts.append("\nElo Rankings:")
        for p in rankings:
            parts.append(f"  {p.name}: Elo={p.elo:.0f}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ReactAgent
# ---------------------------------------------------------------------------

class ReactAgent:
    """ReAct agent that drives the modeling phase through Thought → Action → Observation cycles."""

    def __init__(
        self,
        client: ClaudeClient,
        tool_registry: ToolRegistry,
        cost_tracker: CostTracker,
        max_steps: int = 25,
        patience: int = 8,
        max_parse_failures: int = 3,
        max_wall_seconds: float = 1800.0,
        max_repeated_actions: int = 4,
        max_consecutive_tool_failures: int = 5,
        tool_timeout_seconds: float = 120.0,
    ):
        self.client = client
        self.tool_registry = tool_registry
        self.cost_tracker = cost_tracker
        self.max_steps = max_steps
        self.patience = patience
        self.max_parse_failures = max_parse_failures
        self.max_wall_seconds = max_wall_seconds
        self.max_repeated_actions = max_repeated_actions
        self.max_consecutive_tool_failures = max_consecutive_tool_failures
        self.tool_timeout_seconds = tool_timeout_seconds

    def run(self, state: ReactState, exp_log: Any = None, interactive: bool = False,
            coordinator: Any = None) -> ReactResult | None:
        """Run the ReAct loop. Returns ReactResult or None (triggers deterministic fallback).

        Args:
            interactive: If True, pause after each step for user feedback.
            coordinator: Coordinator instance (needed for interactive chat).
        """
        import time as _time

        scratchpad: list[ScratchpadEntry] = []
        self._interactive = interactive
        self._coordinator = coordinator
        self._user_feedback: str | None = None  # injected into next step
        consecutive_parse_failures = 0
        consecutive_tool_failures = 0
        no_improve_count = 0
        improvements = 0
        stop_reason = ""
        wall_start = _time.time()

        lower_is_better = state.eval_config.primary_metric in ("mse", "rmse", "mae")

        # Build the system prompt with tool descriptions
        tool_descriptions = self.tool_registry.describe_all()
        system_prompt = REACT_AGENT_SYSTEM.format(
            tool_descriptions=tool_descriptions,
            tool_names=", ".join(self.tool_registry.tool_names()),
        )

        console.print("  [bold cyan]ReAct agent starting...[/bold cyan]")

        for step in range(1, self.max_steps + 1):
            # Check wall-clock timeout
            if _time.time() - wall_start > self.max_wall_seconds:
                stop_reason = f"wall_clock_timeout ({self.max_wall_seconds:.0f}s)"
                console.print(f"  [yellow]ReAct: wall-clock timeout ({self.max_wall_seconds:.0f}s)[/yellow]")
                break

            # Check cost budget (respects reservations for future debates)
            if self.cost_tracker.available_budget < 0.01:
                stop_reason = "cost_budget_exhausted"
                console.print("  [yellow]ReAct: LLM budget exhausted[/yellow]")
                break

            # Build user message
            state_summary = _build_state_summary(state)
            if scratchpad:
                # Compress every 8 steps
                if len(scratchpad) >= 8 and len(scratchpad) % 8 == 0:
                    history = _compress_scratchpad(scratchpad)
                else:
                    history = _format_entries(scratchpad)
                user_msg = f"Current state:\n{state_summary}\n\nHistory:\n{history}\n\nStep {step}: What is your next thought and action?"
                # Inject user feedback from interactive mode
                if self._user_feedback:
                    user_msg += f"\n\nIMPORTANT — The user has provided feedback: \"{self._user_feedback}\"\nTake this into account in your next thought and action."
                    self._user_feedback = None
            else:
                user_msg = f"Current state:\n{state_summary}\n\nStep {step}: This is the start. What is your first thought and action?"

            # Call LLM
            response = self.client.ask_text(
                system_prompt=system_prompt,
                user_message=user_msg,
                agent_name="react_agent",
                max_tokens=1024,
                temperature=0.3,
            )

            if response is None:
                # Retry once after a short delay before giving up
                console.print("  [yellow]ReAct: LLM call returned None, retrying...[/yellow]")
                import time as _time2
                _time2.sleep(3)
                response = self.client.ask_text(
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    agent_name="react_agent",
                    max_tokens=1024,
                    temperature=0.3,
                )
                if response is None:
                    stop_reason = "llm_unavailable"
                    console.print("  [yellow]ReAct: LLM still unavailable after retry, stopping[/yellow]")
                    break

            # Parse response
            parsed = _parse_response(response)
            if parsed is None:
                consecutive_parse_failures += 1
                logger.warning("ReAct parse failure %d/%d: %s", consecutive_parse_failures, self.max_parse_failures, response[:200])
                if consecutive_parse_failures >= self.max_parse_failures:
                    stop_reason = "parse_failures"
                    console.print("  [yellow]ReAct: too many parse failures, falling back[/yellow]")
                    return None  # Trigger deterministic fallback
                continue

            consecutive_parse_failures = 0
            thought, action_name, action_params = parsed

            console.print(f"\n  [bold]Step {step}[/bold]")
            if self._interactive:
                # Interactive: show full reasoning so user can follow along
                console.print(f"    [bold]Why:[/bold] {thought}")
                console.print(f"    [bold]Action:[/bold] {action_name}({json.dumps(action_params)})")
            else:
                console.print(f"    Thought: {thought[:120]}{'...' if len(thought) > 120 else ''}")
                console.print(f"    Action: {action_name}({json.dumps(action_params)[:80]})")

            # Update live dashboard
            if state.dashboard:
                state.dashboard.set_agent_step(step, self.max_steps)
                state.dashboard.set_agent_thought(thought)
                state.dashboard.set_agent_action(action_name, action_params)
                state.dashboard.set_cost(self.cost_tracker.total_cost)
                # Also log to agent conversation history
                agent_for_action = {
                    "train_model": "ML Engineer", "tune_hyperparameters": "ML Engineer",
                    "design_model": "ML Engineer", "build_ensemble": "ML Engineer",
                    "get_model_scores": "ML Engineer", "get_rankings": "ML Engineer",
                    "finish": "ML Engineer", "backtrack": "ML Engineer",
                    "analyze_errors": "Data Analyst", "inspect_features": "Data Analyst",
                    "summarize_data": "Data Analyst",
                    # consult_biology and diagnose_data log themselves in tool execute
                }
                agent_name = agent_for_action.get(action_name)
                if agent_name:
                    params_brief = ", ".join(f"{k}={v}" for k, v in list(action_params.items())[:2]) if action_params else ""
                    state.dashboard.add_agent_message(
                        agent=agent_name,
                        stage="react_loop",
                        message=f"{action_name}({params_brief})",
                        msg_type="react_tool",
                    )

            # Execute tool
            tool = self.tool_registry.get(action_name)
            if tool is None:
                observation = f"Error: unknown tool '{action_name}'. Available: {', '.join(self.tool_registry.tool_names())}"
                entry = ScratchpadEntry(
                    step=step, thought=thought, action=action_name,
                    action_params=action_params, observation=observation,
                    score_after=state.best_result.primary_metric_value if state.best_result else None,
                )
                scratchpad.append(entry)
                console.print(f"    Observation: {observation}")
                continue

            # Check for repeated actions — hard stop after max_repeated_actions
            repeat_count = 0
            if scratchpad:
                for e in reversed(scratchpad):
                    if e.action == action_name and e.action_params == action_params:
                        repeat_count += 1
                    else:
                        break
            if repeat_count >= self.max_repeated_actions:
                stop_reason = f"repeated_action ({action_name} x{repeat_count + 1})"
                console.print(f"  [yellow]ReAct: action repeated {repeat_count + 1} times, force stopping[/yellow]")
                break
            elif repeat_count >= 2:
                observation_prefix = f"WARNING: You've called this exact action {repeat_count + 1} times. Try something different. "
            else:
                observation_prefix = ""

            # Execute tool with timeout
            result = _execute_tool_with_timeout(tool, action_params, state, self.tool_timeout_seconds)
            if result is None:
                observation = f"Error: tool '{action_name}' timed out after {self.tool_timeout_seconds:.0f}s"
                entry = ScratchpadEntry(
                    step=step, thought=thought, action=action_name,
                    action_params=action_params, observation=observation,
                    score_after=state.best_result.primary_metric_value if state.best_result else None,
                )
                scratchpad.append(entry)
                consecutive_tool_failures += 1
                if consecutive_tool_failures >= self.max_consecutive_tool_failures:
                    stop_reason = f"consecutive_tool_failures ({self.max_consecutive_tool_failures})"
                    console.print(f"  [yellow]ReAct: {self.max_consecutive_tool_failures} consecutive tool failures[/yellow]")
                    break
                console.print(f"    Observation: {observation}")
                continue

            observation = observation_prefix + result.observation

            # Track consecutive tool failures
            if result.success:
                consecutive_tool_failures = 0
            else:
                consecutive_tool_failures += 1
                if consecutive_tool_failures >= self.max_consecutive_tool_failures:
                    stop_reason = f"consecutive_tool_failures ({self.max_consecutive_tool_failures})"
                    console.print(f"  [yellow]ReAct: {self.max_consecutive_tool_failures} consecutive tool failures[/yellow]")
                    break

            # Handle finish tool
            if action_name == "finish":
                entry = ScratchpadEntry(
                    step=step, thought=thought, action=action_name,
                    action_params=action_params, observation=observation,
                    score_after=state.best_result.primary_metric_value if state.best_result else None,
                )
                scratchpad.append(entry)
                stop_reason = result.data.get("stop_reason", "agent_finished")
                console.print(f"    [cyan]Agent finished: {stop_reason}[/cyan]")
                break

            # Update Elo rankings if active
            if result.success and result.score is not None and state.elo_ranker and state.results:
                state.elo_ranker.update_from_results(state.results, lower_is_better)
                # Push Elo rankings to live dashboard
                if state.dashboard:
                    rankings = state.elo_ranker.get_rankings()
                    state.dashboard.set_elo_rankings([
                        {"name": p.name, "elo": p.elo, "matches": p.matches, "wins": p.wins}
                        for p in rankings
                    ])

            # Update best if improved
            # Only count scoring actions (train/tune/ensemble/design) toward patience.
            # Non-scoring actions (summarize_data, analyze_errors, inspect_features, get_model_scores)
            # should NOT consume patience budget.
            _scoring_actions = {"train_model", "tune_hyperparameters", "build_ensemble", "design_model"}
            if result.success and result.score is not None and state.results:
                _update_best(state, lower_is_better)
                current_best = state.best_result.primary_metric_value if state.best_result else None
                if current_best is not None:
                    # Compare against the previous best to count improvement
                    prev_best = scratchpad[-1].score_after if scratchpad else None
                    if prev_best is not None and current_best != prev_best:
                        improvements += 1
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count = 0  # first model, can't measure improvement yet
            elif action_name in _scoring_actions:
                # Scoring action that failed — count against patience
                no_improve_count += 1
            # else: non-scoring action — do NOT increment no_improve_count

            entry = ScratchpadEntry(
                step=step, thought=thought, action=action_name,
                action_params=action_params, observation=observation,
                score_after=state.best_result.primary_metric_value if state.best_result else None,
            )
            scratchpad.append(entry)
            if self._interactive:
                # Interactive: show full observation + score context
                console.print(f"    [bold]Result:[/bold] {observation}")
                if result.success and result.score is not None:
                    best_score = state.best_result.primary_metric_value if state.best_result else None
                    metric = state.eval_config.primary_metric
                    if best_score is not None and result.score >= best_score:
                        console.print(f"    [bold green]>>> New best! {metric}={result.score:.4f}[/bold green]")
                    elif best_score is not None:
                        gap = best_score - result.score
                        console.print(f"    [dim]Best so far: {metric}={best_score:.4f} (this model is {gap:.4f} behind)[/dim]")
            else:
                console.print(f"    Observation: {observation[:150]}{'...' if len(observation) > 150 else ''}")

            # Interactive checkpoint: let user provide feedback or stop
            if self._interactive:
                user_input = self._interactive_pause(state, step, action_name, observation)
                if user_input == "__EXIT__":
                    stop_reason = "user_stopped"
                    console.print("  [yellow]Stopping ReAct loop as requested.[/yellow]")
                    break

            # Update live dashboard with model result
            if state.dashboard and result.success and result.score is not None:
                # Find the model tier
                tier = "react"
                for t in state.trained_models:
                    if t.config.name == result.model_name:
                        tier = t.config.tier
                        break
                state.dashboard.add_model(
                    result.model_name, tier, result.score,
                    getattr(result, "train_time", 0.0) if hasattr(result, "train_time") else 0.0,
                )
                state.dashboard.set_cost(self.cost_tracker.total_cost)

            # Log to experiment log
            if exp_log:
                exp_log.log("react_step", {
                    "step": step,
                    "thought": thought[:200],
                    "action": action_name,
                    "params": action_params,
                    "success": result.success,
                    "score": result.score,
                })

            # Patience check
            if no_improve_count >= self.patience:
                stop_reason = f"patience_exceeded ({self.patience} steps without improvement)"
                console.print(f"  [yellow]ReAct: {stop_reason}[/yellow]")
                break
        else:
            stop_reason = "max_steps_reached"

        # Must have at least one trained model
        if not state.results:
            console.print("  [yellow]ReAct: no models trained, falling back to deterministic[/yellow]")
            return None

        _update_best(state, lower_is_better)

        console.print(
            f"\n  [bold green]ReAct complete:[/bold green] {len(scratchpad)} steps, "
            f"{improvements} improvement(s), best={state.best_result.model_name} "
            f"({state.eval_config.primary_metric}={state.best_result.primary_metric_value:.4f})"
        )

        elo_rankings = state.elo_ranker.to_dict() if state.elo_ranker else None

        return ReactResult(
            trained_models=state.trained_models,
            results=state.results,
            best_result=state.best_result,
            best_trained=state.best_trained,
            scratchpad=scratchpad,
            stop_reason=stop_reason,
            total_steps=len(scratchpad),
            improvements=improvements,
            elo_rankings=elo_rankings,
            biology_assessment=getattr(state, 'biology_assessment', None),
        )


    def _interactive_pause(self, state: ReactState, step: int, action_name: str, observation: str) -> str:
        """Pause for user input after a ReAct step. Returns "__EXIT__" to stop, or sets self._user_feedback.

        The user can:
        - Press Enter to continue
        - Type "exit"/"stop" to halt the loop
        - Type feedback/instructions that get injected into the next LLM call
        - Ask a question (answered via LLM chat)
        """
        from rich.prompt import Prompt
        from rich.panel import Panel
        from rich.markdown import Markdown

        console.print()
        console.print("  [bold bright_cyan]─── Your Turn ───────────────────────────────────────────[/bold bright_cyan]")
        console.print("  [bright_cyan]Enter[/bright_cyan] = continue  |  [bright_cyan]exit[/bright_cyan] = stop  |  or type a question/instruction")
        user_input = Prompt.ask("  [bold bright_cyan]>[/bold bright_cyan]", default="")
        stripped = user_input.strip().lower()

        if not stripped:
            return ""  # continue

        if stripped in ("exit", "stop", "quit", "cancel", "abort"):
            return "__EXIT__"

        # Check if it's a question (send to LLM chat) or instruction (inject as feedback)
        if self._coordinator and hasattr(self._coordinator, 'client') and self._coordinator.client:
            from co_scientist.agents.interactive import chat_with_agent
            from co_scientist.agents.types import PipelineContext

            # Build minimal context for chat
            context = PipelineContext(
                dataset_path=state.profile.dataset_path,
                modality=state.profile.modality.value,
                task_type=state.profile.task_type.value,
                num_samples=state.profile.num_samples,
                num_features=state.profile.num_features,
                primary_metric=state.eval_config.primary_metric,
                model_scores={r.model_name: r.primary_metric_value for r in state.results},
                best_model_name=state.best_result.model_name if state.best_result else "",
                best_score=state.best_result.primary_metric_value if state.best_result else 0.0,
                stage="react_loop",
            )

            response = chat_with_agent(user_input, self._coordinator, context, "react_loop")
            if response:
                console.print()
                console.print(Panel(
                    Markdown(response),
                    title="[bold green]Co-Scientist[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                ))

        # Store as feedback for next ReAct step
        self._user_feedback = user_input
        console.print(f"  [bold bright_cyan]>>> Feedback noted — agent will consider this next[/bold bright_cyan]")
        console.print("  [bold bright_cyan]─────────────────────────────────────────────────────────[/bold bright_cyan]")
        return ""

    def run_tree_search(self, state: ReactState, exp_log: Any = None) -> ReactResult | None:
        """Run MCTS-inspired tree search instead of linear ReAct loop.

        Algorithm:
        1. Create root node from initial state
        2. While nodes < max_nodes and cost budget OK:
           a. Select node to expand via UCB1
           b. Restore state from that node
           c. Run 3-5 linear ReAct steps from that state
           d. If agent calls backtrack, or no improvement: create branch point
           e. Add child node(s) to tree
           f. Backpropagate scores
        3. Return best result across all nodes
        """
        from co_scientist.agents.tree_search import ExperimentTree

        tree = ExperimentTree(max_depth=4, max_nodes=20)
        lower_is_better = state.eval_config.primary_metric in ("mse", "rmse", "mae")

        # Build system prompt with tree search tools
        tool_descriptions = self.tool_registry.describe_all()
        system_prompt = REACT_TREE_SEARCH_SYSTEM.format(
            tool_descriptions=tool_descriptions,
            tool_names=", ".join(self.tool_registry.tool_names()),
        )

        console.print("  [bold cyan]ReAct tree search starting...[/bold cyan]")

        import time as _time
        wall_start = _time.time()

        # Initialize root
        initial_score = state.best_result.primary_metric_value if state.best_result else 0.0
        all_scratchpad: list[ScratchpadEntry] = []
        tree.init_root(state, all_scratchpad, initial_score, lower_is_better)

        total_steps = 0
        improvements = 0
        stop_reason = ""
        steps_per_branch = min(5, max(3, self.patience // 2))

        while len(tree.nodes) < tree.max_nodes:
            # Wall-clock timeout
            if _time.time() - wall_start > self.max_wall_seconds:
                stop_reason = f"wall_clock_timeout ({self.max_wall_seconds:.0f}s)"
                console.print(f"  [yellow]Tree search: wall-clock timeout[/yellow]")
                break

            if self.cost_tracker.available_budget < 0.01:
                stop_reason = "cost_budget_exhausted"
                break

            # Select node to expand
            node = tree.select_node_to_expand()
            if node.depth >= tree.max_depth:
                stop_reason = "max_depth_reached"
                break

            # Restore state from selected node
            branch_state = tree.restore_state(node)
            branch_scratchpad = list(node.scratchpad)
            branch_improved = False
            branch_score = node.score
            consecutive_parse_failures = 0

            console.print(f"\n  [bold]Branch from node {node.id} (depth={node.depth}, "
                          f"score={node.score:.4f})[/bold]")

            # Update tree search state on dashboard
            if state.dashboard:
                state.dashboard.set_tree_search(
                    active=True,
                    branch_id=node.id,
                    depth=node.depth,
                    score=node.score,
                    total_nodes=len(tree.nodes),
                )

            # Run a few linear steps from this state
            for step_in_branch in range(1, steps_per_branch + 1):
                total_steps += 1
                step = total_steps

                if self.cost_tracker.available_budget < 0.01:
                    break

                state_summary = _build_state_summary(branch_state)
                if branch_scratchpad:
                    history = _compress_scratchpad(branch_scratchpad) if len(branch_scratchpad) >= 8 else _format_entries(branch_scratchpad)
                    user_msg = f"Current state:\n{state_summary}\n\nHistory:\n{history}\n\nStep {step}: What is your next thought and action?"
                else:
                    user_msg = f"Current state:\n{state_summary}\n\nStep {step}: This is the start. What is your first thought and action?"

                response = self.client.ask_text(
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    agent_name="react_tree_search",
                    max_tokens=1024,
                    temperature=0.3,
                )

                if response is None:
                    stop_reason = "llm_unavailable"
                    break

                parsed = _parse_response(response)
                if parsed is None:
                    consecutive_parse_failures += 1
                    if consecutive_parse_failures >= self.max_parse_failures:
                        break
                    continue

                consecutive_parse_failures = 0
                thought, action_name, action_params = parsed

                console.print(f"\n  [bold]Step {step} (branch {node.id})[/bold]")
                console.print(f"    Thought: {thought[:120]}{'...' if len(thought) > 120 else ''}")
                console.print(f"    Action: {action_name}({json.dumps(action_params)[:80]})")

                # Update live dashboard for tree search steps
                if state.dashboard:
                    state.dashboard.set_agent_step(step, self.max_steps)
                    state.dashboard.set_agent_thought(thought)
                    state.dashboard.set_agent_action(action_name, action_params)
                    state.dashboard.set_cost(self.cost_tracker.total_cost)

                tool = self.tool_registry.get(action_name)
                if tool is None:
                    observation = f"Error: unknown tool '{action_name}'"
                    entry = ScratchpadEntry(step=step, thought=thought, action=action_name,
                                            action_params=action_params, observation=observation,
                                            score_after=branch_state.best_result.primary_metric_value if branch_state.best_result else None)
                    branch_scratchpad.append(entry)
                    all_scratchpad.append(entry)
                    continue

                result = tool.execute(action_params, branch_state)

                # Handle backtrack
                if action_name == "backtrack" or result.data.get("backtrack"):
                    entry = ScratchpadEntry(step=step, thought=thought, action=action_name,
                                            action_params=action_params, observation=result.observation,
                                            score_after=branch_state.best_result.primary_metric_value if branch_state.best_result else None)
                    branch_scratchpad.append(entry)
                    all_scratchpad.append(entry)
                    console.print(f"    [yellow]Backtracking: {result.observation}[/yellow]")
                    break

                # Handle finish
                if action_name == "finish":
                    entry = ScratchpadEntry(step=step, thought=thought, action=action_name,
                                            action_params=action_params, observation=result.observation,
                                            score_after=branch_state.best_result.primary_metric_value if branch_state.best_result else None)
                    branch_scratchpad.append(entry)
                    all_scratchpad.append(entry)
                    stop_reason = result.data.get("stop_reason", "agent_finished")
                    break

                # Update best
                if result.success and result.score is not None and branch_state.results:
                    _update_best(branch_state, lower_is_better)
                    new_score = branch_state.best_result.primary_metric_value if branch_state.best_result else branch_score
                    is_better = (new_score < branch_score) if lower_is_better else (new_score > branch_score)
                    if is_better:
                        branch_score = new_score
                        branch_improved = True
                        improvements += 1

                    # Update Elo rankings
                    if branch_state.elo_ranker and branch_state.results:
                        branch_state.elo_ranker.update_from_results(branch_state.results, lower_is_better)
                        if state.dashboard:
                            rankings = branch_state.elo_ranker.get_rankings()
                            state.dashboard.set_elo_rankings([
                                {"name": p.name, "elo": p.elo, "matches": p.matches, "wins": p.wins}
                                for p in rankings
                            ])

                    # Update dashboard model leaderboard
                    if state.dashboard and result.score is not None:
                        tier = "tree_search"
                        for t in branch_state.trained_models:
                            if t.config.name == result.model_name:
                                tier = t.config.tier
                                break
                        state.dashboard.add_model(
                            result.model_name, tier, result.score,
                            getattr(result, "train_time", 0.0) if hasattr(result, "train_time") else 0.0,
                        )

                entry = ScratchpadEntry(step=step, thought=thought, action=action_name,
                                        action_params=action_params, observation=result.observation,
                                        score_after=branch_state.best_result.primary_metric_value if branch_state.best_result else None)
                branch_scratchpad.append(entry)
                all_scratchpad.append(entry)
                console.print(f"    Observation: {result.observation[:150]}{'...' if len(result.observation) > 150 else ''}")

                if exp_log:
                    exp_log.log("react_tree_step", {
                        "step": step, "branch_node": node.id,
                        "thought": thought[:200], "action": action_name,
                        "params": action_params, "success": result.success, "score": result.score,
                    })

            # Add child node for this branch
            tree.add_child(node, branch_state, branch_scratchpad,
                           f"branch_{len(tree.nodes)}", branch_score)
            tree.backpropagate(node, branch_score)

            if stop_reason:
                break

            # Merge best results back into main state
            best_node = tree.global_best_node
            if best_node:
                best_state = tree.restore_state(best_node)
                state.trained_models = best_state.trained_models
                state.results = best_state.results
                state.best_result = best_state.best_result
                state.best_trained = best_state.best_trained

        if not stop_reason:
            stop_reason = "tree_search_complete"

        # Ensure main state has the best results
        best_node = tree.global_best_node
        if best_node:
            best_state = tree.restore_state(best_node)
            state.trained_models = best_state.trained_models
            state.results = best_state.results
            state.best_result = best_state.best_result
            state.best_trained = best_state.best_trained

        # Deactivate tree search on dashboard
        if state.dashboard:
            state.dashboard.set_tree_search(active=False)

        if not state.results:
            console.print("  [yellow]Tree search: no models trained, falling back[/yellow]")
            return None

        _update_best(state, lower_is_better)

        console.print(
            f"\n  [bold green]Tree search complete:[/bold green] {len(tree.nodes)} nodes, "
            f"{total_steps} steps, {improvements} improvement(s), "
            f"best={state.best_result.model_name} "
            f"({state.eval_config.primary_metric}={state.best_result.primary_metric_value:.4f})"
        )

        elo_rankings = state.elo_ranker.to_dict() if state.elo_ranker else None

        return ReactResult(
            trained_models=state.trained_models,
            results=state.results,
            best_result=state.best_result,
            best_trained=state.best_trained,
            scratchpad=all_scratchpad,
            stop_reason=stop_reason,
            total_steps=total_steps,
            improvements=improvements,
            tree_search_log=tree.summary(),
            elo_rankings=elo_rankings,
            biology_assessment=getattr(state, 'biology_assessment', None),
        )


def _execute_tool_with_timeout(
    tool: Any,
    params: dict[str, Any],
    state: "ReactState",
    timeout_seconds: float,
) -> ToolResult | None:
    """Execute a tool in a worker thread so the main thread stays free for dashboard updates.

    The previous segfault issue was caused by forcibly *killing* threads mid-C-extension.
    Here we never kill the thread — we just stop waiting after timeout_seconds.
    The thread is a daemon, so it dies cleanly when the process exits.

    If the tool finishes within timeout: return its result.
    If it exceeds timeout: return None (the thread keeps running but we move on).
    """
    import threading

    tool_name = tool.name
    start = time.time()
    result_holder: list[ToolResult | None] = [None]
    error_holder: list[Exception | None] = [None]
    done_event = threading.Event()

    def _worker():
        try:
            result_holder[0] = tool.execute(params, state)
        except Exception as e:
            error_holder[0] = e
        finally:
            done_event.set()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    # Wait for completion, printing heartbeat every 15s
    while not done_event.is_set():
        done_event.wait(timeout=15)
        if not done_event.is_set():
            elapsed = time.time() - start
            console.print(f"    [dim]... {tool_name} still running ({elapsed:.0f}s)[/dim]")
            if elapsed > timeout_seconds:
                console.print(f"    [yellow]{tool_name} timed out after {timeout_seconds:.0f}s[/yellow]")
                return None

    if error_holder[0] is not None:
        e = error_holder[0]
        return ToolResult(
            success=False,
            observation=f"Error: {tool_name} raised {type(e).__name__}: {e}",
        )

    return result_holder[0]


def _update_best(state: ReactState, lower_is_better: bool) -> None:
    """Update state.best_result and state.best_trained from current results."""
    if not state.results:
        return
    ranked = sorted(
        zip(state.results, state.trained_models),
        key=lambda pair: pair[0].primary_metric_value,
        reverse=not lower_is_better,
    )
    state.best_result, state.best_trained = ranked[0]
