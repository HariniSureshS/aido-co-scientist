"""Rich Live dashboard — real-time updating terminal display during pipeline execution.

Shows a persistent layout that updates in-place with:
- Pipeline progress (current step, elapsed time)
- Model leaderboard (live-updating as models train)
- Agent reasoning (current thought/action)
- Cost tracker (LLM spend)
- Warnings/errors
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


console = Console()


class LiveDashboard:
    """Real-time updating dashboard for the pipeline.

    Usage:
        dash = LiveDashboard(dataset_path="RNA/...", budget=10, max_cost=5.0)
        dash.start()

        dash.set_step("load_profile", 1, 7)
        dash.add_model("xgboost_default", "standard", 0.6279, 0.3)
        dash.set_agent_thought("XGBoost scored 0.63. Let me try LightGBM.")
        dash.set_agent_action("train_model", {"model_type": "lightgbm"})

        dash.stop()
    """

    STEP_LABELS = {
        "load_profile": "Load & Profile",
        "preprocess_split": "Preprocess & Split",
        "baselines": "Train & Model",
        "hp_search": "HP Search",
        "iteration": "Iteration",
        "export": "Export Model",
        "report": "Generate Report",
        "review": "Report Review",
        "validate_data": "Validate Data",
        "validate_profile": "Validate Profile",
        "validate_preprocess": "Validate Preprocessing",
        "validate_split": "Validate Split",
        "validate_export": "Validate Export",
        "validate_train_script": "Test train.py",
        "validate_predict_script": "Test predict.py",
    }

    def __init__(
        self,
        dataset_path: str = "",
        mode: str = "auto",
        budget: int = 10,
        max_cost: float = 5.0,
        version: str = "0.1.0",
        output_dir: str = "",
    ):
        self.dataset_path = dataset_path
        self.mode = mode
        self.budget = budget
        self.max_cost = max_cost
        self.version = version
        self.output_dir = output_dir

        # State
        self._current_step = ""
        self._step_num = 0
        self._total_steps = 7
        self._start_time = time.time()
        self._models: list[dict] = []
        self._best_model = ""
        self._best_score = 0.0
        self._primary_metric = ""
        self._agent_thought = ""
        self._agent_action = ""
        self._agent_step = 0
        self._agent_max_steps = 25
        self._agent_name = "ML Engineer"  # which agent is currently active
        self._cost_spent = 0.0
        self._warnings: list[str] = []
        self._status = "Running"
        self._lower_is_better = False
        self._agent_log: list[dict] = []  # full conversation history
        self._elo_rankings: list[dict] = []  # current Elo rankings
        self._tree_search_active = False
        self._tree_branch_id: int = 0
        self._tree_branch_depth: int = 0
        self._tree_branch_score: float = 0.0
        self._tree_total_nodes: int = 0

        # Validation agent state
        self._validation_checks: list[dict] = []  # {step, status, issues, fixes}

        self._live: Live | None = None

    def start(self) -> None:
        """Start the live display."""
        self._start_time = time.time()
        self._live = Live(
            self._build_layout(),
            console=console,
            refresh_per_second=2,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display and save the full agent conversation log."""
        if self._live:
            self._status = "Complete"
            self._live.update(self._build_layout())
            self._live.stop()
            self._live = None

        # Save full agent conversation log to file
        self._save_agent_log()

    def _refresh(self) -> None:
        """Update the live display."""
        if self._live:
            self._live.update(self._build_layout())

    # ── Update methods (called by pipeline/agent) ──

    def set_step(self, step_key: str, step_num: int, total: int) -> None:
        self._current_step = step_key
        self._step_num = step_num
        self._total_steps = total
        self._refresh()

    def set_metric(self, metric: str, lower_is_better: bool = False) -> None:
        self._primary_metric = metric
        self._lower_is_better = lower_is_better
        self._refresh()

    def add_model(self, name: str, tier: str, score: float, train_time: float) -> None:
        self._models.append({
            "name": name, "tier": tier, "score": score, "time": train_time,
        })
        # Update best
        if not self._best_model:
            self._best_model = name
            self._best_score = score
        else:
            is_better = (score < self._best_score) if self._lower_is_better else (score > self._best_score)
            if is_better:
                self._best_model = name
                self._best_score = score
        self._refresh()

    def set_agent_name(self, name: str) -> None:
        self._agent_name = name
        self._refresh()

    def set_agent_thought(self, thought: str) -> None:
        self._agent_thought = thought
        # Print full thought below dashboard (scrollable) so nothing is lost
        if self._live and thought:
            self._live.console.print(f"  [bold]Thought:[/bold] {thought}")
        self._refresh()

    def set_agent_action(self, action: str, params: dict | None = None) -> None:
        params_str = ""
        if params:
            parts = [f"{k}={v}" for k, v in list(params.items())[:3]]
            params_str = f"({', '.join(parts)})"
        self._agent_action = f"{action}{params_str}"
        self._refresh()

    def set_agent_step(self, step: int, max_steps: int) -> None:
        self._agent_step = step
        self._agent_max_steps = max_steps
        self._refresh()

    def set_cost(self, spent: float) -> None:
        self._cost_spent = spent
        self._refresh()

    def add_warning(self, msg: str) -> None:
        self._warnings.append(msg)
        if len(self._warnings) > 5:
            self._warnings = self._warnings[-5:]
        self._refresh()

    def set_validation_running(self, step: str) -> None:
        """Show that the validation agent is checking a step."""
        self._validation_checks.append({
            "step": step,
            "status": "running",
            "issues": [],
            "fixes": [],
        })
        self._refresh()

    def update_validation_result(self, step: str, passed: bool, issues: list[str], fixes: list[str]) -> None:
        """Update the validation result for a step."""
        # Find and update existing entry, or create new
        for entry in self._validation_checks:
            if entry["step"] == step and entry["status"] == "running":
                entry["status"] = "pass" if passed and not fixes else ("fixed" if fixes else ("fail" if not passed else "pass"))
                entry["issues"] = issues
                entry["fixes"] = fixes
                break
        else:
            self._validation_checks.append({
                "step": step,
                "status": "pass" if passed and not fixes else ("fixed" if fixes else "fail"),
                "issues": issues,
                "fixes": fixes,
            })
        # Print below dashboard for scrollable history
        if self._live:
            status_color = {"pass": "green", "fixed": "yellow", "fail": "red", "running": "cyan"}.get(
                self._validation_checks[-1]["status"], "white"
            )
            self._live.console.print(
                f"  🔍 [bold]Validation[/bold] ({step}): [{status_color}]{self._validation_checks[-1]['status'].upper()}[/{status_color}]"
            )
            for fix in fixes:
                self._live.console.print(f"    [green]Fixed: {fix}[/green]")
            for issue in issues:
                self._live.console.print(f"    [yellow]{issue}[/yellow]")
        self._refresh()

    def set_elo_rankings(self, rankings: list[dict]) -> None:
        """Update the Elo tournament rankings display.

        Args:
            rankings: List of dicts with keys: name, elo, matches, wins.
        """
        self._elo_rankings = rankings[:5]  # top 5
        self._refresh()

    def set_tree_search(
        self,
        active: bool = True,
        branch_id: int = 0,
        depth: int = 0,
        score: float = 0.0,
        total_nodes: int = 0,
    ) -> None:
        """Update tree search state display."""
        self._tree_search_active = active
        self._tree_branch_id = branch_id
        self._tree_branch_depth = depth
        self._tree_branch_score = score
        self._tree_total_nodes = total_nodes
        self._refresh()

    def add_agent_message(
        self,
        agent: str,
        stage: str,
        message: str,
        msg_type: str = "consult",
    ) -> None:
        """Record an agent conversation entry.

        The full message is printed below the dashboard as scrollable output.
        Only the last few entries appear in the live panel as a compact preview.

        Args:
            agent: Agent name (e.g. "Biology Specialist", "Data Analyst").
            stage: Pipeline stage (e.g. "profiling", "preprocessing", "iteration").
            message: The agent's reasoning or output summary.
            msg_type: One of "consult", "debate", "react_tool".
        """
        agent_colors = {
            "Biology Specialist": "green",
            "Data Analyst": "cyan",
            "ML Engineer": "magenta",
            "Research Agent": "blue",
            "Coordinator": "yellow",
            "Validation Agent": "bright_green",
        }
        type_icons = {"consult": "💬", "debate": "⚔️", "react_tool": "🔧"}

        entry = {
            "agent": agent,
            "stage": stage,
            "message": message,
            "type": msg_type,
            "time": time.time() - self._start_time,
        }
        self._agent_log.append(entry)

        # Print full message below dashboard (scrollable)
        color = agent_colors.get(agent, "white")
        icon = type_icons.get(msg_type, "•")
        elapsed = _format_duration(entry["time"])
        stage_tag = f"({stage})" if stage else ""
        if self._live:
            self._live.console.print(
                f"  {icon} [dim]{elapsed}[/dim] [{color}][bold]{agent}[/bold][/{color}] "
                f"[dim]{stage_tag}[/dim] {message}"
            )

        self._refresh()

    def log(self, message: str) -> None:
        """Print a message below the live display."""
        if self._live:
            self._live.console.print(f"  {message}")

    def _save_agent_log(self) -> None:
        """Save the full agent conversation log to a text file."""
        if not self._agent_log:
            return

        import os
        # Try output_dir first, fall back to current directory
        out_dir = self.output_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "agent_conversations.log")

        try:
            with open(path, "w") as f:
                f.write(f"Co-Scientist Agent Conversation Log\n")
                f.write(f"Dataset: {self.dataset_path}\n")
                f.write(f"Mode: {self.mode} | Budget: {self.budget} | Max Cost: ${self.max_cost:.2f}\n")
                f.write(f"{'=' * 80}\n\n")

                for entry in self._agent_log:
                    elapsed = _format_duration(entry["time"])
                    type_label = entry["type"].upper()
                    f.write(f"[{elapsed}] [{type_label}] {entry['agent']} ({entry['stage']})\n")
                    f.write(f"  {entry['message']}\n\n")

                # Also append model leaderboard
                if self._models:
                    f.write(f"{'=' * 80}\n")
                    f.write(f"Model Leaderboard ({self._primary_metric or 'Score'})\n")
                    f.write(f"{'-' * 80}\n")
                    sorted_models = sorted(
                        self._models,
                        key=lambda m: m["score"],
                        reverse=not self._lower_is_better,
                    )
                    for i, m in enumerate(sorted_models, 1):
                        best_marker = " *" if m["name"] == self._best_model else ""
                        f.write(f"  {i}. {m['name']} — {m['score']:.4f} ({m['tier']}, {m['time']:.1f}s){best_marker}\n")

                # Elo rankings
                if self._elo_rankings:
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"Tournament Rankings (Elo)\n")
                    f.write(f"{'-' * 80}\n")
                    for i, r in enumerate(self._elo_rankings, 1):
                        f.write(f"  {i}. {r.get('name', '?')} — Elo: {r.get('elo', 1500):.0f} "
                                f"(W:{r.get('wins', 0)}/M:{r.get('matches', 0)})\n")

            console.print(f"  [dim]Agent conversation log saved to {path}[/dim]")
        except Exception as e:
            console.print(f"  [dim]Could not save agent log: {e}[/dim]")

    # ── Layout building ──

    def _build_layout(self) -> Panel:
        """Build the full dashboard layout."""
        sections = []

        # Header + Progress
        sections.append(self._build_header())

        # Model leaderboard
        if self._models:
            sections.append(self._build_leaderboard())

        # Elo tournament rankings
        if self._elo_rankings:
            sections.append(self._build_elo_panel())

        # Tree search state
        if self._tree_search_active:
            sections.append(self._build_tree_search_panel())

        # Agent conversation log
        if self._agent_log:
            sections.append(self._build_agent_log_panel())

        # Agent reasoning (live ReAct step)
        if self._agent_thought or self._agent_action:
            sections.append(self._build_agent_panel())

        # Validation agent
        if self._validation_checks:
            sections.append(self._build_validation_panel())

        # Cost
        sections.append(self._build_cost_bar())

        # Warnings
        if self._warnings:
            sections.append(self._build_warnings())

        return Panel(
            Group(*sections),
            title=f"[bold green]Co-Scientist v{self.version}[/bold green]",
            subtitle=f"[dim]{self.dataset_path}[/dim]",
            border_style="green",
            padding=(0, 1),
        )

    def _build_header(self) -> Table:
        """Build the header with progress info."""
        elapsed = time.time() - self._start_time
        step_label = self.STEP_LABELS.get(self._current_step, self._current_step)

        grid = Table.grid(padding=(0, 2))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        left = (
            f"[bold]Step:[/bold] {self._step_num}/{self._total_steps} — {step_label}\n"
            f"[bold]Mode:[/bold] {self.mode}  [bold]Budget:[/bold] {self.budget}"
        )
        right = (
            f"[bold]Status:[/bold] {'[green]' + self._status + '[/green]' if self._status == 'Complete' else '[cyan]' + self._status + '[/cyan]'}\n"
            f"[bold]Elapsed:[/bold] {_format_duration(elapsed)}"
        )
        grid.add_row(left, right)
        return grid

    def _build_leaderboard(self) -> Table:
        """Build the model leaderboard table."""
        metric_name = self._primary_metric or "Score"

        table = Table(
            title="[bold]Model Leaderboard[/bold]",
            show_lines=False,
            border_style="blue",
            title_style="bold blue",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("", width=2)
        table.add_column("Model", style="cyan", ratio=3)
        table.add_column("Tier", style="dim", ratio=1)
        table.add_column(metric_name, justify="right", ratio=1)
        table.add_column("Time", justify="right", style="dim", ratio=1)

        # Sort models
        sorted_models = sorted(
            self._models,
            key=lambda m: m["score"],
            reverse=not self._lower_is_better,
        )

        for m in sorted_models[:10]:  # Show top 10
            is_best = m["name"] == self._best_model
            icon = "[bold green]*[/bold green]" if is_best else " "
            name_style = "bold green" if is_best else "cyan"
            table.add_row(
                icon,
                f"[{name_style}]{m['name']}[/{name_style}]",
                m["tier"],
                f"{m['score']:.4f}",
                f"{m['time']:.1f}s",
            )

        return table

    def _build_agent_panel(self) -> Panel:
        """Build the agent reasoning panel."""
        lines = []
        if self._agent_step > 0:
            lines.append(f"[bold]ReAct Step:[/bold] {self._agent_step}/{self._agent_max_steps}")
        if self._agent_thought:
            thought_display = self._agent_thought[:500]
            if len(self._agent_thought) > 500:
                thought_display += "..."
            lines.append(f"[bold]Thought:[/bold] {thought_display}")
        if self._agent_action:
            # Map action to responsible agent
            agent_for_action = {
                "train_model": "ML Engineer",
                "tune_model": "ML Engineer",
                "tune_hyperparameters": "ML Engineer",
                "design_model": "ML Engineer",
                "build_ensemble": "ML Engineer",
                "analyze_features": "Data Analyst",
                "analyze_errors": "Data Analyst",
                "inspect_features": "Data Analyst",
                "summarize_data": "Data Analyst",
                "get_model_scores": "ML Engineer",
                "get_rankings": "ML Engineer",
                "consult_biology": "Biology Specialist",
                "diagnose_data": "Data Analyst",
                "evaluate_test_set": "Evaluator",
                "finish": "ML Engineer",
                "backtrack": "ML Engineer",
            }
            action_name = self._agent_action.split("(")[0].strip()
            active_agent = agent_for_action.get(action_name, self._agent_name)
            lines.append(f"[bold]Agent:[/bold]   [magenta]{active_agent}[/magenta]")
            lines.append(f"[bold]Action:[/bold]  [yellow]{self._agent_action}[/yellow]")

        return Panel(
            "\n".join(lines),
            title="[bold cyan]Agent Reasoning[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )

    def _build_elo_panel(self) -> Table:
        """Build the Elo tournament rankings table."""
        table = Table(
            title="[bold]Tournament Rankings (Elo)[/bold]",
            show_lines=False,
            border_style="bright_magenta",
            title_style="bold bright_magenta",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("#", width=3, justify="right")
        table.add_column("Model", style="cyan", ratio=3)
        table.add_column("Elo", justify="right", ratio=1, style="bold")
        table.add_column("W/M", justify="right", ratio=1, style="dim")

        for i, r in enumerate(self._elo_rankings, 1):
            name = r.get("name", "?")
            elo = r.get("elo", 1500)
            wins = r.get("wins", 0)
            matches = r.get("matches", 0)
            # Color top model
            style = "bold green" if i == 1 else "cyan"
            table.add_row(
                str(i),
                f"[{style}]{name}[/{style}]",
                f"{elo:.0f}",
                f"{wins}/{matches}",
            )

        return table

    def _build_tree_search_panel(self) -> Panel:
        """Build the tree search state panel."""
        lines = [
            f"[bold]Mode:[/bold] Tree Search (MCTS)",
            f"[bold]Nodes:[/bold] {self._tree_total_nodes}",
            f"[bold]Current Branch:[/bold] node {self._tree_branch_id} "
            f"(depth={self._tree_branch_depth}, score={self._tree_branch_score:.4f})",
        ]
        return Panel(
            "\n".join(lines),
            title="[bold bright_yellow]Tree Search[/bold bright_yellow]",
            border_style="bright_yellow",
            padding=(0, 1),
        )

    def _build_agent_log_panel(self) -> Panel:
        """Build a scrolling log of all agent conversations."""
        # Agent name → color mapping
        agent_colors = {
            "Biology Specialist": "green",
            "Data Analyst": "cyan",
            "ML Engineer": "magenta",
            "Research Agent": "blue",
            "Coordinator": "yellow",
            "Validation Agent": "bright_green",
        }
        type_icons = {
            "consult": "💬",
            "debate": "⚔️",
            "react_tool": "🔧",
        }

        # Show only the 2 most recent entries — full history scrolls above
        recent = self._agent_log[-2:]
        lines = []
        for entry in recent:
            color = agent_colors.get(entry["agent"], "white")
            icon = type_icons.get(entry["type"], "•")
            elapsed = _format_duration(entry["time"])
            # Truncate to keep each entry on ~1 line
            msg = entry["message"].replace("\n", " ")
            max_len = 80 if entry["type"] == "react_tool" else 120
            if len(msg) > max_len:
                msg = msg[:max_len] + "..."
            stage_tag = f"[dim]({entry['stage']})[/dim]" if entry["stage"] else ""
            lines.append(
                f"  {icon} [dim]{elapsed}[/dim] [{color}][bold]{entry['agent']}[/bold][/{color}] "
                f"{stage_tag} {msg}"
            )

        return Panel(
            "\n".join(lines),
            title=f"[bold]Agent Conversations[/bold] [dim]({len(self._agent_log)} total)[/dim]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_validation_panel(self) -> Panel:
        """Build the validation agent status panel."""
        status_icons = {
            "running": "[cyan]⟳[/cyan]",
            "pass": "[green]✓[/green]",
            "fixed": "[yellow]⚡[/yellow]",
            "fail": "[red]✗[/red]",
        }
        lines = []
        for entry in self._validation_checks:
            icon = status_icons.get(entry["status"], "•")
            step_label = self.STEP_LABELS.get(entry["step"], entry["step"])
            status_text = entry["status"].upper()
            color = {"pass": "green", "fixed": "yellow", "fail": "red", "running": "cyan"}.get(entry["status"], "white")
            line = f"  {icon} [bold]{step_label}[/bold]: [{color}]{status_text}[/{color}]"
            if entry["fixes"]:
                line += f" [dim]({len(entry['fixes'])} fix{'es' if len(entry['fixes']) != 1 else ''})[/dim]"
            if entry["issues"]:
                line += f" [dim]({len(entry['issues'])} issue{'s' if len(entry['issues']) != 1 else ''})[/dim]"
            lines.append(line)
        return Panel(
            "\n".join(lines),
            title="[bold bright_green]Validation Agent[/bold bright_green] 🔍",
            border_style="bright_green",
            padding=(0, 1),
        )

    def _build_cost_bar(self) -> Text:
        """Build a cost usage bar."""
        pct = min(self._cost_spent / self.max_cost, 1.0) if self.max_cost > 0 else 0
        bar_width = 30
        filled = int(pct * bar_width)
        bar = "[green]" + "█" * filled + "[/green]" + "[dim]░[/dim]" * (bar_width - filled)
        return Text.from_markup(
            f"  [bold]LLM Cost:[/bold] ${self._cost_spent:.3f} / ${self.max_cost:.2f}  {bar}  {pct:.0%}"
        )

    def _build_warnings(self) -> Panel:
        """Build warnings panel."""
        lines = [f"[yellow]• {w}[/yellow]" for w in self._warnings[-3:]]
        return Panel(
            "\n".join(lines),
            title="[yellow]Warnings[/yellow]",
            border_style="yellow",
            padding=(0, 1),
        )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"
