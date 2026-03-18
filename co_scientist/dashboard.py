"""Rich terminal dashboard — styled pipeline output.

Architecture Section 11.2: Live terminal dashboard with header, model table,
progress, and warnings.

Phase B: Static dashboard with styled panels and progress tracking.
Phase C upgrade: Rich Live display for real-time agent loop updates.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()


# ---------------------------------------------------------------------------
# Pipeline steps for progress tracking
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    ("load_profile",     "Load & Profile"),
    ("preprocess_split", "Preprocess & Split"),
    ("baselines",        "Train Baselines"),
    ("hp_search",        "HP Search"),
    ("export",           "Export Model"),
    ("report",           "Generate Report"),
]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def print_header(
    version: str,
    dataset_path: str,
    mode: str,
    budget: int,
    max_cost: float,
    output_dir: str,
    complexity_score: float | None = None,
    complexity_level: str | None = None,
) -> None:
    """Print a styled header panel at pipeline start."""
    lines = []
    lines.append(f"[bold]Dataset:[/bold]    {dataset_path}")
    lines.append(f"[bold]Mode:[/bold]       {mode}")
    lines.append(f"[bold]Budget:[/bold]     {budget} steps")
    lines.append(f"[bold]Max cost:[/bold]   ${max_cost:.2f}")
    lines.append(f"[bold]Output:[/bold]     {output_dir}")
    if complexity_score is not None:
        color = _level_color(complexity_level or "")
        lines.append(f"[bold]Complexity:[/bold] [{color}]{complexity_score:.1f}/10 ({complexity_level})[/{color}]")

    content = "\n".join(lines)
    panel = Panel(
        content,
        title=f"[bold green]Co-Scientist v{version}[/bold green]",
        border_style="green",
        padding=(0, 2),
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# Step progress
# ---------------------------------------------------------------------------

def print_step_start(step_key: str, step_num: int, total: int) -> None:
    """Print a styled step header."""
    label = dict(PIPELINE_STEPS).get(step_key, step_key)
    console.print()
    console.print(Rule(
        f"[bold]Step {step_num}/{total}: {label}[/bold]",
        style="cyan",
    ))


def print_step_resumed(step_key: str, step_num: int, total: int) -> None:
    """Print a dimmed step header for resumed steps."""
    label = dict(PIPELINE_STEPS).get(step_key, step_key)
    console.print(f"  [dim]Step {step_num}/{total}: {label} (resumed)[/dim]")


def print_step_complete(step_key: str, summary: str = "") -> None:
    """Print step completion marker."""
    if summary:
        console.print(f"  [green]Done.[/green] {summary}")


# ---------------------------------------------------------------------------
# Model results table (enhanced)
# ---------------------------------------------------------------------------

def print_model_table(
    results: list[Any],
    eval_config: Any,
    best_name: str = "",
) -> None:
    """Print a styled model comparison table with best model highlighted."""
    metric = eval_config.primary_metric
    lower_is_better = metric in ("mse", "rmse", "mae")
    sorted_results = sorted(
        results,
        key=lambda r: r.primary_metric_value,
        reverse=not lower_is_better,
    )

    table = Table(
        title="Model Comparison",
        show_lines=True,
        title_style="bold",
        border_style="blue",
    )
    table.add_column("", width=2)  # status icon
    table.add_column("Model", style="cyan")
    table.add_column("Tier", style="dim")
    table.add_column(metric, style="bold green", justify="right")

    for m in eval_config.secondary_metrics[:3]:
        table.add_column(m, justify="right")

    table.add_column("Time", justify="right", style="dim")

    for r in sorted_results:
        is_best = r.model_name == best_name
        icon = "[bold green]*[/bold green]" if is_best else " "
        name_style = "bold green" if is_best else "cyan"

        row = [
            icon,
            f"[{name_style}]{r.model_name}[/{name_style}]",
            r.tier,
            f"{r.primary_metric_value:.4f}",
        ]
        for m in eval_config.secondary_metrics[:3]:
            val = r.metrics.get(m)
            row.append(f"{val:.4f}" if val is not None else "-")
        row.append(f"{r.train_time_seconds:.1f}s")
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Final summary panel
# ---------------------------------------------------------------------------

def print_final_summary(
    dataset_path: str,
    best_model: str,
    best_metric: str,
    best_value: float,
    total_models: int,
    elapsed_seconds: float,
    output_dir: str,
    warnings: int = 0,
    complexity_level: str = "",
) -> None:
    """Print a styled final summary panel."""
    lines = []
    lines.append(f"[bold]Best model:[/bold]  [bold green]{best_model}[/bold green]")
    lines.append(f"[bold]Metric:[/bold]      {best_metric} = [bold]{best_value:.4f}[/bold]")
    lines.append(f"[bold]Models tried:[/bold] {total_models}")
    lines.append(f"[bold]Total time:[/bold]  {_format_duration(elapsed_seconds)}")
    lines.append(f"[bold]Output:[/bold]      {output_dir}")

    if warnings > 0:
        lines.append(f"[bold]Warnings:[/bold]    [yellow]{warnings}[/yellow] (see experiment log)")

    content = "\n".join(lines)
    panel = Panel(
        content,
        title="[bold green]Pipeline Complete[/bold green]",
        subtitle=f"[dim]{dataset_path}[/dim]",
        border_style="green",
        padding=(0, 2),
    )
    console.print()
    console.print(panel)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _level_color(level: str) -> str:
    return {
        "simple": "green",
        "moderate": "yellow",
        "complex": "red",
        "very_complex": "bold red",
    }.get(level, "white")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"
