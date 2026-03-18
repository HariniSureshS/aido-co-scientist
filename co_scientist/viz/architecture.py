"""Architecture diagrams for the AIDO Co-Scientist multi-agent system.

Provides two visualizations:
1. generate_architecture_diagram — static overview of the agent architecture,
   pipeline stages, tool nodes, and interaction patterns.
2. generate_agent_flow_diagram — run-specific flowchart showing which agents
   were consulted, what they proposed, any debates, and final decisions.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "coordinator": "#2C3E50",
    "ml_engineer": "#2980B9",
    "data_analyst": "#27AE60",
    "biology_specialist": "#8E44AD",
    "research": "#D35400",
    "tool": "#7F8C8D",
    "pipeline": "#1ABC9C",
    "react": "#F39C12",
    "tree_search": "#E74C3C",
    "debate": "#9B59B6",
    "background": "#FFFFFF",
    "edge_consult": "#2C3E50",
    "edge_decision": "#27AE60",
    "edge_debate": "#9B59B6",
}

TEXT_LIGHT = "#FFFFFF"
TEXT_DARK = "#2C3E50"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: str,
    text_color: str = TEXT_LIGHT,
    fontsize: int = 10,
    fontweight: str = "bold",
    boxstyle: str = "round,pad=0.15",
    alpha: float = 1.0,
    zorder: int = 3,
) -> FancyBboxPatch:
    """Draw a rounded rectangle with centred text and return the patch."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=boxstyle,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight,
        color=text_color, zorder=zorder + 1,
    )
    return box


def _arrow(
    ax: plt.Axes,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    label: str = "",
    color: str = "#2C3E50",
    linestyle: str = "-",
    linewidth: float = 1.5,
    fontsize: int = 7,
    label_offset: tuple[float, float] = (0, 0),
    zorder: int = 2,
    arrowstyle: str = "-|>",
    shrinkA: float = 8,
    shrinkB: float = 8,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    """Draw an annotated arrow between two points."""
    arrow = FancyArrowPatch(
        (x_start, y_start),
        (x_end, y_end),
        arrowstyle=arrowstyle,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        connectionstyle=connectionstyle,
        zorder=zorder,
        mutation_scale=12,
    )
    ax.add_patch(arrow)
    if label:
        mx = (x_start + x_end) / 2 + label_offset[0]
        my = (y_start + y_end) / 2 + label_offset[1]
        ax.text(
            mx, my, label,
            ha="center", va="center",
            fontsize=fontsize, fontstyle="italic",
            color=color, zorder=zorder + 1,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        )


# ---------------------------------------------------------------------------
# 1. Static architecture diagram
# ---------------------------------------------------------------------------

def generate_architecture_diagram(output_dir: Path) -> Path:
    """Generate architecture diagram, save to output_dir/figures/architecture.png"""
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "architecture.png"

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 13)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["background"])

    # ── Title ──────────────────────────────────────────────────────────
    ax.text(
        8, 12.5, "AIDO Co-Scientist Architecture",
        ha="center", va="center",
        fontsize=20, fontweight="bold", color=TEXT_DARK,
    )

    # ── Pipeline flow (top) ────────────────────────────────────────────
    pipeline_stages = [
        "Data", "Profile", "Preprocess", "Search",
        "Train", "HP Tune", "Iterate", "Export", "Test Evaluation", "Report",
    ]
    px_start = 1.0
    px_gap = 1.75
    py = 11.2
    pw, ph = 1.4, 0.5

    for i, stage in enumerate(pipeline_stages):
        px = px_start + i * px_gap
        _rounded_box(ax, px, py, pw, ph, stage, COLORS["pipeline"],
                      fontsize=8, fontweight="bold")
        if i > 0:
            prev_px = px_start + (i - 1) * px_gap
            _arrow(ax, prev_px + pw / 2, py, px - pw / 2, py,
                   color=COLORS["pipeline"], shrinkA=2, shrinkB=2)

    ax.text(
        px_start + (len(pipeline_stages) - 1) * px_gap / 2, 11.8,
        "Pipeline Stages",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=COLORS["pipeline"],
    )

    # ── Coordinator (centre) ───────────────────────────────────────────
    cx, cy = 8, 7.2
    cw, ch = 2.8, 1.0
    _rounded_box(ax, cx, cy, cw, ch, "Coordinator", COLORS["coordinator"],
                  fontsize=14, fontweight="bold")
    ax.text(
        cx, cy - 0.65, "supervisor  |  conflict resolution  |  budget",
        ha="center", va="center", fontsize=7, color="#95A5A6",
    )

    # ── Agent nodes ────────────────────────────────────────────────────
    agents = [
        ("ML Engineer",          COLORS["ml_engineer"],        3.0,  8.5),
        ("Data Analyst",         COLORS["data_analyst"],       13.0, 8.5),
        ("Biology Specialist",   COLORS["biology_specialist"], 3.0,  5.5),
        ("Research Agent",       COLORS["research"],           13.0, 5.5),
    ]
    aw, ah = 2.4, 0.8

    for label, color, ax_, ay_ in agents:
        _rounded_box(ax, ax_, ay_, aw, ah, label, color, fontsize=10)

    # ── Arrows: Coordinator <-> Agents ─────────────────────────────────
    # ML Engineer
    _arrow(ax, cx - cw / 2, cy + 0.2, 3.0 + aw / 2, 8.5 - 0.1,
           label="consult", color=COLORS["edge_consult"],
           label_offset=(0, 0.25))
    _arrow(ax, 3.0 + aw / 2, 8.5 + 0.1, cx - cw / 2, cy + 0.35,
           label="decision", color=COLORS["edge_decision"],
           label_offset=(0, -0.25), linestyle="--")

    # Data Analyst
    _arrow(ax, cx + cw / 2, cy + 0.2, 13.0 - aw / 2, 8.5 - 0.1,
           label="consult", color=COLORS["edge_consult"],
           label_offset=(0, 0.25))
    _arrow(ax, 13.0 - aw / 2, 8.5 + 0.1, cx + cw / 2, cy + 0.35,
           label="decision", color=COLORS["edge_decision"],
           label_offset=(0, -0.25), linestyle="--")

    # Biology Specialist
    _arrow(ax, cx - cw / 2, cy - 0.2, 3.0 + aw / 2, 5.5 + 0.1,
           label="consult", color=COLORS["edge_consult"],
           label_offset=(0, 0.25))
    _arrow(ax, 3.0 + aw / 2, 5.5 - 0.1, cx - cw / 2, cy - 0.35,
           label="decision", color=COLORS["edge_decision"],
           label_offset=(0, -0.25), linestyle="--")

    # Research Agent
    _arrow(ax, cx + cw / 2, cy - 0.2, 13.0 - aw / 2, 5.5 + 0.1,
           label="consult", color=COLORS["edge_consult"],
           label_offset=(0, 0.25))
    _arrow(ax, 13.0 - aw / 2, 5.5 - 0.1, cx + cw / 2, cy - 0.35,
           label="decision", color=COLORS["edge_decision"],
           label_offset=(0, -0.25), linestyle="--")

    # ── Debate arrows (dashed, between agents) ────────────────────────
    _arrow(ax, 3.0, 8.5 - ah / 2, 3.0, 5.5 + ah / 2,
           label="debate", color=COLORS["edge_debate"],
           linestyle=":", label_offset=(-0.6, 0),
           arrowstyle="<|-|>")
    _arrow(ax, 13.0, 8.5 - ah / 2, 13.0, 5.5 + ah / 2,
           label="debate", color=COLORS["edge_debate"],
           linestyle=":", label_offset=(0.6, 0),
           arrowstyle="<|-|>")

    # ── Tool nodes (below ML Engineer) ─────────────────────────────────
    tools = ["summarize_data", "train_model", "tune_model", "design_model", "consult_biology", "diagnose_data", "get_rankings"]
    tx_start = 0.4
    tx_gap = 1.3
    ty = 3.5
    tw, th = 1.2, 0.45

    ax.text(
        tx_start + (len(tools) - 1) * tx_gap / 2, ty + 0.65,
        "ReAct Agent Tools",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=COLORS["tool"],
    )

    for i, tool in enumerate(tools):
        tx = tx_start + i * tx_gap
        _rounded_box(ax, tx, ty, tw, th, tool, COLORS["tool"],
                      fontsize=7, fontweight="normal",
                      boxstyle="round,pad=0.1")
        _arrow(ax, tx, ty + th / 2, 3.0, 5.5 - ah / 2,
               color="#BDC3C7", linewidth=0.8, shrinkA=4, shrinkB=6)

    # ── Available Models annotation ────────────────────────────────────
    ax.text(
        tx_start + (len(tools) - 1) * tx_gap / 2, ty - 0.55,
        "Available Models: XGBoost, LightGBM, Random Forest, Ridge, Lasso,\n"
        "ElasticNet, SVM, KNN, MLP, FT-Transformer, Stacking Ensemble",
        ha="center", va="center", fontsize=7, color="#7F8C8D",
        fontstyle="italic",
    )

    # ── ReAct loop ─────────────────────────────────────────────────────
    rx, ry = 8, 3.8
    rw_total = 5.0
    react_labels = ["Thought", "Action", "Observation"]
    react_box_w = 1.3
    react_box_h = 0.45
    react_gap = rw_total / (len(react_labels) - 1)

    # Background area
    react_bg = FancyBboxPatch(
        (rx - rw_total / 2 - 0.4, ry - 0.7),
        rw_total + 0.8, 1.6,
        boxstyle="round,pad=0.2",
        facecolor=COLORS["react"], edgecolor="none", alpha=0.08, zorder=1,
    )
    ax.add_patch(react_bg)
    ax.text(
        rx, ry + 0.65, "ReAct Loop",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=COLORS["react"],
    )

    for i, lbl in enumerate(react_labels):
        bx = rx - rw_total / 2 + i * react_gap
        _rounded_box(ax, bx, ry, react_box_w, react_box_h, lbl,
                      COLORS["react"], fontsize=8, alpha=0.9)
        if i > 0:
            prev_bx = rx - rw_total / 2 + (i - 1) * react_gap
            _arrow(ax, prev_bx + react_box_w / 2, ry,
                   bx - react_box_w / 2, ry,
                   color=COLORS["react"], shrinkA=2, shrinkB=2, linewidth=1.2)

    # Loop-back arrow from Observation -> Thought
    obs_x = rx - rw_total / 2 + 2 * react_gap
    thought_x = rx - rw_total / 2
    _arrow(ax, obs_x, ry - react_box_h / 2 - 0.05,
           thought_x, ry - react_box_h / 2 - 0.05,
           color=COLORS["react"], linestyle="--", linewidth=1.0,
           connectionstyle="arc3,rad=-0.35",
           label="cycle", label_offset=(0, -0.35))

    # Connect ReAct to Coordinator
    _arrow(ax, cx, cy - ch / 2, cx, ry + 0.7,
           color="#BDC3C7", linewidth=1.0, linestyle="--",
           label="agent reasoning", label_offset=(1.2, 0))

    # ── Tree search branching ──────────────────────────────────────────
    tree_x, tree_y = 14.5, 3.5
    node_r = 0.22

    ax.text(
        tree_x, tree_y + 1.0, "Tree Search",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=COLORS["tree_search"],
    )

    # Root
    root = plt.Circle((tree_x, tree_y + 0.5), node_r,
                       fc=COLORS["tree_search"], ec="none", zorder=3)
    ax.add_patch(root)
    ax.text(tree_x, tree_y + 0.5, "R", ha="center", va="center",
            fontsize=7, color="white", fontweight="bold", zorder=4)

    # Children
    children_offsets = [(-1.0, -0.7), (0, -0.7), (1.0, -0.7)]
    child_labels = ["A", "B", "C"]
    for (dx, dy), lbl in zip(children_offsets, child_labels):
        child_x = tree_x + dx
        child_y = tree_y + 0.5 + dy
        c = plt.Circle((child_x, child_y), node_r,
                        fc=COLORS["tree_search"], ec="none", alpha=0.7, zorder=3)
        ax.add_patch(c)
        ax.text(child_x, child_y, lbl, ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=4)
        _arrow(ax, tree_x, tree_y + 0.5 - node_r, child_x, child_y + node_r,
               color=COLORS["tree_search"], linewidth=1.0, shrinkA=0, shrinkB=0)

    # Grandchildren under B
    for gdx in (-0.4, 0.4):
        gx = tree_x + gdx
        gy = tree_y + 0.5 - 1.4
        g = plt.Circle((gx, gy), node_r * 0.8,
                        fc=COLORS["tree_search"], ec="none", alpha=0.45, zorder=3)
        ax.add_patch(g)
        _arrow(ax, tree_x, tree_y + 0.5 - 0.7 - node_r,
               gx, gy + node_r * 0.8,
               color=COLORS["tree_search"], linewidth=0.8,
               shrinkA=0, shrinkB=0, linestyle="--")

    # ── Debate mechanism box ───────────────────────────────────────────
    deb_x, deb_y = 8, 1.2
    deb_w, deb_h = 4.0, 0.9

    debate_bg = FancyBboxPatch(
        (deb_x - deb_w / 2, deb_y - deb_h / 2),
        deb_w, deb_h,
        boxstyle="round,pad=0.2",
        facecolor=COLORS["debate"], edgecolor="none", alpha=0.10, zorder=1,
    )
    ax.add_patch(debate_bg)
    ax.text(
        deb_x, deb_y + 0.15, "Debate Mechanism",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=COLORS["debate"],
    )
    ax.text(
        deb_x, deb_y - 0.2,
        "Agents propose  \u2192  Argue  \u2192  Coordinator resolves",
        ha="center", va="center", fontsize=8, color="#7F8C8D",
    )

    # ── Legend ──────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=COLORS["coordinator"], label="Coordinator"),
        mpatches.Patch(color=COLORS["ml_engineer"], label="ML Engineer"),
        mpatches.Patch(color=COLORS["data_analyst"], label="Data Analyst"),
        mpatches.Patch(color=COLORS["biology_specialist"], label="Biology Specialist"),
        mpatches.Patch(color=COLORS["research"], label="Research Agent"),
        mpatches.Patch(color=COLORS["tool"], label="Tools"),
        mpatches.Patch(color=COLORS["pipeline"], label="Pipeline Stage"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower left",
        fontsize=8, frameon=True, fancybox=True, framealpha=0.9,
        edgecolor="#DDDDDD",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. Run-specific agent flow diagram
# ---------------------------------------------------------------------------

def generate_agent_flow_diagram(
    output_dir: Path,
    agent_reasoning: list[dict],
    debate_transcripts: list[dict[str, Any]] | None = None,
) -> Path:
    """Generate a diagram showing the actual decision flow for this specific run.

    Shows which agents were consulted, what they proposed, debates that happened,
    and final decisions -- like a flowchart of the run.

    Parameters
    ----------
    output_dir:
        Root output directory; the image is saved under ``output_dir/figures/``.
    agent_reasoning:
        List of dicts, each with at least::

            {
                "agent": str,          # e.g. "ml_engineer"
                "stage": str,          # e.g. "model_selection"
                "action": str,         # chosen action
                "confidence": float,   # 0-1
                "reasoning": str,      # short explanation
            }
    debate_transcripts:
        Optional list of debate dicts::

            {
                "topic": str,
                "agents": list[str],
                "rounds": int,
                "winner": str | None,
                "summary": str,
            }

    Returns
    -------
    Path to the saved PNG.
    """
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "agent_flow.png"

    debate_transcripts = debate_transcripts or []
    # Convert dataclass objects to dicts if needed
    from dataclasses import asdict, fields
    debate_transcripts = [
        asdict(d) if hasattr(d, '__dataclass_fields__') else d
        for d in debate_transcripts
    ]

    # Compute layout dimensions based on content
    n_steps = len(agent_reasoning)
    n_debates = len(debate_transcripts)
    total_rows = n_steps + n_debates
    fig_height = max(6, 2.0 + total_rows * 1.6)
    fig_width = 14

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["background"])

    # Title
    ax.text(
        fig_width / 2, fig_height - 0.5,
        "Agent Decision Flow (This Run)",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=TEXT_DARK,
    )

    # Mapping agent names to colours
    agent_color_map = {
        "coordinator": COLORS["coordinator"],
        "ml_engineer": COLORS["ml_engineer"],
        "data_analyst": COLORS["data_analyst"],
        "biology_specialist": COLORS["biology_specialist"],
        "research": COLORS["research"],
    }

    y_cursor = fig_height - 1.6
    center_x = fig_width / 2
    step_box_w = 6.0
    step_box_h = 1.0

    prev_y: float | None = None

    # ── Decision steps ─────────────────────────────────────────────────
    for i, step in enumerate(agent_reasoning):
        agent_name: str = step.get("agent", "unknown")
        stage: str = step.get("stage", "")
        action: str = step.get("action", "")
        confidence: float = step.get("confidence", 0.0)
        reasoning: str = step.get("reasoning", "")

        color = agent_color_map.get(agent_name, "#95A5A6")

        # Draw connecting arrow from previous step
        if prev_y is not None:
            _arrow(ax, center_x, prev_y - step_box_h / 2,
                   center_x, y_cursor + step_box_h / 2,
                   color="#BDC3C7", linewidth=1.5, shrinkA=4, shrinkB=4)

        # Main box
        _rounded_box(ax, center_x, y_cursor, step_box_w, step_box_h,
                      "", color, alpha=0.15, fontsize=1,
                      boxstyle="round,pad=0.2", zorder=1)

        # Step number badge
        badge_x = center_x - step_box_w / 2 + 0.45
        badge = plt.Circle((badge_x, y_cursor), 0.25,
                            fc=color, ec="none", zorder=3)
        ax.add_patch(badge)
        ax.text(badge_x, y_cursor, str(i + 1),
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=4)

        # Agent label
        display_name = agent_name.replace("_", " ").title()
        ax.text(
            badge_x + 0.55, y_cursor + 0.25,
            display_name, ha="left", va="center",
            fontsize=10, fontweight="bold", color=color, zorder=4,
        )

        # Stage + action
        ax.text(
            badge_x + 0.55, y_cursor + 0.0,
            f"Stage: {stage}  |  Action: {action}",
            ha="left", va="center",
            fontsize=8, color=TEXT_DARK, zorder=4,
        )

        # Confidence bar
        bar_x = badge_x + 0.55
        bar_y = y_cursor - 0.3
        bar_w = 3.5
        bar_h = 0.12
        # Background
        bar_bg = FancyBboxPatch(
            (bar_x, bar_y - bar_h / 2), bar_w, bar_h,
            boxstyle="round,pad=0.02",
            facecolor="#EEEEEE", edgecolor="none", zorder=3,
        )
        ax.add_patch(bar_bg)
        # Fill
        fill_w = max(bar_w * confidence, 0.01)
        bar_fill = FancyBboxPatch(
            (bar_x, bar_y - bar_h / 2), fill_w, bar_h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="none", alpha=0.7, zorder=4,
        )
        ax.add_patch(bar_fill)
        ax.text(
            bar_x + bar_w + 0.15, bar_y,
            f"{confidence:.0%}",
            ha="left", va="center", fontsize=7, color=color, zorder=4,
        )

        # Reasoning snippet (right side)
        wrapped = textwrap.shorten(reasoning, width=55, placeholder="...")
        ax.text(
            center_x + step_box_w / 2 - 0.2, y_cursor,
            wrapped, ha="right", va="center",
            fontsize=7, color="#7F8C8D", fontstyle="italic", zorder=4,
        )

        prev_y = y_cursor
        y_cursor -= 1.6

    # ── Debate transcripts ─────────────────────────────────────────────
    if debate_transcripts:
        # Section header
        y_cursor -= 0.3
        ax.text(
            center_x, y_cursor + 0.5,
            "Debates", ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLORS["debate"],
        )

        for dbt in debate_transcripts:
            topic = dbt.get("topic", "")
            agents_involved = dbt.get("agents", [])
            rounds = dbt.get("rounds", 0)
            winner = dbt.get("winner", None)
            summary = dbt.get("summary", "")

            # Connect from last step
            if prev_y is not None:
                _arrow(ax, center_x, prev_y - step_box_h / 2,
                       center_x, y_cursor + step_box_h / 2,
                       color=COLORS["debate"], linewidth=1.2,
                       linestyle=":", shrinkA=4, shrinkB=4)

            # Debate box
            _rounded_box(ax, center_x, y_cursor, step_box_w, step_box_h,
                          "", COLORS["debate"], alpha=0.10, fontsize=1,
                          boxstyle="round,pad=0.2", zorder=1)

            # Debate icon
            ax.text(
                center_x - step_box_w / 2 + 0.45, y_cursor + 0.2,
                "\u2694", ha="center", va="center",
                fontsize=14, color=COLORS["debate"], zorder=4,
            )

            agents_str = " vs ".join(
                a.replace("_", " ").title() for a in agents_involved
            )
            ax.text(
                center_x - step_box_w / 2 + 1.0, y_cursor + 0.2,
                f"Debate: {topic}",
                ha="left", va="center", fontsize=10,
                fontweight="bold", color=COLORS["debate"], zorder=4,
            )
            ax.text(
                center_x - step_box_w / 2 + 1.0, y_cursor - 0.05,
                f"{agents_str}  |  {rounds} round(s)  |  Winner: {winner or 'none'}",
                ha="left", va="center", fontsize=8, color=TEXT_DARK, zorder=4,
            )

            wrapped_summary = textwrap.shorten(summary, width=60, placeholder="...")
            ax.text(
                center_x + step_box_w / 2 - 0.2, y_cursor - 0.25,
                wrapped_summary, ha="right", va="center",
                fontsize=7, color="#7F8C8D", fontstyle="italic", zorder=4,
            )

            prev_y = y_cursor
            y_cursor -= 1.6

    # ── Final result box ───────────────────────────────────────────────
    if agent_reasoning:
        final = agent_reasoning[-1]
        final_action = final.get("action", "N/A")
        result_y = y_cursor + 0.4
        if prev_y is not None:
            _arrow(ax, center_x, prev_y - step_box_h / 2,
                   center_x, result_y + 0.3,
                   color=COLORS["coordinator"], linewidth=2, shrinkA=4, shrinkB=4)
        _rounded_box(ax, center_x, result_y, 4.0, 0.6,
                      f"Final: {final_action}", COLORS["coordinator"],
                      fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close(fig)
    return out_path
