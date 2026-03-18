"""Generate a one-page visual summary PDF from pipeline results using matplotlib."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from co_scientist import __version__
from co_scientist.data.types import DatasetProfile
from co_scientist.evaluation.types import EvalConfig, ModelResult

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_BG = "#FAFAFA"
_ACCENT = "#2E86AB"
_GREEN = "#4CAF50"
_GREY_TEXT = "#555555"
_DARK_TEXT = "#212121"
_LIGHT_BORDER = "#CCCCCC"
_BAR_DEFAULT = "#90CAF9"
_BAR_BEST = "#4CAF50"
_LINE_COLOR = "#2E86AB"
_HEADER_BG = "#2E86AB"
_HEADER_FG = "#FFFFFF"
_BOX_BG = "#F0F4F8"


def generate_summary_pdf(
    profile: DatasetProfile,
    eval_config: EvalConfig,
    best_result: ModelResult,
    results: list[ModelResult],
    output_dir: Path,
    benchmark_comparison: str = "",
    research_report: dict | None = None,
    react_scratchpad: list[dict] | None = None,
) -> Path:
    """Create a single-page PDF summarising the pipeline run.

    Returns the path to the generated PDF file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "summary.pdf"

    has_scratchpad = bool(react_scratchpad)
    has_benchmark = bool(benchmark_comparison and benchmark_comparison.strip())

    # Determine grid layout — always 12 row-slots high.
    # Row allocation (in 12ths):
    #   header          : 1
    #   key metrics     : 2
    #   bar chart       : 4  (or 5 if no scratchpad & no benchmark)
    #   scratchpad      : 3  (optional)
    #   benchmark       : 1.5 (optional)
    #   footer          : 0.5

    fig = plt.figure(figsize=(11, 8.5), facecolor=_BG, dpi=150)

    # We use a gridspec with fine rows to allocate space dynamically.
    n_extra = int(has_scratchpad) + int(has_benchmark)
    if n_extra == 2:
        heights = [1, 2.2, 3.5, 2.8, 1.2, 0.3]  # header, metrics, bar, scratch, bench, footer
    elif has_scratchpad:
        heights = [1, 2.2, 4.0, 3.0, 0.3]
    elif has_benchmark:
        heights = [1, 2.2, 5.0, 1.5, 0.3]
    else:
        heights = [1, 2.2, 6.0, 0.3]

    gs = fig.add_gridspec(
        nrows=len(heights),
        ncols=2,
        height_ratios=heights,
        hspace=0.45,
        wspace=0.35,
        left=0.06,
        right=0.94,
        top=0.97,
        bottom=0.02,
    )

    # ------------------------------------------------------------------
    # 1. Header
    # ------------------------------------------------------------------
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.set_axis_off()
    ax_header.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_header.transAxes,
                                       facecolor=_HEADER_BG, edgecolor="none", zorder=0))
    title_text = f"AIDO Co-Scientist \u2014 {profile.dataset_name}"
    ax_header.text(0.03, 0.5, title_text, fontsize=16, fontweight="bold",
                   color=_HEADER_FG, va="center", ha="left", transform=ax_header.transAxes)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ax_header.text(0.97, 0.5, timestamp, fontsize=9, color=_HEADER_FG,
                   va="center", ha="right", transform=ax_header.transAxes)

    # ------------------------------------------------------------------
    # 2. Key metrics box
    # ------------------------------------------------------------------
    ax_metrics = fig.add_subplot(gs[1, :])
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_axis_off()
    ax_metrics.add_patch(plt.Rectangle((0.01, 0.05), 0.98, 0.90, transform=ax_metrics.transAxes,
                                        facecolor=_BOX_BG, edgecolor=_LIGHT_BORDER,
                                        linewidth=0.8, zorder=0))

    labels = ["Task Type", "Modality", "Samples", "Features", "Best Model", "Best Score"]
    values = [
        eval_config.task_type.replace("_", " ").title(),
        profile.modality.value.replace("_", " ").title(),
        f"{profile.num_samples:,}",
        f"{profile.num_features:,}" if profile.num_features else "N/A",
        best_result.model_name,
        f"{best_result.primary_metric_value:.4f}  ({eval_config.primary_metric})",
    ]

    n_cols = 3
    n_rows = 2
    for idx, (label, value) in enumerate(zip(labels, values)):
        col = idx % n_cols
        row = idx // n_cols
        x = 0.05 + col * 0.32
        y = 0.75 - row * 0.42
        ax_metrics.text(x, y, label, fontsize=8, color=_GREY_TEXT, fontweight="bold",
                        va="top", transform=ax_metrics.transAxes)
        ax_metrics.text(x, y - 0.16, value, fontsize=10, color=_DARK_TEXT,
                        va="top", transform=ax_metrics.transAxes)

    # ------------------------------------------------------------------
    # 3. Model comparison bar chart
    # ------------------------------------------------------------------
    bar_row = 2
    # If both optional panels exist, bar chart spans full width; otherwise also full width.
    ax_bar = fig.add_subplot(gs[bar_row, :])

    sorted_results = sorted(results, key=lambda r: r.primary_metric_value)
    model_names = [r.model_name for r in sorted_results]
    scores = [r.primary_metric_value for r in sorted_results]
    colors = [_BAR_BEST if r.model_name == best_result.model_name else _BAR_DEFAULT
              for r in sorted_results]

    y_pos = range(len(model_names))
    ax_bar.barh(y_pos, scores, color=colors, edgecolor="white", height=0.6)
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(model_names, fontsize=8, color=_DARK_TEXT)
    ax_bar.set_xlabel(eval_config.primary_metric, fontsize=9, color=_GREY_TEXT)
    ax_bar.set_title("Model Comparison", fontsize=11, fontweight="bold",
                     color=_DARK_TEXT, loc="left", pad=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.tick_params(left=False, bottom=True)
    ax_bar.spines["bottom"].set_color(_LIGHT_BORDER)
    ax_bar.tick_params(axis="x", colors=_GREY_TEXT, labelsize=8)
    ax_bar.set_facecolor(_BG)

    # Add score labels at end of each bar
    for i, score in enumerate(scores):
        ax_bar.text(score + 0.002, i, f"{score:.4f}", va="center", fontsize=7, color=_GREY_TEXT)

    # ------------------------------------------------------------------
    # 4. Score progression (optional)
    # ------------------------------------------------------------------
    next_row = bar_row + 1
    if has_scratchpad:
        ax_prog = fig.add_subplot(gs[next_row, :] if not has_benchmark else gs[next_row, :])

        # Extract scores from scratchpad entries
        step_scores: list[float] = []
        step_labels: list[str] = []
        for i, entry in enumerate(react_scratchpad):  # type: ignore[union-attr]
            score = entry.get("best_score") or entry.get("score") or entry.get("primary_metric_value")
            if score is not None:
                step_scores.append(float(score))
                step_labels.append(entry.get("step", f"Step {i + 1}"))

        if step_scores:
            ax_prog.plot(range(len(step_scores)), step_scores, marker="o", markersize=5,
                         color=_LINE_COLOR, linewidth=2, markerfacecolor="white",
                         markeredgecolor=_LINE_COLOR, markeredgewidth=1.5)
            ax_prog.set_xticks(range(len(step_scores)))
            ax_prog.set_xticklabels(step_labels, fontsize=7, rotation=30, ha="right",
                                     color=_GREY_TEXT)
            ax_prog.set_ylabel(eval_config.primary_metric, fontsize=8, color=_GREY_TEXT)
            ax_prog.set_title("Score Progression (ReAct Steps)", fontsize=11,
                              fontweight="bold", color=_DARK_TEXT, loc="left", pad=8)
            ax_prog.spines["top"].set_visible(False)
            ax_prog.spines["right"].set_visible(False)
            ax_prog.spines["bottom"].set_color(_LIGHT_BORDER)
            ax_prog.spines["left"].set_color(_LIGHT_BORDER)
            ax_prog.tick_params(axis="y", colors=_GREY_TEXT, labelsize=8)
            ax_prog.set_facecolor(_BG)
        else:
            ax_prog.set_axis_off()

        next_row += 1

    # ------------------------------------------------------------------
    # 5. Literature / benchmark comparison (optional)
    # ------------------------------------------------------------------
    if has_benchmark:
        ax_bench = fig.add_subplot(gs[next_row, :])
        ax_bench.set_xlim(0, 1)
        ax_bench.set_ylim(0, 1)
        ax_bench.set_axis_off()
        ax_bench.add_patch(plt.Rectangle((0.01, 0.05), 0.98, 0.90,
                                          transform=ax_bench.transAxes,
                                          facecolor="#FFF8E1", edgecolor="#FFD54F",
                                          linewidth=0.8, zorder=0))
        ax_bench.text(0.03, 0.85, "Literature Comparison", fontsize=10,
                      fontweight="bold", color=_DARK_TEXT, va="top",
                      transform=ax_bench.transAxes)
        # Wrap long text
        wrapped = benchmark_comparison.strip()
        if len(wrapped) > 300:
            wrapped = wrapped[:297] + "..."
        ax_bench.text(0.03, 0.55, wrapped, fontsize=8, color=_GREY_TEXT, va="top",
                      transform=ax_bench.transAxes, wrap=True,
                      fontfamily="sans-serif")
        next_row += 1

    # ------------------------------------------------------------------
    # 6. Footer
    # ------------------------------------------------------------------
    ax_footer = fig.add_subplot(gs[-1, :])
    ax_footer.set_xlim(0, 1)
    ax_footer.set_ylim(0, 1)
    ax_footer.set_axis_off()
    footer_text = f"Generated by AIDO Co-Scientist v{__version__}"
    ax_footer.text(0.5, 0.5, footer_text, fontsize=7, color=_GREY_TEXT,
                   ha="center", va="center", transform=ax_footer.transAxes,
                   fontstyle="italic")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with PdfPages(str(pdf_path)) as pdf:
        pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    return pdf_path
