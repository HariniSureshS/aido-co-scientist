"""Automated report review — verify numbers match experiment log and claims are supported.

After report generation, this module cross-checks the report against the
experiment log to catch numerical mismatches, unsupported claims, and
missing information.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReviewIssue:
    """A single issue found during report review."""

    severity: str  # "error", "warning", "info"
    category: str  # "numerical_mismatch", "unsupported_claim", "missing_info"
    description: str
    location: str = ""
    expected: str | None = None
    found: str | None = None


@dataclass
class ReviewResult:
    """Result of a report review."""

    issues: list[ReviewIssue] = field(default_factory=list)
    llm_review: str | None = None
    passed: bool = True


def review_report(
    report_path: Path | str,
    experiment_log_path: Path | str,
    client: Any = None,
) -> ReviewResult:
    """Review a generated report against the experiment log.

    Steps:
    1. Parse report.md — extract scores, model names, counts via regex
    2. Parse experiment_log.jsonl — extract ground truth
    3. Cross-check: scores match? model counts match? best model correct?
    4. If LLM available: semantic review
    5. Return ReviewResult

    Args:
        report_path: Path to the generated report.md
        experiment_log_path: Path to experiment_log.jsonl
        client: Optional ClaudeClient for semantic review

    Returns:
        ReviewResult with issues found.
    """
    report_path = Path(report_path)
    experiment_log_path = Path(experiment_log_path)

    result = ReviewResult()

    # Read report
    if not report_path.exists():
        result.issues.append(ReviewIssue(
            severity="error", category="missing_info",
            description="Report file not found",
            location=str(report_path),
        ))
        result.passed = False
        return result

    report_text = report_path.read_text(encoding="utf-8")

    # Parse ground truth from experiment log
    ground_truth = _parse_experiment_log(experiment_log_path)
    if ground_truth is None:
        result.issues.append(ReviewIssue(
            severity="warning", category="missing_info",
            description="Experiment log not found or unreadable",
            location=str(experiment_log_path),
        ))
        return result

    # Cross-check model scores
    score_issues = _check_model_scores(report_text, ground_truth)
    result.issues.extend(score_issues)

    # Cross-check best model
    best_issues = _check_best_model(report_text, ground_truth)
    result.issues.extend(best_issues)

    # Cross-check model count
    count_issues = _check_model_count(report_text, ground_truth)
    result.issues.extend(count_issues)

    # LLM semantic review (if available)
    if client is not None:
        llm_review = _llm_semantic_review(report_text, ground_truth, client)
        result.llm_review = llm_review

    # Set passed flag
    result.passed = not any(i.severity == "error" for i in result.issues)

    return result


def _parse_experiment_log(path: Path) -> dict[str, Any] | None:
    """Parse experiment log JSONL and extract ground truth data."""
    if not path.exists():
        return None

    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as e:
        logger.warning("Could not parse experiment log: %s", e)
        return None

    if not entries:
        return None

    ground_truth: dict[str, Any] = {
        "model_scores": {},
        "best_model": None,
        "best_score": None,
        "n_models": 0,
        "primary_metric": None,
    }

    for entry in entries:
        event = entry.get("event", "")

        if event == "evaluation":
            data = entry.get("data", {})
            name = data.get("model_name", "")
            score = data.get("primary_value")
            if name and score is not None:
                ground_truth["model_scores"][name] = score
                ground_truth["primary_metric"] = data.get("primary_metric")

        elif event == "model_trained":
            ground_truth["n_models"] += 1

        elif event == "pipeline_complete":
            data = entry.get("data", {})
            ground_truth["best_model"] = data.get("best_model")
            ground_truth["best_score"] = data.get("best_value")

        elif event == "step_end":
            data = entry.get("data", {})
            if data.get("best_model"):
                ground_truth["best_model"] = data.get("best_model")
            if data.get("best_value"):
                ground_truth["best_score"] = data.get("best_value")

    return ground_truth


def _check_model_scores(report_text: str, ground_truth: dict) -> list[ReviewIssue]:
    """Check that model scores in the report match the experiment log."""
    issues = []
    gt_scores = ground_truth.get("model_scores", {})

    if not gt_scores:
        return issues

    # Find score patterns in report: "model_name ... 0.1234"
    # Sort by name length descending to match longer names first (e.g. random_forest_tuned before random_forest)
    for model_name, expected_score in sorted(gt_scores.items(), key=lambda x: len(x[0]), reverse=True):
        # Use word boundary to avoid matching "random_forest" inside "random_forest_tuned"
        pattern = r"(?<![a-zA-Z0-9_])" + re.escape(model_name) + r"(?![a-zA-Z0-9_])[^0-9]*?(\d+\.\d{4})"
        matches = re.findall(pattern, report_text)

        # Only check first match per model (the baseline table) to avoid false positives
        # from other sections mentioning the model
        if matches:
            reported = float(matches[0])
            if abs(reported - expected_score) > 0.0005:  # tolerance for rounding
                issues.append(ReviewIssue(
                    severity="error",
                    category="numerical_mismatch",
                    description=f"Score mismatch for {model_name}",
                    expected=f"{expected_score:.4f}",
                    found=f"{reported:.4f}",
                ))

    return issues


def _check_best_model(report_text: str, ground_truth: dict) -> list[ReviewIssue]:
    """Check that the best model in the report matches the experiment log."""
    issues = []
    gt_best = ground_truth.get("best_model")

    if not gt_best:
        return issues

    # Look for "Best model" in report
    best_match = re.search(r"Best model[^|]*?\|\s*(\S+)", report_text)
    if best_match:
        reported_best = best_match.group(1).strip("`* ")
        if reported_best != gt_best:
            issues.append(ReviewIssue(
                severity="warning",
                category="numerical_mismatch",
                description="Best model name doesn't match experiment log",
                expected=gt_best,
                found=reported_best,
            ))

    return issues


def _check_model_count(report_text: str, ground_truth: dict) -> list[ReviewIssue]:
    """Check model count consistency."""
    issues = []
    gt_scores = ground_truth.get("model_scores", {})

    if not gt_scores:
        return issues

    # Count model rows in the baseline progression table (section 4.1)
    # Extract just the baseline table by looking between "Baseline Progression" and "Best model:"
    baseline_section = re.search(
        r"### 4\.1 Baseline Progression.*?(?=\*\*Best model)", report_text, re.DOTALL
    )
    if baseline_section:
        section_text = baseline_section.group(0)
    else:
        section_text = report_text
    table_rows = re.findall(r"^\|\s*\S+.*?\|\s*\w+\s*\|.*?\d+\.\d{4}", section_text, re.MULTILINE)
    if table_rows and abs(len(table_rows) - len(gt_scores)) > 2:
        issues.append(ReviewIssue(
            severity="warning",
            category="numerical_mismatch",
            description="Model count in report table doesn't match experiment log",
            expected=str(len(gt_scores)),
            found=str(len(table_rows)),
        ))

    return issues


def _llm_semantic_review(
    report_text: str,
    ground_truth: dict,
    client: Any,
) -> str | None:
    """Use LLM to do a semantic review of the report."""
    from co_scientist.llm.prompts import REPORT_REVIEWER_SYSTEM

    gt_summary = json.dumps({
        "model_scores": {k: round(v, 4) for k, v in ground_truth.get("model_scores", {}).items()},
        "best_model": ground_truth.get("best_model"),
        "best_score": round(ground_truth.get("best_score", 0), 4) if ground_truth.get("best_score") else None,
        "n_models_evaluated": len(ground_truth.get("model_scores", {})),
    }, indent=2)

    # Truncate report for context limits
    report_excerpt = report_text[:4000]

    user_msg = (
        f"Review this report for accuracy and completeness.\n\n"
        f"## Ground truth from experiment log:\n```json\n{gt_summary}\n```\n\n"
        f"## Report (excerpt):\n{report_excerpt}\n\n"
        f"Check:\n"
        f"1. Do the scores in the report match the experiment log?\n"
        f"2. Is the best model correctly identified?\n"
        f"3. Are any claims unsupported by the data?\n"
        f"4. Is any important information missing?\n\n"
        f"Reply with a brief review (3-5 bullet points)."
    )

    return client.ask_text(
        system_prompt=REPORT_REVIEWER_SYSTEM,
        user_message=user_msg,
        agent_name="report_reviewer",
        max_tokens=512,
        temperature=0.1,
    )


def format_review_for_report(review: ReviewResult) -> str:
    """Format a ReviewResult as markdown for appending to the report."""
    lines = ["### Report Review\n"]

    if review.passed:
        lines.append("Automated review: **PASSED**\n")
    else:
        lines.append("Automated review: **ISSUES FOUND**\n")

    if review.issues:
        lines.append("| Severity | Category | Description | Expected | Found |")
        lines.append("|----------|----------|-------------|----------|-------|")
        for issue in review.issues:
            lines.append(
                f"| {issue.severity.upper()} | {issue.category} | "
                f"{issue.description} | {issue.expected or '-'} | {issue.found or '-'} |"
            )
        lines.append("")

    if review.llm_review:
        lines.append("**LLM Review:**\n")
        lines.append(review.llm_review)
        lines.append("")

    return "\n".join(lines)
