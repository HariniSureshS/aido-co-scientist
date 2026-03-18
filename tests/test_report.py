"""Tests for report generation helpers (benchmark comparison)."""

from __future__ import annotations

from co_scientist.evaluation.types import EvalConfig, ModelResult
from co_scientist.report.generator import _build_benchmark_comparison


def _make_result(score: float) -> ModelResult:
    return ModelResult(
        model_name="xgboost_tuned",
        tier="standard",
        metrics={"accuracy": score},
        primary_metric_name="accuracy",
        primary_metric_value=score,
        train_time_seconds=2.0,
    )


def _make_eval_config(metric: str = "accuracy") -> EvalConfig:
    return EvalConfig(
        task_type="binary_classification",
        primary_metric=metric,
    )


def test_benchmark_comparison_outperforms():
    """When our score exceeds published benchmarks, output should contain 'outperforms'."""
    best = _make_result(0.95)
    eval_cfg = _make_eval_config()
    research_report = {
        "benchmarks_found": [
            "accuracy: 0.88 (Smith et al. 2024)",
            "accuracy: 0.91 (Jones et al. 2025)",
        ],
    }

    result = _build_benchmark_comparison(best, eval_cfg, research_report)
    assert "outperforms" in result.lower()
    assert "0.95" in result or "0.9500" in result


def test_benchmark_comparison_no_benchmarks():
    """When no benchmarks exist, the function returns an empty string."""
    best = _make_result(0.90)
    eval_cfg = _make_eval_config()

    result = _build_benchmark_comparison(best, eval_cfg, {})
    assert result == ""

    result2 = _build_benchmark_comparison(best, eval_cfg, {"benchmarks_found": []})
    assert result2 == ""
