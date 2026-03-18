"""Tests for RunConfig defaults and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from co_scientist.config import DEFAULT_BUDGET, DEFAULT_MAX_COST, DEFAULT_MODE, RunConfig


def test_default_config():
    """Verify that RunConfig defaults are applied correctly."""
    cfg = RunConfig(dataset_path="RNA/translation_efficiency_muscle")
    assert cfg.dataset_path == "RNA/translation_efficiency_muscle"
    assert cfg.mode == DEFAULT_MODE
    assert cfg.budget == DEFAULT_BUDGET
    assert cfg.max_cost == DEFAULT_MAX_COST
    assert cfg.seed == 42
    assert isinstance(cfg.output_dir, Path)


def test_invalid_mode():
    """A mode that is neither 'auto' nor 'interactive' should be rejected."""
    with pytest.raises(ValidationError):
        RunConfig(dataset_path="test/data", mode="turbo")


def test_task_output_dir():
    """The task_output_dir property should sanitize slashes."""
    cfg = RunConfig(dataset_path="RNA/translation_efficiency_muscle")
    assert "RNA__translation_efficiency_muscle" in str(cfg.task_output_dir)
