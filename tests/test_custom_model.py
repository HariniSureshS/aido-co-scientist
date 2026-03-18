"""Tests for custom model code validation and extraction."""

from __future__ import annotations

from co_scientist.modeling.custom_model import extract_code_from_response, validate_model_code


# ---------------------------------------------------------------------------
# Valid code
# ---------------------------------------------------------------------------

_SAFE_CODE = """\
import numpy as np
import torch
import torch.nn as nn

class SimpleModel:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.ones((len(X), 2)) * 0.5
"""


def test_validate_safe_code():
    errors = validate_model_code(_SAFE_CODE)
    assert errors == [], f"Expected no errors, got: {errors}"


# ---------------------------------------------------------------------------
# Disallowed imports
# ---------------------------------------------------------------------------

def test_validate_blocks_os_import():
    code = "import os\nclass M:\n    def fit(self, X, y): pass\n    def predict(self, X): pass\n"
    errors = validate_model_code(code)
    assert any("os" in e for e in errors), f"Expected os import blocked, got: {errors}"


def test_validate_blocks_subprocess():
    code = "import subprocess\nclass M:\n    def fit(self, X, y): pass\n    def predict(self, X): pass\n"
    errors = validate_model_code(code)
    assert any("subprocess" in e for e in errors), f"Expected subprocess blocked, got: {errors}"


# ---------------------------------------------------------------------------
# Disallowed builtins
# ---------------------------------------------------------------------------

def test_validate_blocks_eval():
    code = (
        "import numpy as np\n"
        "class M:\n"
        "    def fit(self, X, y):\n"
        "        eval('1+1')\n"
        "        return self\n"
        "    def predict(self, X):\n"
        "        return np.zeros(len(X))\n"
    )
    errors = validate_model_code(code)
    assert any("eval" in e for e in errors), f"Expected eval() blocked, got: {errors}"


def test_validate_blocks_open():
    code = (
        "import numpy as np\n"
        "class M:\n"
        "    def fit(self, X, y):\n"
        "        f = open('data.txt')\n"
        "        return self\n"
        "    def predict(self, X):\n"
        "        return np.zeros(len(X))\n"
    )
    errors = validate_model_code(code)
    assert any("open" in e for e in errors), f"Expected open() blocked, got: {errors}"


# ---------------------------------------------------------------------------
# Code extraction from markdown
# ---------------------------------------------------------------------------

def test_extract_code_from_response():
    response = (
        "Here is the model:\n\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "class MyModel:\n"
        "    def fit(self, X, y): return self\n"
        "    def predict(self, X): return np.zeros(len(X))\n"
        "```\n"
    )
    code = extract_code_from_response(response)
    assert code is not None
    assert "class MyModel" in code
    assert "import numpy" in code


def test_extract_code_returns_none_for_no_fences():
    assert extract_code_from_response("No code here.") is None
