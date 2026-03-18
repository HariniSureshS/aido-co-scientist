"""Dynamic custom model generation — LLM designs PyTorch architectures at runtime.

The design_model tool uses this module to:
1. Prompt the LLM with dataset characteristics
2. Validate generated code via AST
3. Dynamically load and instantiate the model
4. Enforce sklearn-compatible interface (fit/predict/predict_proba)
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import types
from typing import Any

logger = logging.getLogger(__name__)

# Imports disallowed in generated code for basic safety
_DISALLOWED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
}

_DISALLOWED_BUILTINS = {"exec", "eval", "__import__", "compile", "open"}


CUSTOM_MODEL_PROMPT = """\
Design a custom PyTorch model for the following dataset and task.

## Dataset Characteristics
- Task: {task_type}
- Number of features: {n_features}
- Number of samples: {n_samples}
- Number of classes: {n_classes} (classification only, 0 for regression)
- Modality: {modality}
- Dataset: {dataset_name}
- Primary metric: {primary_metric}

## Architecture Request
{architecture_hint}

## Requirements

Write a SINGLE Python class that:
1. Has `__init__(self, **kwargs)` accepting optional hyperparameters
2. Has `fit(self, X: np.ndarray, y: np.ndarray) -> self` that trains the model
3. Has `predict(self, X: np.ndarray) -> np.ndarray` that returns predictions
4. For classification: has `predict_proba(self, X: np.ndarray) -> np.ndarray` returning class probabilities
5. Uses only: `torch`, `torch.nn`, `numpy`, `sklearn` (for utilities like train_test_split)
6. Includes early stopping in the training loop
7. Sets `torch.manual_seed` from a `random_state` kwarg for reproducibility

## Output Format

Return ONLY the Python code inside a ```python code fence. Include all imports at the top.
The class name should be descriptive (e.g., `TabularResNet`, `AttentionMLP`, `CodonAwareCNN`).

```python
import numpy as np
import torch
import torch.nn as nn

class YourModelName:
    def __init__(self, random_state=42, **kwargs):
        ...

    def fit(self, X, y):
        ...
        return self

    def predict(self, X):
        ...

    def predict_proba(self, X):  # classification only
        ...
```
"""


def validate_model_code(code: str) -> list[str]:
    """Validate generated model code via AST analysis.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    # 1. Parse
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    # 2. Check for disallowed imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in _DISALLOWED_IMPORTS:
                    errors.append(f"Disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split(".")[0]
                if module_root in _DISALLOWED_IMPORTS:
                    errors.append(f"Disallowed import: {node.module}")

    # 3. Check for disallowed builtins
    # Only block standalone calls like eval(...), not method calls like model.eval()
    # PyTorch models need .eval() for inference mode, so attribute calls are allowed
    _DISALLOWED_ATTRS = _DISALLOWED_BUILTINS - {"eval"}  # .eval() is fine (PyTorch)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DISALLOWED_BUILTINS:
                errors.append(f"Disallowed builtin: {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in _DISALLOWED_ATTRS:
                errors.append(f"Disallowed call: .{func.attr}()")

    # 4. Must define at least one class
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if not classes:
        errors.append("No class definition found in generated code")

    # 5. Check the first class has fit and predict methods
    if classes:
        cls = classes[0]
        methods = {node.name for node in ast.walk(cls) if isinstance(node, ast.FunctionDef)}
        if "fit" not in methods:
            errors.append(f"Class '{cls.name}' missing fit() method")
        if "predict" not in methods:
            errors.append(f"Class '{cls.name}' missing predict() method")

    return errors


def extract_code_from_response(response: str) -> str | None:
    """Extract Python code from markdown code fences."""
    # Try ```python ... ``` first
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ```
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def load_custom_model(
    code: str,
    class_name: str | None = None,
    task_type: str = "classification",
    hyperparameters: dict[str, Any] | None = None,
) -> Any:
    """Dynamically load and instantiate a model from generated code.

    Args:
        code: Python source code defining a model class.
        class_name: Class to instantiate. Auto-detected if None.
        task_type: "classification" or "regression".
        hyperparameters: Kwargs passed to the class constructor.

    Returns:
        An instantiated model with fit/predict interface.

    Raises:
        ValueError: If code validation fails.
        TypeError: If the model doesn't have required methods.
    """
    # Validate
    errors = validate_model_code(code)
    if errors:
        raise ValueError(f"Code validation failed: {'; '.join(errors)}")

    # Auto-detect class name
    if class_name is None:
        tree = ast.parse(code)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        if not classes:
            raise ValueError("No class found in code")
        class_name = classes[0].name

    # Dynamic loading
    module = types.ModuleType(f"custom_model_{class_name}")
    exec(code, module.__dict__)  # noqa: S102

    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"Class '{class_name}' not found in generated code")

    # Instantiate
    hp = dict(hyperparameters or {})
    try:
        instance = cls(**hp)
    except Exception as e:
        raise TypeError(f"Failed to instantiate {class_name}: {e}") from e

    # Verify interface
    if not callable(getattr(instance, "fit", None)):
        raise TypeError(f"{class_name} missing callable fit() method")
    if not callable(getattr(instance, "predict", None)):
        raise TypeError(f"{class_name} missing callable predict() method")
    if task_type == "classification" and not callable(getattr(instance, "predict_proba", None)):
        logger.warning("%s missing predict_proba() — probabilities unavailable", class_name)

    return instance


def generate_model_name(architecture_hint: str) -> str:
    """Generate a short, unique model name from the architecture hint."""
    short_hash = hashlib.md5(architecture_hint.encode()).hexdigest()[:6]
    # Clean the hint into a short prefix
    words = architecture_hint.lower().split()[:3]
    prefix = "_".join(w for w in words if w.isalnum())[:20]
    return f"custom_{prefix}_{short_hash}"
