"""Step-level validation + auto-repair — catches silent errors and fixes them.

Every validator follows the pattern:
  1. Detect the issue
  2. Fix it deterministically if possible
  3. Use LLM to fix if deterministic fix isn't available
  4. Return the (possibly repaired) object

The pipeline calls validate_and_fix_*() after each step. The returned object
replaces the original — the pipeline always moves forward with clean data.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ValidationResult:
    """Result of a validation + fix step."""

    def __init__(self, step: str):
        self.step = step
        self.passed = True
        self.issues: list[str] = []
        self.fixes_applied: list[str] = []

    def fail(self, msg: str) -> None:
        self.passed = False
        self.issues.append(msg)

    def warn(self, msg: str) -> None:
        self.issues.append(f"[warning] {msg}")

    def fix(self, msg: str) -> None:
        self.fixes_applied.append(msg)
        # If we fixed all failures, mark as passed
        # (caller decides by checking fixes_applied)

    def print_summary(self) -> None:
        if self.passed and not self.fixes_applied:
            console.print(f"  Validation ({self.step}): [green]PASS[/green]")
            return
        status = "[green]FIXED[/green]" if self.fixes_applied and not self.issues else (
            "[green]PASS[/green]" if self.passed else "[red]FAIL[/red]"
        )
        console.print(f"  Validation ({self.step}): {status}")
        for issue in self.issues:
            console.print(f"    [yellow]{issue}[/yellow]")
        for fix in self.fixes_applied:
            console.print(f"    [green]Fixed: {fix}[/green]")


# ---------------------------------------------------------------------------
# Step 1: Validate + fix loaded dataset
# ---------------------------------------------------------------------------

def validate_and_fix_loaded_data(dataset, profile=None) -> tuple[Any, ValidationResult]:
    """Validate dataset after loading. Fix issues where possible.

    Returns (dataset, validation_result). Dataset may be modified in-place.
    """
    v = ValidationResult("data_loading")

    # Non-empty
    if dataset.X is None or len(dataset.y) == 0:
        v.fail("Dataset is empty (0 samples) — cannot fix")
        v.print_summary()
        return dataset, v

    # Shape consistency
    if hasattr(dataset.X, 'shape'):
        n_x = dataset.X.shape[0]
        n_y = len(dataset.y)
        if n_x != n_y:
            # Fix: truncate to minimum length
            n_min = min(n_x, n_y)
            dataset.X = dataset.X[:n_min]
            dataset.y = dataset.y[:n_min]
            v.fix(f"Truncated X ({n_x}) and y ({n_y}) to {n_min} rows")

    # Target: all null/NaN
    import pandas as pd
    y = np.array(dataset.y)
    if y.dtype == float and np.all(np.isnan(y)):
        v.fail("Target column is all NaN — cannot fix")
    elif y.dtype == object:
        # Check for None/"nan" strings
        null_mask = pd.isna(dataset.y)
        if null_mask.all():
            v.fail("Target column is all null — cannot fix")
        elif null_mask.any():
            # Fix: drop rows with null targets
            keep = ~null_mask
            n_dropped = (~keep).sum()
            if hasattr(dataset.X, 'loc'):
                dataset.X = dataset.X.loc[keep].reset_index(drop=True)
            else:
                dataset.X = dataset.X[keep]
            dataset.y = dataset.y[keep]
            v.fix(f"Dropped {n_dropped} rows with null target values")

    # Check splits if predefined
    if dataset.info.has_predefined_splits and dataset.raw_data is not None:
        if hasattr(dataset.raw_data, 'columns') and '_split' in dataset.raw_data.columns:
            splits = set(dataset.raw_data['_split'].unique())
            if 'train' not in splits:
                # Fix: rename the largest split to 'train'
                split_counts = dataset.raw_data['_split'].value_counts()
                largest = split_counts.index[0]
                dataset.raw_data['_split'] = dataset.raw_data['_split'].replace(largest, 'train')
                v.fix(f"Renamed largest split '{largest}' to 'train'")

    n = len(dataset.y)
    if n < 10:
        v.warn(f"Very small dataset ({n} samples) — results may be unreliable")

    v.print_summary()
    return dataset, v


# ---------------------------------------------------------------------------
# Step 2: Validate + fix profiling
# ---------------------------------------------------------------------------

def validate_and_fix_profile(profile, dataset=None) -> tuple[Any, ValidationResult]:
    """Validate dataset profile. Fix issues where possible."""
    from co_scientist.data.types import Modality, TaskType

    v = ValidationResult("profiling")

    # Fix unknown modality from dataset content
    if profile.modality == Modality.UNKNOWN and dataset is not None:
        import pandas as pd
        if isinstance(dataset.X, pd.DataFrame):
            # Check for sequence columns
            for col in dataset.X.columns:
                if col.lower() in ("sequences", "sequence", "seq"):
                    sample = str(dataset.X[col].iloc[0])[:100].upper()
                    if set(sample) <= set("ACGTN"):
                        profile.modality = Modality.RNA
                        v.fix("Detected RNA modality from sequence column content")
                    elif set(sample) <= set("ACDEFGHIKLMNPQRSTVWY"):
                        profile.modality = Modality.PROTEIN
                        v.fix("Detected PROTEIN modality from sequence column content")
                    break
            if profile.modality == Modality.UNKNOWN:
                # Heuristic: many numeric cols → tabular
                numeric_cols = dataset.X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 100:
                    profile.modality = Modality.CELL_EXPRESSION
                    v.fix("Detected CELL_EXPRESSION modality (>100 numeric features)")
                else:
                    profile.modality = Modality.TABULAR
                    v.fix("Defaulted to TABULAR modality")

    if profile.modality == Modality.UNKNOWN:
        v.warn("Modality still UNKNOWN — using TABULAR as fallback")
        profile.modality = Modality.TABULAR
        v.fix("Set modality to TABULAR as fallback")

    # Fix unknown task type from target values
    if profile.task_type == TaskType.UNKNOWN and dataset is not None:
        y = np.array(dataset.y)
        if y.dtype == object or str(y.dtype) == "category":
            n_unique = len(set(y))
            profile.task_type = TaskType.BINARY_CLASSIFICATION if n_unique <= 2 else TaskType.MULTICLASS_CLASSIFICATION
            profile.num_classes = n_unique
            v.fix(f"Inferred {profile.task_type.value} from target ({n_unique} unique values)")
        elif np.issubdtype(y.dtype, np.integer):
            n_unique = len(set(y))
            if n_unique <= 2:
                profile.task_type = TaskType.BINARY_CLASSIFICATION
            elif n_unique <= 50:
                profile.task_type = TaskType.MULTICLASS_CLASSIFICATION
            else:
                profile.task_type = TaskType.REGRESSION
            profile.num_classes = n_unique if profile.task_type != TaskType.REGRESSION else None
            v.fix(f"Inferred {profile.task_type.value} from target dtype + cardinality")
        else:
            profile.task_type = TaskType.REGRESSION
            v.fix("Defaulted to REGRESSION for float target")

    if profile.task_type == TaskType.UNKNOWN:
        v.fail("Task type is UNKNOWN — cannot select models/metrics")

    if profile.num_samples == 0:
        v.fail("Profile reports 0 samples")

    if profile.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        if profile.num_classes is not None and profile.num_classes < 2:
            v.fail(f"Classification task but only {profile.num_classes} class(es)")

    v.print_summary()
    return profile, v


# ---------------------------------------------------------------------------
# Step 3: Validate + fix preprocessing
# ---------------------------------------------------------------------------

def validate_and_fix_preprocessing(preprocessed, profile) -> tuple[Any, ValidationResult]:
    """Validate preprocessed data. Fix NaN/Inf/constant features."""
    v = ValidationResult("preprocessing")

    X = preprocessed.X
    y = preprocessed.y

    if X.shape[0] == 0:
        v.fail("Preprocessed X has 0 rows — cannot fix")
        v.print_summary()
        return preprocessed, v

    if X.shape[1] == 0:
        v.fail("Preprocessed X has 0 features — cannot fix")
        v.print_summary()
        return preprocessed, v

    if X.shape[0] != len(y):
        # Fix: truncate
        n_min = min(X.shape[0], len(y))
        preprocessed.X = X[:n_min]
        preprocessed.y = y[:n_min]
        v.fix(f"Truncated X and y to {n_min} rows")
        X = preprocessed.X
        y = preprocessed.y

    # Fix NaN values
    if np.issubdtype(X.dtype, np.floating):
        n_nan = np.isnan(X).sum()
        if n_nan > 0:
            pct = n_nan / X.size * 100
            # Fix: replace NaN with column mean (or 0 if all NaN)
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0, col_means)
            nan_mask = np.isnan(X)
            X_fixed = X.copy()
            for j in range(X.shape[1]):
                X_fixed[nan_mask[:, j], j] = col_means[j]
            preprocessed.X = X_fixed
            X = preprocessed.X
            v.fix(f"Replaced {n_nan} NaN values ({pct:.1f}%) with column means")

        # Fix Inf values
        n_inf = np.isinf(X).sum()
        if n_inf > 0:
            # Fix: clip to finite range
            finite_mask = np.isfinite(X)
            if finite_mask.any():
                finite_max = np.abs(X[finite_mask]).max() * 10
            else:
                finite_max = 1e6
            X_fixed = np.clip(X, -finite_max, finite_max)
            preprocessed.X = X_fixed
            X = preprocessed.X
            v.fix(f"Clipped {n_inf} Inf values to ±{finite_max:.0f}")

    # Warn about constant features (don't remove — could break feature alignment)
    if X.shape[0] > 1:
        variances = np.var(X, axis=0)
        n_zero_var = np.sum(variances == 0)
        if n_zero_var > X.shape[1] * 0.5:
            v.warn(f"{n_zero_var}/{X.shape[1]} features have zero variance")

    # Fix NaN in target
    y_arr = np.array(y)
    if np.issubdtype(y_arr.dtype, np.floating) and np.any(np.isnan(y_arr)):
        nan_rows = np.isnan(y_arr)
        n_nan_y = nan_rows.sum()
        # Fix: drop rows with NaN target
        keep = ~nan_rows
        preprocessed.X = X[keep]
        preprocessed.y = y_arr[keep]
        v.fix(f"Dropped {n_nan_y} rows with NaN target values")

    v.print_summary()
    return preprocessed, v


# ---------------------------------------------------------------------------
# Step 4: Validate + fix split
# ---------------------------------------------------------------------------

def validate_and_fix_split(split, preprocessed=None, profile=None, seed=42) -> tuple[Any, ValidationResult]:
    """Validate train/val/test split. Fix empty splits, NaN values."""
    v = ValidationResult("splitting")

    # Check for empty splits
    empty_splits = []
    for name, X, y in [
        ("train", split.X_train, split.y_train),
        ("val", split.X_val, split.y_val),
        ("test", split.X_test, split.y_test),
    ]:
        if len(y) == 0:
            empty_splits.append(name)

    # Fix empty val/test by carving from train
    if "train" not in empty_splits and empty_splits:
        n_train = len(split.y_train)
        rng = np.random.RandomState(seed)

        if "val" in empty_splits and n_train > 10:
            n_val = max(1, int(n_train * 0.15))
            idx = rng.permutation(n_train)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            split.X_val = split.X_train[val_idx]
            split.y_val = split.y_train[val_idx]
            split.X_train = split.X_train[train_idx]
            split.y_train = split.y_train[train_idx]
            v.fix(f"Carved {n_val} validation samples from train")

        if "test" in empty_splits and len(split.y_train) > 10:
            n_test = max(1, int(len(split.y_train) * 0.15))
            n_remaining = len(split.y_train)
            idx = rng.permutation(n_remaining)
            test_idx, train_idx = idx[:n_test], idx[n_test:]
            split.X_test = split.X_train[test_idx]
            split.y_test = split.y_train[test_idx]
            split.X_train = split.X_train[train_idx]
            split.y_train = split.y_train[train_idx]
            v.fix(f"Carved {n_test} test samples from train")

    elif "train" in empty_splits:
        v.fail("Train split is empty — cannot fix")
        v.print_summary()
        return split, v

    # Fix feature count mismatch
    n_train_f = split.X_train.shape[1]
    for name, attr in [("val", "X_val"), ("test", "X_test")]:
        X = getattr(split, attr)
        if X.shape[1] != n_train_f:
            # Fix: pad or truncate features
            if X.shape[1] < n_train_f:
                pad = np.zeros((X.shape[0], n_train_f - X.shape[1]))
                setattr(split, attr, np.hstack([X, pad]))
                v.fix(f"Padded {name} features from {X.shape[1]} to {n_train_f}")
            else:
                setattr(split, attr, X[:, :n_train_f])
                v.fix(f"Truncated {name} features from {X.shape[1]} to {n_train_f}")

    # Fix NaN in splits
    for name, attr in [("train", "X_train"), ("val", "X_val"), ("test", "X_test")]:
        X = getattr(split, attr)
        if np.issubdtype(X.dtype, np.floating) and np.any(np.isnan(X)):
            n_nan = np.isnan(X).sum()
            col_means = np.nanmean(split.X_train, axis=0)  # always use train stats
            col_means = np.where(np.isnan(col_means), 0, col_means)
            X_fixed = X.copy()
            for j in range(X.shape[1]):
                mask = np.isnan(X_fixed[:, j])
                if mask.any():
                    X_fixed[mask, j] = col_means[j]
            setattr(split, attr, X_fixed)
            v.fix(f"Replaced {n_nan} NaN values in {name} split with train column means")

    # Check for data leakage
    if len(split.y_train) > 0 and len(split.y_test) > 0:
        n_check = min(3, split.X_train.shape[1])
        train_hashes = set(map(tuple, split.X_train[:, :n_check].round(6)))
        test_hashes = set(map(tuple, split.X_test[:, :n_check].round(6)))
        overlap = len(train_hashes & test_hashes)
        if overlap > len(test_hashes) * 0.5:
            v.warn(f"Possible train/test leakage: {overlap} overlapping feature vectors")

    # Split proportions warning
    n_total = len(split.y_train) + len(split.y_val) + len(split.y_test)
    if n_total > 0:
        train_pct = len(split.y_train) / n_total * 100
        if train_pct < 10:
            v.warn(f"Train split is unusually small ({train_pct:.0f}% of data)")

    v.print_summary()
    return split, v


# ---------------------------------------------------------------------------
# Step 5: Validate trained model (detection only — no auto-fix)
# ---------------------------------------------------------------------------

def validate_trained_model(trained, split, eval_config=None) -> ValidationResult:
    """Validate a trained model produces correct predictions."""
    v = ValidationResult(f"model_{trained.config.name}")

    try:
        # Route to correct feature set based on model type
        from co_scientist.modeling.types import _CONCAT_MODELS
        if trained.config.model_type in _CONCAT_MODELS and split.X_embed_val is not None:
            n_check = min(20, len(split.X_val))
            X_check = np.hstack([split.X_val[:n_check], split.X_embed_val[:n_check]])
            y_pred = trained.model.predict(X_check)
        elif trained.needs_embeddings and split.X_embed_val is not None:
            n_check = min(20, len(split.X_embed_val))
            X_check = split.X_embed_val[:n_check]
            y_pred = trained.model.predict(X_check)
        elif trained.needs_sequences and split.seqs_val:
            n_check = min(20, len(split.X_val))
            X_check = split.X_val[:n_check]
            seqs_check = split.seqs_val[:n_check]
            y_pred = trained.model.predict(X_check, sequences=seqs_check)
        else:
            n_check = min(20, len(split.X_val))
            X_check = split.X_val[:n_check]
            y_pred = trained.model.predict(X_check)

        if len(y_pred) != n_check:
            v.fail(f"predict() returned {len(y_pred)} values, expected {n_check}")

        y_arr = np.array(y_pred, dtype=float)
        if np.any(np.isnan(y_arr)):
            v.fail("Model predictions contain NaN")
        if np.any(np.isinf(y_arr)):
            v.fail("Model predictions contain Inf")
        if len(set(y_pred)) == 1 and n_check > 5:
            v.warn(f"Model predicts constant value for all {n_check} samples")

    except Exception as e:
        v.fail(f"Model prediction failed: {type(e).__name__}: {e}")

    return v


# ---------------------------------------------------------------------------
# Step 6: Validate + fix exported scripts
# ---------------------------------------------------------------------------

def validate_and_fix_export(output_dir: Path, profile=None, client=None) -> ValidationResult:
    """Validate exported train.py/predict.py. Auto-fix syntax and import errors.

    Args:
        output_dir: Pipeline output directory
        profile: Dataset profile (for finding subdirectory names)
        client: ClaudeClient for LLM-assisted repair (optional)
    """
    v = ValidationResult("export")
    ds_name = profile.dataset_name.replace("/", "_").replace(" ", "_") if profile else "*"

    reproduce_dir = output_dir / f"reproduce_{ds_name}"
    inference_dir = output_dir / f"inference_{ds_name}"

    for script_path in [reproduce_dir / "train.py", inference_dir / "predict.py"]:
        if not script_path.exists():
            v.fail(f"Missing: {script_path.name}")
            continue

        source = script_path.read_text()

        # 1. Syntax check + auto-fix
        try:
            compile(source, str(script_path), "exec")
        except SyntaxError as e:
            issue = f"{script_path.name}: SyntaxError at line {e.lineno}: {e.msg}"
            # Try deterministic fixes first
            fixed_source = _auto_fix_syntax(source, e)
            if fixed_source:
                try:
                    compile(fixed_source, str(script_path), "exec")
                    script_path.write_text(fixed_source)
                    v.fix(f"Auto-fixed syntax error in {script_path.name}")
                    source = fixed_source
                except SyntaxError:
                    pass  # deterministic fix didn't work

            # If still broken, try LLM
            if not _compiles(source):
                if client and _llm_fix_script(script_path, issue, client):
                    v.fix(f"LLM-fixed syntax error in {script_path.name}")
                    source = script_path.read_text()
                else:
                    v.fail(issue)
                    continue

        # 2. Import check + auto-fix
        missing = _check_imports(script_path)
        if missing:
            # Try LLM fix for missing imports
            issue = f"{script_path.name}: {missing}"
            if client and _llm_fix_script(script_path, issue, client):
                v.fix(f"LLM-fixed import issue in {script_path.name}")
            else:
                v.fail(issue)

    # Check model files
    model_dir = inference_dir / "model"
    if model_dir.exists():
        has_pkl = (model_dir / "best_model.pkl").exists()
        has_pt = (model_dir / "best_model.pt").exists()
        if not has_pkl and not has_pt:
            v.fail("No model file found (best_model.pkl or best_model.pt)")
        if not (model_dir / "model_config.json").exists():
            v.fail("Missing model_config.json")

    # Check requirements.txt
    for req_path in [reproduce_dir / "requirements.txt", inference_dir / "requirements.txt"]:
        if req_path.exists():
            reqs = req_path.read_text().strip()
            if not reqs:
                v.warn(f"Empty {req_path.name}")
        elif req_path.parent.exists():
            v.warn(f"Missing {req_path.name}")

    # 3. Execution test — actually run the scripts
    train_script = reproduce_dir / "train.py"
    predict_script = inference_dir / "predict.py"

    if train_script.exists() and _compiles(train_script.read_text()):
        train_ok, train_err = _test_run_script(train_script, timeout=600)
        if train_ok:
            v.fix("train.py executed successfully") if v.issues else None
            console.print(f"    [green]train.py: execution OK[/green]")
        else:
            v.fail(f"train.py execution failed: {train_err}")
            # Try LLM fix
            if client and train_err:
                if _llm_fix_script(train_script, f"Runtime error: {train_err}", client):
                    v.fix(f"LLM-fixed runtime error in train.py")
                    # Re-test after fix
                    ok2, err2 = _test_run_script(train_script, timeout=300)
                    if ok2:
                        console.print(f"    [green]train.py: execution OK after LLM fix[/green]")
                    else:
                        v.fail(f"train.py still fails after LLM fix: {err2}")

    if predict_script.exists() and _compiles(predict_script.read_text()):
        predict_ok, predict_err = _test_run_script(predict_script, timeout=120)
        if predict_ok:
            console.print(f"    [green]predict.py: execution OK[/green]")
        else:
            # predict.py may fail without real input data — only warn
            v.warn(f"predict.py execution: {predict_err}")

    v.print_summary()
    return v


def _compiles(source: str) -> bool:
    """Check if Python source compiles without error."""
    try:
        compile(source, "<check>", "exec")
        return True
    except SyntaxError:
        return False


def _test_run_script(script_path: Path, timeout: int = 300) -> tuple[bool, str | None]:
    """Run a script in a subprocess and check if it completes without error.

    Returns (success: bool, error_message: str | None).
    """
    import os

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    try:
        abs_script = script_path.resolve()
        result = subprocess.run(
            [sys.executable, str(abs_script)],
            cwd=str(abs_script.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode == 0:
            return True, None

        # Extract last meaningful error from stderr
        stderr = result.stderr.strip()
        if stderr:
            lines = stderr.split("\n")
            # Get last traceback line or last few lines
            error_lines = []
            for line in reversed(lines):
                error_lines.insert(0, line)
                if line.startswith("Traceback") or len(error_lines) >= 5:
                    break
            error_msg = "\n".join(error_lines)
        else:
            error_msg = f"Exit code {result.returncode}"

        # Truncate for display
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."

        return False, error_msg

    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _auto_fix_syntax(source: str, error: SyntaxError) -> str | None:
    """Try deterministic syntax fixes for common patterns."""
    lines = source.split("\n")
    line_idx = (error.lineno or 1) - 1

    if line_idx >= len(lines):
        return None

    line = lines[line_idx]

    # Fix 1: backslash in f-string (common exporter bug)
    # e.g., f"...{f1_score(y, p, average=\'macro\'):.4f}"
    if "\\'" in line and "f\"" in line or "f'" in line:
        fixed_line = line.replace("\\'", '"')
        lines[line_idx] = fixed_line
        result = "\n".join(lines)
        if _compiles(result):
            return result

    # Fix 2: function call with format spec inside f-string
    # e.g., f"{func(x, average='macro'):.4f}" → compute first, then format
    if "f\"" in line or "f'" in line:
        # Extract the problematic expression and move it out
        # This is hard to do generically — fall through to LLM
        pass

    return None


def _check_imports(script_path: Path) -> str | None:
    """Check that a script's imports are all available."""
    source = script_path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])

    skip = {
        "os", "sys", "json", "pickle", "argparse", "warnings", "collections",
        "itertools", "pathlib", "typing", "math", "time", "inspect",
        "importlib", "abc", "dataclasses", "enum", "functools", "re",
        "copy", "io", "tempfile", "hashlib", "subprocess", "ast",
    }
    modules -= skip

    missing = []
    for mod in sorted(modules):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        return f"Missing imports: {', '.join(missing)}"
    return None


def _llm_fix_script(script_path: Path, error_msg: str, client) -> bool:
    """Use LLM to diagnose and fix a broken script."""
    if client is None:
        return False

    source = script_path.read_text()
    prompt = (
        f"The following Python script has an error:\n\n"
        f"File: {script_path.name}\n"
        f"Error: {error_msg}\n\n"
        f"```python\n{source}\n```\n\n"
        f"Fix the error and return the COMPLETE corrected script. "
        f"Return ONLY the Python code inside a ```python code fence."
    )

    try:
        response = client.ask_text(
            prompt,
            system="You are a Python debugging expert. Fix the script and return the complete corrected version."
        )

        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.strip()

        compile(code, str(script_path), "exec")
        script_path.write_text(code)
        console.print(f"    [green]LLM fix applied to {script_path.name}[/green]")
        return True

    except Exception as e:
        console.print(f"    [red]LLM fix failed: {e}[/red]")
        return False


# ---------------------------------------------------------------------------
# Backward-compatible aliases (for existing CLI calls)
# ---------------------------------------------------------------------------

def validate_loaded_data(dataset, profile=None) -> ValidationResult:
    _, v = validate_and_fix_loaded_data(dataset, profile)
    return v

def validate_profile(profile) -> ValidationResult:
    _, v = validate_and_fix_profile(profile)
    return v

def validate_preprocessing(preprocessed, profile) -> ValidationResult:
    _, v = validate_and_fix_preprocessing(preprocessed, profile)
    return v

def validate_split(split) -> ValidationResult:
    _, v = validate_and_fix_split(split)
    return v

def validate_export(output_dir, profile=None) -> ValidationResult:
    return validate_and_fix_export(output_dir, profile)

def llm_fix_script(script_path, error_msg, client) -> bool:
    return _llm_fix_script(script_path, error_msg, client)
