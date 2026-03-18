"""Error recovery and graceful degradation — the system always produces output.

Adopted from:
  - CellAgent (Architecture Section 2.3): "Error-feedback self-correction —
    when training fails, feed the exact error message and context back for
    intelligent recovery."
  - Architecture Section 10.1: Infrastructure failure handling
  - Architecture Section 10.2: Graceful degradation chain

Deterministic version: catch errors, log structured context, retry with
fallback config, then skip. The LLM-driven version (Phase C) will use the
logged error context for intelligent recovery.

Segfault protection: critical operations can be run in isolated subprocesses
so a native library crash (OMP conflict, corrupted memory) kills only the
subprocess, not the entire pipeline.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console

from co_scientist.experiment_log import ExperimentLog

console = Console()

# Ensure OMP safety in any process that imports this module
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


@dataclass
class ErrorContext:
    """Structured error information for recovery (CellAgent pattern)."""

    step: str
    error_type: str           # e.g., "MemoryError", "ValueError"
    error_message: str        # the str(exception)
    traceback: str            # full traceback for debugging
    model_name: str = ""      # which model failed (if applicable)
    attempted_config: dict = field(default_factory=dict)  # what was tried
    recovery_action: str = "" # what we did about it

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "model_name": self.model_name,
            "attempted_config": self.attempted_config,
            "recovery_action": self.recovery_action,
        }


# ---------------------------------------------------------------------------
# Fallback configurations for common failures
# ---------------------------------------------------------------------------

_FALLBACK_CONFIGS: dict[str, dict[str, Any]] = {
    # OOM or slow training → reduce model capacity
    "MemoryError": {
        "xgboost": {"n_estimators": 50, "max_depth": 4},
        "lightgbm": {"n_estimators": 50, "max_depth": 4, "num_leaves": 15},
        "random_forest": {"n_estimators": 50, "max_depth": 6},
        "mlp": {"hidden_dims": [64, 32], "batch_size": 128, "max_epochs": 20},
    },
    # Numerical issues → increase regularization
    "ValueError": {
        "xgboost": {"reg_alpha": 1.0, "reg_lambda": 5.0},
        "lightgbm": {"reg_alpha": 1.0, "reg_lambda": 5.0},
        "random_forest": {"min_samples_leaf": 5, "max_depth": 8},
        "mlp": {"dropout": 0.5, "weight_decay": 0.01},
    },
    # Generic fallback
    "_default": {
        "xgboost": {"n_estimators": 50, "max_depth": 3},
        "lightgbm": {"n_estimators": 50, "max_depth": 3, "num_leaves": 15},
        "random_forest": {"n_estimators": 50, "max_depth": 5},
        "mlp": {"hidden_dims": [64], "max_epochs": 10},
    },
}


def get_fallback_hp(error_type: str, model_type: str) -> dict[str, Any] | None:
    """Get fallback hyperparameters for a given error type and model."""
    configs = _FALLBACK_CONFIGS.get(error_type, _FALLBACK_CONFIGS["_default"])
    return configs.get(model_type)


# ---------------------------------------------------------------------------
# Resilient execution wrapper
# ---------------------------------------------------------------------------

def run_with_recovery(
    fn: Callable,
    step: str,
    exp_log: ExperimentLog,
    model_name: str = "",
    fallback_fn: Callable | None = None,
    max_retries: int = 1,
) -> Any:
    """Execute a function with error recovery.

    Architecture Section 10.1 recovery chain:
      1. Try the original function
      2. On failure: log error context, try fallback if provided
      3. On second failure: log and return None (skip)

    Returns the function result, or None if all attempts fail.
    """
    for attempt in range(1 + max_retries):
        try:
            if attempt == 0:
                return fn()
            elif fallback_fn is not None:
                console.print(f"    [yellow]Retrying {model_name or step} with fallback config...[/yellow]")
                return fallback_fn()
            else:
                return None
        except KeyboardInterrupt:
            raise  # never catch user interrupts
        except Exception as e:
            tb = traceback.format_exc()
            ctx = ErrorContext(
                step=step,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=tb,
                model_name=model_name,
                recovery_action="retry_fallback" if attempt == 0 and fallback_fn else "skip",
            )

            exp_log.log("error_recovery", ctx.to_dict())

            if attempt == 0:
                console.print(f"    [yellow]{type(e).__name__}: {e}[/yellow]")
                if fallback_fn is None:
                    console.print(f"    [yellow]Skipping {model_name or step}[/yellow]")
                    return None
            else:
                console.print(f"    [red]Fallback also failed: {e}[/red]")
                console.print(f"    [yellow]Skipping {model_name or step}[/yellow]")
                return None

    return None


# ---------------------------------------------------------------------------
# Resilient model training
# ---------------------------------------------------------------------------

def train_model_resilient(
    config: Any,
    split: Any,
    exp_log: ExperimentLog,
) -> Any:
    """Train a single model with error recovery.

    On failure:
      1. Log structured error context (CellAgent pattern)
      2. Try fallback config (reduced capacity / increased regularization)
      3. If fallback fails, skip and return None
    """
    from co_scientist.modeling.trainer import train_model
    from co_scientist.modeling.types import ModelConfig

    model_name = config.name
    model_type = config.model_type

    def try_train():
        return train_model(config, split)

    def try_fallback():
        fallback_hp = get_fallback_hp("_default", model_type)
        if fallback_hp is None:
            raise RuntimeError(f"No fallback config for {model_type}")
        merged_hp = dict(config.hyperparameters)
        merged_hp.update(fallback_hp)
        fallback_config = ModelConfig(
            name=f"{model_name}_fallback",
            tier=config.tier,
            model_type=model_type,
            task_type=config.task_type,
            hyperparameters=merged_hp,
        )
        return train_model(fallback_config, split)

    return run_with_recovery(
        fn=try_train,
        step="baselines",
        exp_log=exp_log,
        model_name=model_name,
        fallback_fn=try_fallback,
    )


def train_baselines_resilient(
    configs: list,
    split: Any,
    exp_log: ExperimentLog,
) -> list:
    """Train all baselines with per-model error recovery.

    If a model fails, tries fallback config. If that fails too, skips it.
    The pipeline continues with whatever models succeed.
    """
    models = []
    for config in configs:
        trained = train_model_resilient(config, split, exp_log)
        if trained is not None:
            models.append(trained)
    return models


# ---------------------------------------------------------------------------
# Resilient pipeline step wrapper
# ---------------------------------------------------------------------------

def run_step_resilient(
    step_name: str,
    fn: Callable,
    exp_log: ExperimentLog,
    critical: bool = False,
) -> Any:
    """Run a pipeline step with graceful degradation.

    Architecture Section 10.2:
      Full agent → Rule-based only → Baselines only → Partial report

    If critical=True, re-raises the exception (pipeline can't continue).
    If critical=False, logs error and returns None (pipeline continues with partial results).
    """
    try:
        return fn()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        ctx = ErrorContext(
            step=step_name,
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=tb,
            recovery_action="halt" if critical else "skip",
        )
        exp_log.log("error_recovery", ctx.to_dict())

        if critical:
            console.print(f"  [bold red]Critical step '{step_name}' failed: {e}[/bold red]")
            raise
        else:
            console.print(f"  [yellow]Step '{step_name}' failed: {e}. Continuing with partial results.[/yellow]")
            return None


# ---------------------------------------------------------------------------
# Subprocess-isolated execution (survives segfaults)
# ---------------------------------------------------------------------------

def run_in_subprocess(fn: Callable, timeout: int = 600) -> Any:
    """Run a callable in an isolated subprocess.

    If the subprocess segfaults, this returns None instead of killing the
    entire pipeline. Uses multiprocessing with 'spawn' to get a clean
    process (no inherited OMP state).

    Args:
        fn: A picklable callable (e.g., a top-level function or lambda-free callable).
        timeout: Max seconds to wait.

    Returns:
        The function result, or None if the subprocess crashed/timed out.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")  # clean process, no fork hazards
    result_queue = ctx.Queue()

    def _worker(q, func):
        """Subprocess worker: run func, put result in queue."""
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            result = func()
            q.put(("ok", result))
        except Exception as e:
            q.put(("error", (type(e).__name__, str(e), traceback.format_exc())))

    proc = ctx.Process(target=_worker, args=(result_queue, fn))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        console.print(f"  [red]Subprocess timed out after {timeout}s — killing[/red]")
        proc.kill()
        proc.join(timeout=5)
        return None

    if proc.exitcode != 0:
        if proc.exitcode == -11:  # SIGSEGV
            console.print("  [red]Subprocess crashed with segmentation fault (SIGSEGV)[/red]")
            console.print("  [yellow]This is usually an OpenMP library conflict. Skipping this model.[/yellow]")
        elif proc.exitcode and proc.exitcode < 0:
            import signal
            sig_name = signal.Signals(-proc.exitcode).name if hasattr(signal.Signals, '__call__') else f"signal {-proc.exitcode}"
            console.print(f"  [red]Subprocess killed by {sig_name}[/red]")
        else:
            console.print(f"  [red]Subprocess exited with code {proc.exitcode}[/red]")
        return None

    try:
        status, payload = result_queue.get_nowait()
        if status == "ok":
            return payload
        else:
            err_type, err_msg, err_tb = payload
            console.print(f"  [yellow]Subprocess error: {err_type}: {err_msg}[/yellow]")
            return None
    except Exception:
        console.print("  [red]Could not retrieve subprocess result[/red]")
        return None
