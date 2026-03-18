"""Tool registry for the ReAct agent.

Each tool is a thin wrapper around existing pipeline infrastructure.
Tools execute actions and return structured observations for the agent's scratchpad.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from co_scientist.agents.react import ReactState

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    success: bool
    observation: str  # Human-readable for scratchpad
    data: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    score: float | None = None


class Tool(ABC):
    """Base class for ReAct agent tools."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    @abstractmethod
    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        ...

    def describe(self) -> str:
        """Return a description for the system prompt."""
        param_parts = []
        for pname, pinfo in self.parameters_schema.items():
            req = " (required)" if pinfo.get("required", False) else ""
            param_parts.append(f'    "{pname}": {pinfo.get("description", pinfo.get("type", "any"))}{req}')
        params_str = ",\n".join(param_parts)
        return f"{self.name} — {self.description}\n  Parameters: {{\n{params_str}\n  }}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class TrainModelTool(Tool):
    """Train a model type and evaluate on validation set."""

    name = "train_model"
    description = "Train a model type and return validation score. Available: xgboost, lightgbm, random_forest, ridge_regression, logistic_regression, svm, knn, mlp, ft_transformer. GPU-only (embeddings): embed_xgboost, embed_mlp. GPU-only (hybrid features+embeddings): concat_xgboost, concat_mlp. GPU-only (fine-tuning): aido_finetune. Each model has a 120s timeout."
    parameters_schema = {
        "model_type": {"type": "string", "required": True, "description": "Model type to train"},
        "hyperparameters": {"type": "object", "required": False, "description": "Optional HP overrides"},
    }

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        from co_scientist.modeling.registry import build_model
        from co_scientist.modeling.types import ModelConfig, TrainedModel
        from co_scientist.evaluation.metrics import evaluate_model

        model_type = params.get("model_type", "")
        if not model_type:
            return ToolResult(success=False, observation="Error: model_type is required")

        # Check if already trained
        existing = [r for r in state.results if r.model_name.startswith(model_type)]
        suffix = f"_react_{len(existing)}" if existing else "_default"
        model_name = f"{model_type}{suffix}"

        task_family = "classification" if "classification" in state.eval_config.task_type else "regression"
        hp = dict(params.get("hyperparameters", {}) or {})
        _seed_models = {"xgboost", "lightgbm", "random_forest", "mlp", "bio_cnn", "svm",
                        "ft_transformer", "embed_xgboost", "embed_mlp", "aido_finetune"}
        if model_type in _seed_models:
            hp.setdefault("random_state", state.seed)
        if model_type == "lightgbm":
            hp.setdefault("verbose", -1)
        # Inject AIDO model name for fine-tuning
        if model_type == "aido_finetune":
            from co_scientist.modeling.foundation import get_foundation_model_name
            hp.setdefault("model_name", get_foundation_model_name(state.profile.modality.value))

        tier = "foundation" if model_type in ("embed_xgboost", "embed_mlp", "aido_finetune") else "react"
        config = ModelConfig(
            name=model_name,
            tier=tier,
            model_type=model_type,
            task_type=task_family,
            hyperparameters=hp,
        )

        # Check prerequisites for foundation models
        _embedding_models = {"embed_xgboost", "embed_mlp"}
        _concat_models = {"concat_xgboost", "concat_mlp"}
        if (model_type in _embedding_models or model_type in _concat_models) and state.split.X_embed_train is None:
            return ToolResult(success=False, observation=f"Error: {model_type} requires AIDO embeddings but none available (no GPU?)")

        try:
            model = build_model(config)
        except Exception as e:
            return ToolResult(success=False, observation=f"Error: could not build model '{model_type}': {e}")

        try:
            console.print(f"    [dim]Training {model_name}...[/dim]")
            start = time.time()
            _seq_models = {"bio_cnn", "aido_finetune"}
            y_train = state.split.y_train
            if config.model_type in _concat_models:
                import numpy as _np
                X_combined = _np.hstack([state.split.X_train, state.split.X_embed_train])
                model.fit(X_combined, y_train)
            elif config.model_type in _embedding_models:
                model.fit(state.split.X_embed_train, y_train)
            elif config.model_type in _seq_models:
                model.fit(state.split.X_train, state.split.y_train, sequences=state.split.seqs_train)
            else:
                model.fit(state.split.X_train, y_train)
            train_time = time.time() - start
        except Exception as e:
            return ToolResult(success=False, observation=f"Error: training failed for '{model_type}': {e}")

        trained = TrainedModel(config=config, model=model, train_time_seconds=train_time)
        result = evaluate_model(trained, state.split, state.eval_config, use_test=False)

        state.trained_models.append(trained)
        state.results.append(result)

        score = result.primary_metric_value
        obs = (
            f"Trained {model_name}: "
            f"{state.eval_config.primary_metric}={score:.4f} "
            f"({train_time:.1f}s)"
        )
        console.print(f"    [green]{obs}[/green]")

        return ToolResult(
            success=True,
            observation=obs,
            data={"metrics": result.metrics},
            model_name=model_name,
            score=score,
        )


class TuneHyperparametersTool(Tool):
    """Run Optuna HP search on a model type."""

    name = "tune_hyperparameters"
    description = "Run Optuna hyperparameter search on a trained model type. Returns before/after scores."
    parameters_schema = {
        "model_type": {"type": "string", "required": True, "description": "Model type to tune (must already be trained)"},
        "n_trials": {"type": "integer", "required": False, "description": "Number of Optuna trials (default: 20)"},
    }

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        from co_scientist.modeling.hp_search import run_hp_search

        model_type = params.get("model_type", "")
        if not model_type:
            return ToolResult(success=False, observation="Error: model_type is required")

        # Find base config for this model type
        base_trained = None
        base_result = None
        for t, r in zip(state.trained_models, state.results):
            if t.config.model_type == model_type:
                base_trained = t
                base_result = r
                break

        if base_trained is None:
            return ToolResult(
                success=False,
                observation=f"Error: no trained model of type '{model_type}' found. Train it first.",
            )

        n_trials = params.get("n_trials", 15)
        timeout = 120  # 2 min default

        try:
            hp_result = run_hp_search(
                base_config=base_trained.config,
                split=state.split,
                eval_config=state.eval_config,
                profile=state.profile,
                seed=state.seed,
                n_trials_override=n_trials,
                timeout_override=timeout,
            )
        except Exception as e:
            return ToolResult(success=False, observation=f"Error: HP search failed: {e}")

        if hp_result is None:
            return ToolResult(
                success=False,
                observation=f"HP search returned no result for {model_type} (no search space defined or no improvement).",
            )

        tuned_trained, tuned_result = hp_result
        state.trained_models.append(tuned_trained)
        state.results.append(tuned_result)

        before = base_result.primary_metric_value
        after = tuned_result.primary_metric_value
        obs = (
            f"Tuned {model_type}: "
            f"{state.eval_config.primary_metric} {before:.4f} → {after:.4f} "
            f"({'improved' if after > before else 'no improvement'})"
        )
        console.print(f"    [green]{obs}[/green]")

        return ToolResult(
            success=True,
            observation=obs,
            data={"before": before, "after": after, "hp": tuned_trained.config.hyperparameters},
            model_name=tuned_result.model_name,
            score=after,
        )


class GetModelScoresTool(Tool):
    """Return a table of all trained models and their scores."""

    name = "get_model_scores"
    description = "Get a table of all trained models with their validation scores. No parameters needed."
    parameters_schema = {}

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        if not state.results:
            return ToolResult(success=True, observation="No models trained yet.")

        lines = [f"{'Model':<30} {'Tier':<10} {state.eval_config.primary_metric:>12} {'Time':>8}"]
        lines.append("-" * 65)

        lower_is_better = state.eval_config.primary_metric in ("mse", "rmse", "mae")
        sorted_results = sorted(
            state.results,
            key=lambda r: r.primary_metric_value,
            reverse=not lower_is_better,
        )
        for r in sorted_results:
            lines.append(
                f"{r.model_name:<30} {r.tier:<10} {r.primary_metric_value:>12.4f} {r.train_time_seconds:>7.1f}s"
            )

        obs = "\n".join(lines)
        return ToolResult(
            success=True,
            observation=obs,
            data={"models": [{
                "name": r.model_name,
                "tier": r.tier,
                "score": r.primary_metric_value,
            } for r in sorted_results]},
        )


class AnalyzeErrorsTool(Tool):
    """Analyze errors for a trained model (confusion matrix or residual stats)."""

    name = "analyze_errors"
    description = "Analyze prediction errors for a model. Returns confusion matrix (classification) or residual statistics (regression)."
    parameters_schema = {
        "model_name": {"type": "string", "required": False, "description": "Model to analyze (default: best model)"},
    }

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        target_name = params.get("model_name", "")
        trained = None
        for t in state.trained_models:
            if t.config.name == target_name:
                trained = t
                break
        if trained is None and state.best_trained:
            trained = state.best_trained
            target_name = trained.config.name

        if trained is None:
            return ToolResult(success=False, observation="Error: no model to analyze")

        y_val = state.split.y_val
        seqs_val = state.split.seqs_val

        # Route to correct feature set
        _concat = {"concat_xgboost", "concat_mlp"}
        if trained.config.model_type in _concat and state.split.X_embed_val is not None:
            X_val = np.hstack([state.split.X_val, state.split.X_embed_val])
        elif trained.needs_embeddings and state.split.X_embed_val is not None:
            X_val = state.split.X_embed_val
        else:
            X_val = state.split.X_val

        y_pred = trained.predict(X_val, sequences=seqs_val)

        is_clf = "classification" in state.eval_config.task_type
        if is_clf:
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_val, y_pred)
            report = classification_report(y_val, y_pred, zero_division=0)
            obs = f"Error analysis for {target_name}:\n{report}"
            data = {"confusion_matrix": cm.tolist(), "report": report}
        else:
            residuals = y_val - y_pred
            obs = (
                f"Error analysis for {target_name}:\n"
                f"  Mean residual: {np.mean(residuals):.4f}\n"
                f"  Std residual:  {np.std(residuals):.4f}\n"
                f"  Max error:     {np.max(np.abs(residuals)):.4f}\n"
                f"  Median |err|:  {np.median(np.abs(residuals)):.4f}"
            )
            data = {
                "mean_residual": float(np.mean(residuals)),
                "std_residual": float(np.std(residuals)),
                "max_error": float(np.max(np.abs(residuals))),
                "median_abs_error": float(np.median(np.abs(residuals))),
            }

        return ToolResult(success=True, observation=obs, data=data)


class BuildEnsembleTool(Tool):
    """Build a stacking ensemble from all trained models."""

    name = "build_ensemble"
    description = "Build a stacking ensemble from all trained non-trivial models. Requires at least 2 base models."
    parameters_schema = {}

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        from co_scientist.modeling.ensemble import build_stacking_ensemble
        from co_scientist.evaluation.metrics import evaluate_model

        is_clf = state.eval_config.task_type in ("binary_classification", "multiclass_classification")
        task = "classification" if is_clf else "regression"

        try:
            ensemble_trained = build_stacking_ensemble(
                trained_models=state.trained_models,
                split=state.split,
                task_type=task,
                seed=state.seed,
            )
        except Exception as e:
            return ToolResult(success=False, observation=f"Error: ensemble build failed: {e}")

        if ensemble_trained is None:
            return ToolResult(
                success=False,
                observation="Ensemble build failed: need at least 2 non-trivial base models.",
            )

        result = evaluate_model(ensemble_trained, state.split, state.eval_config, use_test=False)
        state.trained_models.append(ensemble_trained)
        state.results.append(result)

        score = result.primary_metric_value
        obs = f"Stacking ensemble: {state.eval_config.primary_metric}={score:.4f}"
        console.print(f"    [green]{obs}[/green]")

        return ToolResult(
            success=True,
            observation=obs,
            data={"metrics": result.metrics},
            model_name=result.model_name,
            score=score,
        )


class InspectFeaturesTool(Tool):
    """Get top-N feature importances from a trained model."""

    name = "inspect_features"
    description = "Get top feature importances from a model (tree-based or linear). Returns feature names and importance scores."
    parameters_schema = {
        "model_name": {"type": "string", "required": False, "description": "Model to inspect (default: best)"},
        "top_n": {"type": "integer", "required": False, "description": "Number of top features (default: 10)"},
    }

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        target_name = params.get("model_name", "")
        top_n = params.get("top_n", 10)

        trained = None
        for t in state.trained_models:
            if t.config.name == target_name:
                trained = t
                break
        if trained is None and state.best_trained:
            trained = state.best_trained

        if trained is None:
            return ToolResult(success=False, observation="Error: no model to inspect")

        model = trained.model
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances is None:
            return ToolResult(
                success=False,
                observation=f"Model {trained.config.name} does not expose feature importances.",
            )

        feature_names = getattr(state.split, "feature_names", None) or [
            f"f{i}" for i in range(len(importances))
        ]
        if len(importances) != len(feature_names):
            feature_names = [f"f{i}" for i in range(len(importances))]

        indices = np.argsort(importances)[::-1][:top_n]
        lines = [f"Top {min(top_n, len(indices))} features for {trained.config.name}:"]
        features_data = []
        for rank, idx in enumerate(indices, 1):
            name = feature_names[idx]
            imp = importances[idx]
            lines.append(f"  {rank}. {name}: {imp:.4f}")
            features_data.append({"name": name, "importance": float(imp)})

        obs = "\n".join(lines)
        return ToolResult(success=True, observation=obs, data={"features": features_data})


class FinishTool(Tool):
    """Signal that the agent wants to stop modeling."""

    name = "finish"
    description = "Signal that you are done with the modeling phase. Call this when you believe no further improvement is likely."
    parameters_schema = {
        "reason": {"type": "string", "required": True, "description": "Why you are stopping"},
    }

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        reason = params.get("reason", "Agent decided to stop")
        return ToolResult(
            success=True,
            observation=f"Finished: {reason}",
            data={"stop_reason": reason},
        )


class DesignModelTool(Tool):
    """Design a custom PyTorch model architecture using the LLM, then train and evaluate it."""

    name = "design_model"
    description = (
        "Design a custom neural network architecture tailored to this dataset. "
        "The LLM generates PyTorch code based on your architecture hint, validates it, "
        "trains the model, and returns the validation score. Use this when standard models "
        "plateau and you want to try a novel architecture (e.g., residual networks, "
        "attention mechanisms, custom loss functions)."
    )
    parameters_schema = {
        "architecture_hint": {
            "type": "string",
            "required": True,
            "description": (
                "Description of desired architecture, e.g. 'ResNet-style tabular model with skip connections', "
                "'Attention-based model focusing on codon features', 'Multi-head network with separate pathway for sequence length'"
            ),
        },
        "hyperparameters": {
            "type": "object",
            "required": False,
            "description": "Optional HP overrides passed to the generated model (e.g. learning_rate, dropout)",
        },
    }

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        from co_scientist.modeling.custom_model import (
            CUSTOM_MODEL_PROMPT,
            extract_code_from_response,
            generate_model_name,
            load_custom_model,
        )
        from co_scientist.modeling.types import ModelConfig, TrainedModel
        from co_scientist.evaluation.metrics import evaluate_model

        architecture_hint = params.get("architecture_hint", "")
        if not architecture_hint:
            return ToolResult(success=False, observation="Error: architecture_hint is required")

        # Check LLM availability
        llm_client = getattr(state, "llm_client", None)
        if llm_client is None or not llm_client.available:
            return ToolResult(
                success=False,
                observation="Error: design_model requires LLM access but no client is available",
            )

        # Build the prompt
        is_clf = "classification" in state.eval_config.task_type
        task_family = "classification" if is_clf else "regression"
        n_classes = state.profile.num_classes if is_clf else 0

        prompt = CUSTOM_MODEL_PROMPT.format(
            task_type=task_family,
            n_features=state.split.X_train.shape[1],
            n_samples=state.split.X_train.shape[0],
            n_classes=n_classes,
            modality=state.profile.modality.value,
            dataset_name=getattr(state.profile, "dataset_path", "unknown"),
            primary_metric=state.eval_config.primary_metric,
            architecture_hint=architecture_hint,
        )

        # Call LLM to generate code
        response = llm_client.ask_text(
            system_prompt="You are an expert ML engineer who designs custom PyTorch model architectures.",
            user_message=prompt,
            agent_name="design_model",
            max_tokens=4096,
            temperature=0.3,
        )

        if response is None:
            return ToolResult(success=False, observation="Error: LLM failed to generate model code")

        # Extract code
        code = extract_code_from_response(response)
        if code is None:
            return ToolResult(
                success=False,
                observation=f"Error: could not extract code from LLM response. Response: {response[:200]}...",
            )

        # Load and validate
        model_name = generate_model_name(architecture_hint)
        hp = dict(params.get("hyperparameters", {}) or {})
        hp.setdefault("random_state", state.seed)

        try:
            model = load_custom_model(code, task_type=task_family, hyperparameters=hp)
        except (ValueError, TypeError) as e:
            return ToolResult(success=False, observation=f"Error: generated model is invalid: {e}")

        # Train
        try:
            start = time.time()
            model.fit(state.split.X_train, state.split.y_train)
            train_time = time.time() - start
        except Exception as e:
            return ToolResult(
                success=False,
                observation=f"Error: custom model training failed: {e}",
            )

        # Evaluate
        config = ModelConfig(
            name=model_name,
            tier="custom",
            model_type="custom",
            task_type=task_family,
            hyperparameters=hp,
        )
        trained = TrainedModel(config=config, model=model, train_time_seconds=train_time)
        # Store source code so the model can be saved/exported later
        trained.custom_model_code = code
        result = evaluate_model(trained, state.split, state.eval_config, use_test=False)

        state.trained_models.append(trained)
        state.results.append(result)

        score = result.primary_metric_value
        obs = (
            f"Designed and trained {model_name}: "
            f"{state.eval_config.primary_metric}={score:.4f} "
            f"({train_time:.1f}s)"
        )
        console.print(f"    [green]{obs}[/green]")

        return ToolResult(
            success=True,
            observation=obs,
            data={"metrics": result.metrics, "architecture_hint": architecture_hint},
            model_name=model_name,
            score=score,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of all available tools for the ReAct agent."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def describe_all(self) -> str:
        """Return descriptions of all tools for injection into the system prompt."""
        parts = []
        for tool in self.tools.values():
            parts.append(tool.describe())
        return "\n\n".join(parts)

    def tool_names(self) -> list[str]:
        return list(self.tools.keys())


class BacktrackTool(Tool):
    """Signal that the agent wants to backtrack to a previous state (tree search only)."""

    name = "backtrack"
    description = "Backtrack to a previous state in the experiment tree. Use when current approach is not improving."
    parameters_schema = {
        "reason": {"type": "string", "required": True, "description": "Why you want to backtrack"},
    }

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        reason = params.get("reason", "Agent wants to try a different approach")
        return ToolResult(
            success=True,
            observation=f"Backtracking: {reason}",
            data={"backtrack": True, "reason": reason},
        )


class GetRankingsTool(Tool):
    """Return the current Elo tournament rankings."""

    name = "get_rankings"
    description = "Get the Elo tournament rankings of all trained models. Shows relative model quality via pairwise matchups."
    parameters_schema = {}

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        elo_ranker = getattr(state, "elo_ranker", None)
        if elo_ranker is None:
            return ToolResult(success=True, observation="No tournament rankings available (Elo ranker not active).")

        table = elo_ranker.format_table()
        return ToolResult(
            success=True,
            observation=table,
            data=elo_ranker.to_dict(),
        )


class SummarizeDataTool(Tool):
    """Give the agent a detailed summary of the dataset to reason about modeling strategy."""

    name = "summarize_data"
    description = (
        "Get a detailed summary of the dataset characteristics to inform modeling decisions. "
        "Returns sample size, feature count, modality, class balance, feature statistics, "
        "and any detected issues. Use this at the START to plan your strategy."
    )
    parameters_schema = {}

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        profile = state.profile
        split = state.split

        lines = [
            f"Dataset: {profile.dataset_name}",
            f"Modality: {profile.modality.value}",
            f"Task: {profile.task_type.value}",
            f"Samples: {profile.num_samples:,} (train={split.X_train.shape[0]}, val={split.X_val.shape[0]}, test={split.X_test.shape[0]})",
            f"Features after preprocessing: {split.X_train.shape[1]}",
            f"Raw features: {profile.num_features}",
        ]

        # Feature statistics
        X = split.X_train
        lines.append(f"Feature sparsity: {(X == 0).mean() * 100:.1f}% zeros")
        lines.append(f"Feature value range: [{X.min():.4f}, {X.max():.4f}]")
        feature_variance = np.var(X, axis=0)
        lines.append(f"Feature variance: mean={feature_variance.mean():.4f}, max={feature_variance.max():.4f}, min={feature_variance.min():.6f}")
        n_low_var = int((feature_variance < 0.01).sum())
        if n_low_var > 0:
            lines.append(f"Low-variance features (<0.01): {n_low_var}/{X.shape[1]} — consider these may be noise")

        # Correlation
        if X.shape[1] <= 500:
            try:
                corr_matrix = np.corrcoef(X.T)
                upper_tri = np.triu(corr_matrix, k=1)
                high_corr = (np.abs(upper_tri) > 0.9).sum()
                lines.append(f"Highly correlated feature pairs (|r|>0.9): {high_corr}")
            except Exception:
                pass

        # Task-specific stats
        if profile.num_classes > 0:
            lines.append(f"Classes: {profile.num_classes}")
            if profile.class_distribution:
                sorted_classes = sorted(profile.class_distribution.items(), key=lambda x: x[1], reverse=True)
                lines.append(f"Largest class: {sorted_classes[0][0]} ({sorted_classes[0][1]} samples)")
                lines.append(f"Smallest class: {sorted_classes[-1][0]} ({sorted_classes[-1][1]} samples)")
                imbalance_ratio = sorted_classes[0][1] / max(sorted_classes[-1][1], 1)
                lines.append(f"Imbalance ratio: {imbalance_ratio:.1f}x")
        else:
            y = split.y_train
            lines.append(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
            lines.append(f"Target skewness: {float(np.mean(((y - y.mean()) / max(y.std(), 1e-8)) ** 3)):.2f}")

        if profile.detected_issues:
            lines.append(f"Detected issues: {', '.join(profile.detected_issues)}")

        if profile.modality.value in ("rna", "dna", "protein"):
            if profile.sequence_length_stats:
                lines.append(f"Sequence lengths: {profile.sequence_length_stats}")

        # Foundation model embeddings availability
        if split.X_embed_train is not None:
            lines.append(f"AIDO embeddings available: {split.X_embed_train.shape[1]} dims (GPU detected)")
            lines.append("  → You can train embed_xgboost, embed_mlp on these embeddings")
            lines.append("  → You can also try aido_finetune for end-to-end fine-tuning")
        else:
            from co_scientist.modeling.foundation import gpu_available
            if gpu_available():
                lines.append("GPU detected but no embeddings extracted yet")
            else:
                lines.append("No GPU — foundation models (embed_xgboost, embed_mlp, aido_finetune) not available")

        lines.append("")
        lines.append("Based on this data, think about:")
        lines.append("- Is the feature space sparse? → Tree models handle sparsity well")
        lines.append("- High feature correlation? → PCA-like compression or regularized models may help")
        lines.append("- Small sample size? → Avoid complex models, use regularization")
        lines.append("- Imbalanced classes? → Use class weights or focal loss")
        lines.append("- Sequence data? → Consider CNN for motif detection alongside k-mer features")

        return ToolResult(success=True, observation="\n".join(lines))


class ConsultBiologyTool(Tool):
    """Consult the Biology Specialist to validate results and get biological context."""

    name = "consult_biology"
    description = (
        "Consult the Biology Specialist agent to: validate whether model performance is "
        "biologically plausible, get domain-specific interpretation of results, check if "
        "the chosen metric is appropriate, and identify biological signals the model should capture. "
        "Use this after training a few models to get biological validation."
    )
    parameters_schema = {}

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        from co_scientist.agents.biology import BiologySpecialistAgent
        from co_scientist.agents.types import PipelineContext

        profile = state.profile
        best = state.best_result

        # Build context
        model_scores = {r.model_name: r.primary_metric_value for r in state.results}
        context = PipelineContext(
            dataset_path=getattr(profile, "dataset_path", profile.dataset_name),
            modality=profile.modality.value,
            task_type=profile.task_type.value,
            num_samples=profile.num_samples,
            num_features=profile.num_features,
            num_classes=profile.num_classes,
            stage="post_training",
            trained_model_names=[r.model_name for r in state.results],
            model_scores=model_scores,
            best_model_name=best.model_name if best else "",
            best_score=best.primary_metric_value if best else 0.0,
            primary_metric=state.eval_config.primary_metric,
        )

        bio_agent = BiologySpecialistAgent(client=getattr(state, "llm_client", None))
        decision = bio_agent.decide(context)

        # Format the response
        p = decision.parameters
        lines = [f"Biology Specialist Assessment:"]
        lines.append(f"  Plausibility: {p.get('plausibility', 'unknown')}")
        if p.get("plausibility_detail"):
            lines.append(f"  Detail: {p['plausibility_detail']}")
        if p.get("biological_context"):
            lines.append(f"  Context: {p['biological_context'][:300]}")
        if p.get("metric_note"):
            lines.append(f"  Metric: {p['metric_note']}")
        if p.get("suggested_features"):
            lines.append(f"  Suggested features: {', '.join(p['suggested_features'][:5])}")
        signals = p.get("biological_signals", [])
        if signals:
            lines.append(f"  Key signals: {'; '.join(signals[:3])}")

        obs = "\n".join(lines)
        console.print(f"    [magenta]{obs}[/magenta]")

        # Push to live dashboard
        if getattr(state, "dashboard", None):
            summary = p.get("plausibility_detail") or p.get("plausibility", decision.reasoning)
            state.dashboard.add_agent_message(
                agent="Biology Specialist",
                stage="react_loop",
                message=str(summary)[:300],
                msg_type="react_tool",
            )

        # Store latest biology assessment on state for report generation
        state.biology_assessment = p

        return ToolResult(success=True, observation=obs, data=p)


class DiagnoseDataTool(Tool):
    """Consult the Data Analyst to diagnose data issues that might explain poor model performance."""

    name = "diagnose_data"
    description = (
        "Consult the Data Analyst agent to diagnose potential data issues: "
        "class imbalance, feature quality, train-validation distribution shift, "
        "high-dimensional noise features, or data leakage. "
        "Use this when models are underperforming or showing unexpected behavior."
    )
    parameters_schema = {}

    def execute(self, params: dict[str, Any], state: "ReactState") -> ToolResult:
        profile = state.profile
        split = state.split
        best = state.best_result

        lines = ["Data Analyst Diagnosis:"]

        # 1. Class imbalance check
        if profile.num_classes > 0 and profile.class_distribution:
            sorted_classes = sorted(profile.class_distribution.values())
            ratio = sorted_classes[-1] / max(sorted_classes[0], 1)
            if ratio > 10:
                lines.append(f"  WARNING: Severe class imbalance — {ratio:.0f}:1 ratio. Consider class weights or oversampling.")
            elif ratio > 3:
                lines.append(f"  NOTE: Moderate class imbalance — {ratio:.1f}:1 ratio. Use class_weight='balanced'.")
            else:
                lines.append(f"  Class balance: OK ({ratio:.1f}:1 ratio)")

        # 2. Feature quality
        X_train = split.X_train
        feature_variance = np.var(X_train, axis=0)
        n_zero_var = int((feature_variance < 1e-10).sum())
        n_low_var = int((feature_variance < 0.01).sum())
        if n_zero_var > 0:
            lines.append(f"  WARNING: {n_zero_var} zero-variance features — these add noise with no signal.")
        if n_low_var > X_train.shape[1] * 0.3:
            lines.append(f"  WARNING: {n_low_var}/{X_train.shape[1]} low-variance features (<0.01) — consider feature selection.")
        else:
            lines.append(f"  Feature quality: {n_low_var} low-variance features out of {X_train.shape[1]}")

        # 3. Train-val distribution shift
        try:
            train_mean = X_train.mean(axis=0)
            val_mean = split.X_val.mean(axis=0)
            max_shift = np.max(np.abs(train_mean - val_mean))
            mean_shift = np.mean(np.abs(train_mean - val_mean))
            if max_shift > 2.0:
                lines.append(f"  WARNING: Distribution shift detected — max feature mean diff: {max_shift:.3f}")
            else:
                lines.append(f"  Train-val distribution: consistent (max shift: {max_shift:.3f})")
        except Exception:
            pass

        # 4. Train-val gap (overfitting) check
        if state.results:
            for r in state.results[-3:]:
                m = r.metrics
                # Check if train scores exist
                for key in ("train_accuracy", "train_spearman", "train_r2"):
                    if key in m:
                        val_key = key.replace("train_", "")
                        if val_key in m:
                            gap = abs(m[key] - m[val_key])
                            if gap > 0.3:
                                lines.append(f"  WARNING: {r.model_name} shows overfitting — train-val gap: {gap:.3f}")

        # 5. Sample size assessment
        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]
        if n_features > n_samples * 5:
            lines.append(f"  WARNING: High-dimensional data ({n_features} features, {n_samples} samples) — risk of overfitting. Use regularized models.")
        elif n_features > n_samples:
            lines.append(f"  NOTE: More features ({n_features}) than samples ({n_samples}) — prefer tree models or regularized linear models.")

        # 6. Performance diagnosis
        if best:
            metric = state.eval_config.primary_metric
            score = best.primary_metric_value
            if metric == "spearman" and score < 0.3:
                lines.append(f"  Diagnosis: {metric}={score:.4f} is low. Possible causes: (1) weak signal in features, (2) non-linear relationships not captured, (3) noisy labels.")
            elif metric in ("accuracy", "f1_macro") and score < 0.5:
                lines.append(f"  Diagnosis: {metric}={score:.4f} is low. Check: (1) class imbalance, (2) insufficient features, (3) label noise.")

        obs = "\n".join(lines)
        console.print(f"    [cyan]{obs}[/cyan]")

        # Push to live dashboard
        if getattr(state, "dashboard", None):
            # Pick the most informative line as summary
            diagnosis_lines = [l.strip() for l in lines[1:] if l.strip().startswith(("WARNING", "NOTE", "Diagnosis"))]
            summary = "; ".join(diagnosis_lines[:3]) if diagnosis_lines else "No issues found"
            state.dashboard.add_agent_message(
                agent="Data Analyst",
                stage="react_loop",
                message=summary[:300],
                msg_type="react_tool",
            )

        return ToolResult(success=True, observation=obs)


def build_default_registry() -> ToolRegistry:
    """Build the default tool registry with all tools."""
    registry = ToolRegistry()
    registry.register(SummarizeDataTool())
    registry.register(TrainModelTool())
    registry.register(TuneHyperparametersTool())
    registry.register(GetModelScoresTool())
    registry.register(AnalyzeErrorsTool())
    registry.register(BuildEnsembleTool())
    registry.register(InspectFeaturesTool())
    registry.register(GetRankingsTool())
    registry.register(ConsultBiologyTool())
    registry.register(DiagnoseDataTool())
    registry.register(DesignModelTool())
    registry.register(FinishTool())
    return registry


def build_tree_search_registry() -> ToolRegistry:
    """Build a tool registry that includes the backtrack tool for tree search mode."""
    registry = build_default_registry()
    registry.register(BacktrackTool())
    return registry
