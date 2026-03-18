"""ML Engineer agent — advises on model selection, HP tuning, and iteration strategy."""

from __future__ import annotations

from co_scientist.agents.base import BaseAgent
from co_scientist.agents.types import AgentRole, Decision, PipelineContext
from co_scientist.llm.prompts import ML_ENGINEER_SYSTEM


class MLEngineerAgent(BaseAgent):
    """Advises on model selection, hyperparameter tuning, and next iteration strategy.

    Deterministic fallback replicates the current rule-based model selection
    from defaults.yaml and the existing HP search logic.
    """

    role = AgentRole.ML_ENGINEER

    def system_prompt(self) -> str:
        return ML_ENGINEER_SYSTEM

    def decide_deterministic(self, context: PipelineContext) -> Decision:
        """Rule-based model selection and iteration strategy."""
        stage = context.stage

        if stage == "model_selection":
            return self._select_models(context)
        elif stage == "hp_search":
            return self._should_tune(context)
        elif stage == "iteration":
            return self._next_iteration(context)
        else:
            return Decision(
                action="no_op",
                parameters={},
                reasoning=f"No deterministic logic for stage: {stage}",
                confidence=0.5,
            )

    def _select_models(self, context: PipelineContext) -> Decision:
        """Select model candidates based on dataset characteristics."""
        modality = context.modality
        n_samples = context.num_samples
        task = context.task_type

        # Base models — always include tree ensembles
        models = ["xgboost", "lightgbm", "random_forest"]
        reasons = [
            "Tree ensembles (XGBoost, LightGBM, Random Forest) included as strong defaults — "
            "they handle non-linear interactions, require no feature scaling, and are robust to irrelevant features"
        ]

        # Simple models
        if "classification" in task:
            models.extend(["logistic_regression", "elastic_net_clf"])
            reasons.append(
                "Linear models (Logistic Regression, Elastic Net) added to test if classes are linearly separable — "
                "if they perform well, complex models may be unnecessary"
            )
        else:
            models.extend(["ridge_regression", "elastic_net_reg"])
            reasons.append(
                "Regularized linear models (Ridge, Elastic Net) added to test linearity of the signal — "
                "Ridge handles correlated features, Elastic Net adds feature selection via L1 penalty"
            )

        # Advanced models
        if n_samples > 300:
            models.append("mlp")
            reasons.append(
                f"MLP neural network included because {n_samples} samples is sufficient to train shallow networks — "
                "can discover non-linear feature combinations that tree models miss"
            )
        if modality in ("rna", "dna", "protein") and n_samples > 200:
            models.append("bio_cnn")
            reasons.append(
                f"1D CNN designed for {modality.upper()} sequences — convolutional filters detect local motifs "
                "that k-mer frequency features may under-represent"
            )

        # Attention-based models
        if n_samples > 800:
            models.append("ft_transformer")
            reasons.append(
                f"FT-Transformer included — applies self-attention across features, capturing complex "
                f"inter-feature dependencies that tree models handle greedily. "
                f"With {n_samples:,} samples, there is enough data to train transformer weights"
            )

        # KNN as a non-parametric baseline
        models.append("knn")
        reasons.append(
            "KNN added as a non-parametric baseline — if it performs well, the feature space has "
            "good local structure; if poorly, the signal requires more complex modeling"
        )

        # SVM for kernel-based learning
        models.append("svm")
        reasons.append(
            "SVM with RBF kernel included — can find non-linear decision boundaries in "
            "high-dimensional feature spaces via the kernel trick"
        )

        # Foundation models (GPU-gated)
        from co_scientist.modeling.foundation import gpu_available
        if gpu_available():
            models.extend(["embed_xgboost", "concat_xgboost", "concat_mlp", "aido_finetune"])
            reasons.append(
                "GPU detected — adding foundation models: embed_xgboost (AIDO embeddings + XGBoost), "
                "concat_xgboost/concat_mlp (handcrafted + AIDO embeddings — often best overall), "
                "aido_finetune (end-to-end AIDO backbone fine-tuning — strongest for sequence data)"
            )

        # Priority: tree models first for tabular, CNN for sequences
        if modality in ("rna", "dna", "protein"):
            priority = "bio_cnn"
            priority_reason = f"Bio-CNN prioritized for {modality.upper()} data — sequence-aware architecture may capture motifs better than hand-crafted features"
        else:
            priority = "xgboost"
            priority_reason = "XGBoost prioritized for tabular data — typically the strongest general-purpose model"

        reasons.append(f"Priority: {priority_reason}")

        return Decision(
            action="select_models",
            parameters={
                "models": models,
                "priority": priority,
                "selection_reasons": reasons,
            },
            reasoning=". ".join([
                f"Selected {len(models)} models for {modality}/{task} with {n_samples:,} samples",
                priority_reason,
            ]),
            confidence=0.9,
        )

    def _should_tune(self, context: PipelineContext) -> Decision:
        """Decide whether HP tuning is worthwhile."""
        best = context.best_score
        metric = context.primary_metric
        target = context.best_model_name
        n_trials = 20 if context.complexity_level in ("simple", "moderate") else 30

        # Don't tune if already excellent
        if metric in ("mse", "rmse", "mae"):
            should_tune = True
            reason = (
                f"Current best: {target} with {metric}={best:.4f}. "
                f"Tuning with {n_trials} Optuna trials to search for better hyperparameters — "
                f"for regression tasks, there is almost always room to reduce error through HP optimization"
            )
        elif best >= 0.95:
            should_tune = False
            reason = (
                f"Current best: {target} with {metric}={best:.4f}. "
                f"Score is already near-perfect (≥0.95) — HP tuning unlikely to yield meaningful improvement "
                f"and risks overfitting to the validation set"
            )
        else:
            should_tune = True
            gap_to_perfect = 1.0 - best
            reason = (
                f"Current best: {target} with {metric}={best:.4f} (gap to perfect: {gap_to_perfect:.4f}). "
                f"Running {n_trials} Optuna trials on {target} — default hyperparameters are rarely optimal, "
                f"and tuning typically improves scores by 1-5%"
            )

        return Decision(
            action="hp_tune" if should_tune else "skip_hp_search",
            parameters={
                "target_model": target,
                "n_trials": n_trials,
            },
            reasoning=reason,
            confidence=0.85 if should_tune else 0.9,
        )

    def _next_iteration(self, context: PipelineContext) -> Decision:
        """Decide what to try next in the iteration loop.

        Strategy priority:
        1. HP-tune best model (if not already done)
        2. HP-tune second-best model (if different type from best)
        3. Try untrained model types
        4. Rebuild ensemble (if new models have been added)
        5. Stop
        """
        scores = context.model_scores
        best_name = context.best_model_name
        best_score = context.best_score
        iteration = context.iteration
        decisions = context.decisions_so_far

        # Check for stagnation — 3 consecutive no-improvement strategies
        if len(decisions) >= 3:
            recent = decisions[-3:]
            if all(d in ("hp_tune", "try_model") for d in recent):
                return Decision(
                    action="stop",
                    parameters={"reason": "stagnation"},
                    reasoning="3 consecutive attempts without clear direction — diminishing returns",
                    confidence=0.8,
                )

        # Find which model types have been HP-tuned (look for "tuned" in names)
        tuned_types = set()
        for name in scores:
            if "tuned" in name.lower():
                # Extract base type: "xgboost_tuned" → "xgboost"
                base = name.lower().replace("_tuned", "").replace("tuned_", "")
                tuned_types.add(base)

        # Also track what's been tried in this iteration loop
        hp_tuned_in_loop = sum(1 for d in decisions if d == "hp_tune")

        # Strategy 1: HP-tune best model if not yet tuned (and tunable)
        _NON_TUNABLE_BEST = {"stacking", "mean_predictor", "majority_class"}
        best_type = self._extract_model_type(best_name)
        if best_type not in tuned_types and best_type not in _NON_TUNABLE_BEST and hp_tuned_in_loop == 0:
            return Decision(
                action="hp_tune",
                parameters={
                    "target_model": best_name,
                    "n_trials": 15,
                },
                reasoning=f"HP-tune best model ({best_name}) — first priority",
                confidence=0.85,
            )

        # Strategy 2: HP-tune second-best model (if different type, not ensemble)
        _NON_TUNABLE = {"stacking", "mean_predictor", "majority_class"}
        if len(scores) >= 2:
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_models[1:3]:
                model_type = self._extract_model_type(name)
                if (model_type != best_type
                        and model_type not in tuned_types
                        and model_type not in _NON_TUNABLE
                        and hp_tuned_in_loop < 2):
                    return Decision(
                        action="hp_tune",
                        parameters={
                            "target_model": name,
                            "n_trials": 10,
                        },
                        reasoning=f"HP-tune runner-up ({name}, {score:.4f}) — different model type from best",
                        confidence=0.7,
                    )

        # Strategy 3: Try untrained model types
        trained_types = {self._extract_model_type(n) for n in scores}
        candidates = self._suggest_untrained_models(context, trained_types)
        if candidates:
            return Decision(
                action="try_model",
                parameters={
                    "target_model": candidates[0],
                    "model_type": candidates[0],
                },
                reasoning=f"Try untrained model type: {candidates[0]}",
                confidence=0.65,
            )

        # Strategy 4: Rebuild ensemble if we have new models since last ensemble
        ensemble_names = [n for n in scores if "ensemble" in n.lower() or "stacking" in n.lower()]
        n_non_ensemble = len(scores) - len(ensemble_names)
        if n_non_ensemble >= 3 and (
            not ensemble_names  # no ensemble yet
            or "try_ensemble" not in decisions  # haven't tried in this loop
        ):
            return Decision(
                action="try_ensemble",
                parameters={"type": "stacking"},
                reasoning=f"Rebuild ensemble with {n_non_ensemble} base models",
                confidence=0.6,
            )

        # Default: stop
        return Decision(
            action="stop",
            parameters={"reason": "no_clear_improvement_path"},
            reasoning=f"Best model {best_name}={best_score:.4f}, all strategies exhausted",
            confidence=0.7,
        )

    def _extract_model_type(self, model_name: str) -> str:
        """Extract base model type from a model name."""
        name = model_name.lower()
        # Strip common suffixes
        for suffix in ("_default", "_tuned", "_iter1", "_iter2", "_iter3", "_iter4", "_iter5"):
            name = name.replace(suffix, "")
        # Map common names to types
        _TYPE_MAP = {
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "random_forest": "random_forest",
            "ridge": "ridge",
            "logistic": "logistic",
            "elastic": "elastic_net",
            "mlp": "mlp",
            "bio_cnn": "bio_cnn",
            "stacking": "stacking",
            "ensemble": "stacking",
        }
        for key, typ in _TYPE_MAP.items():
            if key in name:
                return typ
        return name

    def _suggest_untrained_models(self, context: PipelineContext, trained_types: set[str]) -> list[str]:
        """Suggest model types that haven't been tried yet."""
        candidates = []
        modality = context.modality
        n_samples = context.num_samples
        task = context.task_type

        # All possible models
        all_models = ["xgboost", "lightgbm", "random_forest"]
        if "classification" in task:
            all_models.extend(["logistic_regression", "elastic_net_clf"])
        else:
            all_models.extend(["ridge_regression", "elastic_net_reg"])
        if n_samples > 300:
            all_models.append("mlp")
        if modality in ("rna", "dna", "protein") and n_samples > 200:
            all_models.append("bio_cnn")

        # Foundation models (GPU-gated)
        from co_scientist.modeling.foundation import gpu_available
        if gpu_available():
            all_models.extend(["embed_xgboost", "concat_xgboost", "concat_mlp", "aido_finetune"])

        for m in all_models:
            base_type = self._extract_model_type(m)
            if base_type not in trained_types:
                candidates.append(m)

        return candidates

    def diagnose_failure(self, context: PipelineContext, error: str) -> Decision:
        """Diagnose a training failure and suggest recovery."""
        error_lower = error.lower()

        if "memory" in error_lower or "oom" in error_lower:
            return Decision(
                action="retry_with_fallback",
                parameters={
                    "reduce": ["batch_size", "n_estimators", "max_depth"],
                    "strategy": "reduce_memory",
                },
                reasoning="OOM error — reduce model size",
                confidence=0.8,
            )
        elif "convergence" in error_lower or "nan" in error_lower:
            return Decision(
                action="retry_with_fallback",
                parameters={
                    "reduce": ["learning_rate"],
                    "increase": ["regularization"],
                    "strategy": "stabilize",
                },
                reasoning="Convergence issue — reduce LR, increase regularization",
                confidence=0.7,
            )
        else:
            return Decision(
                action="skip_model",
                parameters={"reason": error[:200]},
                reasoning=f"Unknown error, skip this model: {error[:100]}",
                confidence=0.5,
                fallback="try_next_model",
            )
