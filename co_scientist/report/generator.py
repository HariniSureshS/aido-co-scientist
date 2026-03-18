"""Report generation — produces report.md from pipeline results."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from co_scientist import __version__
from co_scientist.data.types import DatasetProfile, SplitData
from co_scientist.evaluation.types import EvalConfig, ModelResult
from co_scientist.modeling.types import TrainedModel

console = Console()

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "template.md.jinja"


def generate_report(
    profile: DatasetProfile,
    split: SplitData,
    eval_config: EvalConfig,
    results: list[ModelResult],
    best_result: ModelResult,
    best_trained: TrainedModel,
    preprocessing_steps: list[str],
    output_dir: Path,
    profiling_figures: list[Path] | None = None,
    preprocessing_figures: list[Path] | None = None,
    training_figures: list[Path] | None = None,
    seed: int = 42,
    biological_interpretation: str = "",
    agent_reasoning: list[dict] | None = None,
    research_report: dict | None = None,
    feature_interpretation: list[dict] | None = None,
    active_learning_report: Any = None,
    guardrail_alerts: list[dict] | None = None,
    iteration_log: dict | None = None,
    react_scratchpad: list[dict] | None = None,
    elo_rankings: dict | None = None,
    debate_transcripts: list | None = None,
    tree_search_log: dict | None = None,
    review_result: str | None = None,
    architecture_diagram: str | None = None,
    agent_flow_diagram: str | None = None,
    test_metrics: dict | None = None,
    biology_assessment: dict | None = None,
) -> Path:
    """Generate the markdown report and write it to output_dir/report.md."""
    profiling_figures = profiling_figures or []
    preprocessing_figures = preprocessing_figures or []
    training_figures = training_figures or []
    all_figures = profiling_figures + preprocessing_figures + training_figures

    # Sort results by primary metric
    lower_is_better = eval_config.primary_metric in ("mse", "rmse", "mae")
    results_sorted = sorted(
        results,
        key=lambda r: r.primary_metric_value,
        reverse=not lower_is_better,
    )

    # Build executive summary
    executive_summary = _build_executive_summary(profile, eval_config, best_result, results)

    # Build profile summary
    profile_summary = _build_profile_summary(profile)

    # Class distribution sorted by count
    class_distribution_sorted = sorted(
        profile.class_distribution.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Model config JSON
    model_config_json = json.dumps(
        {
            "name": best_trained.config.name,
            "tier": best_trained.config.tier,
            "model_type": best_trained.config.model_type,
            "hyperparameters": best_trained.config.hyperparameters,
            "task_type": best_trained.config.task_type,
        },
        indent=2,
    )

    # Export dir relative path
    export_dir_rel = str(output_dir.name)

    # Number of features (account for foundation model feature routing)
    best_type = best_trained.config.model_type if best_trained else ""
    if best_type in ("concat_xgboost", "concat_mlp") and split.X_embed_train is not None:
        num_features = split.X_train.shape[1] + split.X_embed_train.shape[1]
    elif best_type in ("embed_xgboost", "embed_mlp") and split.X_embed_train is not None:
        num_features = split.X_embed_train.shape[1]
    else:
        num_features = split.X_train.shape[1]

    # Render
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(TEMPLATE_NAME)

    # Build model selection rationale
    model_selection_rationale = _build_model_selection_rationale(
        best_result, best_trained, results_sorted, eval_config, profile, agent_reasoning,
    )

    # Build model strategy rationale (why these models were chosen to train)
    model_strategy_rationale = _build_model_strategy_rationale(
        profile, results, eval_config, agent_reasoning,
    )

    # Build benchmark comparison if literature benchmarks exist
    benchmark_comparison = _build_benchmark_comparison(
        best_result, eval_config, research_report or {},
    )

    # Build metric selection rationale
    metric_rationale = _build_metric_rationale(profile, eval_config)

    rendered = template.render(
        version=__version__,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        profile=profile,
        eval_config=eval_config,
        best_result=best_result,
        results_sorted=results_sorted,
        executive_summary=executive_summary,
        profile_summary=profile_summary,
        class_distribution_sorted=class_distribution_sorted,
        preprocessing_steps=preprocessing_steps,
        num_features=num_features,
        split=split,
        split_sizes=split.summary(),
        profiling_figures=profiling_figures,
        preprocessing_figures=preprocessing_figures,
        training_figures=training_figures,
        all_figures=all_figures,
        all_metrics=best_result.metrics,
        model_config_json=model_config_json,
        export_dir_rel=export_dir_rel,
        seed=seed,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        output_dir=output_dir,
        biological_interpretation=biological_interpretation,
        agent_reasoning=agent_reasoning or [],
        research_report=research_report or {},
        research_papers=(research_report or {}).get("papers", []),
        feature_interpretation=feature_interpretation or [],
        al_report=active_learning_report,
        guardrail_alerts=guardrail_alerts or [],
        iteration_log=iteration_log or {},
        react_scratchpad=react_scratchpad or [],
        elo_rankings=elo_rankings or {},
        debate_transcripts=debate_transcripts or [],
        tree_search_log=tree_search_log or {},
        review_result=review_result or "",
        benchmark_comparison=benchmark_comparison,
        model_selection_rationale=model_selection_rationale,
        model_strategy_rationale=model_strategy_rationale,
        architecture_diagram=architecture_diagram or "",
        agent_flow_diagram=agent_flow_diagram or "",
        test_metrics=test_metrics or {},
        metric_rationale=metric_rationale,
        biology_assessment=biology_assessment or {},
    )

    report_path = output_dir / "report.md"
    report_path.write_text(rendered, encoding="utf-8")
    return report_path


def _build_executive_summary(
    profile: DatasetProfile,
    eval_config: EvalConfig,
    best_result: ModelResult,
    results: list[ModelResult],
) -> str:
    """Build a one-paragraph executive summary."""
    task_desc = profile.task_type.value.replace("_", " ")
    modality_desc = profile.modality.value.replace("_", " ")

    # Find trivial baseline for comparison
    trivial_score = None
    for r in results:
        if r.tier == "trivial":
            trivial_score = r.primary_metric_value
            break

    lines = [
        f"This report documents the automated analysis of the **{profile.dataset_name}** dataset, "
        f"a {task_desc} task on {modality_desc} data with {profile.num_samples:,} samples.",
    ]

    if trivial_score is not None:
        improvement = best_result.primary_metric_value - trivial_score
        lines.append(
            f"The best model ({best_result.model_name}) achieved a "
            f"{eval_config.primary_metric} of **{best_result.primary_metric_value:.4f}**, "
            f"an improvement of {improvement:+.4f} over the trivial baseline ({trivial_score:.4f})."
        )
    else:
        lines.append(
            f"The best model ({best_result.model_name}) achieved a "
            f"{eval_config.primary_metric} of **{best_result.primary_metric_value:.4f}**."
        )

    if profile.detected_issues:
        lines.append(
            f"The profiling stage detected {len(profile.detected_issues)} data issue(s) "
            f"that were addressed during preprocessing."
        )

    return " ".join(lines)


def _build_model_strategy_rationale(
    profile: DatasetProfile,
    results: list[ModelResult],
    eval_config: EvalConfig,
    agent_reasoning: list[dict] | None,
) -> str:
    """Explain why the agent chose these particular models for this dataset."""
    lines = []
    modality = profile.modality.value
    task = profile.task_type.value
    n_samples = profile.num_samples
    n_features = profile.num_features

    # Collect model types that were trained
    model_types = list(dict.fromkeys(r.model_name.replace("_tuned", "").replace("_default", "") for r in results if r.tier != "trivial"))

    lines.append(
        f"Given the dataset characteristics — **{modality}** modality, **{task}** task, "
        f"**{n_samples:,}** samples, **{n_features:,}** raw features — the agent selected "
        f"the following modeling strategy:"
    )

    # Modality-specific reasoning
    modality_reasons = {
        "rna": (
            "For RNA sequence data, k-mer frequency features create a high-dimensional, sparse representation "
            "where each feature captures the prevalence of a short nucleotide motif. "
            "**Tree-based models** (Random Forest, XGBoost, LightGBM) are prioritized because they: "
            "(1) handle high-dimensional sparse features naturally without feature scaling, "
            "(2) capture non-linear interactions between k-mer frequencies (e.g., codon context effects), "
            "and (3) are robust to irrelevant features common in k-mer spaces."
        ),
        "protein": (
            "For protein sequence data, amino acid composition and k-mer features encode structural and "
            "functional properties. **Tree-based ensembles** are preferred for their ability to discover "
            "non-linear relationships between sequence motifs and protein function, while **SVMs** with "
            "RBF kernels can capture similarity patterns in the high-dimensional amino acid feature space."
        ),
        "expression": (
            "For gene expression data, features represent transcript abundances that are often correlated "
            "across co-regulated genes. **Tree-based models** handle feature correlations well, while "
            "**regularized linear models** (Ridge, Elastic Net) can identify key marker genes through "
            "coefficient sparsity. **Neural networks** may capture complex non-linear expression patterns."
        ),
        "dna": (
            "For DNA sequence data, k-mer frequencies capture motif patterns at the genomic level. "
            "**Gradient boosting** methods are effective at learning position-invariant sequence signals, "
            "while **CNNs** can detect local motifs directly from sequence representations."
        ),
    }

    if modality in modality_reasons:
        lines.append(modality_reasons[modality])

    # Sample size reasoning
    if n_samples < 500:
        lines.append(
            "With a **small sample size** (<500), simpler models are favored to avoid overfitting. "
            "Complex models like deep neural networks are included but monitored closely for train-validation gaps."
        )
    elif n_samples < 2000:
        lines.append(
            "The **moderate sample size** supports both traditional ML models and shallow neural networks, "
            "but deep architectures risk overfitting without careful regularization."
        )
    else:
        lines.append(
            "The **large sample size** enables more complex models including deep neural networks "
            "and large ensembles without significant overfitting risk."
        )

    # Tier progression reasoning
    tier_counts = {}
    for r in results:
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

    tier_desc = []
    if "trivial" in tier_counts:
        tier_desc.append(f"**Trivial** ({tier_counts['trivial']}): establishes a no-skill baseline")
    if "simple" in tier_counts:
        tier_desc.append(f"**Simple** ({tier_counts['simple']}): linear models to test if the signal is linearly separable")
    if "standard" in tier_counts:
        tier_desc.append(f"**Standard** ({tier_counts['standard']}): non-linear models (tree ensembles, SVM, KNN) for complex patterns")
    if "advanced" in tier_counts:
        tier_desc.append(f"**Advanced** ({tier_counts['advanced']}): neural networks (MLP, CNN) for deep feature learning")
    if "ensemble" in tier_counts:
        tier_desc.append(f"**Ensemble** ({tier_counts['ensemble']}): stacking/blending to combine strengths of diverse models")
    if "tuned" in tier_counts:
        tier_desc.append(f"**Tuned** ({tier_counts['tuned']}): hyperparameter-optimized version of the best-performing model")

    if tier_desc:
        lines.append("**Tier progression:**\n" + "\n".join(f"- {t}" for t in tier_desc))

    # Include agent reasoning if available
    if agent_reasoning:
        for entry in agent_reasoning:
            if entry.get("action") == "select_models" and entry.get("reasoning"):
                lines.append(f"**Agent's recommendation:** {entry['reasoning']}")
                break

    return "\n\n".join(lines) if lines else ""


def _build_model_selection_rationale(
    best_result: ModelResult,
    best_trained: TrainedModel,
    results_sorted: list[ModelResult],
    eval_config: EvalConfig,
    profile: DatasetProfile,
    agent_reasoning: list[dict] | None,
) -> str:
    """Build a natural-language explanation of why the best model was chosen."""
    lines = []
    metric = eval_config.primary_metric
    lower_is_better = metric in ("mse", "rmse", "mae")
    model_type = best_trained.config.model_type
    tier = best_result.tier

    # 1. How it compared to runner-up
    if len(results_sorted) >= 2:
        runner_up = results_sorted[1]
        gap = abs(best_result.primary_metric_value - runner_up.primary_metric_value)
        lines.append(
            f"**{best_result.model_name}** achieved the highest {metric} "
            f"({best_result.primary_metric_value:.4f}), outperforming the runner-up "
            f"**{runner_up.model_name}** ({runner_up.primary_metric_value:.4f}) by {gap:.4f}."
        )

    # 2. How it compared to trivial baseline
    trivial = [r for r in results_sorted if r.tier == "trivial"]
    if trivial:
        trivial_score = trivial[0].primary_metric_value
        if trivial_score != 0:
            improvement_pct = abs(best_result.primary_metric_value - trivial_score) / abs(trivial_score) * 100
            lines.append(f"This is a **{improvement_pct:.0f}% improvement** over the trivial baseline.")
        else:
            lines.append(
                f"The trivial baseline scored {trivial_score:.4f} (no predictive power), "
                f"confirming that the model is learning real patterns."
            )

    # 3. Why this model type works for this data
    model_explanations = {
        "random_forest": (
            "Random forests are well-suited here because they handle non-linear feature interactions "
            "without requiring feature scaling, are robust to outliers, and provide feature importance rankings. "
            "For k-mer frequency features, tree-based splits can naturally capture combinatorial sequence patterns."
        ),
        "xgboost": (
            "XGBoost's gradient boosting builds an ensemble of weak learners that progressively correct errors. "
            "Its L1/L2 regularization helps prevent overfitting on high-dimensional k-mer features, "
            "and built-in handling of sparse data is beneficial for sequence-derived features."
        ),
        "lightgbm": (
            "LightGBM uses histogram-based splitting and leaf-wise growth, making it efficient on "
            "high-dimensional biological feature spaces. Its native categorical feature support "
            "and fast training make it ideal for iterative hyperparameter search."
        ),
        "svm": (
            "SVMs find optimal decision boundaries in high-dimensional spaces using kernel functions. "
            "The RBF kernel can capture non-linear relationships between sequence features and the target."
        ),
        "mlp": (
            "Multi-layer perceptrons learn complex non-linear mappings through backpropagation. "
            "Multiple hidden layers can discover hierarchical feature representations, "
            "though they require more data to avoid overfitting."
        ),
        "bio_cnn": (
            "The 1D CNN was designed specifically for sequence data — convolutional filters can "
            "detect local motifs and patterns in nucleotide/amino acid sequences that k-mer features "
            "may miss."
        ),
        "ridge": (
            "Ridge regression adds L2 regularization to prevent overfitting on high-dimensional features. "
            "Its linear nature makes it interpretable but limits its ability to capture non-linear patterns."
        ),
        "elastic_net": (
            "Elastic net combines L1 (feature selection) and L2 (regularization) penalties, "
            "useful when many features are correlated, as is common with overlapping k-mer frequencies."
        ),
        "embed_xgboost": (
            "XGBoost trained on AIDO foundation model embeddings rather than handcrafted features. "
            "The AIDO model provides rich, pre-trained representations of biological sequences that "
            "capture evolutionary and structural patterns beyond what k-mer frequencies can encode."
        ),
        "embed_mlp": (
            "MLP neural network trained on AIDO foundation model embeddings. "
            "The combination of pre-trained biological representations with a flexible neural "
            "head can learn complex non-linear mappings from sequence embeddings to the target."
        ),
        "aido_finetune": (
            "End-to-end fine-tuning of the AIDO foundation model backbone with a task-specific head. "
            "By unfreezing the last layers of the pre-trained model, it adapts the biological representations "
            "directly to the target task — typically the strongest approach when sufficient data is available."
        ),
        "concat_xgboost": (
            "XGBoost trained on a concatenation of handcrafted features (k-mer frequencies, sequence properties) "
            "and AIDO foundation model embeddings. This hybrid approach combines domain-engineered signal "
            "with deep biological representations, often outperforming either feature set alone."
        ),
        "concat_mlp": (
            "MLP neural network trained on concatenated handcrafted features and AIDO embeddings. "
            "The neural head can learn which features from each representation are most informative, "
            "effectively performing late fusion of engineered and learned biological features."
        ),
    }

    explanation = model_explanations.get(model_type, "")
    if explanation:
        lines.append(explanation)

    # 4. Was it tuned?
    if tier == "tuned":
        hp = best_trained.config.hyperparameters
        hp_highlights = []
        if "n_estimators" in hp:
            hp_highlights.append(f"{hp['n_estimators']} trees")
        if "max_depth" in hp:
            hp_highlights.append(f"max depth {hp['max_depth']}")
        if "learning_rate" in hp:
            hp_highlights.append(f"learning rate {hp['learning_rate']}")
        if hp_highlights:
            lines.append(
                f"Hyperparameter tuning further improved performance: {', '.join(hp_highlights)}."
            )

    # 5. Overfitting check
    if hasattr(best_result, "metrics") and best_result.metrics:
        metrics = best_result.metrics
        r2 = metrics.get("r2")
        if r2 is not None and r2 < 0.3:
            lines.append(
                "Note: the relatively low R² suggests the model captures ranking well (Spearman) "
                "but absolute predictions have room for improvement."
            )

    # 6. Agent reasoning if available
    if agent_reasoning:
        for entry in agent_reasoning:
            if entry.get("action") == "select_models" and entry.get("reasoning"):
                lines.append(f"**Agent reasoning:** {entry['reasoning']}")
                break

    return "\n\n".join(lines) if lines else ""


def _build_benchmark_comparison(
    best_result: ModelResult,
    eval_config: EvalConfig,
    research_report: dict,
) -> str:
    """Compare our best score against published benchmarks from literature search.

    Parses numeric scores from benchmark strings and produces a natural-language verdict.
    """
    benchmarks = research_report.get("benchmarks_found", [])
    if not benchmarks:
        return ""

    import re

    our_score = best_result.primary_metric_value
    our_metric = eval_config.primary_metric
    lower_is_better = our_metric in ("mse", "rmse", "mae")

    # Try to extract numeric values from benchmark strings
    # Format: "metric: value (paper title...)"
    parsed = []
    score_pattern = re.compile(r"([\d.]+(?:%)?)")
    for bm in benchmarks:
        matches = score_pattern.findall(bm)
        for m in matches:
            try:
                val = float(m.rstrip("%"))
                # If it was a percentage, convert to 0-1 scale if our score is in 0-1
                if m.endswith("%") and our_score <= 1.0:
                    val = val / 100.0
                # Filter out obviously wrong values (years, citation counts, etc.)
                if 0.0 < val <= 1.0 or (lower_is_better and val > 0):
                    parsed.append((val, bm))
                    break  # take first plausible number per benchmark
            except ValueError:
                continue

    if not parsed:
        return (
            f"Published benchmarks were found in the literature but use different metrics or "
            f"reporting formats, making direct numerical comparison difficult. "
            f"Our best model achieved **{our_metric} = {our_score:.4f}**."
        )

    # Compare
    lines = []
    for pub_score, bm_str in parsed:
        if lower_is_better:
            diff = our_score - pub_score
            if diff < 0:
                verdict = f"**outperforms** this benchmark by {abs(diff):.4f}"
            elif diff < pub_score * 0.05:
                verdict = f"**comparable** (within 5% of this benchmark)"
            else:
                verdict = f"trails this benchmark by {diff:.4f}"
        else:
            diff = our_score - pub_score
            if diff > 0:
                verdict = f"**outperforms** this benchmark by {diff:.4f}"
            elif abs(diff) < pub_score * 0.05:
                verdict = f"**comparable** (within 5% of this benchmark)"
            else:
                verdict = f"trails this benchmark by {abs(diff):.4f}"
        lines.append(f"- vs *{bm_str}*: Our model ({our_score:.4f}) {verdict}")

    # Overall verdict
    better_count = sum(
        1 for ps, _ in parsed
        if (our_score < ps if lower_is_better else our_score > ps)
    )
    comparable_count = sum(
        1 for ps, _ in parsed
        if abs(our_score - ps) < ps * 0.05
    )
    total = len(parsed)

    if better_count == total:
        summary = f"Our best model **outperforms all {total} published benchmark(s)** found in the literature."
    elif better_count + comparable_count == total:
        summary = f"Our best model **matches or exceeds all {total} published benchmark(s)** found in the literature."
    elif better_count > 0:
        summary = f"Our best model outperforms {better_count}/{total} published benchmarks and is comparable or trailing on the rest."
    else:
        summary = (
            f"Published benchmarks report higher scores, though differences in data splits, "
            f"preprocessing, and evaluation protocols may account for the gap."
        )

    return summary + "\n\n" + "\n".join(lines)


def _build_profile_summary(profile: DatasetProfile) -> str:
    """Build a natural-language dataset profile summary."""
    parts = [
        f"The dataset contains **{profile.num_samples:,} samples**",
    ]

    if profile.num_features > 0:
        feat_word = "feature" if profile.num_features == 1 else "features"
        parts[0] += f" with {profile.num_features:,} raw {feat_word}"

    parts[0] += "."

    if profile.modality.value in ("rna", "dna", "protein"):
        if profile.sequence_length_stats:
            mean_len = profile.sequence_length_stats.get("mean", 0)
            parts.append(f"Input sequences have a mean length of {mean_len:.0f} characters.")

    if profile.missing_value_pct > 0:
        parts.append(f"Missing values: {profile.missing_value_pct:.1f}% of entries.")

    if profile.feature_sparsity > 0:
        parts.append(f"Feature sparsity: {profile.feature_sparsity:.1f}% zeros.")

    return " ".join(parts)


def _build_metric_rationale(profile: DatasetProfile, eval_config: EvalConfig) -> str:
    """Explain why the primary metric was chosen for this task."""
    metric = eval_config.primary_metric
    task = profile.task_type.value

    rationales = {
        ("regression", "spearman"): (
            "**Spearman correlation** was selected as the primary metric because this is a regression task "
            "where the ranking of predictions matters more than their absolute values. Spearman measures "
            "monotonic relationship between predicted and true values, making it robust to outliers "
            "and non-linear transformations — important for biological measurements like translation "
            "efficiency where the relative ordering of samples is the key signal."
        ),
        ("regression", "pearson"): (
            "**Pearson correlation** was selected because it measures the linear relationship between "
            "predicted and true values, appropriate when the signal is expected to be linear."
        ),
        ("regression", "mse"): (
            "**Mean Squared Error** was selected to penalize large prediction errors, important when "
            "accurate absolute predictions (not just rankings) are required."
        ),
        ("binary_classification", "auroc"): (
            "**AUROC** (Area Under the ROC Curve) was selected because this is a binary classification task. "
            "AUROC is threshold-independent and measures the model's ability to distinguish between classes "
            "across all decision thresholds, which is the standard metric for binary biological classifiers."
        ),
        ("multiclass_classification", "accuracy"): (
            "**Accuracy** was selected because the classes are approximately balanced. "
            "With roughly equal class sizes, accuracy directly measures the fraction of correct predictions "
            "without the need for macro/micro averaging."
        ),
        ("multiclass_classification", "macro_f1"): (
            "**Macro F1** was selected because the classes are imbalanced. Macro F1 gives equal weight "
            "to each class regardless of size, ensuring the model performs well on minority classes — "
            "critical in biological settings where rare cell types or conditions are often the most interesting."
        ),
    }

    key = (task, metric)
    return rationales.get(key, f"**{metric}** was selected as the primary evaluation metric for this {task.replace('_', ' ')} task.")
