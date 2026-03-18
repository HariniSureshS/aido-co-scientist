"""Active learning analysis — identifies what additional data would help most.

Three types of analysis:
1. Class-level data need (classification): which classes bottleneck performance?
2. Uncertainty-based prioritization: which samples is the model most uncertain about?
3. Residual analysis (regression): where does the model fail?

Plus a feature gap analysis via the Biology Specialist agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassAnalysis:
    """Per-class performance analysis for classification."""

    class_name: str
    support: int  # number of samples
    f1: float
    precision: float
    recall: float
    most_confused_with: str = ""  # class most often confused with
    confusion_count: int = 0


@dataclass
class UncertainSample:
    """A sample the model is uncertain about."""

    index: int
    entropy: float
    predicted_label: str
    true_label: str
    top_proba: float  # highest class probability


@dataclass
class ResidualBin:
    """Residual analysis for a target value range."""

    bin_label: str
    mean_target: float
    mean_prediction: float
    mean_abs_error: float
    count: int


@dataclass
class ActiveLearningReport:
    """Full active learning analysis output."""

    # Classification
    class_analyses: list[ClassAnalysis] = field(default_factory=list)
    bottleneck_classes: list[str] = field(default_factory=list)
    uncertain_samples: list[UncertainSample] = field(default_factory=list)

    # Regression
    residual_bins: list[ResidualBin] = field(default_factory=list)
    worst_predicted_range: str = ""

    # Feature gap (filled by biology specialist)
    feature_gap_suggestions: list[str] = field(default_factory=list)

    # Summary
    summary: str = ""


def run_active_learning_analysis(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    task_type: str,
    label_encoder=None,
    sequences: list[str] | None = None,
    top_uncertain: int = 20,
) -> ActiveLearningReport:
    """Run active learning analysis on the test set.

    Args:
        model: TrainedModel instance
        X: Feature matrix (test set)
        y_true: True labels (test set)
        task_type: "classification" or "regression" (or full task type string)
        label_encoder: LabelEncoder for class names (optional)
        sequences: Raw sequences for CNN models
        top_uncertain: Number of most uncertain samples to report
    """
    report = ActiveLearningReport()
    is_classification = "classification" in task_type

    try:
        if is_classification:
            _analyze_classification(report, model, X, y_true, label_encoder, sequences, top_uncertain)
        else:
            _analyze_regression(report, model, X, y_true, sequences)

        report.summary = _build_summary(report, is_classification)
    except Exception as e:
        logger.warning("Active learning analysis failed: %s", e)
        report.summary = f"Analysis could not be completed: {e}"

    return report


def _analyze_classification(
    report: ActiveLearningReport,
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    label_encoder,
    sequences: list[str] | None,
    top_uncertain: int,
) -> None:
    """Classification-specific analysis."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X, sequences=sequences)
    y_proba = model.predict_proba(X, sequences=sequences)

    # Class names
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if label_encoder is not None:
        class_names = [str(label_encoder.inverse_transform([c])[0]) for c in classes]
    else:
        class_names = [str(c) for c in classes]

    # Per-class metrics
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix for "most confused with"
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    for i, (cls, name) in enumerate(zip(classes, class_names)):
        cls_key = str(cls)
        if cls_key not in clf_report:
            continue

        r = clf_report[cls_key]

        # Find most confused class (highest off-diagonal in row i)
        confused_with = ""
        confusion_count = 0
        if cm.shape[0] > 1:
            row = cm[i].copy()
            row[i] = 0  # exclude self
            max_idx = np.argmax(row)
            if row[max_idx] > 0:
                confused_with = class_names[max_idx]
                confusion_count = int(row[max_idx])

        report.class_analyses.append(ClassAnalysis(
            class_name=name,
            support=int(r.get("support", 0)),
            f1=float(r.get("f1-score", 0)),
            precision=float(r.get("precision", 0)),
            recall=float(r.get("recall", 0)),
            most_confused_with=confused_with,
            confusion_count=confusion_count,
        ))

    # Sort by F1 ascending — worst classes first
    report.class_analyses.sort(key=lambda c: c.f1)

    # Bottleneck classes: F1 < 0.5 or bottom 3
    f1_threshold = 0.5
    bottlenecks = [c.class_name for c in report.class_analyses if c.f1 < f1_threshold]
    if not bottlenecks and len(report.class_analyses) > 2:
        bottlenecks = [c.class_name for c in report.class_analyses[:3]]
    report.bottleneck_classes = bottlenecks

    # Uncertainty analysis (needs probabilities)
    if y_proba is not None and y_proba.ndim == 2:
        _analyze_uncertainty(report, y_proba, y_true, y_pred, class_names, classes, label_encoder, top_uncertain)


def _analyze_uncertainty(
    report: ActiveLearningReport,
    y_proba: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    classes: np.ndarray,
    label_encoder,
    top_n: int,
) -> None:
    """Find most uncertain samples by prediction entropy."""
    # Compute entropy per sample
    proba_clipped = np.clip(y_proba, 1e-10, 1.0)
    entropy = -np.sum(proba_clipped * np.log(proba_clipped), axis=1)

    # Sort by entropy descending
    uncertain_indices = np.argsort(entropy)[::-1][:top_n]

    # Map class indices to names
    cls_to_name = {c: n for c, n in zip(classes, class_names)}

    for idx in uncertain_indices:
        pred_label = cls_to_name.get(y_pred[idx], str(y_pred[idx]))
        true_label = cls_to_name.get(y_true[idx], str(y_true[idx]))

        report.uncertain_samples.append(UncertainSample(
            index=int(idx),
            entropy=float(entropy[idx]),
            predicted_label=pred_label,
            true_label=true_label,
            top_proba=float(np.max(y_proba[idx])),
        ))


def _analyze_regression(
    report: ActiveLearningReport,
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    sequences: list[str] | None,
) -> None:
    """Regression-specific analysis: residuals by target range."""
    y_pred = model.predict(X, sequences=sequences)
    residuals = np.abs(y_true - y_pred)

    # Bin by target value quantiles
    n_bins = min(5, max(2, len(y_true) // 20))
    try:
        bin_edges = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            # Fall back to uniform bins
            bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    except Exception:
        return

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (y_true >= low) & (y_true < high)
        else:
            mask = (y_true >= low) & (y_true <= high)

        count = int(mask.sum())
        if count == 0:
            continue

        report.residual_bins.append(ResidualBin(
            bin_label=f"[{low:.2f}, {high:.2f}]",
            mean_target=float(y_true[mask].mean()),
            mean_prediction=float(y_pred[mask].mean()),
            mean_abs_error=float(residuals[mask].mean()),
            count=count,
        ))

    # Sort by error descending
    report.residual_bins.sort(key=lambda b: b.mean_abs_error, reverse=True)

    if report.residual_bins:
        report.worst_predicted_range = report.residual_bins[0].bin_label


def _build_summary(report: ActiveLearningReport, is_classification: bool) -> str:
    """Build a natural-language summary of the analysis."""
    parts = []

    if is_classification:
        if report.bottleneck_classes:
            parts.append(
                f"**Bottleneck classes:** {', '.join(report.bottleneck_classes)}. "
                f"These classes have the lowest F1 scores and would benefit most from additional training data."
            )

        if report.class_analyses:
            worst = report.class_analyses[0]
            if worst.most_confused_with:
                parts.append(
                    f"The worst-performing class is **{worst.class_name}** "
                    f"(F1={worst.f1:.2f}), most often confused with **{worst.most_confused_with}** "
                    f"({worst.confusion_count} misclassifications)."
                )

        if report.uncertain_samples:
            avg_entropy = np.mean([s.entropy for s in report.uncertain_samples])
            n_wrong = sum(1 for s in report.uncertain_samples if s.predicted_label != s.true_label)
            parts.append(
                f"Among the top {len(report.uncertain_samples)} most uncertain samples "
                f"(mean entropy={avg_entropy:.3f}), {n_wrong} were misclassified. "
                f"These samples would be most informative for active learning."
            )
    else:
        if report.residual_bins:
            worst = report.residual_bins[0]
            parts.append(
                f"The model struggles most in the target range **{worst.bin_label}** "
                f"(MAE={worst.mean_abs_error:.4f}, n={worst.count}). "
                f"Additional samples in this range would help improve predictions."
            )

            # Check if errors are skewed toward extremes
            if len(report.residual_bins) >= 3:
                extreme_error = max(report.residual_bins[0].mean_abs_error, report.residual_bins[-1].mean_abs_error)
                mid_errors = [b.mean_abs_error for b in report.residual_bins[1:-1]]
                if mid_errors and extreme_error > 1.5 * np.mean(mid_errors):
                    parts.append(
                        "Prediction errors are higher at the extremes of the target distribution, "
                        "suggesting the model has insufficient training data in the tails."
                    )

    if report.feature_gap_suggestions:
        parts.append(
            "**Suggested additional data types:** " + "; ".join(report.feature_gap_suggestions)
        )

    return "\n\n".join(parts) if parts else "No significant data needs identified."


def get_feature_gap_suggestions(
    modality: str,
    task_type: str,
    bottleneck_classes: list[str] | None = None,
    worst_range: str = "",
) -> list[str]:
    """Suggest what additional data types would help, based on biology.

    This is the deterministic path — the Biology Specialist agent can provide
    richer suggestions when LLM is available.
    """
    suggestions = []

    if modality == "rna":
        suggestions.append("Ribosome profiling (Ribo-seq) data for direct translation measurement")
        suggestions.append("RNA structure probing (DMS-seq, SHAPE) for secondary structure features")
        if "translation" in task_type or "stability" in task_type:
            suggestions.append("Polysome fractionation data to distinguish translational regulation from mRNA abundance")

    elif modality == "dna":
        suggestions.append("Epigenomic data (ATAC-seq, ChIP-seq) for chromatin accessibility context")
        suggestions.append("Evolutionary conservation scores (PhyloP, phastCons) as features")

    elif modality == "protein":
        suggestions.append("Protein structure predictions (AlphaFold) as 3D features")
        suggestions.append("Evolutionary profiles (MSA depth) from sequence databases")

    elif modality == "cell_expression":
        if bottleneck_classes:
            suggestions.append(
                f"FACS-sorted populations for bottleneck classes ({', '.join(bottleneck_classes[:3])}) "
                f"to get pure reference profiles"
            )
        suggestions.append("Surface protein measurements (CITE-seq) for orthogonal cell type markers")
        suggestions.append("Spatial transcriptomics to add tissue context to cell type assignments")

    # General suggestions
    if not suggestions:
        suggestions.append("Additional samples from underrepresented conditions or classes")
        suggestions.append("Multi-modal measurements to provide complementary biological signals")

    return suggestions
