"""Data Analyst agent — advises on preprocessing, features, and data quality."""

from __future__ import annotations

from co_scientist.agents.base import BaseAgent
from co_scientist.agents.types import AgentRole, Decision, PipelineContext
from co_scientist.llm.prompts import DATA_ANALYST_SYSTEM


class DataAnalystAgent(BaseAgent):
    """Advises on preprocessing, feature engineering, and data quality.

    Deterministic fallback replicates the current rule-based behavior from
    defaults.yaml — the same logic that ran before agents existed.
    """

    role = AgentRole.DATA_ANALYST

    def system_prompt(self) -> str:
        return DATA_ANALYST_SYSTEM

    def decide_deterministic(self, context: PipelineContext) -> Decision:
        """Rule-based preprocessing/feature decisions based on modality."""
        modality = context.modality
        n_samples = context.num_samples

        if modality in ("rna", "dna", "protein"):
            # Sequence data: k-mer features + optional CNN
            k = 4 if modality == "rna" else 3
            steps = [f"kmer_frequency_k{k}", "standard_scale"]
            reasons = []
            if modality == "rna":
                reasons.append(
                    f"Using k={k} (4-mers) for RNA because 4-mers capture codon context + flanking nucleotide, "
                    f"producing {4**k} features that encode translation-relevant motifs"
                )
            elif modality == "protein":
                reasons.append(
                    f"Using k={k} (3-mers) for protein because tripeptide frequencies capture "
                    f"local structural motifs and biochemical properties"
                )
            else:
                reasons.append(
                    f"Using k={k} for DNA to capture regulatory motifs like transcription factor binding sites"
                )
            reasons.append(
                "Standard scaling applied so distance-based models (SVM, KNN) are not dominated by high-frequency k-mers"
            )
            if n_samples > 500:
                steps.append("recommend_cnn")
                reasons.append(
                    f"With {n_samples:,} samples, a 1D CNN is also recommended — it can learn positional motifs "
                    f"directly from sequences without relying on k-mer bag-of-words"
                )
            return Decision(
                action="set_preprocessing",
                parameters={
                    "steps": steps,
                    "kmer_k": k,
                    "scaling": "standard",
                    "selection_reasons": reasons,
                },
                reasoning=(
                    f"Sequence modality ({modality.upper()}): converting sequences to {k}-mer frequency vectors "
                    f"({4**k} features) to capture nucleotide/amino acid composition patterns. "
                    f"This transforms variable-length sequences into fixed-size numeric vectors suitable for ML models"
                ),
                confidence=0.9,
            )

        elif modality == "cell_expression":
            # Expression: log1p + HVG + scale
            hvg_count = min(2000, max(500, context.num_features // 5))
            reasons = [
                "log1p transform applied to stabilize variance in count data — raw counts span orders of magnitude",
                f"Selecting top {hvg_count} highly variable genes (HVGs) to focus on informative features "
                f"and reduce noise from housekeeping genes",
                "Standard scaling ensures each gene contributes equally regardless of expression level",
            ]
            return Decision(
                action="set_preprocessing",
                parameters={
                    "steps": ["log1p", "hvg_selection", "standard_scale"],
                    "hvg_count": hvg_count,
                    "scaling": "standard",
                    "selection_reasons": reasons,
                },
                reasoning=(
                    f"Expression data: log1p normalization → top {hvg_count} HVG selection → standard scaling. "
                    f"This is the standard single-cell analysis pipeline (Scanpy/Seurat-inspired)"
                ),
                confidence=0.9,
            )

        else:
            # Generic tabular
            return Decision(
                action="set_preprocessing",
                parameters={
                    "steps": ["standard_scale"],
                    "scaling": "standard",
                    "selection_reasons": [
                        "Standard scaling (zero mean, unit variance) applied to ensure all features "
                        "contribute equally to distance-based and gradient-based models",
                    ],
                },
                reasoning="Generic tabular data: standard scaling to normalize feature ranges",
                confidence=0.7,
            )

    def assess_data_quality(self, context: PipelineContext) -> Decision:
        """Assess data quality and flag issues."""
        issues = []

        if context.num_samples < 100:
            issues.append("very_small_dataset")
        if context.num_features > context.num_samples * 10:
            issues.append("high_dimensional")
        if context.num_classes > 0 and context.num_classes > 50:
            issues.append("many_classes")

        severity = "warning" if issues else "ok"

        return Decision(
            action="data_quality_assessment",
            parameters={
                "issues": issues,
                "severity": severity,
                "recommendations": self._quality_recommendations(issues),
            },
            reasoning=f"Found {len(issues)} data quality issue(s)" if issues else "Data quality looks good",
            confidence=0.8,
        )

    def _quality_recommendations(self, issues: list[str]) -> list[str]:
        recs = []
        if "very_small_dataset" in issues:
            recs.append("Use simple models (ridge/logistic) to avoid overfitting")
        if "high_dimensional" in issues:
            recs.append("Apply aggressive feature selection (HVG or L1)")
        if "many_classes" in issues:
            recs.append("Check for class imbalance; consider grouping rare classes")
        return recs
