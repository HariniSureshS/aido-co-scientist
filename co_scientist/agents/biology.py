"""Biology Specialist agent — provides biological context, validation, and interpretation.

Enhanced with:
- Rich biological knowledge base per modality/task/dataset
- Feature importance interpretation (maps ML features → biological meaning)
- Metric validation (is the metric appropriate for this biological question?)
- Task-specific plausibility ranges from literature
- Research-informed assessment (uses search results when available)
- Detailed report text generation
"""

from __future__ import annotations

from co_scientist.agents.base import BaseAgent
from co_scientist.agents.types import AgentRole, Decision, PipelineContext
from co_scientist.llm.prompts import BIOLOGY_SPECIALIST_SYSTEM


# ── Biological Knowledge Base ─────────────────────────────────────────────

# Task-specific biological context — keyed by (modality, task_hint keywords)
_TASK_KNOWLEDGE: dict[str, dict] = {
    # RNA translation efficiency
    "rna:translation": {
        "context": (
            "Translation efficiency (TE) measures how effectively an mRNA is translated "
            "into protein. It is influenced by codon usage bias, 5'UTR structure and length, "
            "Kozak consensus sequence strength, mRNA secondary structure (particularly in the "
            "5'UTR), codon optimality (tAI, CAI), poly(A) tail length, and tissue-specific "
            "tRNA pools. In muscle tissue, codon usage bias is especially pronounced due to "
            "the high expression of a limited set of structural proteins (actins, myosins)."
        ),
        "key_features": {
            "codon_usage_bias": "Reflects adaptation to tissue-specific tRNA pools — highly optimized codons are translated faster",
            "gc_content": "GC-rich regions form more stable secondary structures that can impede ribosome scanning",
            "utr_length": "Longer 5'UTRs with upstream ORFs reduce translation initiation efficiency",
            "kozak_score": "Strong Kozak context (GCC[A/G]CCAUGG) enhances start codon recognition",
            "rna_folding_energy": "Lower minimum free energy (more stable structure) in 5'UTR correlates with lower TE",
            "seq_length": "mRNA length affects ribosome density and mRNA half-life",
        },
        "expected_metric_range": {"spearman": (0.3, 0.85), "pearson": (0.3, 0.85)},
        "appropriate_metrics": ["spearman", "pearson", "r2", "mse", "rmse"],
        "inappropriate_metrics": ["accuracy", "f1_macro", "auroc"],
        "biological_signals": [
            "Codon usage features should dominate — if not, check for sequence composition bias",
            "k-mer features capture codon effects indirectly via trinucleotide frequencies",
            "Tree models may outperform linear models because TE has nonlinear dependencies on multiple sequence features",
        ],
    },
    # Cell type classification
    "cell_expression:cell_type": {
        "context": (
            "Cell type classification from scRNA-seq relies on marker gene expression patterns. "
            "Each cell type has a characteristic transcriptional program driven by lineage-specific "
            "transcription factors. Key challenges include: dropout noise (zero-inflation), batch "
            "effects between experiments, continuous transitions between cell states, and rare cell "
            "types with few training examples. The Segerstolpe pancreas dataset contains endocrine "
            "(alpha, beta, delta, gamma/PP, epsilon) and exocrine (acinar, ductal) cell types, "
            "plus immune cells (mast, macrophage) and stellate cells."
        ),
        "key_features": {
            "pathway_scores": "Aggregate expression of genes in known pathways (e.g., insulin signaling for beta cells)",
            "cell_cycle_markers": "G1/S/G2/M phase genes — proliferating cells may confound type assignment",
            "mitochondrial_fraction": "High MT gene fraction indicates stressed or dying cells (quality filter)",
        },
        "marker_genes": {
            "alpha": ["GCG", "TTR", "IRX2"],
            "beta": ["INS", "IAPP", "MAFA", "NKX6-1"],
            "delta": ["SST", "HHEX", "RBP4"],
            "gamma": ["PPY", "MEIS2"],
            "epsilon": ["GHRL", "ARX"],
            "acinar": ["PRSS1", "CPA1", "CELA3A"],
            "ductal": ["KRT19", "SPP1", "MUC1"],
            "stellate": ["COL1A1", "SPARC", "DCN"],
        },
        "expected_metric_range": {"accuracy": (0.75, 0.98), "f1_macro": (0.65, 0.95), "f1_weighted": (0.75, 0.98)},
        "appropriate_metrics": ["accuracy", "f1_macro", "f1_weighted", "auroc"],
        "inappropriate_metrics": ["mse", "rmse", "r2"],
        "biological_signals": [
            "Top features should include known marker genes for the most abundant cell types",
            "Confusion between delta and gamma cells is biologically expected — they share hormonal programs",
            "If ductal/acinar separation is hard, check for contaminating doublets",
            "High accuracy on rare types (epsilon, mast) suggests the model captures subtle transcriptional signatures",
        ],
    },
    # RNA stability
    "rna:stability": {
        "context": (
            "mRNA stability (half-life) is determined by cis-regulatory elements in the UTRs, "
            "codon optimality, polyA tail length, m6A methylation sites, and RNA-binding protein "
            "motifs. ARE (AU-rich elements) in 3'UTRs promote rapid degradation, while stable "
            "stem-loop structures can protect against exonuclease activity."
        ),
        "key_features": {
            "gc_content": "GC-rich mRNAs tend to be more stable — G:C base pairs in stems are thermodynamically stronger",
            "codon_usage_bias": "Optimal codons are associated with higher mRNA stability (codon-mediated decay)",
            "seq_length": "Longer mRNAs have more potential degradation sites",
        },
        "expected_metric_range": {"spearman": (0.25, 0.75), "pearson": (0.25, 0.75)},
        "appropriate_metrics": ["spearman", "pearson", "r2", "mse", "rmse"],
        "inappropriate_metrics": ["accuracy", "f1_macro"],
        "biological_signals": [
            "3'UTR features should be informative — this is where most stability elements reside",
            "Moderate correlations (0.3-0.6) are expected — stability is regulated post-transcriptionally by many factors not in sequence alone",
        ],
    },
    # Protein function
    "protein:function": {
        "context": (
            "Protein function prediction from sequence leverages amino acid composition, "
            "evolutionary conservation, domain architecture, and physicochemical properties. "
            "Machine learning models capture statistical patterns in k-mer frequencies that "
            "correlate with protein families and functional annotations."
        ),
        "key_features": {
            "amino_acid_composition": "Distribution of 20 amino acids reflects structural/functional constraints",
            "molecular_weight": "Correlates with protein size and domain content",
            "isoelectric_point": "Affects subcellular localization (secreted proteins tend to be acidic)",
            "hydrophobicity": "Membrane proteins have distinct hydrophobic profiles",
        },
        "expected_metric_range": {"accuracy": (0.6, 0.95), "f1_macro": (0.5, 0.90)},
        "appropriate_metrics": ["accuracy", "f1_macro", "f1_weighted"],
        "inappropriate_metrics": ["mse", "spearman"],
        "biological_signals": [
            "k-mer features capture short sequence motifs associated with protein families",
            "Amino acid composition features should be informative — this is a classic bioinformatics approach",
        ],
    },
}

# Generic fallback knowledge per modality
_MODALITY_KNOWLEDGE: dict[str, dict] = {
    "rna": {
        "context": (
            "RNA sequence features can capture codon usage bias, UTR regulatory elements, "
            "and sequence composition effects on gene expression and regulation."
        ),
        "suggested_features": ["codon_usage_bias", "gc_content", "utr_length", "kozak_score", "rna_folding_energy"],
    },
    "dna": {
        "context": (
            "DNA sequence features can capture promoter motifs, transcription factor binding sites, "
            "CpG islands, repeat elements, and epigenetic regulatory contexts."
        ),
        "suggested_features": ["gc_content", "cpg_density", "repeat_content", "conservation_score"],
    },
    "protein": {
        "context": (
            "Protein sequence features capture amino acid composition, physicochemical properties, "
            "secondary structure propensities, and functional domain signatures."
        ),
        "suggested_features": ["amino_acid_composition", "molecular_weight", "isoelectric_point", "hydrophobicity"],
    },
    "cell_expression": {
        "context": (
            "Gene expression profiles capture cellular state through marker genes, pathway activity "
            "levels, and transcriptional programs that define cell identity and function."
        ),
        "suggested_features": ["pathway_scores", "cell_cycle_markers", "mitochondrial_fraction"],
    },
}

# Known plausibility ranges by metric (very general fallback)
_GENERAL_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "accuracy": (0.05, 0.999),
    "f1_macro": (0.02, 0.999),
    "f1_weighted": (0.05, 0.999),
    "auroc": (0.45, 0.999),
    "spearman": (-0.1, 0.999),
    "pearson": (-0.1, 0.999),
    "r2": (-0.5, 0.999),
    "mse": (0.0, 1e6),
    "rmse": (0.0, 1e3),
    "mae": (0.0, 1e3),
}


class BiologySpecialistAgent(BaseAgent):
    """Provides biological context, validates plausibility, suggests domain features.

    Enhanced capabilities:
    - Rich knowledge base with task-specific biological context
    - Feature importance interpretation mapping ML features to biology
    - Metric appropriateness validation
    - Task-specific plausibility ranges from literature
    - Research-informed assessment when search results available
    """

    role = AgentRole.BIOLOGY_SPECIALIST

    def system_prompt(self) -> str:
        return BIOLOGY_SPECIALIST_SYSTEM

    def decide_deterministic(self, context: PipelineContext) -> Decision:
        """Rule-based biological assessment with rich knowledge base."""
        knowledge = self._get_knowledge(context)
        bio_context = knowledge.get("context", "")
        features = self._suggest_features(context, knowledge)
        plausibility = self._check_plausibility(context, knowledge)
        metric_check = self._validate_metric(context, knowledge)

        return Decision(
            action="biological_assessment",
            parameters={
                "plausibility": plausibility["status"],
                "plausibility_detail": plausibility["detail"],
                "biological_context": bio_context,
                "suggested_features": features,
                "metric_appropriate": metric_check["appropriate"],
                "metric_note": metric_check["note"],
                "biological_signals": knowledge.get("biological_signals", []),
            },
            reasoning=f"Bio assessment for {context.modality} / {context.dataset_path}: {plausibility['status']}",
            confidence=0.6 if bio_context else 0.4,
        )

    # ── Knowledge Retrieval ───────────────────────────────────────────────

    def _get_knowledge(self, context: PipelineContext) -> dict:
        """Look up the best matching knowledge entry for this dataset."""
        task_hint = context.dataset_path.split("/")[-1] if "/" in context.dataset_path else ""
        modality = context.modality

        # Try task-specific knowledge first
        for key, knowledge in _TASK_KNOWLEDGE.items():
            mod_part, task_part = key.split(":", 1)
            if mod_part == modality and task_part in task_hint:
                return knowledge

        # Fall back to modality-level knowledge
        mod_knowledge = _MODALITY_KNOWLEDGE.get(modality, {})
        return {
            "context": mod_knowledge.get("context", "General biological dataset."),
            "key_features": {},
            "expected_metric_range": {},
            "appropriate_metrics": [],
            "inappropriate_metrics": [],
            "biological_signals": [],
            "suggested_features": mod_knowledge.get("suggested_features", []),
        }

    # ── Feature Suggestions ───────────────────────────────────────────────

    def _suggest_features(self, context: PipelineContext, knowledge: dict) -> list[str]:
        """Suggest domain-specific features from knowledge base."""
        # Task-specific features
        if "key_features" in knowledge and knowledge["key_features"]:
            return list(knowledge["key_features"].keys())
        # Modality-level fallback
        if "suggested_features" in knowledge:
            return knowledge["suggested_features"]
        return []

    # ── Feature Importance Interpretation ──────────────────────────────────

    def interpret_features(
        self,
        context: PipelineContext,
        top_features: list[tuple[str, float]],
    ) -> list[dict[str, str]]:
        """Map top ML features to biological meaning.

        Args:
            context: Pipeline context for knowledge lookup
            top_features: List of (feature_name, importance) tuples, sorted by importance desc

        Returns:
            List of dicts with keys: feature, importance, biological_meaning
        """
        knowledge = self._get_knowledge(context)
        key_features = knowledge.get("key_features", {})
        marker_genes = knowledge.get("marker_genes", {})

        # Build reverse lookup: gene → cell type
        gene_to_type: dict[str, str] = {}
        for cell_type, genes in marker_genes.items():
            for gene in genes:
                gene_to_type[gene.upper()] = cell_type

        interpretations = []
        for feat_name, importance in top_features[:15]:  # top 15
            meaning = self._interpret_single_feature(feat_name, importance, key_features, gene_to_type)
            interpretations.append(meaning)

        return interpretations

    def _interpret_single_feature(
        self,
        feat_name: str,
        importance: float,
        key_features: dict[str, str],
        gene_to_type: dict[str, str],
    ) -> dict[str, str]:
        """Interpret a single feature biologically."""
        bio_meaning = ""

        # Check against known key features
        for key, desc in key_features.items():
            if key in feat_name.lower():
                bio_meaning = desc
                break

        # Check if it's a known marker gene
        if not bio_meaning:
            upper_name = feat_name.upper().replace("GENE_", "").replace("HVG_", "")
            if upper_name in gene_to_type:
                bio_meaning = f"Known marker gene for {gene_to_type[upper_name]} cells"

        # k-mer feature interpretation
        if not bio_meaning and feat_name.startswith("kmer_"):
            kmer = feat_name.replace("kmer_", "")
            bio_meaning = self._interpret_kmer(kmer)

        # Amino acid frequency
        if not bio_meaning and feat_name.startswith("aa_freq_"):
            aa = feat_name.replace("aa_freq_", "")
            bio_meaning = self._interpret_amino_acid(aa)

        # Generic feature pattern matching
        if not bio_meaning:
            bio_meaning = self._interpret_generic_feature(feat_name)

        return {
            "feature": feat_name,
            "importance": f"{importance:.4f}",
            "biological_meaning": bio_meaning or "No specific biological annotation available",
        }

    def _interpret_kmer(self, kmer: str) -> str:
        """Interpret a k-mer feature biologically."""
        kmer = kmer.upper()
        # Known biological signals in k-mers
        if len(kmer) == 3:
            # Codon interpretation
            _NOTABLE_CODONS = {
                "ATG": "Start codon — frequency reflects upstream ORF density",
                "TAA": "Stop codon (ochre) — frequency in ORFs reflects coding density",
                "TAG": "Stop codon (amber) — rare in highly expressed genes",
                "TGA": "Stop codon (opal) — context-dependent selenocysteine codon",
                "CGA": "Rare arginine codon — associated with low translation efficiency",
                "AGA": "Common arginine codon in mammals — associated with higher TE",
                "GCG": "Alanine codon — GC-rich, associated with stable mRNA structures",
            }
            if kmer in _NOTABLE_CODONS:
                return _NOTABLE_CODONS[kmer]
            return f"Trinucleotide {kmer} — captures codon usage and dinucleotide composition"

        if len(kmer) == 4:
            gc_count = kmer.count("G") + kmer.count("C")
            if gc_count >= 3:
                return f"GC-rich tetramer — associated with stable RNA secondary structures"
            elif gc_count <= 1:
                return f"AT-rich tetramer — may reflect AU-rich elements (AREs) in UTRs"
            return f"Tetranucleotide {kmer} — captures local sequence composition"

        return f"k-mer {kmer} — captures sequence composition patterns"

    def _interpret_amino_acid(self, aa: str) -> str:
        """Interpret amino acid frequency feature."""
        _AA_INFO = {
            "A": "Alanine — small, hydrophobic, common in alpha-helices",
            "C": "Cysteine — disulfide bonds, redox-sensitive, rare",
            "D": "Aspartate — negative charge, common in active sites",
            "E": "Glutamate — negative charge, common in surface-exposed regions",
            "F": "Phenylalanine — aromatic, hydrophobic core packing",
            "G": "Glycine — smallest, enables backbone flexibility",
            "H": "Histidine — pH-sensitive, common in catalytic sites",
            "I": "Isoleucine — hydrophobic, beta-branched",
            "K": "Lysine — positive charge, post-translational modification site",
            "L": "Leucine — most common amino acid, hydrophobic",
            "M": "Methionine — start codon, oxidation-sensitive",
            "N": "Asparagine — glycosylation site (N-X-S/T motif)",
            "P": "Proline — rigid, disrupts secondary structure",
            "Q": "Glutamine — polar, deamidation-prone",
            "R": "Arginine — positive charge, strong interactions",
            "S": "Serine — phosphorylation site, small polar",
            "T": "Threonine — phosphorylation site, beta-branched",
            "V": "Valine — hydrophobic, beta-branched",
            "W": "Tryptophan — largest, aromatic, rare",
            "Y": "Tyrosine — phosphorylation site, aromatic",
        }
        return _AA_INFO.get(aa.upper(), f"Amino acid {aa} frequency")

    def _interpret_generic_feature(self, feat_name: str) -> str:
        """Interpret common feature name patterns."""
        name_lower = feat_name.lower()
        if "gc_content" in name_lower:
            return "GC content — reflects thermodynamic stability and codon bias"
        if "seq_length" in name_lower or "sequence_length" in name_lower:
            return "Sequence length — affects mRNA stability, ribosome transit time"
        if "molecular_weight" in name_lower:
            return "Protein molecular weight — correlates with domain complexity"
        if "hydrophobicity" in name_lower:
            return "Average hydrophobicity — distinguishes membrane vs soluble proteins"
        if "isoelectric" in name_lower:
            return "Isoelectric point — relates to protein charge and localization"
        if "mitochondrial" in name_lower or name_lower.startswith("mt-"):
            return "Mitochondrial gene — high levels indicate cell stress or apoptosis"
        return ""

    # ── Metric Validation ─────────────────────────────────────────────────

    def _validate_metric(self, context: PipelineContext, knowledge: dict) -> dict[str, any]:
        """Check if the chosen metric is appropriate for this biological task."""
        metric = context.primary_metric
        if not metric:
            return {"appropriate": True, "note": "No metric to validate yet"}

        appropriate_metrics = knowledge.get("appropriate_metrics", [])
        inappropriate_metrics = knowledge.get("inappropriate_metrics", [])

        if metric in inappropriate_metrics:
            return {
                "appropriate": False,
                "note": (
                    f"Metric '{metric}' may not be appropriate for this task. "
                    f"Consider: {', '.join(appropriate_metrics[:3])}"
                ),
            }

        if appropriate_metrics and metric in appropriate_metrics:
            return {"appropriate": True, "note": f"'{metric}' is a standard metric for this task"}

        return {"appropriate": True, "note": f"No specific recommendation for metric '{metric}'"}

    # ── Plausibility Checking ─────────────────────────────────────────────

    def _check_plausibility(self, context: PipelineContext, knowledge: dict) -> dict[str, str]:
        """Detailed plausibility check using task-specific expected ranges."""
        if not context.best_score:
            return {"status": "no_results", "detail": "No model results to assess"}

        metric = context.primary_metric
        score = context.best_score

        # Try task-specific range first
        expected_range = knowledge.get("expected_metric_range", {}).get(metric)
        if expected_range:
            low, high = expected_range
            return self._assess_score_in_range(score, low, high, metric, specific=True)

        # Fall back to general plausibility
        general_range = _GENERAL_PLAUSIBILITY.get(metric)
        if general_range:
            low, high = general_range
            return self._assess_score_in_range(score, low, high, metric, specific=False)

        return {"status": "unknown", "detail": f"No plausibility range for metric '{metric}'"}

    def _assess_score_in_range(
        self, score: float, low: float, high: float, metric: str, specific: bool
    ) -> dict[str, str]:
        """Assess a score against expected range."""
        source = "task-specific literature" if specific else "general guidelines"

        if score > high:
            return {
                "status": "suspicious",
                "detail": (
                    f"Score {score:.4f} exceeds expected range [{low:.2f}, {high:.2f}] "
                    f"based on {source}. Possible data leakage or train/test overlap."
                ),
            }

        if metric in ("mse", "rmse", "mae"):
            # Lower is better — high scores are bad, not suspicious
            return {"status": "plausible", "detail": f"Score {score:.4f} — within expected range"}

        if score < low:
            return {
                "status": "implausible",
                "detail": (
                    f"Score {score:.4f} is below expected range [{low:.2f}, {high:.2f}] "
                    f"based on {source}. Model may not be capturing meaningful signal."
                ),
            }

        # Within range — provide context on where in the range
        range_width = high - low
        if range_width > 0:
            position = (score - low) / range_width
            if position > 0.8:
                quality = "excellent (upper range)"
            elif position > 0.5:
                quality = "good (mid-upper range)"
            elif position > 0.2:
                quality = "moderate (mid range)"
            else:
                quality = "low but within expected range"
        else:
            quality = "within range"

        return {
            "status": "plausible",
            "detail": (
                f"Score {score:.4f} is {quality} for this task "
                f"(expected [{low:.2f}, {high:.2f}] based on {source})."
            ),
        }

    # ── Biological Interpretation for Report ──────────────────────────────

    def generate_interpretation(
        self,
        context: PipelineContext,
        feature_names: list[str] | None = None,
        research_papers: list | None = None,
    ) -> str:
        """Generate a rich biological interpretation paragraph for the report.

        This is the deterministic path — the LLM path is handled in analysis.py.
        """
        knowledge = self._get_knowledge(context)
        parts = []

        # Opening context
        bio_context = knowledge.get("context", "")
        if bio_context:
            parts.append(bio_context)

        # Score interpretation
        plausibility = self._check_plausibility(context, knowledge)
        if plausibility["detail"]:
            parts.append(plausibility["detail"])

        # Metric appropriateness
        metric_check = self._validate_metric(context, knowledge)
        if not metric_check["appropriate"]:
            parts.append(f"**Note:** {metric_check['note']}")

        # Biological signals to look for
        signals = knowledge.get("biological_signals", [])
        if signals:
            parts.append("**Key biological signals:** " + signals[0])

        # Research context (if available)
        if research_papers:
            n = len(research_papers)
            parts.append(
                f"The literature search found {n} relevant paper(s) for this task, "
                f"providing context for interpreting these results."
            )

        return "\n\n".join(parts)

    # ── Known Marker Gene Check ───────────────────────────────────────────

    def check_marker_genes(
        self,
        context: PipelineContext,
        feature_names: list[str],
    ) -> dict[str, list[str]]:
        """Check which known marker genes appear in the feature set.

        Returns dict mapping cell_type → list of marker genes found in features.
        """
        knowledge = self._get_knowledge(context)
        marker_genes = knowledge.get("marker_genes", {})
        if not marker_genes:
            return {}

        feature_set = {f.upper() for f in feature_names}
        found = {}
        for cell_type, genes in marker_genes.items():
            matches = [g for g in genes if g.upper() in feature_set]
            if matches:
                found[cell_type] = matches

        return found
