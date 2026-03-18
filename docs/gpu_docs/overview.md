# GPU-Accelerated Foundation Model Pipeline — Overview

## Why GPU Matters for Biological ML

Traditional ML on biological data relies on **handcrafted features**: k-mer frequencies for DNA/RNA, amino acid composition for proteins, highly variable genes for expression data. These features are well-understood, fast to compute, and work on CPU — but they have a ceiling. They encode **local statistics** (what 3-mers appear?) without capturing **global context** (how does codon 50 relate to codon 200?).

Foundation models like GenBio's AIDO family change this. Pre-trained on billions of biological sequences, they learn rich representations that encode:
- **Evolutionary conservation patterns** — which positions are functionally constrained
- **Structural context** — how sequence folds into 3D structure
- **Long-range dependencies** — how distant elements interact (e.g., UTR structure affecting translation)
- **Cross-species patterns** — conserved motifs across species that handcrafted features miss

The catch: these models require a GPU. The AIDO models range from 300M to 16B parameters — inference alone needs CUDA. The co-scientist pipeline is designed to **transparently leverage GPU when available** while running identically on CPU.

## The Performance Gap

Based on the genbio-leaderboard tasks, the expected performance delta:

| Approach | RNA Translation Efficiency | Cell Type Classification | Protein Function |
|----------|---------------------------|--------------------------|------------------|
| Handcrafted features (CPU) | Spearman ~0.55-0.65 | Accuracy ~0.85-0.92 | F1 ~0.70-0.80 |
| AIDO embeddings + XGBoost | Spearman ~0.65-0.75 | Accuracy ~0.90-0.95 | F1 ~0.80-0.88 |
| Concat (handcrafted + embeddings) | Spearman ~0.70-0.80 | Accuracy ~0.92-0.96 | F1 ~0.83-0.90 |
| AIDO fine-tuning | Spearman ~0.75-0.85 | Accuracy ~0.94-0.97 | F1 ~0.85-0.93 |

The key insight: **concat models often outperform pure embedding models** because handcrafted features and learned representations capture complementary signals.

## Architecture: CPU Pipeline is Untouched

```
                    ┌──────────────────────────────────────────────┐
                    │              Data Loading + Profiling          │
                    └──────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────────┐
                    │           Preprocessing (modality-specific)    │
                    │  k-mers / HVG / AA composition  [always runs]  │
                    │                                               │
                    │  ┌─── GPU available? ────────────────────┐    │
                    │  │  YES: extract AIDO embeddings → X_embed │    │
                    │  │  NO:  X_embed = None (skip silently)   │    │
                    │  └────────────────────────────────────────┘    │
                    └──────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────────┐
                    │              Train/Val/Test Split              │
                    │  X_train, X_val, X_test      [always]         │
                    │  X_embed_train/val/test       [if GPU]         │
                    │  seqs_train/val/test          [if sequence]    │
                    └──────────────┬───────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
     ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐
     │  CPU Models     │  │ Embedding Models │  │ Fine-tune Model  │
     │  (always run)   │  │ (GPU only)       │  │ (GPU only)       │
     │                │  │                 │  │                  │
     │ trivial        │  │ embed_xgboost   │  │ aido_finetune    │
     │ logistic/ridge │  │ embed_mlp       │  │                  │
     │ xgboost/lgbm   │  │ concat_xgboost  │  │ Unfreezes last   │
     │ random_forest  │  │ concat_mlp      │  │ N layers + head  │
     │ svm/knn        │  │                 │  │                  │
     │ mlp/bio_cnn    │  │ Uses X_embed or │  │ Uses raw seqs    │
     │ ft_transformer │  │ hstack(X,X_emb) │  │ Mixed precision  │
     │ stacking       │  │                 │  │                  │
     └───────┬────────┘  └───────┬─────────┘  └────────┬─────────┘
              │                    │                      │
              └────────────────────┼──────────────────────┘
                                   ▼
                    ┌──────────────────────────────────────────────┐
                    │        Evaluation (same val set for all)       │
                    │        Best model wins regardless of tier      │
                    └──────────────────────────────────────────────┘
```

## Three Implementation Phases

The GPU integration was implemented in three phases:

1. **Phase 1: Frozen Embeddings** (`embed_xgboost`, `embed_mlp`) — Extract embeddings once, train standard models on them
2. **Phase 2: End-to-End Fine-Tuning** (`aido_finetune`) — Unfreeze AIDO backbone layers, train with task head
3. **Phase 3: Hybrid Features + Integration** (`concat_xgboost`, `concat_mlp`) — Combine handcrafted + embeddings; full agent awareness, HP search, validation, export

See the phase-specific documents for implementation details:
- [Phase 1: Frozen Embeddings](phase1_frozen_embeddings.md)
- [Phase 2: Fine-Tuning](phase2_finetuning.md)
- [Phase 3: Hybrid + Integration](phase3_hybrid_integration.md)
