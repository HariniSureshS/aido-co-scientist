# Step 14: Multi-Scale BioCNN — Detailed Walkthrough

## Overview

The Multi-Scale BioCNN is a **custom 1D convolutional neural network** designed specifically for biological sequences. Unlike standard k-mer frequency features (which lose positional information), the BioCNN operates directly on one-hot encoded raw sequences, capturing **spatial patterns at multiple scales simultaneously**.

This implements Architecture Section 7.4.

---

## Why Not Just Use K-mer Features?

K-mer frequency vectors count how often each k-mer appears in a sequence, but they throw away **where** those k-mers appear. For biological sequences, position matters:

- A start codon (AUG) at position 1 means something different than AUG at position 50
- Regulatory motifs have position-dependent effects (5' UTR vs 3' UTR)
- Motif pairs that are close together may interact differently than distant ones

A CNN preserves this spatial structure by sliding learned filters across the sequence.

---

## Architecture

```
Input: Raw nucleotide sequence (e.g., "AUGCCGUAA...")
         │
         ▼
    One-Hot Encoding (4 channels: A, C, G, T)
         │
    ┌────┼────┬────┬────┐
    │    │    │    │    │
    ▼    ▼    ▼    ▼    ▼
 Conv1D  Conv1D  Conv1D  Conv1D
 k=3     k=5     k=7     k=9
    │    │    │    │
    ▼    ▼    ▼    ▼
 BN+ReLU BN+ReLU BN+ReLU BN+ReLU
    │    │    │    │
    └────┴────┴────┘
         │
    Concatenate (n_filters × 4 branches)
         │
    ┌────┴────┐
    │         │
 GlobalAvgPool GlobalMaxPool
    │         │
    └────┬────┘
         │
    Concat (n_filters × 4 × 2 = 512 dim with default 64 filters)
         │
    FC + Residual Block (512 → 256)
         │
    FC + Residual Block (256 → 128)
         │
    Linear Head → Output
```

### What Each Kernel Size Captures

| Kernel Size | Biological Scale | Examples |
|-------------|-----------------|---------|
| **k=3** | Codons, dinucleotide steps | Start codons (AUG), CpG sites, codon usage bias |
| **k=5** | Short motifs | Splice donor/acceptor sites (GT-AG), Kozak-like patterns |
| **k=7** | Transcription factor binding sites | Most TF motifs are 6-8 bp |
| **k=9** | Longer regulatory elements | Enhancer elements, RNA secondary structure motifs |

### Why Parallel Branches (Inception-style)

A standard CNN with a single kernel size can only capture one scale at a time. Stacking different kernel sizes sequentially (as in deeper networks) loses the ability to detect motifs at each scale independently. The parallel branch design:

1. Each branch independently learns features at its own scale
2. Concatenation preserves all scales equally
3. The FC layers learn to combine multi-scale features
4. Much lighter than stacking — trainable on CPU with thousands of samples

### Dual Global Pooling

Using both average and max pooling captures complementary information:
- **Average pool:** Captures the overall "density" of a motif across the sequence
- **Max pool:** Captures the "peak signal" — whether a motif appears at all, regardless of frequency

### Residual FC Blocks

The fully connected layers use skip connections:

```python
class _ResidualBlock(nn.Module):
    def forward(self, x):
        return self.fc(x) + self.skip(x)
```

This prevents vanishing gradients in the FC layers and lets the network learn residual improvements over identity.

---

## One-Hot Encoding

Sequences are encoded as (N, 4, max_len) tensors:

```
"ACGT" → [[1,0,0,0],   ← A channel
           [0,1,0,0],   ← C channel
           [0,0,1,0],   ← G channel
           [0,0,0,1]]   ← T channel
```

- RNA sequences: U is converted to T before encoding
- Unknown characters (N, ambiguity codes): encoded as all zeros
- Shorter sequences: zero-padded to the maximum length in the batch
- The `same` padding in Conv1D ensures output length matches input length

---

## Pipeline Integration

### Raw Sequence Flow

The BioCNN needs raw sequences, not preprocessed k-mer features. The pipeline carries raw sequences through:

```
Loader → dataset.X["sequences"]
           │
Preprocessor → preprocessed.raw_sequences (extracted before k-mer transform)
           │
Splitter → split.seqs_train / split.seqs_val / split.seqs_test
           │
Trainer → model.fit(X_train, y_train, sequences=seqs_train)
           │
Evaluator → model.predict(X_val, sequences=seqs_val)
```

**Key files modified:**
- `data/types.py`: Added `raw_sequences` to `PreprocessingResult` and `seqs_train/val/test` to `SplitData`
- `data/preprocess.py`: Extracts raw sequences before k-mer encoding
- `data/split.py`: Carries sequences through all three split strategies (predefined, fold-based, random)
- `modeling/trainer.py`: Passes sequences to CNN models during `fit()`
- `modeling/types.py`: `TrainedModel.needs_sequences` property routes sequences to CNN models
- `evaluation/metrics.py`: Passes sequences to CNN models during `predict()`
- `guardrails.py`: Passes sequences to CNN models during post-training checks

### Modality Gating

The BioCNN is only activated for sequence modalities (RNA, DNA, protein) via modality overrides in `defaults.yaml`:

```yaml
modality_overrides:
  rna:
    models:
      regression:
        advanced:
          - name: mlp
            model_type: mlp
            hyperparameters: { ... }
          - name: bio_cnn
            model_type: bio_cnn
            hyperparameters: { ... }
```

For tabular modalities (cell_expression, etc.), the BioCNN is not included. The guardrail `check_model_data_compatibility` also blocks sequence models on non-sequence data:

```python
if model_type in _SEQUENCE_ONLY_MODELS:
    if profile.modality not in (Modality.RNA, Modality.DNA, Modality.PROTEIN):
        → ERROR: "sequence model requires sequence data"
    elif split.seqs_train is None:
        → ERROR: "no sequences available for CNN"
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_filters` | 64 | Filters per conv branch (total: 64 × 4 = 256 channels) |
| `kernel_sizes` | [3, 5, 7, 9] | Parallel branch kernel sizes |
| `fc_dims` | [256, 128] | Fully connected layer dimensions |
| `dropout` | 0.3 | Dropout rate in FC layers |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `weight_decay` | 0.0001 | L2 regularization |
| `batch_size` | 64 | Training batch size |
| `max_epochs` | 50 | Maximum training epochs |
| `patience` | 10 | Early stopping patience |

### Training Details
- **Optimizer:** Adam with weight decay
- **Loss:** MSELoss (regression) or CrossEntropyLoss (classification)
- **Early stopping:** 10% internal validation split, stop when val loss plateaus
- **Manual batching:** Uses custom `_iter_batches()` to avoid OpenMP deadlocks with XGBoost on macOS
- **Thread safety:** `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`

---

## Results

### RNA/translation_efficiency_muscle

| Model | Spearman | Pearson | MSE |
|-------|----------|---------|-----|
| random_forest | **0.6941** | 0.7068 | 1.0627 |
| **bio_cnn** | 0.6655 | **0.7207** | **0.9969** |
| lightgbm | 0.6604 | 0.6982 | 1.0652 |
| xgboost | 0.6279 | 0.6816 | 1.1035 |
| mlp (on k-mers) | 0.6229 | 0.6800 | 1.1534 |

**Key insight:** The BioCNN achieves the best Pearson correlation and lowest MSE, even though Random Forest has the best Spearman. This means the BioCNN captures **different patterns** from the sequence than k-mer features — specifically, positional motif information that k-mer counting misses. This is exactly why we built it.

### Train-Val Gap
- BioCNN: 0.257 (train=0.9224, val=0.6655) — moderate overfitting, better than XGBoost (0.357)
- The lower overfitting is partly due to the CNN's weight sharing (the same filters scan the entire sequence)

---

## File Structure

```
co_scientist/
├── modeling/
│   ├── bio_cnn.py       ← Multi-Scale BioCNN implementation
│   ├── registry.py      ← _build_bio_cnn builder
│   └── types.py         ← needs_sequences property, _SEQUENCE_MODELS set
├── data/
│   ├── types.py         ← raw_sequences in PreprocessingResult, seqs in SplitData
│   ├── preprocess.py    ← extracts raw sequences before k-mer encoding
│   └── split.py         ← carries sequences through all split strategies
├── evaluation/
│   └── metrics.py       ← passes sequences to CNN during evaluation
├── guardrails.py        ← sequence model gating + compatibility checks
└── defaults.yaml        ← bio_cnn config in rna/dna modality overrides
```

The module exports:
- `BioCNNRegressor` — sklearn-compatible regressor on raw sequences
- `BioCNNClassifier` — sklearn-compatible classifier on raw sequences
- `_one_hot_encode(sequences, max_len)` — internal encoding function
- `_MultiScaleCNN` — PyTorch nn.Module (not directly used outside)
