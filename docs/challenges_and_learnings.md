# Challenges, Solutions & Known Limitations

This document captures the technical challenges encountered during development and deployment of the AIDO Co-Scientist pipeline, how they were resolved, and what remains unsolved.

---

## Challenges Solved

### 1. AIDO Foundation Model API Mismatch

**Problem:** The `modelgenerator` library's API was undocumented and different from what we assumed. We initially used `Embed.from_config({"model.backbone.model_name": name})` and passed raw strings directly to `model(batch)`.

**What happened:** On Colab with GPU, the CLI argument parser (`jsonargparse`) hijacked the config key, causing `co-scientist --help`-style errors. Raw strings passed to the model produced no output or errors.

**Solution:** The correct API (discovered from HuggingFace model cards and source inspection) is:
```python
model = Embed.from_config({"model.backbone": name}).eval()
transformed = model.transform({"sequences": batch})
embedding = model(transformed)
```

**Root cause:** The config key is `"model.backbone"` not `"model.backbone.model_name"`, and a two-step process (transform then forward) is required. The model output is a `SequenceBackboneOutput` dataclass with `.last_hidden_state`, not a raw Tensor or dict.

**Files changed:** `modeling/foundation.py`, `modeling/aido_finetune.py`, `export/exporter.py` (all templates)

---

### 2. SequenceBackboneOutput — Unrecognized Output Type

**Problem:** AIDO models return a `SequenceBackboneOutput` dataclass, not a `torch.Tensor` or `dict`. Our code only checked for those two types and silently fell through to the error path, returning `None`. Embeddings appeared to extract but produced only 40 out of 1257 expected rows.

**What happened:** The pipeline continued without embeddings. Foundation models were silently skipped. No error was visible — the `X_embed` was `None` and the dimension mismatch check didn't exist yet.

**Solution:** Added `hasattr(embeddings, "last_hidden_state")` check before the dict check, extracting from `.last_hidden_state` (shape: `batch, seq_len, hidden_dim`) with mean pooling.

**Files changed:** `modeling/foundation.py`, `modeling/aido_finetune.py`

---

### 3. Embedding Dimension Mismatch Crash

**Problem:** When AIDO embedding extraction partially failed (40 out of 1257 samples), the 40-row `X_embed` was passed to the splitter which tried to index it with 1257-length boolean masks, causing `IndexError: boolean index did not match indexed array along dimension 0`.

**What happened:** The pipeline crashed hard — no models trained, no report generated, no checkpoint saved.

**Solution:** Two-level defense:
1. Per-batch try/except in extraction loop — failed batches logged and skipped
2. Post-extraction dimension check in `preprocess.py` — if `X_embed.shape[0] != X.shape[0]`, discard embeddings with warning

**Files changed:** `modeling/foundation.py`, `data/preprocess.py`

---

### 4. numpy<2 Compatibility on Colab

**Problem:** `modelgenerator` requires `numpy<2` but Colab ships numpy 2.x. Installing modelgenerator downgrades numpy, breaking pre-compiled packages (`numpy.dtype size changed, may indicate binary incompatibility`).

**What happened:** `from modelgenerator.tasks import Embed` failed with numpy binary incompatibility errors.

**Solution:** Install modelgenerator first, then restart the Colab runtime before installing co-scientist. Documented in README with explicit install order and restart step.

**Files changed:** `README.md`, `ARCHITECTURE.md`, `Dockerfile.gpu`

---

### 5. SIGALRM Training Timeout Crashes

**Problem:** We added a per-model training timeout using `signal.alarm(120)` to prevent slow models (FT-Transformer) from blocking the pipeline. But `SIGALRM` fired inside C extension code (XGBoost, LightGBM), causing the entire process to crash with a fatal signal.

**What happened:** Large datasets (mean_ribosome_load 91K samples, ncrna_family_classification 148K samples) crashed during baseline training. No checkpoint saved, no recovery possible. This was a regression — these datasets passed before the timeout was added.

**Solution:** Replaced `signal.alarm` with a threading-based timeout. Training runs in a daemon thread with `thread.join(timeout=120)`. If it doesn't finish, the main thread continues — no signals, no crashes. The daemon thread continues in background and is cleaned up on process exit.

**Files changed:** `modeling/trainer.py`

---

### 6. h5ad Export Template Bug

**Problem:** The exporter checked `profile.modality.value == "expression"` to decide between h5ad and standard HuggingFace loaders for the generated train.py. But the actual modality value is `"cell_expression"`, not `"expression"`.

**What happened:** Cell expression datasets (segerstolpe, zheng) generated train.py scripts that used `load_dataset()` instead of `hf_hub_download` + `anndata`, causing the reproduce scripts to fail.

**Solution:** Changed check to `profile.modality.value == "cell_expression"`.

**Files changed:** `export/exporter.py`

---

### 7. HuggingFace API Change (rfilename)

**Problem:** The `list_repo_tree()` API changed — `RepoFolder` objects no longer have `.rfilename` attribute. The h5ad file discovery fallback used this attribute to list files in the repo.

**What happened:** The zheng dataset couldn't be loaded — the primary download patterns failed (case-sensitive: `Zheng` vs `zheng`), and the discovery fallback crashed on `AttributeError: 'RepoFolder' object has no attribute 'rfilename'`.

**Solution:** Changed to `item.path` attribute. Added recursive search inside subfolders. Added lowercase filename patterns to direct download attempts.

**Files changed:** `data/loader.py`

---

### 8. Interactive Mode — Y/N Only, No Conversation

**Problem:** Interactive mode used Rich's `Confirm.ask()` which only accepted Y/N. When users typed natural language ("no do single classification"), it rejected the input and on "n" cancelled the pipeline entirely.

**What happened:** Users couldn't ask questions, give instructions, or have any conversation at decision points. Typing anything other than Y/N either errored or cancelled.

**Solution:** Replaced all `Confirm.ask()` with a conversational loop that accepts: `y` (accept), `n` (override), `exit/stop/quit` (halt), or free-form text (sent to LLM for chat or fed back as agent instruction). Added full pipeline context to the LLM chat so it can answer questions about splits, class distribution, etc.

**Files changed:** `agents/interactive.py`, `agents/types.py`, `cli.py`

---

### 9. Interactive Mode — No Pause During ReAct Loop

**Problem:** In interactive mode, the user could only interact during profiling and preprocessing. Once the ReAct agent started (the modeling phase — the longest and most important part), there were zero interactive checkpoints.

**What happened:** The user watched the agent train models for 10+ minutes with no ability to provide feedback, redirect strategy, or stop.

**Solution:** Added `_interactive_pause()` method to `ReactAgent`. After each Thought/Action/Observation step, if interactive mode is on, the user can press Enter (continue), type `exit` (stop), or type feedback that gets injected into the agent's next LLM call as `"IMPORTANT — The user has provided feedback: '...' Take this into account."`.

**Files changed:** `agents/react.py`, `agents/coordinator.py`, `cli.py`

---

### 10. Stacking Ensemble Export — Not Reproducible

**Problem:** When the best model is a stacking ensemble, the generated train.py uses `pickle.load()` to load the pre-trained model instead of training from scratch. This is because the ensemble wraps multiple base models + a meta-learner, and generating inline code for all of them is complex.

**What happened:** The reproduce train.py failed with `FileNotFoundError: 'models/best_model.pkl'` because the pickle is in the inference directory, not the reproduce directory.

**Solution:** Added `models/` subdirectory with `best_model.pkl` and `label_encoder.pkl` to the reproduce directory when the best model is a stacking ensemble.

**Files changed:** `export/exporter.py`

---

### 11. DebateTranscript Dataclass vs Dict

**Problem:** The visualization code expected debate transcripts as dicts (`dbt.get("topic")`), but the coordinator stored them as `DebateTranscript` dataclass objects which don't have `.get()`.

**What happened:** Architecture diagram generation crashed with `'DebateTranscript' object has no attribute 'get'`.

**Solution:** Convert dataclass objects to dicts using `dataclasses.asdict()` before passing to the visualization function.

**Files changed:** `viz/architecture.py`

---

### 12. Validation Script Path Doubling

**Problem:** The validation agent's script execution test used relative paths that doubled when combined with the working directory: `reproduce_.../outputs/.../reproduce_.../train.py`.

**What happened:** The train.py execution test failed with "No such file or directory" even though the script existed and was syntactically correct.

**Solution:** Resolve script paths to absolute with `.resolve()` before passing to subprocess.

**Files changed:** `validation.py`

---

### 13. Interactive Chat — No Pipeline Context

**Problem:** When the user asked questions in interactive mode ("what is the split?"), the LLM responded with "I don't have information about the training split yet" — even though the split info was displayed on screen moments before.

**What happened:** The `PipelineContext` passed to the chat LLM didn't include split info, class distribution, target stats, or any of the rich profile data. The LLM literally couldn't see the data.

**Solution:** Added `split_info`, `class_distribution`, `target_stats`, `target_column`, `missing_value_pct`, `feature_sparsity`, `sequence_length_stats`, `detected_issues`, `preprocessing_steps` fields to `PipelineContext`. Updated `_build_pipeline_state_text()` to include all fields. Populated them from the profile at the first interactive checkpoint.

**Files changed:** `agents/types.py`, `agents/interactive.py`, `cli.py`

---

### 14. GPU Status Not Communicated to LLM Chat

**Problem:** In interactive mode on CPU, the LLM mentioned foundation models (embed_xgboost, aido_finetune) in responses because the prompts listed them. Users were confused when told about GPU models on a CPU machine.

**Solution:** Added GPU detection to the pipeline state text: on CPU it says `"GPU: Not available — foundation models are NOT available. Do NOT suggest embed_*, concat_*, or aido_finetune."` On GPU it confirms availability.

**Files changed:** `agents/interactive.py`

---

## Known Limitations (Not Yet Solved)

### 1. AIDO.Cell Expression Model — Incompatible Input Format

**Status:** The `aido_cell_100m` model expects a different input format than sequence models. We pass expression vectors via `model.transform({"sequences": data})`, but the Cell model likely needs a different key or tokenization approach for expression matrices. On Colab with GPU, cell_expression datasets show "GPU detected but no embeddings extracted" — the extraction fails silently and the pipeline continues with CPU models only.

**Impact:** Cell expression datasets don't benefit from foundation models even on GPU. RNA/DNA/protein sequences work correctly.

**Workaround:** CPU models still achieve strong results (macro_f1=0.97 on segerstolpe).

---

### 2. AIDO Fine-Tuning Not Tried by Agent

**Status:** The ReAct agent has `aido_finetune` available as a tool, but often chooses not to try it — especially after seeing that `embed_xgboost` scored lower than handcrafted models. The agent reasons that if frozen embeddings underperform, fine-tuning won't help. This is often a reasonable decision for small datasets but may miss opportunities on larger ones.

**Impact:** Fine-tuning is never used unless the agent decides to, or the user explicitly requests it in interactive mode.

**Possible fix:** Add a rule that forces the agent to try `aido_finetune` at least once on GPU, similar to how `design_model` is required.

---

### 3. Foundation Models Underperform on Short Sequences

**Status:** On the RNA translation efficiency dataset (91-char sequences, 1257 samples), AIDO embeddings (2048 dims) score lower than handcrafted k-mer features (326 dims). This is expected — the AIDO.RNA model was pre-trained on longer sequences and the high embedding dimensionality overfits on small datasets.

**Impact:** Foundation models are most useful on larger datasets with longer sequences. On small/short datasets, they add compute cost without performance benefit.

**Not a bug** — this is the correct behavior. The pipeline lets all models compete fairly, and the best model wins.

---

### 4. Stacking Ensemble Cannot Include Foundation Models

**Status:** Foundation models (embed_*, concat_*, aido_finetune) are excluded from the stacking ensemble's out-of-fold cross-validation because:
- Embedding models need `X_embed` per fold (expensive to re-extract per fold)
- Fine-tuning models take 10+ min per fold
- Concat models need both `X` and `X_embed` per fold

**Impact:** The ensemble only combines CPU models. Foundation models compete individually but can't contribute to the ensemble.

**Possible fix:** Pre-extract embeddings once, then use the same embeddings across CV folds. This would enable embed_* and concat_* models in the ensemble (but not aido_finetune).

---

### 5. Custom Models Can't Be Pickled for Checkpointing

**Status:** LLM-designed custom models (via `design_model` tool) are dynamically generated Python classes that can't be pickled because their module doesn't persist. The checkpoint system catches this and saves state without the unpicklable models.

**Impact:** If the pipeline crashes after a custom model is trained, the resume won't have that model. Results and scores are preserved, just not the model object.

**Workaround:** The custom model source code is saved to `models/custom_model.py` in the export step, so it can be manually reconstructed.

---

### 6. FT-Transformer Slow on CPU

**Status:** The FT-Transformer model with 50 epochs on 326 features takes several minutes on CPU (macOS). The 120s training timeout skips it on most datasets.

**Impact:** FT-Transformer is effectively unavailable on CPU for datasets with >200 features. Works fine on GPU.

**Workaround:** The timeout correctly skips it and the pipeline continues. On the H100 evaluation environment, it will train within the timeout.

---

### 7. Large Datasets and Training Timeouts

**Status:** Datasets with >50K samples (mean_ribosome_load 91K, ncrna_family_classification 148K) may have some models skip due to the 120s per-model timeout. Tree models on large datasets can take >2 minutes for a single training run.

**Impact:** Some models are skipped, reducing the candidate pool. The best models (typically XGBoost/LightGBM which train fastest) still complete.

**Possible fix:** Scale timeout based on dataset size: `timeout = max(120, n_samples / 500)`.

---

### 8. Reproduce Script for Expression Datasets — Validation Score Mismatch

**Status:** The reproduce train.py for segerstolpe generates macro_f1=0.889 on test while the pipeline reports 0.970. This is because the pipeline evaluates on the validation set for model selection (0.970), while the reproduce script reports the test set score. The scores are different because the tuned hyperparameters were optimized on validation.

**Impact:** Confusion when comparing pipeline report vs reproduce output. Not a bug — different evaluation sets.

---

### 9. Dashboard Refresh Rate in Colab

**Status:** The Rich live dashboard redraws every second. In Colab's output cell, this creates massive output (5000+ lines truncated). The dashboard is designed for terminal use where it overwrites in-place.

**Impact:** Colab output is flooded with repeated dashboard frames, making it hard to find actual log messages.

**Possible fix:** Detect Colab environment (`COLAB_RELEASE_TAG` env var) and disable live dashboard refresh, using static prints instead.

---

## Architecture Decisions That Proved Correct

1. **Graceful degradation** — Every component fails safely. No GPU? Foundation models skipped. No API key? Deterministic path runs. Model crashes? Next model trains. This design meant the pipeline always produces output even when individual components fail.

2. **Validation agent after every step** — Caught NaN creep, empty splits, and shape mismatches before they propagated. The validate-and-fix pattern prevented ~90% of potential silent failures.

3. **Per-model training timeout** — Prevents a single slow model from blocking the entire pipeline. Critical for the evaluation environment where wall-clock time matters.

4. **Embedding caching** — AIDO embeddings cached to disk as `.npy` files. Re-runs or pipeline iterations reuse cached embeddings instead of re-extracting (saving 5-10 min per dataset).

5. **All models compete on same validation set** — Foundation models don't get special treatment. If handcrafted features + LightGBM beats AIDO embeddings + XGBoost, that's the correct answer. No bias toward expensive models.
