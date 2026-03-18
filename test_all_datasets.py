#!/usr/bin/env python3
"""End-to-end QC script — runs co-scientist on every dataset, then generates a
detailed inspection report covering every artifact the pipeline should produce.

Usage:
    python test_all_datasets.py                              # run all, sequential
    python test_all_datasets.py --quick                      # deterministic (no LLM)
    python test_all_datasets.py --filter RNA                 # only RNA datasets
    python test_all_datasets.py --filter splice_site         # single dataset
    python test_all_datasets.py --parallel 2                 # parallel (optional)
    python test_all_datasets.py --inspect-only               # skip runs, just inspect existing outputs
    python test_all_datasets.py --dry-run                    # print plan and exit
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ── Python interpreter that has the packages installed ──────────────────────
# co-scientist is installed under python3.10, so use the same one for scripts.
_CO_SCIENTIST_BIN = subprocess.run(
    ["which", "co-scientist"], capture_output=True, text=True,
).stdout.strip()
if _CO_SCIENTIST_BIN and Path(_CO_SCIENTIST_BIN).exists():
    _shebang = Path(_CO_SCIENTIST_BIN).read_text().split("\n")[0]
    PYTHON = _shebang.replace("#!", "").strip() if _shebang.startswith("#!") else sys.executable
else:
    PYTHON = sys.executable


# ── All supported datasets ──────────────────────────────────────────────────

DATASETS = [
    # RNA — regression
    "RNA/translation_efficiency_muscle",
    "RNA/translation_efficiency_hek",
    "RNA/translation_efficiency_pc3",
    "RNA/expression_muscle",
    "RNA/expression_hek",
    "RNA/expression_pc3",
    "RNA/mean_ribosome_load",
    # RNA — classification
    "RNA/splice_site_prediction",
    "RNA/ncrna_family_classification",
    # Expression — classification (h5ad)
    "expression/cell_type_classification_segerstolpe",
    "expression/cell_type_classification_zheng",
]


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Issue:
    severity: str   # "error", "warning", "info"
    category: str   # "missing_file", "syntax", "metric", "validation", etc.
    message: str

@dataclass
class DatasetReport:
    dataset: str
    output_dir: str = ""
    run_success: bool = False
    run_exit_code: int = -1
    run_error: str = ""
    run_elapsed: float = 0.0

    # Pipeline completeness
    steps_completed: list = field(default_factory=list)
    steps_expected: list = field(default_factory=lambda: [
        "load_profile", "preprocess_split", "baselines",
        "hp_search", "iteration", "export", "report",
    ])

    # ── Artifacts (file existence) ──
    has_report: bool = False
    has_summary_pdf: bool = False
    has_experiment_log: bool = False
    has_checkpoint: bool = False
    has_agent_log: bool = False
    has_top_level_requirements: bool = False
    # models/
    has_model_pkl: bool = False
    has_model_config: bool = False
    model_pkl_size_kb: float = 0.0
    # inference package
    has_inference_dir: bool = False
    has_inference_model: bool = False
    has_inference_config: bool = False
    has_predict_py: bool = False
    has_inference_requirements: bool = False
    # reproduce package
    has_reproduce_dir: bool = False
    has_train_py: bool = False
    has_evaluate_py: bool = False
    has_reproduce_requirements: bool = False
    has_reproduce_model: bool = False
    has_label_encoder: bool = False
    # code/
    has_code_dir: bool = False
    code_files: list = field(default_factory=list)
    # figures
    has_figures: bool = False
    num_figures: int = 0
    figure_subdirs: list = field(default_factory=list)

    # ── Script quality ──
    train_py_compiles: bool = False
    predict_py_compiles: bool = False
    evaluate_py_compiles: bool = False
    train_py_runs: bool = False
    train_py_run_error: str = ""

    # ── Model quality ──
    best_model: str = ""
    primary_metric: str = ""
    primary_value: float = 0.0
    all_metrics: dict = field(default_factory=dict)
    num_models_trained: int = 0
    model_type: str = ""
    task_type: str = ""
    modality: str = ""
    num_samples: int = 0
    num_features: int = 0
    num_classes: int = 0
    hyperparameters: dict = field(default_factory=dict)
    preprocessing_steps: list = field(default_factory=list)
    feature_names_count: int = 0
    model_loads: bool = False
    model_predicts: bool = False

    # ── Cross-file consistency ──
    best_model_matches: bool = True   # same across checkpoint, config, log
    metric_matches: bool = True       # same across config and log
    feature_count_matches: bool = True  # config vs actual

    # ── Report quality ──
    report_sections_found: list = field(default_factory=list)
    report_sections_missing: list = field(default_factory=list)
    report_broken_images: list = field(default_factory=list)
    report_best_model_mentioned: bool = False
    report_metric_mentioned: bool = False

    # ── Experiment log details ──
    log_total_events: int = 0
    log_react_steps: int = 0
    log_total_elapsed: float = 0.0

    # ── Validation agent activity ──
    validation_fixes: list = field(default_factory=list)
    validation_failures: list = field(default_factory=list)

    # ── Debates ──
    num_debates: int = 0
    debate_fallbacks: int = 0
    debate_topics: list = field(default_factory=list)

    # ── LLM cost ──
    llm_cost: float = 0.0
    llm_calls: int = 0

    # ── Issues ──

    # Issues found during inspection
    issues: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.run_success and not any(i.severity == "error" for i in self.issues)


# ── Pipeline runner ─────────────────────────────────────────────────────────

def run_dataset(dataset: str, quick: bool = False, budget: int = 5, max_cost: float = 2.0) -> float:
    """Run co-scientist on a dataset. Returns elapsed seconds."""
    cmd = ["co-scientist", "run", dataset, "--budget", str(budget)]
    if quick:
        cmd += ["--max-cost", "0"]
    else:
        cmd += ["--max-cost", str(max_cost)]

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"\n{'='*70}")
    print(f"  RUNNING: {dataset}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env,
        )
        elapsed = time.time() - start
        if proc.returncode != 0:
            print(f"  \033[91mFAILED (exit {proc.returncode})\033[0m")
            stderr_tail = "\n".join((proc.stderr or "").strip().split("\n")[-8:])
            if stderr_tail:
                print(f"  stderr:\n{stderr_tail}")
        else:
            print(f"  \033[92mOK\033[0m ({elapsed:.0f}s)")
        return elapsed
    except subprocess.TimeoutExpired:
        print(f"  \033[91mTIMEOUT (30 min)\033[0m")
        return 1800.0
    except Exception as e:
        print(f"  \033[91mERROR: {e}\033[0m")
        return time.time() - start


# ── Deep inspection ─────────────────────────────────────────────────────────

def find_output_dir(dataset: str) -> Path | None:
    """Find the most recent output directory for a dataset."""
    ds_slug = dataset.replace("/", "__")
    outputs = Path("outputs")
    if not outputs.exists():
        return None
    matches = sorted(
        outputs.glob(f"{ds_slug}_*"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return matches[0] if matches else None


def _detect_ds_name(out: Path, dataset: str) -> str:
    """Detect the actual dataset name used in reproduce_*/inference_* subdirectories."""
    # Look for reproduce_* dirs
    reproduce_dirs = list(out.glob("reproduce_*"))
    if reproduce_dirs:
        name = reproduce_dirs[0].name.replace("reproduce_", "", 1)
        return name

    inference_dirs = list(out.glob("inference_*"))
    if inference_dirs:
        name = inference_dirs[0].name.replace("inference_", "", 1)
        return name

    # Fallback: try common patterns
    # "RNA/splice_site_prediction" -> try "splice_site_prediction" first, then "RNA_splice_site_prediction"
    parts = dataset.split("/")
    if len(parts) == 2:
        # Task name only (most common)
        candidate = parts[1].replace(" ", "_")
        if (out / f"reproduce_{candidate}").exists():
            return candidate
        # Full path with modality
        candidate = dataset.replace("/", "_").replace(" ", "_")
        if (out / f"reproduce_{candidate}").exists():
            return candidate
        # Return task name as best guess
        return parts[1].replace(" ", "_")

    return dataset.replace("/", "_").replace(" ", "_")


def inspect_dataset(dataset: str) -> DatasetReport:
    """Deep-inspect the output directory for a dataset. No running — just checks."""
    r = DatasetReport(dataset=dataset)
    out = find_output_dir(dataset)
    if out is None:
        r.issues.append(Issue("error", "missing_output", "No output directory found"))
        return r
    r.output_dir = str(out)
    out = out.resolve()
    ds_name = _detect_ds_name(out, dataset)

    reproduce_dir = out / f"reproduce_{ds_name}"
    inference_dir = out / f"inference_{ds_name}"

    # ── 1. Top-level files ──
    _check_top_level(out, r)

    # ── 2. Checkpoint meta — pipeline completeness ──
    _check_checkpoint(out, r)

    # ── 3. Experiment log — events, costs, debates, validation ──
    _check_experiment_log(out, r)

    # ── 4. Report — sections, images, mentions ──
    _check_report(out, r)

    # ── 5. Models directory (top-level models/) ──
    _check_models_dir(out, r)

    # ── 6. Inference package ──
    _check_inference_package(inference_dir, r)

    # ── 7. Reproduce package ──
    _check_reproduce_package(reproduce_dir, r)

    # ── 8. Code directory ──
    _check_code_dir(out, r)

    # ── 9. Figures ──
    _check_figures(out, r)

    # ── 10. Model loading + prediction test ──
    _check_model_loads(out, r)

    # ── 11. Script execution ──
    _check_scripts_run(reproduce_dir, inference_dir, r)

    # ── 12. Cross-file consistency ──
    _check_consistency(r)

    # ── 13. Metric sanity ──
    _check_metrics(r)

    return r


# ── Inspection helpers ──────────────────────────────────────────────────────

def _check_top_level(out: Path, r: DatasetReport) -> None:
    """Check top-level files: report.md, summary.pdf, requirements.txt, agent_conversations.log."""
    if (out / "report.md").exists():
        r.has_report = True
    else:
        r.issues.append(Issue("error", "missing_file", "Missing report.md"))

    if (out / "summary.pdf").exists():
        r.has_summary_pdf = True
    else:
        r.issues.append(Issue("warning", "missing_file", "Missing summary.pdf"))

    if (out / "requirements.txt").exists():
        r.has_top_level_requirements = True
        content = (out / "requirements.txt").read_text().strip()
        if not content:
            r.issues.append(Issue("warning", "empty_file", "Top-level requirements.txt is empty"))
    else:
        r.issues.append(Issue("warning", "missing_file", "Missing top-level requirements.txt"))

    if (out / "agent_conversations.log").exists():
        r.has_agent_log = True
        size = (out / "agent_conversations.log").stat().st_size
        if size < 50:
            r.issues.append(Issue("warning", "empty_file", "agent_conversations.log is nearly empty"))
    else:
        r.issues.append(Issue("info", "missing_file", "Missing agent_conversations.log (no LLM mode?)"))


def _check_checkpoint(out: Path, r: DatasetReport) -> None:
    """Check checkpoint_meta.json and pipeline completeness."""
    meta_path = out / "logs" / "checkpoint_meta.json"
    pkl_path = out / "logs" / "checkpoint.pkl"

    if pkl_path.exists():
        r.has_checkpoint = True
    else:
        r.issues.append(Issue("warning", "missing_file", "Missing logs/checkpoint.pkl"))

    if not meta_path.exists():
        r.issues.append(Issue("error", "missing_file", "Missing logs/checkpoint_meta.json"))
        return

    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        r.issues.append(Issue("error", "corrupt_file", "checkpoint_meta.json is invalid JSON"))
        return

    r.steps_completed = meta.get("completed_steps", [])
    r.num_models_trained = meta.get("n_models_trained", 0)
    r.best_model = meta.get("best_model", "")

    missing = [s for s in r.steps_expected if s not in r.steps_completed]
    if missing:
        r.run_success = False
        r.issues.append(Issue("error", "incomplete_pipeline", f"Steps not completed: {', '.join(missing)}"))
    else:
        r.run_success = True


def _check_experiment_log(out: Path, r: DatasetReport) -> None:
    """Parse experiment_log.jsonl — costs, debates, validation, react steps."""
    log_path = out / "logs" / "experiment_log.jsonl"
    if not log_path.exists():
        r.has_experiment_log = False
        r.issues.append(Issue("error", "missing_file", "Missing logs/experiment_log.jsonl"))
        return
    r.has_experiment_log = True

    events = []
    for line in log_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            r.issues.append(Issue("warning", "corrupt_log", "Invalid JSON line in experiment_log.jsonl"))
    r.log_total_events = len(events)

    # Track best model from different sources for consistency check
    log_best_model = ""
    log_best_metric = ""
    log_best_value = 0.0

    for ev in events:
        event_type = ev.get("event", "")
        data = ev.get("data", {})

        if event_type == "pipeline_start":
            pass  # good, exists

        elif event_type == "pipeline_complete":
            log_best_metric = data.get("best_metric", "")
            log_best_value = data.get("best_value", 0.0)
            log_best_model = data.get("best_model", "")
            r.log_total_elapsed = data.get("total_elapsed_seconds", 0.0)

        elif event_type == "llm_costs":
            r.llm_cost = data.get("total_cost", 0.0)
            r.llm_calls = data.get("num_calls", 0)

        elif event_type == "react_step":
            r.log_react_steps += 1

        elif event_type == "test_evaluation":
            # Final held-out test metrics
            pass

        elif event_type == "validation_failure":
            r.validation_failures.extend(data.get("issues", []))

        elif event_type == "validation_fix":
            r.validation_fixes.extend(data.get("fixes", []))

        elif event_type == "agent_decision":
            stage = data.get("stage", "")
            agent = data.get("agent", "")
            # Count debates (logged as agent="debate" or stage="pre_react_strategy")
            if agent == "debate" or stage == "pre_react_strategy":
                r.num_debates += 1
                topic = stage if stage == "pre_react_strategy" else data.get("action", "")
                r.debate_topics.append(topic)

    # Set from log if not already set
    r.primary_metric = r.primary_metric or log_best_metric
    r.primary_value = r.primary_value or log_best_value
    r.best_model = r.best_model or log_best_model

    # Store for consistency check
    r._log_best_model = log_best_model
    r._log_best_value = log_best_value
    r._log_best_metric = log_best_metric

    event_types = {ev.get("event") for ev in events}
    if "pipeline_start" not in event_types:
        r.issues.append(Issue("warning", "log_incomplete", "No pipeline_start event in log"))
    if "pipeline_complete" not in event_types:
        r.issues.append(Issue("error", "log_incomplete", "No pipeline_complete event — pipeline may have crashed"))


def _check_report(out: Path, r: DatasetReport) -> None:
    """Check report.md structure, sections, images, mentions."""
    report_path = out / "report.md"
    if not report_path.exists():
        return

    text = report_path.read_text()

    expected_sections = [
        ("Executive Summary", "## 1"),
        ("Dataset Profile", "## 2"),
        ("Preprocessing", "## 3"),
        ("Model Development", "## 4"),
    ]
    for name, marker in expected_sections:
        if marker in text:
            r.report_sections_found.append(name)
        else:
            r.report_sections_missing.append(name)
            r.issues.append(Issue("warning", "report_section", f"report.md missing section: {name}"))

    # Broken image references
    import re
    image_refs = re.findall(r'!\[.*?\]\((.*?)\)', text)
    for ref in image_refs:
        img_path = out / ref
        if not img_path.exists():
            r.report_broken_images.append(ref)
            r.issues.append(Issue("warning", "broken_image", f"report.md references missing image: {ref}"))

    # Does report mention the best model?
    if r.best_model and r.best_model in text:
        r.report_best_model_mentioned = True
    elif r.best_model:
        r.issues.append(Issue("warning", "report_content", f"report.md does not mention best model '{r.best_model}'"))

    # Does report mention the primary metric value?
    if r.primary_value:
        val_str = f"{r.primary_value:.4f}"
        # Also check truncated versions
        if val_str in text or f"{r.primary_value:.3f}" in text or f"{r.primary_value:.2f}" in text:
            r.report_metric_mentioned = True
        else:
            r.issues.append(Issue("warning", "report_content",
                f"report.md does not contain primary metric value {val_str}"))

    # Report file size sanity
    size_kb = len(text.encode()) / 1024
    if size_kb < 1:
        r.issues.append(Issue("warning", "report_content", f"report.md is suspiciously small ({size_kb:.1f} KB)"))


def _check_models_dir(out: Path, r: DatasetReport) -> None:
    """Check top-level models/ directory."""
    models_dir = out / "models"
    if not models_dir.exists():
        r.issues.append(Issue("error", "missing_file", "Missing models/ directory"))
        return

    pkl = models_dir / "best_model.pkl"
    cfg = models_dir / "model_config.json"

    if pkl.exists():
        r.has_model_pkl = True
        r.model_pkl_size_kb = pkl.stat().st_size / 1024
        if r.model_pkl_size_kb < 0.1:
            r.issues.append(Issue("warning", "model", f"best_model.pkl is very small ({r.model_pkl_size_kb:.1f} KB)"))
    else:
        r.issues.append(Issue("error", "missing_file", "Missing models/best_model.pkl"))

    if cfg.exists():
        r.has_model_config = True
        _parse_model_config(cfg, r)
    else:
        r.issues.append(Issue("error", "missing_file", "Missing models/model_config.json"))


def _parse_model_config(config_path: Path, r: DatasetReport) -> None:
    """Parse model_config.json and populate report fields."""
    try:
        cfg = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        r.issues.append(Issue("error", "corrupt_file", f"{config_path.parent.name}/model_config.json is invalid JSON"))
        return

    r.model_type = cfg.get("model_type", "")
    r.task_type = r.task_type or cfg.get("dataset", {}).get("task_type", "")
    r.modality = r.modality or cfg.get("dataset", {}).get("modality", "")
    r.num_samples = r.num_samples or cfg.get("dataset", {}).get("num_samples", 0)
    r.num_features = r.num_features or cfg.get("dataset", {}).get("num_features", 0)
    r.num_classes = cfg.get("dataset", {}).get("num_classes", 0) or 0
    r.hyperparameters = cfg.get("hyperparameters", {})
    r.preprocessing_steps = cfg.get("preprocessing_steps", [])
    r.feature_names_count = len(cfg.get("feature_names", []))

    # Store config's best model name for consistency check
    r._config_best_model = cfg.get("model_name", "")

    required_keys = ["model_name", "model_type", "hyperparameters", "dataset", "evaluation"]
    for key in required_keys:
        if key not in cfg:
            r.issues.append(Issue("error", "model_config", f"model_config.json missing key: {key}"))

    eval_section = cfg.get("evaluation", {})
    if eval_section:
        r._config_metric = eval_section.get("primary_metric", "")
        r._config_value = eval_section.get("primary_value", 0.0)
        r.primary_metric = r.primary_metric or r._config_metric
        r.primary_value = r.primary_value or r._config_value
        r.all_metrics = eval_section.get("all_metrics", {})
        if not r.all_metrics:
            r.issues.append(Issue("warning", "model_config", "model_config.json has no all_metrics"))
    else:
        r.issues.append(Issue("error", "model_config", "model_config.json missing evaluation section"))


def _check_inference_package(inference_dir: Path, r: DatasetReport) -> None:
    """Check inference package completeness."""
    if not inference_dir.exists():
        r.issues.append(Issue("error", "missing_file", f"Missing {inference_dir.name}/ directory"))
        return
    r.has_inference_dir = True

    model_dir = inference_dir / "model"
    if model_dir.exists():
        if (model_dir / "best_model.pkl").exists() or (model_dir / "best_model.pt").exists():
            r.has_inference_model = True
        else:
            r.issues.append(Issue("error", "missing_file", f"{inference_dir.name}/model/ has no model file"))

        if (model_dir / "model_config.json").exists():
            r.has_inference_config = True
        else:
            r.issues.append(Issue("error", "missing_file", f"{inference_dir.name}/model/model_config.json missing"))
    else:
        r.issues.append(Issue("error", "missing_file", f"{inference_dir.name}/model/ directory missing"))

    predict_path = inference_dir / "predict.py"
    if predict_path.exists():
        r.has_predict_py = True
        try:
            compile(predict_path.read_text(), str(predict_path), "exec")
            r.predict_py_compiles = True
        except SyntaxError as e:
            r.issues.append(Issue("error", "syntax", f"predict.py: SyntaxError line {e.lineno}: {e.msg}"))
    else:
        r.issues.append(Issue("error", "missing_file", f"{inference_dir.name}/predict.py missing"))

    req = inference_dir / "requirements.txt"
    if req.exists():
        r.has_inference_requirements = True
        if not req.read_text().strip():
            r.issues.append(Issue("warning", "empty_file", f"{inference_dir.name}/requirements.txt is empty"))
    else:
        r.issues.append(Issue("warning", "missing_file", f"{inference_dir.name}/requirements.txt missing"))


def _check_reproduce_package(reproduce_dir: Path, r: DatasetReport) -> None:
    """Check reproduce package completeness."""
    if not reproduce_dir.exists():
        r.issues.append(Issue("error", "missing_file", f"Missing {reproduce_dir.name}/ directory"))
        return
    r.has_reproduce_dir = True

    train_path = reproduce_dir / "train.py"
    if train_path.exists():
        r.has_train_py = True
        try:
            compile(train_path.read_text(), str(train_path), "exec")
            r.train_py_compiles = True
        except SyntaxError as e:
            r.issues.append(Issue("error", "syntax", f"train.py: SyntaxError line {e.lineno}: {e.msg}"))
    else:
        r.issues.append(Issue("error", "missing_file", f"{reproduce_dir.name}/train.py missing"))

    eval_path = reproduce_dir / "evaluate.py"
    if eval_path.exists():
        r.has_evaluate_py = True
        try:
            compile(eval_path.read_text(), str(eval_path), "exec")
            r.evaluate_py_compiles = True
        except SyntaxError as e:
            r.issues.append(Issue("error", "syntax", f"evaluate.py: SyntaxError line {e.lineno}: {e.msg}"))
    else:
        r.issues.append(Issue("warning", "missing_file", f"{reproduce_dir.name}/evaluate.py missing"))

    req = reproduce_dir / "requirements.txt"
    if req.exists():
        r.has_reproduce_requirements = True
        if not req.read_text().strip():
            r.issues.append(Issue("warning", "empty_file", f"{reproduce_dir.name}/requirements.txt is empty"))
    else:
        r.issues.append(Issue("warning", "missing_file", f"{reproduce_dir.name}/requirements.txt missing"))

    # Check models subdirectory in reproduce
    models_sub = reproduce_dir / "models"
    if models_sub.exists():
        if (models_sub / "best_model.pkl").exists():
            r.has_reproduce_model = True
        else:
            r.issues.append(Issue("warning", "missing_file", f"{reproduce_dir.name}/models/best_model.pkl missing"))
        if (models_sub / "label_encoder.pkl").exists():
            r.has_label_encoder = True
        elif "classification" in r.task_type:
            r.issues.append(Issue("warning", "missing_file",
                f"{reproduce_dir.name}/models/label_encoder.pkl missing (classification task)"))
    else:
        r.issues.append(Issue("warning", "missing_file", f"{reproduce_dir.name}/models/ directory missing"))


def _check_code_dir(out: Path, r: DatasetReport) -> None:
    """Check code/ directory (agent-generated scripts)."""
    code_dir = out / "code"
    if not code_dir.exists():
        r.issues.append(Issue("warning", "missing_file", "Missing code/ directory"))
        return
    r.has_code_dir = True
    r.code_files = [f.name for f in code_dir.iterdir() if f.is_file()]
    if not r.code_files:
        r.issues.append(Issue("warning", "empty_file", "code/ directory is empty"))


def _check_figures(out: Path, r: DatasetReport) -> None:
    """Check figures directory."""
    fig_dir = out / "figures"
    if not fig_dir.exists():
        r.issues.append(Issue("warning", "missing_figures", "No figures/ directory"))
        return

    pngs = list(fig_dir.rglob("*.png"))
    r.num_figures = len(pngs)
    r.has_figures = r.num_figures > 0
    r.figure_subdirs = [d.name for d in fig_dir.iterdir() if d.is_dir()]

    if r.num_figures == 0:
        r.issues.append(Issue("warning", "missing_figures", "figures/ directory has no PNGs"))

    # Expected figure subdirectories
    for expected in ["01_profiling", "02_preprocessing"]:
        if expected not in r.figure_subdirs:
            r.issues.append(Issue("warning", "missing_figures", f"figures/{expected}/ missing"))


def _check_model_loads(out: Path, r: DatasetReport) -> None:
    """Try to load the model pickle and make a prediction (in subprocess with correct python)."""
    pkl_path = out / "models" / "best_model.pkl"
    if not pkl_path.exists():
        return

    # Run model load + predict test in a subprocess using the correct python
    test_script = f"""
import pickle, sys, numpy as np
try:
    with open("{pkl_path}", "rb") as f:
        model = pickle.load(f)
    print("LOAD_OK")
    n_features = {r.num_features or 0}
    if n_features > 0:
        X = np.random.randn(5, n_features).astype(np.float32)
        preds = model.predict(X)
        if len(preds) == 5:
            print("PREDICT_OK")
        else:
            print(f"PREDICT_WRONG_LEN:{{len(preds)}}")
    else:
        print("PREDICT_SKIP")
except Exception as e:
    print(f"ERROR:{{type(e).__name__}}:{{e}}")
"""
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"

    try:
        result = subprocess.run(
            [PYTHON, "-c", test_script],
            capture_output=True, text=True, timeout=30, env=env,
        )
        output = result.stdout.strip()

        if "LOAD_OK" in output:
            r.model_loads = True
        if "PREDICT_OK" in output:
            r.model_predicts = True
        elif "PREDICT_WRONG_LEN" in output:
            r.model_loads = True
            count = output.split("PREDICT_WRONG_LEN:")[1].split("\n")[0]
            r.issues.append(Issue("warning", "model", f"predict() returned {count} values for 5 samples"))

        if "ERROR" in output:
            err_line = [l for l in output.split("\n") if l.startswith("ERROR:")][0]
            r.issues.append(Issue("error", "model", f"Model test failed: {err_line[6:]}"))

        if result.returncode != 0 and "LOAD_OK" not in output:
            stderr = result.stderr.strip().split("\n")[-3:]
            r.issues.append(Issue("error", "model", f"Model load subprocess failed: {' '.join(stderr)[:200]}"))

    except subprocess.TimeoutExpired:
        r.issues.append(Issue("warning", "model", "Model load test timed out (30s)"))
    except Exception as e:
        r.issues.append(Issue("error", "model", f"Model load test error: {e}"))


def _check_scripts_run(reproduce_dir: Path, inference_dir: Path, r: DatasetReport) -> None:
    """Actually execute train.py and check predict.py compiles."""
    train_path = reproduce_dir / "train.py"
    if train_path.exists() and r.train_py_compiles:
        ok, err = _test_run_script(train_path, timeout=300)
        r.train_py_runs = ok
        if not ok:
            r.train_py_run_error = err or "unknown"
            r.issues.append(Issue("error", "runtime", f"train.py execution failed: {(err or '')[:200]}"))


def _check_consistency(r: DatasetReport) -> None:
    """Cross-check data across checkpoint_meta, model_config, experiment_log."""
    # Best model name consistency
    names = set()
    if r.best_model:
        names.add(r.best_model)
    if hasattr(r, '_config_best_model') and r._config_best_model:
        names.add(r._config_best_model)
    if hasattr(r, '_log_best_model') and r._log_best_model:
        names.add(r._log_best_model)
    if len(names) > 1:
        r.best_model_matches = False
        r.issues.append(Issue("warning", "consistency",
            f"Best model name mismatch across files: {names}"))

    # Metric value consistency (config vs log)
    if hasattr(r, '_config_value') and hasattr(r, '_log_best_value'):
        cv = r._config_value or 0
        lv = r._log_best_value or 0
        if cv and lv and abs(cv - lv) > 0.001:
            r.metric_matches = False
            r.issues.append(Issue("warning", "consistency",
                f"Metric value mismatch: config={cv:.4f} vs log={lv:.4f}"))

    # Feature count: config vs feature_names
    if r.num_features and r.feature_names_count and r.num_features != r.feature_names_count:
        r.feature_count_matches = False
        r.issues.append(Issue("warning", "consistency",
            f"Feature count mismatch: dataset.num_features={r.num_features} vs feature_names={r.feature_names_count}"))


def _check_metrics(r: DatasetReport) -> None:
    """Sanity-check model metrics."""
    if r.primary_value == 0.0 and r.run_success:
        r.issues.append(Issue("warning", "metric", "Primary metric value is 0.0"))
        return

    if "classification" in r.task_type:
        if r.primary_metric in ("accuracy", "auroc", "f1", "macro_f1", "weighted_f1", "balanced_accuracy"):
            if not (0.0 <= r.primary_value <= 1.0):
                r.issues.append(Issue("error", "metric", f"{r.primary_metric}={r.primary_value:.4f} outside [0, 1]"))
            elif r.primary_value < 0.3:
                r.issues.append(Issue("warning", "metric", f"{r.primary_metric}={r.primary_value:.4f} suspiciously low"))

    if "regression" in r.task_type:
        if r.primary_metric == "spearman" and not (-1.0 <= r.primary_value <= 1.0):
            r.issues.append(Issue("error", "metric", f"spearman={r.primary_value:.4f} outside [-1, 1]"))

    if r.num_models_trained == 0 and r.run_success:
        r.issues.append(Issue("warning", "metric", "0 models trained despite successful run"))

    # Check all_metrics for NaN/Inf
    for name, val in r.all_metrics.items():
        if val is None:
            continue
        try:
            import math
            if math.isnan(val) or math.isinf(val):
                r.issues.append(Issue("error", "metric", f"Metric '{name}' is {val}"))
        except (TypeError, ValueError):
            pass


def _test_run_script(script_path: Path, timeout: int = 300) -> tuple:
    """Run a script in subprocess, return (success, error_message)."""
    script_path = script_path.resolve()
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        if result.returncode == 0:
            return True, None
        stderr = result.stderr.strip()
        lines = stderr.split("\n")
        err = "\n".join(lines[-5:]) if lines else f"Exit code {result.returncode}"
        return False, err[:500]
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except Exception as e:
        return False, str(e)[:500]


# ── QC Report generation ───────────────────────────────────────────────────

def generate_qc_report(reports: list[DatasetReport], output_path: Path) -> None:
    """Generate a detailed QC report in Markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    failed = total - passed

    lines = [
        f"# Co-Scientist QC Report",
        f"",
        f"**Date:** {now}",
        f"**Datasets tested:** {total}",
        f"**Passed:** {passed}  |  **Failed:** {failed}",
        f"",
        f"---",
        f"",
        f"## Summary Table",
        f"",
        f"| Dataset | Status | Models | Best Score | Metric | train.py | predict.py | Time | Issues |",
        f"|---------|--------|--------|------------|--------|----------|------------|------|--------|",
    ]

    for r in reports:
        status = "PASS" if r.passed else "**FAIL**"
        score = f"{r.primary_value:.4f}" if r.primary_value else "—"
        metric = r.primary_metric or "—"
        train = "OK" if r.train_py_runs else ("compiles" if r.train_py_compiles else ("exists" if r.has_train_py else "MISSING"))
        predict = "OK" if r.predict_py_compiles else ("exists" if r.has_predict_py else "MISSING")
        elapsed = f"{r.run_elapsed:.0f}s" if r.run_elapsed else "—"
        n_errors = sum(1 for i in r.issues if i.severity == "error")
        n_warnings = sum(1 for i in r.issues if i.severity == "warning")
        issues_str = ""
        if n_errors:
            issues_str += f"{n_errors}E"
        if n_warnings:
            issues_str += f" {n_warnings}W" if issues_str else f"{n_warnings}W"
        issues_str = issues_str or "—"

        lines.append(
            f"| {r.dataset} | {status} | {r.num_models_trained} | {score} | {metric} | {train} | {predict} | {elapsed} | {issues_str} |"
        )

    lines += ["", "---", ""]

    # ── Per-dataset details ──
    lines.append("## Per-Dataset Details")
    lines.append("")

    for r in reports:
        status_emoji = "PASS" if r.passed else "FAIL"
        lines.append(f"### {r.dataset} — {status_emoji}")
        lines.append("")

        # ── Quick facts ──
        pv = f"{r.primary_value:.4f}" if r.primary_value else "—"
        lines.append(f"| Property | Value |")
        lines.append(f"|----------|-------|")
        lines.append(f"| Output dir | `{r.output_dir or 'N/A'}` |")
        lines.append(f"| Modality | {r.modality or '—'} |")
        lines.append(f"| Task type | {r.task_type or '—'} |")
        lines.append(f"| Samples | {r.num_samples or '—'} |")
        lines.append(f"| Features | {r.num_features or '—'} |")
        lines.append(f"| Classes | {r.num_classes or '—'} |")
        lines.append(f"| Best model | {r.best_model or '—'} ({r.model_type or '?'}) |")
        lines.append(f"| Primary metric | {r.primary_metric or '—'} = {pv} |")
        lines.append(f"| Models trained | {r.num_models_trained} |")
        lines.append(f"| LLM cost | ${r.llm_cost:.3f} ({r.llm_calls} calls) |")
        lines.append(f"| Figures | {r.num_figures} |")
        lines.append(f"| Debates | {r.num_debates} ({', '.join(r.debate_topics) if r.debate_topics else '—'}) |")
        lines.append(f"| ReAct steps | {r.log_react_steps} |")
        lines.append(f"| Log events | {r.log_total_events} |")
        lines.append("")

        # ── All metrics (if available) ──
        if r.all_metrics:
            lines.append("**All metrics:**")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for name, val in sorted(r.all_metrics.items()):
                v_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                lines.append(f"| {name} | {v_str} |")
            lines.append("")

        # ── Hyperparameters ──
        if r.hyperparameters:
            lines.append("**Hyperparameters:**")
            lines.append("| Param | Value |")
            lines.append("|-------|-------|")
            for k, v in sorted(r.hyperparameters.items()):
                lines.append(f"| {k} | {v} |")
            lines.append("")

        # ── Preprocessing steps ──
        if r.preprocessing_steps:
            lines.append("**Preprocessing:**")
            for step in r.preprocessing_steps:
                lines.append(f"- {step}")
            lines.append("")

        # ── Pipeline steps ──
        lines.append("**Pipeline steps:**")
        for step in r.steps_expected:
            mark = "[x]" if step in r.steps_completed else "[ ]"
            lines.append(f"- {mark} {step}")
        lines.append("")

        # ── Artifact checklist ──
        lines.append("**Artifacts:**")
        artifact_groups = [
            ("Top-level", [
                ("report.md", r.has_report),
                ("summary.pdf", r.has_summary_pdf),
                ("requirements.txt", r.has_top_level_requirements),
                ("agent_conversations.log", r.has_agent_log),
            ]),
            ("Logs", [
                ("experiment_log.jsonl", r.has_experiment_log),
                ("checkpoint.pkl", r.has_checkpoint),
            ]),
            ("Models", [
                ("models/best_model.pkl", r.has_model_pkl),
                ("models/model_config.json", r.has_model_config),
            ]),
            ("Inference package", [
                ("inference dir", r.has_inference_dir),
                ("model file", r.has_inference_model),
                ("model_config.json", r.has_inference_config),
                ("predict.py", r.has_predict_py),
                ("requirements.txt", r.has_inference_requirements),
            ]),
            ("Reproduce package", [
                ("reproduce dir", r.has_reproduce_dir),
                ("train.py", r.has_train_py),
                ("evaluate.py", r.has_evaluate_py),
                ("requirements.txt", r.has_reproduce_requirements),
                ("models/best_model.pkl", r.has_reproduce_model),
                ("models/label_encoder.pkl", r.has_label_encoder),
            ]),
            ("Other", [
                ("code/ directory", r.has_code_dir),
                ("figures/", r.has_figures),
            ]),
        ]
        for group_name, items in artifact_groups:
            lines.append(f"  *{group_name}:*")
            for name, present in items:
                mark = "[x]" if present else "[ ]"
                lines.append(f"  - {mark} {name}")
        lines.append("")

        if r.code_files:
            lines.append(f"  Code files: {', '.join(r.code_files)}")
            lines.append("")
        if r.figure_subdirs:
            lines.append(f"  Figure dirs: {', '.join(r.figure_subdirs)} ({r.num_figures} PNGs)")
            lines.append("")

        # ── Model verification ──
        lines.append("**Model verification:**")
        lines.append(f"- Pickle loads: {'yes' if r.model_loads else 'NO'}")
        lines.append(f"- Predicts on dummy data: {'yes' if r.model_predicts else 'NO'}")
        lines.append(f"- Model size: {r.model_pkl_size_kb:.1f} KB")
        lines.append("")

        # ── Script verification ──
        lines.append("**Script verification:**")
        lines.append(f"- train.py compiles: {'yes' if r.train_py_compiles else 'NO'}")
        lines.append(f"- train.py executes: {'yes' if r.train_py_runs else 'NO'}")
        if r.train_py_run_error:
            lines.append(f"  - Error: `{r.train_py_run_error[:300]}`")
        lines.append(f"- predict.py compiles: {'yes' if r.predict_py_compiles else 'NO'}")
        lines.append(f"- evaluate.py compiles: {'yes' if r.evaluate_py_compiles else 'NO'}")
        lines.append("")

        # ── Cross-file consistency ──
        lines.append("**Consistency checks:**")
        lines.append(f"- Best model name matches across files: {'yes' if r.best_model_matches else 'NO'}")
        lines.append(f"- Metric value matches across files: {'yes' if r.metric_matches else 'NO'}")
        lines.append(f"- Feature count matches: {'yes' if r.feature_count_matches else 'NO'}")
        lines.append(f"- Report mentions best model: {'yes' if r.report_best_model_mentioned else 'NO'}")
        lines.append(f"- Report mentions metric value: {'yes' if r.report_metric_mentioned else 'NO'}")
        lines.append("")

        # ── Validation agent activity ──
        if r.validation_fixes or r.validation_failures:
            lines.append("**Validation agent:**")
            for fix in r.validation_fixes:
                lines.append(f"- Fixed: {fix}")
            for fail in r.validation_failures:
                lines.append(f"- FAILED: {fail}")
            lines.append("")

        # ── Issues ──
        if r.issues:
            lines.append("**Issues found:**")
            for issue in r.issues:
                icon = {"error": "ERROR", "warning": "WARN", "info": "INFO"}[issue.severity]
                lines.append(f"- [{icon}] ({issue.category}) {issue.message}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── Aggregate issues ──
    all_errors = [(r.dataset, i) for r in reports for i in r.issues if i.severity == "error"]
    all_warnings = [(r.dataset, i) for r in reports for i in r.issues if i.severity == "warning"]

    if all_errors:
        lines.append("## All Errors")
        lines.append("")
        for ds, issue in all_errors:
            lines.append(f"- **{ds}** — ({issue.category}) {issue.message}")
        lines.append("")

    if all_warnings:
        lines.append("## All Warnings")
        lines.append("")
        for ds, issue in all_warnings:
            lines.append(f"- **{ds}** — ({issue.category}) {issue.message}")
        lines.append("")

    # ── Write ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def print_terminal_summary(reports: list[DatasetReport]) -> None:
    """Print colored summary to terminal."""
    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    failed = total - passed

    G = "\033[92m"  # green
    R = "\033[91m"  # red
    Y = "\033[93m"  # yellow
    D = "\033[2m"   # dim
    B = "\033[1m"   # bold
    X = "\033[0m"   # reset

    print(f"\n{'='*95}")
    print(f"  {B}QC REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}{X}")
    print(f"{'='*95}")

    # Header
    print(f"\n  {'Dataset':<45} {'Status':>6} {'Models':>7} {'Score':>10} {'train.py':>10} {'Issues':>7}")
    print(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*7}")

    for r in reports:
        status = f"{G}PASS{X}" if r.passed else f"{R}FAIL{X}"
        score = f"{r.primary_value:.4f}" if r.primary_value else "  —"
        train = f"{G}runs{X}" if r.train_py_runs else (f"{Y}compiles{X}" if r.train_py_compiles else f"{R}FAIL{X}")
        n_err = sum(1 for i in r.issues if i.severity == "error")
        n_warn = sum(1 for i in r.issues if i.severity == "warning")
        issues = ""
        if n_err:
            issues += f"{R}{n_err}E{X}"
        if n_warn:
            issues += f" {Y}{n_warn}W{X}" if issues else f"{Y}{n_warn}W{X}"
        issues = issues or f"{D}—{X}"

        print(f"  {r.dataset:<45} {status:>15} {r.num_models_trained:>7} {score:>10} {train:>19} {issues:>16}")

    total_time = sum(r.run_elapsed for r in reports)
    print(f"\n  {'-'*95}")
    print(f"  Total: {total}  |  {G}Passed: {passed}{X}  |  {R}Failed: {failed}{X}  |  Time: {total_time:.0f}s ({total_time/60:.1f}m)")

    # Show top errors
    all_errors = [(r.dataset, i) for r in reports for i in r.issues if i.severity == "error"]
    if all_errors:
        print(f"\n  {R}{B}ERRORS ({len(all_errors)}):{X}")
        for ds, issue in all_errors[:15]:
            print(f"    {R}•{X} {ds}: {issue.message[:100]}")
        if len(all_errors) > 15:
            print(f"    ... and {len(all_errors) - 15} more (see full report)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QC: run co-scientist on all datasets and generate report")
    parser.add_argument("--parallel", type=int, default=0, help="Max parallel runs (0 = sequential)")
    parser.add_argument("--quick", action="store_true", help="Deterministic mode (no LLM, --max-cost 0)")
    parser.add_argument("--budget", type=int, default=5, help="Model budget per run (default: 5)")
    parser.add_argument("--max-cost", type=float, default=2.0, help="Max LLM cost per run (default: $2)")
    parser.add_argument("--filter", type=str, default="", help="Only run datasets matching this substring")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    parser.add_argument("--inspect-only", action="store_true", help="Skip runs, just inspect existing outputs")
    parser.add_argument("--report-path", type=str, default="outputs/qc_report.md", help="Output report path")
    args = parser.parse_args()

    datasets = DATASETS
    if args.filter:
        datasets = [d for d in datasets if args.filter.lower() in d.lower()]

    if not datasets:
        print(f"No datasets match filter '{args.filter}'")
        sys.exit(1)

    print(f"\n  Co-Scientist QC Runner")
    print(f"  ─────────────────────")
    print(f"  Datasets:     {len(datasets)}")
    print(f"  Mode:         {'inspect only' if args.inspect_only else ('sequential' if args.parallel == 0 else f'parallel ({args.parallel})')}")
    print(f"  Quick:        {args.quick}")
    print(f"  Budget:       {args.budget} models")
    print(f"  Max cost:     ${0 if args.quick else args.max_cost:.2f}/run")
    print(f"  Report path:  {args.report_path}")
    print()

    for i, d in enumerate(datasets, 1):
        existing = find_output_dir(d)
        tag = f"  ({existing.name})" if existing else "  (no prior output)"
        print(f"  {i:>2}. {d}{tag}")

    if args.dry_run:
        print(f"\n  (dry run — exiting)")
        sys.exit(0)

    # ── Phase 1: Run pipelines (unless --inspect-only) ──
    if not args.inspect_only:
        print(f"\n  Starting runs in 3 seconds... (Ctrl+C to cancel)")
        time.sleep(3)

        if args.parallel > 0:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.parallel) as pool:
                futures = {
                    pool.submit(run_dataset, d, args.quick, args.budget, args.max_cost): d
                    for d in datasets
                }
                for future in as_completed(futures):
                    future.result()  # just wait for completion
        else:
            for i, dataset in enumerate(datasets, 1):
                print(f"\n  [{i}/{len(datasets)}]")
                run_dataset(dataset, args.quick, args.budget, args.max_cost)

    # ── Phase 2: Inspect all outputs ──
    print(f"\n{'='*70}")
    print(f"  INSPECTING OUTPUTS...")
    print(f"{'='*70}")

    reports = []
    for dataset in datasets:
        print(f"\n  Inspecting {dataset}...")
        report = inspect_dataset(dataset)
        n_err = sum(1 for i in report.issues if i.severity == "error")
        n_warn = sum(1 for i in report.issues if i.severity == "warning")
        status = "\033[92mOK\033[0m" if report.passed else f"\033[91m{n_err} errors\033[0m"
        if n_warn:
            status += f", \033[93m{n_warn} warnings\033[0m"
        print(f"    {status}")
        reports.append(report)

    # ── Phase 3: Generate report ──
    report_path = Path(args.report_path)
    generate_qc_report(reports, report_path)
    print(f"\n  QC report written to: {report_path}")

    # Also save structured JSON
    json_path = report_path.with_suffix(".json")
    json_data = []
    for r in reports:
        json_data.append({
            "dataset": r.dataset,
            "passed": r.passed,
            "output_dir": r.output_dir,
            "steps_completed": r.steps_completed,
            # Model info
            "best_model": r.best_model,
            "model_type": r.model_type,
            "primary_metric": r.primary_metric,
            "primary_value": round(r.primary_value, 4) if r.primary_value else None,
            "all_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in r.all_metrics.items()},
            "num_models_trained": r.num_models_trained,
            "hyperparameters": r.hyperparameters,
            "preprocessing_steps": r.preprocessing_steps,
            # Dataset info
            "task_type": r.task_type,
            "modality": r.modality,
            "num_samples": r.num_samples,
            "num_features": r.num_features,
            "num_classes": r.num_classes,
            # Model verification
            "model_loads": r.model_loads,
            "model_predicts": r.model_predicts,
            "model_pkl_size_kb": round(r.model_pkl_size_kb, 1),
            # Script verification
            "train_py_compiles": r.train_py_compiles,
            "train_py_runs": r.train_py_runs,
            "train_py_run_error": r.train_py_run_error[:300] if r.train_py_run_error else "",
            "predict_py_compiles": r.predict_py_compiles,
            "evaluate_py_compiles": r.evaluate_py_compiles,
            # Artifacts
            "has_report": r.has_report,
            "has_summary_pdf": r.has_summary_pdf,
            "has_agent_log": r.has_agent_log,
            "code_files": r.code_files,
            "num_figures": r.num_figures,
            "figure_subdirs": r.figure_subdirs,
            # Consistency
            "best_model_matches": r.best_model_matches,
            "metric_matches": r.metric_matches,
            "feature_count_matches": r.feature_count_matches,
            "report_best_model_mentioned": r.report_best_model_mentioned,
            "report_metric_mentioned": r.report_metric_mentioned,
            # Agent activity
            "llm_cost": round(r.llm_cost, 4),
            "llm_calls": r.llm_calls,
            "num_debates": r.num_debates,
            "debate_topics": r.debate_topics,
            "log_react_steps": r.log_react_steps,
            "validation_fixes": r.validation_fixes,
            "validation_failures": r.validation_failures,
            # Issues
            "errors": [{"category": i.category, "message": i.message} for i in r.issues if i.severity == "error"],
            "warnings": [{"category": i.category, "message": i.message} for i in r.issues if i.severity == "warning"],
        })
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"  JSON results: {json_path}")

    # ── Phase 4: Terminal summary ──
    print_terminal_summary(reports)

    print(f"\n  Full report: {report_path}")

    if any(not r.passed for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
