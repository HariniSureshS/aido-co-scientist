"""Export best model: weights, config, standalone code."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from rich.console import Console

from co_scientist.data.types import DatasetProfile, SplitData
from co_scientist.evaluation.types import EvalConfig, ModelResult
from co_scientist.modeling.types import TrainedModel

console = Console()


def export_model(
    trained: TrainedModel,
    result: ModelResult,
    profile: DatasetProfile,
    eval_config: EvalConfig,
    split: SplitData,
    output_dir: Path,
    preprocessing_steps: list[str],
) -> Path:
    """Export the best model to the output directory.

    Organized into two clear sections:

    (a) reproduce/ — Reproduce training and analysis from scratch
        train.py            — downloads dataset, preprocesses, trains, evaluates
        evaluate.py         — evaluate predictions against ground truth
        requirements.txt    — all dependencies needed to reproduce

    (b) inference/ — Port the model and run inference on new data
        predict.py          — load model + run predictions on new data
        model/
            best_model.pkl      — trained model weights
            model_config.json   — hyperparameters + metadata
            label_encoder.pkl   — label encoder (classification only)
        requirements.txt    — minimal deps for inference only
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset name for folder naming
    ds_name = profile.dataset_name.replace("/", "_").replace(" ", "_")

    # ── (a) reproduce_{dataset}/ — Reproduce training and analysis ───
    reproduce_dir = output_dir / f"reproduce_{ds_name}"
    reproduce_dir.mkdir(exist_ok=True)

    _EMBEDDING_TYPES = ("embed_xgboost", "embed_mlp")
    _CONCAT_TYPES = ("concat_xgboost", "concat_mlp")
    is_embedding = trained.config.model_type in _EMBEDDING_TYPES
    is_concat = trained.config.model_type in _CONCAT_TYPES
    is_foundation = is_embedding or is_concat
    is_finetune = trained.config.model_type == "aido_finetune"
    if is_finetune:
        _generate_train_py_finetune(reproduce_dir, trained, profile, split, preprocessing_steps)
    elif is_concat:
        _generate_train_py_concat(reproduce_dir, trained, profile, split, preprocessing_steps)
    elif is_embedding:
        _generate_train_py_foundation(reproduce_dir, trained, profile, split, preprocessing_steps)
    else:
        _generate_train_py(reproduce_dir, trained, profile, split, preprocessing_steps)
    _generate_evaluate_py(reproduce_dir, eval_config, profile)
    _generate_requirements(reproduce_dir, trained, profile=profile, minimal=False)

    console.print(f"  [bold]reproduce_{ds_name}/[/bold] — Retrain from scratch:")
    console.print(f"    train.py         — downloads data, preprocesses, trains, evaluates")
    console.print(f"    evaluate.py      — evaluate predictions against ground truth")
    console.print(f"    requirements.txt — full dependencies")

    # ── (b) inference_{dataset}/ — Port model and run inference ──────
    inference_dir = output_dir / f"inference_{ds_name}"
    model_dir = inference_dir / "model"
    inference_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Save model weights
    model_path = model_dir / "best_model.pkl"
    custom_code = getattr(trained, "custom_model_code", None)
    if trained.config.model_type == "aido_finetune":
        # AIDO fine-tuned models: save state_dict + config (too large to pickle)
        try:
            import torch
            save_dict = trained.model._model.get_save_dict() if trained.model._model else {}
            save_dict["hyperparameters"] = trained.config.hyperparameters
            torch.save(save_dict, model_dir / "best_model.pt")
            console.print("  [dim]AIDO fine-tuned model saved as state_dict[/dim]")
        except Exception as e:
            console.print(f"  [yellow]AIDO fine-tune export: {e}[/yellow]")
    elif custom_code and trained.config.model_type == "custom":
        # Custom models can't be pickled — save PyTorch state_dict + source code
        try:
            import torch
            torch.save(trained.model.model.state_dict(), model_dir / "best_model.pt")
            (model_dir / "custom_model.py").write_text(custom_code)
            console.print("  [dim]Custom model saved as state_dict + source code[/dim]")
            # Also try pickle as a fallback (may work for some custom models)
            try:
                with open(model_path, "wb") as f:
                    pickle.dump(trained.model, f)
            except Exception:
                console.print("  [dim]Pickle export skipped for custom model (expected)[/dim]")
        except Exception as e:
            console.print(f"  [yellow]Custom model export: {e}[/yellow]")
            (model_dir / "custom_model.py").write_text(custom_code)
    else:
        with open(model_path, "wb") as f:
            pickle.dump(trained.model, f)

    # Save model config
    config_data = {
        "model_name": trained.config.name,
        "model_type": trained.config.model_type,
        "tier": trained.config.tier,
        "task_type": trained.config.task_type,
        "hyperparameters": trained.config.hyperparameters,
        "dataset": {
            "name": profile.dataset_name,
            "path": profile.dataset_path,
            "modality": profile.modality.value,
            "task_type": profile.task_type.value,
            "num_samples": profile.num_samples,
            "num_features": (
                split.X_train.shape[1] + split.X_embed_train.shape[1] if is_concat and split.X_embed_train is not None
                else split.X_embed_train.shape[1] if is_embedding and split.X_embed_train is not None
                else split.X_train.shape[1]
            ),
            "num_classes": profile.num_classes,
        },
        "evaluation": {
            "primary_metric": eval_config.primary_metric,
            "primary_value": result.primary_metric_value,
            "all_metrics": result.metrics,
            "evaluated_on": "validation",
        },
        "preprocessing_steps": preprocessing_steps,
        "feature_names": split.feature_names,
    }
    config_path = model_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2, default=str)

    # Save label encoder if present
    if split.label_encoder is not None:
        le_path = model_dir / "label_encoder.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(split.label_encoder, f)

    if is_finetune:
        _generate_predict_py_finetune(inference_dir, trained, profile, split)
    elif is_concat:
        _generate_predict_py_concat(inference_dir, trained, profile, split)
    elif is_embedding:
        _generate_predict_py_foundation(inference_dir, trained, profile, split)
    else:
        _generate_predict_py(inference_dir, trained, profile, split)
    _generate_requirements(inference_dir, trained, profile=profile, minimal=True)

    console.print(f"  [bold]inference_{ds_name}/[/bold]  — Port model to new environments:")
    console.print(f"    predict.py       — load model, run predictions on new data")
    console.print(f"    model/           — model weights + config (self-contained)")
    console.print(f"    requirements.txt — minimal inference dependencies")

    # ── Also keep top-level models/ and code/ for backward compat ────
    # (Some downstream code references these paths)
    models_dir = output_dir / "models"
    code_dir = output_dir / "code"
    models_dir.mkdir(exist_ok=True)
    code_dir.mkdir(exist_ok=True)

    # Copy model to top-level models/
    import shutil
    if model_path.exists():
        shutil.copy2(model_path, models_dir / "best_model.pkl")
    # Also copy custom model artifacts if present
    if (model_dir / "best_model.pt").exists():
        shutil.copy2(model_dir / "best_model.pt", models_dir / "best_model.pt")
    if (model_dir / "custom_model.py").exists():
        shutil.copy2(model_dir / "custom_model.py", models_dir / "custom_model.py")
    shutil.copy2(config_path, models_dir / "model_config.json")
    if split.label_encoder is not None:
        shutil.copy2(model_dir / "label_encoder.pkl", models_dir / "label_encoder.pkl")

    # Copy scripts to top-level code/
    shutil.copy2(reproduce_dir / "train.py", code_dir / "train.py")
    shutil.copy2(inference_dir / "predict.py", code_dir / "predict.py")
    shutil.copy2(reproduce_dir / "evaluate.py", code_dir / "evaluate.py")

    # Top-level requirements (full)
    shutil.copy2(reproduce_dir / "requirements.txt", output_dir / "requirements.txt")

    return output_dir


def _generate_train_py(
    code_dir: Path,
    trained: TrainedModel,
    profile: DatasetProfile,
    split: SplitData,
    preprocessing_steps: list[str],
) -> None:
    """Generate a standalone train.py that reproduces training."""
    model_type = trained.config.model_type
    hparams = trained.config.hyperparameters
    task = trained.config.task_type
    n_features = split.X_train.shape[1]

    # Resolve the real HuggingFace repo + subset for the train script
    try:
        from co_scientist.data.loader import resolve_dataset_path
        hf_repo, hf_subset, _, _ = resolve_dataset_path(profile.dataset_path)
    except Exception:
        hf_repo = profile.dataset_path
        hf_subset = ""

    # Build model construction code for ALL supported model types
    model_code, model_imports = _build_model_code(model_type, task, hparams)

    # Build preprocessing code based on modality
    preprocess_code = _build_preprocess_code(profile, split)

    is_h5ad = profile.modality.value == "cell_expression"

    # Build the loading + preprocessing function based on data format
    if is_h5ad:
        load_code = _build_h5ad_load_code(hf_repo, hf_subset, profile, split)
        extra_imports = "import anndata as ad\nfrom huggingface_hub import hf_hub_download"
        dataset_lib_import = ""
    else:
        load_code = _build_hf_load_code(hf_repo, hf_subset, profile, preprocess_code)
        extra_imports = ""
        dataset_lib_import = "from datasets import load_dataset"

    script = f'''"""Standalone training script — fully reproducible, no co-scientist dependency.

Dataset: {profile.dataset_name}
Source:  {hf_repo} (subset: {hf_subset})
Model:   {trained.config.name} ({trained.config.tier})
Task:    {profile.task_type.value}
Metric:  Evaluated on validation set

Preprocessing steps applied:
{chr(10).join(f"  - {s}" for s in preprocessing_steps)}

Usage:
    python train.py
    python train.py --seed 42
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
{dataset_lib_import}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
{extra_imports}
{model_imports}

warnings.filterwarnings("ignore")

DATASET_PATH = "{hf_repo}"
DATASET_SUBSET = "{hf_subset}"
TARGET_COLUMN = "{profile.target_column}"
TASK_TYPE = "{profile.task_type.value}"
SEED = 42

{load_code}


def build_model(seed: int = SEED):
    """Construct the model with the exact hyperparameters from the pipeline run."""
{model_code}
    return model


def main():
    parser = argparse.ArgumentParser(description="Reproduce training")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess(args.seed)

    print(f"\\nTraining {{type(build_model()).__name__}}...")
    model = build_model(args.seed)
    model.fit(X_train, y_train)
    print("  Training complete.")

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    print(f"\\nValidation results:")
    _evaluate(y_val, y_pred)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    print(f"\\nTest results:")
    _evaluate(y_test, y_test_pred)

    # Save model
    import os
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\\nModel saved to models/best_model.pkl")

    if label_encoder is not None:
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)


def _evaluate(y_true, y_pred):
    if TASK_TYPE in ("binary_classification", "multiclass_classification"):
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"  Accuracy:    {{acc:.4f}}")
        print(f"  Macro F1:    {{mf1:.4f}}")
        print(f"  Weighted F1: {{wf1:.4f}}")
    else:
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_squared_error, r2_score
        sp = spearmanr(y_true, y_pred).correlation
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        print(f"  Spearman: {{sp:.4f}}")
        print(f"  RMSE:     {{rmse:.4f}}")
        print(f"  R²:       {{r2:.4f}}")


if __name__ == "__main__":
    main()
'''
    (code_dir / "train.py").write_text(script)


def _generate_predict_py(
    code_dir: Path,
    trained: TrainedModel,
    profile: DatasetProfile,
    split: SplitData,
) -> None:
    """Generate a standalone predict.py."""
    is_clf = trained.config.task_type == "classification"
    n_features = split.X_train.shape[1]

    script = f'''"""Standalone prediction script.

Usage:
    python predict.py --input data.csv --output predictions.csv

The model expects preprocessed numeric features of shape (n_samples, {n_features}).
"""

import argparse
import pickle

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run predictions with trained model")
    parser.add_argument("--input", required=True, help="Path to input CSV (preprocessed features)")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    parser.add_argument("--model", default="model/best_model.pkl", help="Path to model file")
    args = parser.parse_args()

    # Load model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Load input data
    df = pd.read_csv(args.input)
    X = df.values.astype(np.float64)

    # Predict
    predictions = model.predict(X)
'''

    if is_clf:
        script += '''
    # Decode class labels if encoder exists
    try:
        with open("model/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        predictions = le.inverse_transform(predictions)
    except FileNotFoundError:
        pass
'''

    script += f'''
    # Save predictions
    out_df = pd.DataFrame({{"prediction": predictions}})
    out_df.to_csv(args.output, index=False)
    print(f"Saved {{len(predictions)}} predictions to {{args.output}}")


if __name__ == "__main__":
    main()
'''
    (code_dir / "predict.py").write_text(script)


def _generate_evaluate_py(
    code_dir: Path,
    eval_config: EvalConfig,
    profile: DatasetProfile,
) -> None:
    """Generate a standalone evaluate.py."""
    is_clf = eval_config.task_type in ("binary_classification", "multiclass_classification")

    if is_clf:
        metric_code = '''
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:    {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print()
    print(classification_report(y_true, y_pred, zero_division=0))
'''
    else:
        metric_code = '''
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print(f"Spearman:  {spearmanr(y_true, y_pred).correlation:.4f}")
    print(f"Pearson:   {pearsonr(y_true, y_pred)[0]:.4f}")
    print(f"MSE:       {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE:      {mean_squared_error(y_true, y_pred)**0.5:.4f}")
    print(f"MAE:       {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R²:        {r2_score(y_true, y_pred):.4f}")
'''

    script = f'''"""Standalone evaluation script.

Usage:
    python evaluate.py --predictions predictions.csv --ground-truth labels.csv
"""

import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions", required=True, help="CSV with 'prediction' column")
    parser.add_argument("--ground-truth", required=True, help="CSV with 'label' column")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions)
    true_df = pd.read_csv(args.ground_truth)

    y_pred = pred_df["prediction"].values
    y_true = true_df["label"].values

    print(f"Evaluating {{len(y_pred)}} predictions")
    print(f"Primary metric: {eval_config.primary_metric}")
    print()
{metric_code}

if __name__ == "__main__":
    main()
'''
    (code_dir / "evaluate.py").write_text(script)


def _generate_train_py_foundation(
    code_dir: Path,
    trained: TrainedModel,
    profile: DatasetProfile,
    split: SplitData,
    preprocessing_steps: list[str],
) -> None:
    """Generate a standalone train.py for foundation-tier (embedding-based) models."""
    model_type = trained.config.model_type
    hparams = trained.config.hyperparameters
    task = trained.config.task_type
    is_clf = task == "classification"

    try:
        from co_scientist.data.loader import resolve_dataset_path
        hf_repo, hf_subset, _, _ = resolve_dataset_path(profile.dataset_path)
    except Exception:
        hf_repo = profile.dataset_path
        hf_subset = ""

    from co_scientist.defaults import get_defaults
    fm_config = get_defaults(profile.modality.value, profile.dataset_path).get("foundation_models", {})
    from co_scientist.modeling.foundation import get_foundation_model_name
    aido_model = get_foundation_model_name(profile.modality.value, fm_config)

    # Build downstream model code
    model_code, model_imports = _build_model_code(
        model_type.replace("embed_", ""),  # reuse xgboost/mlp builder
        task, hparams,
    )

    script = f'''"""Standalone training script — foundation model embeddings + downstream model.

Dataset: {profile.dataset_name}
Source:  {hf_repo} (subset: {hf_subset})
Model:   {trained.config.name} ({trained.config.tier})
Task:    {profile.task_type.value}
Foundation Model: {aido_model}

REQUIRES: CUDA-capable GPU + modelgenerator package

Usage:
    python train.py
    python train.py --seed 42
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from modelgenerator.tasks import Embed
{model_imports}

warnings.filterwarnings("ignore")

DATASET_PATH = "{hf_repo}"
DATASET_SUBSET = "{hf_subset}"
TARGET_COLUMN = "{profile.target_column}"
TASK_TYPE = "{profile.task_type.value}"
AIDO_MODEL = "{aido_model}"
SEED = 42


def extract_embeddings(sequences, model_name=AIDO_MODEL, batch_size=32, max_length=1024):
    """Extract embeddings using AIDO foundation model."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for foundation model embeddings")

    print(f"Loading AIDO model: {{model_name}}")
    model = Embed.from_config({{"model.backbone": model_name}}).eval()
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)

    all_embeddings = []
    n = len(sequences)
    print(f"Extracting embeddings for {{n}} samples...")

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = sequences[start:start + batch_size]
            batch = [s[:max_length] if isinstance(s, str) else s for s in batch]
            transformed = model.transform({{"sequences": batch}})
            if isinstance(transformed, dict):
                transformed = {{k: v.to(device) if hasattr(v, "to") else v for k, v in transformed.items()}}
            embeddings = model(transformed)
            if isinstance(embeddings, torch.Tensor):
                emb_np = embeddings.cpu().numpy()
            elif isinstance(embeddings, dict):
                emb_tensor = embeddings.get("embeddings", embeddings.get("last_hidden_state"))
                if emb_tensor.ndim == 3:
                    emb_np = emb_tensor.mean(dim=1).cpu().numpy()
                else:
                    emb_np = emb_tensor.cpu().numpy()
            else:
                emb_np = np.array(embeddings)
            all_embeddings.append(emb_np)

    del model
    torch.cuda.empty_cache()
    result = np.vstack(all_embeddings)
    print(f"Embeddings shape: {{result.shape}}")
    return result


def load_and_preprocess(seed=SEED):
    """Load dataset, extract embeddings, and split."""
    print(f"Loading dataset: {{DATASET_PATH}} (subset: {{DATASET_SUBSET}})")
    if DATASET_SUBSET:
        ds = load_dataset(DATASET_PATH, DATASET_SUBSET)
    else:
        ds = load_dataset(DATASET_PATH)

    split_names = list(ds.keys())
    has_multiple_splits = len(split_names) > 1

    if has_multiple_splits:
        frames = []
        for split_name in split_names:
            df_split = ds[split_name].to_pandas()
            if split_name.startswith("test"):
                df_split["_split"] = "test"
            elif split_name in ("validation", "valid", "val"):
                df_split["_split"] = "valid"
            else:
                df_split["_split"] = split_name
            frames.append(df_split)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = ds[split_names[0]].to_pandas()

    y = df[TARGET_COLUMN].values

    # Find sequence column
    seq_col = None
    for c in df.columns:
        if c.lower() in ("sequences", "sequence", "seq"):
            seq_col = c
            break
    if seq_col is None:
        for c in df.columns:
            if df[c].dtype == object and c != TARGET_COLUMN and c != "_split":
                seq_col = c
                break
    if seq_col is None:
        raise ValueError("No sequence column found")

    sequences = df[seq_col].astype(str).tolist()

    # Extract embeddings
    X = extract_embeddings(sequences)

    # Encode labels
    label_encoder = None
    if TASK_TYPE in ("binary_classification", "multiclass_classification"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Split
    if has_multiple_splits and "_split" in df.columns:
        train_mask = df["_split"].values == "train"
        val_mask = df["_split"].values == "valid"
        test_mask = df["_split"].values == "test"
        if not val_mask.any():
            train_indices = np.where(train_mask)[0]
            n_val = max(1, int(len(train_indices) * 0.2))
            rng = np.random.RandomState(seed)
            rng.shuffle(train_indices)
            val_mask = np.zeros(len(X), dtype=bool)
            val_mask[train_indices[:n_val]] = True
            train_mask = np.zeros(len(X), dtype=bool)
            train_mask[train_indices[n_val:]] = True
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    print(f"Train: {{len(X_train)}}, Val: {{len(X_val)}}, Test: {{len(X_test)}}")
    print(f"Embedding features: {{X_train.shape[1]}}")
    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder


def build_model(seed=SEED):
    """Construct the downstream model."""
{model_code}
    return model


def main():
    parser = argparse.ArgumentParser(description="Reproduce training (foundation model)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess(args.seed)

    print(f"\\nTraining {{type(build_model()).__name__}} on AIDO embeddings...")
    model = build_model(args.seed)
    model.fit(X_train, y_train)
    print("  Training complete.")

    y_pred = model.predict(X_val)
    print(f"\\nValidation results:")
    _evaluate(y_val, y_pred)

    y_test_pred = model.predict(X_test)
    print(f"\\nTest results:")
    _evaluate(y_test, y_test_pred)

    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\\nModel saved to models/best_model.pkl")

    if label_encoder is not None:
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)


def _evaluate(y_true, y_pred):
    if TASK_TYPE in ("binary_classification", "multiclass_classification"):
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"  Accuracy:    {{acc:.4f}}")
        print(f"  Macro F1:    {{mf1:.4f}}")
        print(f"  Weighted F1: {{wf1:.4f}}")
    else:
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_squared_error, r2_score
        sp = spearmanr(y_true, y_pred).correlation
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        print(f"  Spearman: {{sp:.4f}}")
        print(f"  RMSE:     {{rmse:.4f}}")
        print(f"  R²:       {{r2:.4f}}")


if __name__ == "__main__":
    main()
'''
    (code_dir / "train.py").write_text(script)


def _generate_predict_py_foundation(
    code_dir: Path,
    trained: TrainedModel,
    profile: DatasetProfile,
    split: SplitData,
) -> None:
    """Generate a standalone predict.py for foundation-tier models."""
    is_clf = trained.config.task_type == "classification"

    from co_scientist.defaults import get_defaults
    fm_config = get_defaults(profile.modality.value, profile.dataset_path).get("foundation_models", {})
    from co_scientist.modeling.foundation import get_foundation_model_name
    aido_model = get_foundation_model_name(profile.modality.value, fm_config)

    script = f'''"""Standalone prediction script — foundation model embeddings.

REQUIRES: CUDA-capable GPU + modelgenerator package

Usage:
    python predict.py --input data.csv --output predictions.csv
    python predict.py --input data.csv --seq-column sequence
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from modelgenerator.tasks import Embed

AIDO_MODEL = "{aido_model}"


def extract_embeddings(sequences, model_name=AIDO_MODEL, batch_size=32, max_length=1024):
    """Extract embeddings using AIDO foundation model."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for foundation model embeddings")

    model = Embed.from_config({{"model.backbone": model_name}}).eval()
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = sequences[start:start + batch_size]
            batch = [s[:max_length] if isinstance(s, str) else s for s in batch]
            transformed = model.transform({{"sequences": batch}})
            if isinstance(transformed, dict):
                transformed = {{k: v.to(device) if hasattr(v, "to") else v for k, v in transformed.items()}}
            embeddings = model(transformed)
            if isinstance(embeddings, torch.Tensor):
                emb_np = embeddings.cpu().numpy()
            elif isinstance(embeddings, dict):
                emb_tensor = embeddings.get("embeddings", embeddings.get("last_hidden_state"))
                if emb_tensor.ndim == 3:
                    emb_np = emb_tensor.mean(dim=1).cpu().numpy()
                else:
                    emb_np = emb_tensor.cpu().numpy()
            else:
                emb_np = np.array(embeddings)
            all_embeddings.append(emb_np)

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Run predictions with foundation model embeddings")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    parser.add_argument("--model", default="model/best_model.pkl", help="Path to model file")
    parser.add_argument("--seq-column", default="sequence", help="Name of sequence column")
    args = parser.parse_args()

    # Load downstream model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Load and extract embeddings from input data
    df = pd.read_csv(args.input)
    sequences = df[args.seq_column].astype(str).tolist()
    print(f"Extracting embeddings for {{len(sequences)}} sequences...")
    X = extract_embeddings(sequences)

    # Predict
    predictions = model.predict(X)
'''

    if is_clf:
        script += '''
    # Decode class labels if encoder exists
    try:
        with open("model/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        predictions = le.inverse_transform(predictions)
    except FileNotFoundError:
        pass
'''

    script += f'''
    # Save predictions
    out_df = pd.DataFrame({{"prediction": predictions}})
    out_df.to_csv(args.output, index=False)
    print(f"Saved {{len(predictions)}} predictions to {{args.output}}")


if __name__ == "__main__":
    main()
'''
    (code_dir / "predict.py").write_text(script)


def _generate_train_py_finetune(
    code_dir: Path,
    trained: TrainedModel,
    profile: DatasetProfile,
    split: SplitData,
    preprocessing_steps: list[str],
) -> None:
    """Generate a standalone train.py for AIDO fine-tuning models."""
    hparams = trained.config.hyperparameters
    task = trained.config.task_type
    is_clf = task == "classification"

    try:
        from co_scientist.data.loader import resolve_dataset_path
        hf_repo, hf_subset, _, _ = resolve_dataset_path(profile.dataset_path)
    except Exception:
        hf_repo = profile.dataset_path
        hf_subset = ""

    aido_model = hparams.get("model_name", "")
    unfreeze_layers = hparams.get("unfreeze_layers", 2)
    head_hidden = hparams.get("head_hidden", 256)
    head_dropout = hparams.get("head_dropout", 0.3)
    lr = hparams.get("learning_rate", 2e-5)
    wd = hparams.get("weight_decay", 0.01)
    bs = hparams.get("batch_size", 16)
    max_epochs = hparams.get("max_epochs", 20)
    patience = hparams.get("patience", 5)
    max_length = hparams.get("max_length", 1024)
    n_classes_or_1 = profile.num_classes if is_clf else 1
    loss_fn_str = "nn.CrossEntropyLoss()" if is_clf else "nn.MSELoss()"
    y_dtype = "torch.long" if is_clf else "torch.float32"
    y_unsqueeze = "" if is_clf else ".unsqueeze(1)"
    pred_code = "logits.argmax(dim=1)" if is_clf else "preds.squeeze(1)"

    script = f'''"""Standalone training script — AIDO fine-tuning with task head.

Dataset: {profile.dataset_name}
Source:  {hf_repo} (subset: {hf_subset})
Model:   {trained.config.name} ({trained.config.tier})
Task:    {profile.task_type.value}
Foundation Model: {aido_model}
Strategy: Fine-tune last {unfreeze_layers} layers + task head

REQUIRES: CUDA-capable GPU + modelgenerator package

Usage:
    python train.py
    python train.py --seed 42
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from modelgenerator.tasks import Embed

warnings.filterwarnings("ignore")

DATASET_PATH = "{hf_repo}"
DATASET_SUBSET = "{hf_subset}"
TARGET_COLUMN = "{profile.target_column}"
TASK_TYPE = "{profile.task_type.value}"
AIDO_MODEL = "{aido_model}"
SEED = 42

# Hyperparameters
UNFREEZE_LAYERS = {unfreeze_layers}
HEAD_HIDDEN = {head_hidden}
HEAD_DROPOUT = {head_dropout}
LEARNING_RATE = {lr}
WEIGHT_DECAY = {wd}
BATCH_SIZE = {bs}
MAX_EPOCHS = {max_epochs}
PATIENCE = {patience}
MAX_LENGTH = {max_length}
OUTPUT_DIM = {n_classes_or_1}


class TaskHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=HEAD_HIDDEN, dropout=HEAD_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def load_aido_model():
    """Load AIDO backbone and build fine-tuning model."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for AIDO fine-tuning")

    print(f"Loading AIDO model: {{AIDO_MODEL}}")
    embed_model = Embed.from_config({{"model.backbone": AIDO_MODEL}}).eval()

    # Probe embedding dim
    embed_model.eval()
    with torch.no_grad():
        _probe = embed_model.transform({{"sequences": ["ACGT"]}})
        out = embed_model(_probe)
        if isinstance(out, torch.Tensor):
            embed_dim = out.shape[-1]
        elif isinstance(out, dict):
            for k in ("embeddings", "last_hidden_state"):
                if k in out:
                    embed_dim = out[k].shape[-1]
                    break
        else:
            embed_dim = 768

    # Freeze all, unfreeze last N layers
    for param in embed_model.parameters():
        param.requires_grad = False

    # Find and unfreeze transformer layers
    layers = None
    for attr in ("encoder.layer", "layers", "transformer.layer", "blocks"):
        obj = embed_model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__"):
                layers = obj
                break
        except AttributeError:
            continue

    if layers is not None:
        for layer in layers[-UNFREEZE_LAYERS:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  Unfroze last {{UNFREEZE_LAYERS}} of {{len(layers)}} layers")
    else:
        all_params = list(embed_model.named_parameters())
        n_unfreeze = max(1, len(all_params) // 4)
        for _, param in all_params[-n_unfreeze:]:
            param.requires_grad = True

    head = TaskHead(embed_dim, OUTPUT_DIM)
    return embed_model, head, embed_dim


def forward_pass(embed_model, head, batch_seqs):
    """Run sequences through backbone + head."""
    transformed = embed_model.transform({{"sequences": batch_seqs}})
    if isinstance(transformed, dict):
        transformed = {{k: v.to(device) if hasattr(v, "to") else v for k, v in transformed.items()}}
    embeddings = embed_model(transformed)
    if isinstance(embeddings, torch.Tensor):
        pooled = embeddings.mean(dim=1) if embeddings.ndim == 3 else embeddings
    elif isinstance(embeddings, dict):
        for k in ("embeddings", "last_hidden_state"):
            if k in embeddings:
                e = embeddings[k]
                pooled = e.mean(dim=1) if e.ndim == 3 else e
                break
    else:
        pooled = torch.tensor(np.array(embeddings), dtype=torch.float32)
    return head(pooled)


def load_and_preprocess(seed=SEED):
    """Load dataset and return sequences + labels + splits."""
    print(f"Loading dataset: {{DATASET_PATH}} (subset: {{DATASET_SUBSET}})")
    if DATASET_SUBSET:
        ds = load_dataset(DATASET_PATH, DATASET_SUBSET)
    else:
        ds = load_dataset(DATASET_PATH)

    split_names = list(ds.keys())
    has_multiple_splits = len(split_names) > 1

    if has_multiple_splits:
        frames = []
        for sn in split_names:
            df_s = ds[sn].to_pandas()
            if sn.startswith("test"):
                df_s["_split"] = "test"
            elif sn in ("validation", "valid", "val"):
                df_s["_split"] = "valid"
            else:
                df_s["_split"] = sn
            frames.append(df_s)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = ds[split_names[0]].to_pandas()

    y = df[TARGET_COLUMN].values

    # Find sequence column
    seq_col = None
    for c in df.columns:
        if c.lower() in ("sequences", "sequence", "seq"):
            seq_col = c
            break
    if seq_col is None:
        for c in df.columns:
            if df[c].dtype == object and c not in (TARGET_COLUMN, "_split"):
                seq_col = c
                break
    sequences = df[seq_col].astype(str).tolist()

    # Encode labels
    label_encoder = None
    if TASK_TYPE in ("binary_classification", "multiclass_classification"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Split
    if has_multiple_splits and "_split" in df.columns:
        splits = df["_split"].values
        train_mask = splits == "train"
        val_mask = splits == "valid"
        test_mask = splits == "test"
        if not val_mask.any():
            train_idx = np.where(train_mask)[0]
            n_val = max(1, int(len(train_idx) * 0.15))
            rng = np.random.RandomState(seed)
            rng.shuffle(train_idx)
            val_mask = np.zeros(len(y), dtype=bool)
            val_mask[train_idx[:n_val]] = True
            train_mask[train_idx[:n_val]] = False
        return (
            [sequences[i] for i in np.where(train_mask)[0]], y[train_mask],
            [sequences[i] for i in np.where(val_mask)[0]], y[val_mask],
            [sequences[i] for i in np.where(test_mask)[0]], y[test_mask],
            label_encoder,
        )
    else:
        idx = np.arange(len(sequences))
        idx_train, idx_temp, y_train, y_temp = train_test_split(idx, y, test_size=0.3, random_state=seed)
        idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.5, random_state=seed)
        return (
            [sequences[i] for i in idx_train], y_train,
            [sequences[i] for i in idx_val], y_val,
            [sequences[i] for i in idx_test], y_test,
            label_encoder,
        )


def main():
    parser = argparse.ArgumentParser(description="AIDO fine-tuning")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    seqs_train, y_train, seqs_val, y_val, seqs_test, y_test, label_encoder = load_and_preprocess(args.seed)
    print(f"Train: {{len(seqs_train)}}, Val: {{len(seqs_val)}}, Test: {{len(seqs_test)}}")

    device = torch.device("cuda")
    embed_model, head, embed_dim = load_aido_model()
    embed_model = embed_model.to(device)
    head = head.to(device)

    backbone_params = [p for p in embed_model.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    optimizer = torch.optim.AdamW([
        {{"params": backbone_params, "lr": LEARNING_RATE}},
        {{"params": head_params, "lr": LEARNING_RATE * 10}},
    ], weight_decay=WEIGHT_DECAY)

    loss_fn = {loss_fn_str}
    scaler = torch.amp.GradScaler("cuda")
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        embed_model.train()
        head.train()
        train_loss, n_b = 0.0, 0
        for start in range(0, len(seqs_train), BATCH_SIZE):
            batch_seqs = [s[:MAX_LENGTH] for s in seqs_train[start:start+BATCH_SIZE]]
            batch_y = torch.tensor(y_train[start:start+BATCH_SIZE], dtype={y_dtype}){y_unsqueeze}.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out = forward_pass(embed_model, head, batch_seqs)
                loss = loss_fn(out, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(embed_model.parameters()) + list(head.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_b += 1

        # Validate
        embed_model.eval()
        head.eval()
        val_loss, n_vb = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(seqs_val), BATCH_SIZE):
                batch_seqs = [s[:MAX_LENGTH] for s in seqs_val[start:start+BATCH_SIZE]]
                batch_y = torch.tensor(y_val[start:start+BATCH_SIZE], dtype={y_dtype}){y_unsqueeze}.to(device)
                with torch.amp.autocast("cuda"):
                    out = forward_pass(embed_model, head, batch_seqs)
                    loss = loss_fn(out, batch_y)
                val_loss += loss.item()
                n_vb += 1

        avg_val = val_loss / max(n_vb, 1)
        print(f"  Epoch {{epoch+1}}: train_loss={{train_loss/max(n_b,1):.4f}} val_loss={{avg_val:.4f}}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({{"head": head.state_dict(), "embed_model": embed_model.state_dict()}}, "best_model.pt")
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stopping at epoch {{epoch+1}}")
                break

    # Reload best
    ckpt = torch.load("best_model.pt", weights_only=True)
    head.load_state_dict(ckpt["head"])
    embed_model.load_state_dict(ckpt["embed_model"])

    # Evaluate
    embed_model.eval()
    head.eval()
    for name, seqs, y_true in [("Validation", seqs_val, y_val), ("Test", seqs_test, y_test)]:
        all_preds = []
        with torch.no_grad():
            for start in range(0, len(seqs), BATCH_SIZE):
                batch = [s[:MAX_LENGTH] for s in seqs[start:start+BATCH_SIZE]]
                with torch.amp.autocast("cuda"):
                    out = forward_pass(embed_model, head, batch)
                all_preds.append({pred_code}.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        print(f"\\n{{name}} results:")
        _evaluate(y_true, y_pred)

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save(ckpt, "models/best_model.pt")
    if label_encoder is not None:
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
    print("\\nModel saved to models/best_model.pt")


def _evaluate(y_true, y_pred):
    if TASK_TYPE in ("binary_classification", "multiclass_classification"):
        from sklearn.metrics import accuracy_score, f1_score
        print(f"  Accuracy:    {{accuracy_score(y_true, y_pred):.4f}}")
        print(f"  Macro F1:    {{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}}")
    else:
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_squared_error, r2_score
        print(f"  Spearman: {{spearmanr(y_true, y_pred).correlation:.4f}}")
        print(f"  RMSE:     {{mean_squared_error(y_true, y_pred)**0.5:.4f}}")
        print(f"  R²:       {{r2_score(y_true, y_pred):.4f}}")


if __name__ == "__main__":
    main()
'''
    (code_dir / "train.py").write_text(script)


def _generate_predict_py_finetune(
    code_dir: Path,
    trained: TrainedModel,
    profile: DatasetProfile,
    split: SplitData,
) -> None:
    """Generate standalone predict.py for AIDO fine-tuned models."""
    hparams = trained.config.hyperparameters
    is_clf = trained.config.task_type == "classification"
    aido_model = hparams.get("model_name", "")
    head_hidden = hparams.get("head_hidden", 256)
    head_dropout = hparams.get("head_dropout", 0.3)
    max_length = hparams.get("max_length", 1024)
    bs = hparams.get("batch_size", 16)
    n_classes_or_1 = profile.num_classes if is_clf else 1
    pred_code = "out.argmax(dim=1)" if is_clf else "out.squeeze(1)"

    script = f'''"""Standalone prediction script — AIDO fine-tuned model.

REQUIRES: CUDA-capable GPU + modelgenerator package

Usage:
    python predict.py --input data.csv --output predictions.csv
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from modelgenerator.tasks import Embed

AIDO_MODEL = "{aido_model}"
HEAD_HIDDEN = {head_hidden}
HEAD_DROPOUT = {head_dropout}
MAX_LENGTH = {max_length}
BATCH_SIZE = {bs}
OUTPUT_DIM = {n_classes_or_1}


class TaskHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=HEAD_HIDDEN, dropout=HEAD_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def forward_pass(embed_model, head, batch_seqs):
    transformed = embed_model.transform({{"sequences": batch_seqs}})
    if isinstance(transformed, dict):
        transformed = {{k: v.to(device) if hasattr(v, "to") else v for k, v in transformed.items()}}
    embeddings = embed_model(transformed)
    if isinstance(embeddings, torch.Tensor):
        pooled = embeddings.mean(dim=1) if embeddings.ndim == 3 else embeddings
    elif isinstance(embeddings, dict):
        for k in ("embeddings", "last_hidden_state"):
            if k in embeddings:
                e = embeddings[k]
                pooled = e.mean(dim=1) if e.ndim == 3 else e
                break
    else:
        pooled = torch.tensor(np.array(embeddings), dtype=torch.float32)
    return head(pooled)


def main():
    parser = argparse.ArgumentParser(description="Predict with AIDO fine-tuned model")
    parser.add_argument("--input", required=True, help="CSV with sequence column")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default="model/best_model.pt")
    parser.add_argument("--seq-column", default="sequence")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for AIDO model inference")
    device = torch.device("cuda")

    # Load AIDO backbone
    print(f"Loading AIDO model: {{AIDO_MODEL}}")
    embed_model = Embed.from_config({{"model.backbone": AIDO_MODEL}}).eval()

    # Probe embed dim
    embed_model.eval()
    with torch.no_grad():
        _probe = embed_model.transform({{"sequences": ["ACGT"]}})
        out = embed_model(_probe)
        if isinstance(out, torch.Tensor):
            embed_dim = out.shape[-1]
        elif isinstance(out, dict):
            for k in ("embeddings", "last_hidden_state"):
                if k in out:
                    embed_dim = out[k].shape[-1]
                    break
        else:
            embed_dim = 768

    head = TaskHead(embed_dim, OUTPUT_DIM)

    # Load saved weights
    ckpt = torch.load(args.model, weights_only=True, map_location="cpu")
    if "embed_model" in ckpt:
        embed_model.load_state_dict(ckpt["embed_model"])
    if "head" in ckpt:
        head.load_state_dict(ckpt["head"])

    embed_model = embed_model.to(device)
    head = head.to(device)
    embed_model.eval()
    head.eval()

    # Load input
    df = pd.read_csv(args.input)
    sequences = df[args.seq_column].astype(str).tolist()
    print(f"Running inference on {{len(sequences)}} sequences...")

    all_preds = []
    with torch.no_grad():
        for start in range(0, len(sequences), BATCH_SIZE):
            batch = [s[:MAX_LENGTH] for s in sequences[start:start+BATCH_SIZE]]
            with torch.amp.autocast("cuda"):
                out = forward_pass(embed_model, head, batch)
            all_preds.append({pred_code}.cpu().numpy())

    predictions = np.concatenate(all_preds)
'''

    if is_clf:
        script += '''
    try:
        with open("model/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        predictions = le.inverse_transform(predictions)
    except FileNotFoundError:
        pass
'''

    script += f'''
    out_df = pd.DataFrame({{"prediction": predictions}})
    out_df.to_csv(args.output, index=False)
    print(f"Saved {{len(predictions)}} predictions to {{args.output}}")


if __name__ == "__main__":
    main()
'''
    (code_dir / "predict.py").write_text(script)


def _generate_train_py_concat(
    code_dir: Path,
    trained: TrainedModel,
    profile: DatasetProfile,
    split: SplitData,
    preprocessing_steps: list[str],
) -> None:
    """Generate standalone train.py for concat models (handcrafted + AIDO embeddings)."""
    model_type = trained.config.model_type
    hparams = trained.config.hyperparameters
    task = trained.config.task_type
    is_clf = task == "classification"

    try:
        from co_scientist.data.loader import resolve_dataset_path
        hf_repo, hf_subset, _, _ = resolve_dataset_path(profile.dataset_path)
    except Exception:
        hf_repo = profile.dataset_path
        hf_subset = ""

    from co_scientist.defaults import get_defaults
    fm_config = get_defaults(profile.modality.value, profile.dataset_path).get("foundation_models", {})
    from co_scientist.modeling.foundation import get_foundation_model_name
    aido_model = get_foundation_model_name(profile.modality.value, fm_config)

    # Reuse the downstream model builder
    base_type = model_type.replace("concat_", "")
    model_code, model_imports = _build_model_code(base_type, task, hparams)

    # Get the handcrafted preprocessing code
    preprocess_code = _build_preprocess_code(profile, split)

    script = f'''"""Standalone training script — concat (handcrafted features + AIDO embeddings).

Dataset: {profile.dataset_name}
Source:  {hf_repo} (subset: {hf_subset})
Model:   {trained.config.name} ({trained.config.tier})
Task:    {profile.task_type.value}
Foundation Model: {aido_model}
Strategy: Concatenate handcrafted features + AIDO embeddings

REQUIRES: CUDA-capable GPU + modelgenerator package

Usage:
    python train.py
    python train.py --seed 42
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from modelgenerator.tasks import Embed
{model_imports}

warnings.filterwarnings("ignore")

DATASET_PATH = "{hf_repo}"
DATASET_SUBSET = "{hf_subset}"
TARGET_COLUMN = "{profile.target_column}"
TASK_TYPE = "{profile.task_type.value}"
AIDO_MODEL = "{aido_model}"
SEED = 42


def extract_embeddings(sequences, model_name=AIDO_MODEL, batch_size=32, max_length=1024):
    """Extract embeddings using AIDO foundation model."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for foundation model embeddings")

    print(f"Loading AIDO model: {{model_name}}")
    model = Embed.from_config({{"model.backbone": model_name}}).eval()
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)

    all_embeddings = []
    n = len(sequences)
    print(f"Extracting embeddings for {{n}} samples...")

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = sequences[start:start + batch_size]
            batch = [s[:max_length] if isinstance(s, str) else s for s in batch]
            transformed = model.transform({{"sequences": batch}})
            if isinstance(transformed, dict):
                transformed = {{k: v.to(device) if hasattr(v, "to") else v for k, v in transformed.items()}}
            embeddings = model(transformed)
            if isinstance(embeddings, torch.Tensor):
                emb_np = embeddings.cpu().numpy()
            elif isinstance(embeddings, dict):
                emb_tensor = embeddings.get("embeddings", embeddings.get("last_hidden_state"))
                if emb_tensor.ndim == 3:
                    emb_np = emb_tensor.mean(dim=1).cpu().numpy()
                else:
                    emb_np = emb_tensor.cpu().numpy()
            else:
                emb_np = np.array(embeddings)
            all_embeddings.append(emb_np)

    del model
    torch.cuda.empty_cache()
    result = np.vstack(all_embeddings)
    print(f"Embeddings shape: {{result.shape}}")
    return result


def load_and_preprocess(seed=SEED):
    """Load dataset, compute handcrafted features + AIDO embeddings, concatenate, and split."""
    print(f"Loading dataset: {{DATASET_PATH}} (subset: {{DATASET_SUBSET}})")
    if DATASET_SUBSET:
        ds = load_dataset(DATASET_PATH, DATASET_SUBSET)
    else:
        ds = load_dataset(DATASET_PATH)

    split_names = list(ds.keys())
    has_multiple_splits = len(split_names) > 1

    if has_multiple_splits:
        frames = []
        for split_name in split_names:
            df_split = ds[split_name].to_pandas()
            if split_name.startswith("test"):
                df_split["_split"] = "test"
            elif split_name in ("validation", "valid", "val"):
                df_split["_split"] = "valid"
            else:
                df_split["_split"] = split_name
            frames.append(df_split)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = ds[split_names[0]].to_pandas()

    y = df[TARGET_COLUMN].values
    X_df = df.drop(columns=[TARGET_COLUMN, "_split", "fold_id", "fold"], errors="ignore")

    # --- Step 1: Handcrafted features ---
{preprocess_code}
    X_handcrafted = X
    print(f"  Handcrafted features: {{X_handcrafted.shape[1]}}")

    # --- Step 2: AIDO embeddings ---
    # Find sequence column for embedding extraction
    seq_col = None
    for c in X_df.columns:
        if c.lower() in ("sequences", "sequence", "seq"):
            seq_col = c
            break
    if seq_col is None:
        for c in X_df.columns:
            if X_df[c].dtype == object:
                seq_col = c
                break
    if seq_col is not None:
        sequences = X_df[seq_col].astype(str).tolist()
        X_embed = extract_embeddings(sequences)
    else:
        # Expression/tabular data: pass raw numeric values
        X_embed = extract_embeddings(X_df.select_dtypes(include=[np.number]).values)
    print(f"  AIDO embeddings: {{X_embed.shape[1]}}")

    # --- Step 3: Concatenate ---
    X = np.hstack([X_handcrafted, X_embed])
    print(f"  Concatenated features: {{X.shape[1]}} (handcrafted + embeddings)")

    # Encode labels
    label_encoder = None
    if TASK_TYPE in ("binary_classification", "multiclass_classification"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Split
    if has_multiple_splits and "_split" in df.columns:
        train_mask = df["_split"].values == "train"
        val_mask = df["_split"].values == "valid"
        test_mask = df["_split"].values == "test"
        if not val_mask.any():
            train_indices = np.where(train_mask)[0]
            n_val = max(1, int(len(train_indices) * 0.2))
            rng = np.random.RandomState(seed)
            rng.shuffle(train_indices)
            val_mask = np.zeros(len(X), dtype=bool)
            val_mask[train_indices[:n_val]] = True
            train_mask = np.zeros(len(X), dtype=bool)
            train_mask[train_indices[n_val:]] = True
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    print(f"Train: {{len(X_train)}}, Val: {{len(X_val)}}, Test: {{len(X_test)}}")
    print(f"Combined features: {{X_train.shape[1]}}")
    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder


def build_model(seed=SEED):
    """Construct the downstream model."""
{model_code}
    return model


def main():
    parser = argparse.ArgumentParser(description="Reproduce training (concat features + embeddings)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess(args.seed)

    print(f"\\nTraining {{type(build_model()).__name__}} on concatenated features + AIDO embeddings...")
    model = build_model(args.seed)
    model.fit(X_train, y_train)
    print("  Training complete.")

    y_pred = model.predict(X_val)
    print(f"\\nValidation results:")
    _evaluate(y_val, y_pred)

    y_test_pred = model.predict(X_test)
    print(f"\\nTest results:")
    _evaluate(y_test, y_test_pred)

    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\\nModel saved to models/best_model.pkl")

    if label_encoder is not None:
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)


def _evaluate(y_true, y_pred):
    if TASK_TYPE in ("binary_classification", "multiclass_classification"):
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"  Accuracy:    {{acc:.4f}}")
        print(f"  Macro F1:    {{mf1:.4f}}")
        print(f"  Weighted F1: {{wf1:.4f}}")
    else:
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_squared_error, r2_score
        sp = spearmanr(y_true, y_pred).correlation
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        print(f"  Spearman: {{sp:.4f}}")
        print(f"  RMSE:     {{rmse:.4f}}")
        print(f"  R²:       {{r2:.4f}}")


if __name__ == "__main__":
    main()
'''
    (code_dir / "train.py").write_text(script)


def _generate_predict_py_concat(
    code_dir: Path,
    trained: TrainedModel,
    profile: DatasetProfile,
    split: SplitData,
) -> None:
    """Generate standalone predict.py for concat models."""
    is_clf = trained.config.task_type == "classification"

    from co_scientist.defaults import get_defaults
    fm_config = get_defaults(profile.modality.value, profile.dataset_path).get("foundation_models", {})
    from co_scientist.modeling.foundation import get_foundation_model_name
    aido_model = get_foundation_model_name(profile.modality.value, fm_config)

    preprocess_code = _build_preprocess_code(profile, split)

    script = f'''"""Standalone prediction script — concat (handcrafted features + AIDO embeddings).

REQUIRES: CUDA-capable GPU + modelgenerator package

Usage:
    python predict.py --input data.csv --output predictions.csv
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from modelgenerator.tasks import Embed

AIDO_MODEL = "{aido_model}"


def extract_embeddings(sequences, model_name=AIDO_MODEL, batch_size=32, max_length=1024):
    """Extract embeddings using AIDO foundation model."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for foundation model embeddings")

    model = Embed.from_config({{"model.backbone": model_name}}).eval()
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = sequences[start:start + batch_size]
            batch = [s[:max_length] if isinstance(s, str) else s for s in batch]
            transformed = model.transform({{"sequences": batch}})
            if isinstance(transformed, dict):
                transformed = {{k: v.to(device) if hasattr(v, "to") else v for k, v in transformed.items()}}
            embeddings = model(transformed)
            if isinstance(embeddings, torch.Tensor):
                emb_np = embeddings.cpu().numpy()
            elif isinstance(embeddings, dict):
                emb_tensor = embeddings.get("embeddings", embeddings.get("last_hidden_state"))
                if emb_tensor.ndim == 3:
                    emb_np = emb_tensor.mean(dim=1).cpu().numpy()
                else:
                    emb_np = emb_tensor.cpu().numpy()
            else:
                emb_np = np.array(embeddings)
            all_embeddings.append(emb_np)

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Predict with concat model (features + embeddings)")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    parser.add_argument("--model", default="model/best_model.pkl", help="Path to model file")
    parser.add_argument("--seq-column", default="sequence", help="Name of sequence column")
    args = parser.parse_args()

    # Load downstream model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Load input data
    X_df = pd.read_csv(args.input)

    # --- Step 1: Handcrafted features ---
{preprocess_code}
    X_handcrafted = X
    print(f"Handcrafted features: {{X_handcrafted.shape[1]}}")

    # --- Step 2: AIDO embeddings ---
    if args.seq_column in X_df.columns:
        sequences = X_df[args.seq_column].astype(str).tolist()
    else:
        # Try to find a sequence column
        seq_col = None
        for c in X_df.columns:
            if c.lower() in ("sequences", "sequence", "seq"):
                seq_col = c
                break
        if seq_col is None:
            for c in X_df.columns:
                if X_df[c].dtype == object:
                    seq_col = c
                    break
        if seq_col is not None:
            sequences = X_df[seq_col].astype(str).tolist()
        else:
            sequences = X_df.select_dtypes(include=[np.number]).values

    print(f"Extracting AIDO embeddings...")
    X_embed = extract_embeddings(sequences)
    print(f"AIDO embeddings: {{X_embed.shape[1]}}")

    # --- Step 3: Concatenate ---
    X = np.hstack([X_handcrafted, X_embed])
    print(f"Concatenated features: {{X.shape[1]}}")

    # Predict
    predictions = model.predict(X)
'''

    if is_clf:
        script += '''
    # Decode class labels if encoder exists
    try:
        with open("model/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        predictions = le.inverse_transform(predictions)
    except FileNotFoundError:
        pass
'''

    script += f'''
    # Save predictions
    out_df = pd.DataFrame({{"prediction": predictions}})
    out_df.to_csv(args.output, index=False)
    print(f"Saved {{len(predictions)}} predictions to {{args.output}}")


if __name__ == "__main__":
    main()
'''
    (code_dir / "predict.py").write_text(script)


def _build_model_code(model_type: str, task: str, hparams: dict) -> tuple[str, str]:
    """Return (model_construction_code, import_statements) for the given model type."""
    params_str = _format_params(hparams)
    is_clf = task == "classification"

    # Filter out parameters that MLP/neural models handle internally
    # (these are passed to the wrapper class, not to sklearn)
    _mlp_params = {"hidden_dims", "dropout", "learning_rate", "weight_decay",
                   "batch_size", "max_epochs", "patience", "random_state"}

    model_map = {
        "xgboost": (
            f"model = xgb.XGBClassifier({params_str}, eval_metric='mlogloss')" if is_clf
            else f"model = xgb.XGBRegressor({params_str})",
            "import xgboost as xgb",
        ),
        "lightgbm": (
            f"model = lgb.LGBMClassifier({params_str})" if is_clf
            else f"model = lgb.LGBMRegressor({params_str})",
            "import lightgbm as lgb",
        ),
        "random_forest": (
            f"model = RandomForestClassifier({params_str})" if is_clf
            else f"model = RandomForestRegressor({params_str})",
            "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
        ),
        "logistic_regression": (
            f"model = LogisticRegression({params_str})",
            "from sklearn.linear_model import LogisticRegression",
        ),
        "ridge_regression": (
            f"model = Ridge({params_str})",
            "from sklearn.linear_model import Ridge",
        ),
        "elastic_net_clf": (
            f"model = LogisticRegression(penalty='elasticnet', solver='saga', {params_str})",
            "from sklearn.linear_model import LogisticRegression",
        ),
        "elastic_net_reg": (
            f"model = ElasticNet({params_str})",
            "from sklearn.linear_model import ElasticNet",
        ),
        "svm": (
            f"model = SVC(probability=True, {params_str})" if is_clf
            else f"model = SVR({_format_params({k: v for k, v in hparams.items() if k != 'random_state'})})",
            "from sklearn.svm import SVC, SVR",
        ),
        "knn": (
            f"model = KNeighborsClassifier({_format_params({k: v for k, v in hparams.items() if k != 'random_state'})})" if is_clf
            else f"model = KNeighborsRegressor({_format_params({k: v for k, v in hparams.items() if k != 'random_state'})})",
            "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor",
        ),
        "majority_class": (
            "model = DummyClassifier(strategy='most_frequent')",
            "from sklearn.dummy import DummyClassifier",
        ),
        "mean_predictor": (
            "model = DummyRegressor(strategy='mean')",
            "from sklearn.dummy import DummyRegressor",
        ),
    }

    if model_type in model_map:
        code, imports = model_map[model_type]
        return f"    {code}", imports

    # MLP: construct the model from hyperparameters (trains from scratch)
    if model_type == "mlp":
        mlp_hp = {k: v for k, v in hparams.items() if k in _mlp_params}
        mlp_params_str = _format_params(mlp_hp)
        if is_clf:
            return (
                f"    model = MLPClassifier({mlp_params_str})",
                _MLP_IMPORT_CODE,
            )
        else:
            return (
                f"    model = MLPRegressor({mlp_params_str})",
                _MLP_IMPORT_CODE,
            )

    # FT-Transformer: construct from hyperparameters
    if model_type == "ft_transformer":
        ft_hp = {k: v for k, v in hparams.items()}
        ft_params_str = _format_params(ft_hp)
        if is_clf:
            return (
                f"    model = FTTransformerClassifier({ft_params_str})",
                _FT_TRANSFORMER_IMPORT_CODE,
            )
        else:
            return (
                f"    model = FTTransformerRegressor({ft_params_str})",
                _FT_TRANSFORMER_IMPORT_CODE,
            )

    # Custom models: load from state_dict + source code
    if model_type == "custom":
        return (
            f"    # Custom model — load from saved state_dict + source code\n"
            f"    import torch, importlib.util, sys\n"
            f"    spec = importlib.util.spec_from_file_location('custom_model', 'models/custom_model.py')\n"
            f"    mod = importlib.util.module_from_spec(spec)\n"
            f"    spec.loader.exec_module(mod)\n"
            f"    # Find the model class\n"
            f"    import inspect\n"
            f"    cls = [c for _, c in inspect.getmembers(mod, inspect.isclass) if hasattr(c, 'fit')][0]\n"
            f"    model = cls({params_str})\n"
            f"    model.model.load_state_dict(torch.load('models/best_model.pt', weights_only=True))",
            "import torch",
        )

    # Fallback: load from pickle (stacking, etc.)
    return (
        f"    # Model type: {model_type} — loading pre-trained model\n"
        f"    with open('models/best_model.pkl', 'rb') as f:\n"
        f"        model = pickle.load(f)",
        "",
    )


# ---------------------------------------------------------------------------
# Inline MLP code for standalone train.py (no co-scientist dependency)
# ---------------------------------------------------------------------------

_MLP_IMPORT_CODE = '''
import torch
import torch.nn as nn

class _MLPModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def _batches(X, y, batch_size, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start+batch_size]
        yield X[batch_idx], y[batch_idx]

class MLPClassifier:
    def __init__(self, hidden_dims=None, dropout=0.3, learning_rate=1e-3,
                 weight_decay=1e-4, batch_size=64, max_epochs=100,
                 patience=10, random_state=42):
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._model = None
        self._classes = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._model = _MLPModule(X.shape[1], n_classes, self.hidden_dims, self.dropout)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        best_loss, wait = float("inf"), 0
        for epoch in range(self.max_epochs):
            self._model.train()
            epoch_loss = 0.0
            for xb, yb in _batches(X, y, self.batch_size):
                xb_t, yb_t = torch.FloatTensor(xb), torch.LongTensor(yb)
                opt.zero_grad()
                loss = loss_fn(self._model(xb_t), yb_t)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return self

    def predict(self, X):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.FloatTensor(X))
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.FloatTensor(X))
        return torch.softmax(logits, dim=1).numpy()

class MLPRegressor:
    def __init__(self, hidden_dims=None, dropout=0.3, learning_rate=1e-3,
                 weight_decay=1e-4, batch_size=64, max_epochs=100,
                 patience=10, random_state=42):
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._model = _MLPModule(X.shape[1], 1, self.hidden_dims, self.dropout)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best_loss, wait = float("inf"), 0
        for epoch in range(self.max_epochs):
            self._model.train()
            epoch_loss = 0.0
            for xb, yb in _batches(X, y, self.batch_size):
                xb_t = torch.FloatTensor(xb)
                yb_t = torch.FloatTensor(yb).unsqueeze(1)
                opt.zero_grad()
                loss = loss_fn(self._model(xb_t), yb_t)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return self

    def predict(self, X):
        self._model.eval()
        with torch.no_grad():
            return self._model(torch.FloatTensor(X)).squeeze(1).numpy()
'''

_FT_TRANSFORMER_IMPORT_CODE = '''import torch
# FT-Transformer requires co-scientist installation or the model pickle
'''


def _build_h5ad_load_code(hf_repo: str, hf_subset: str, profile: DatasetProfile, split: SplitData) -> str:
    """Generate load_and_preprocess() for h5ad (expression) datasets."""
    n_hvg = split.X_train.shape[1]  # use the same number of features as the pipeline
    return f'''
def load_and_preprocess(seed: int = SEED):
    """Load h5ad dataset from HuggingFace and preprocess."""
    print(f"Loading dataset: {{DATASET_PATH}} (subset: {{DATASET_SUBSET}})")

    # Download h5ad split files
    split_data = {{}}
    for split_name in ["train", "valid", "test"]:
        subset_lower = DATASET_SUBSET.lower()
        for pattern in [f"{{DATASET_SUBSET}}_{{split_name}}.h5ad",
                        f"{{DATASET_SUBSET}}/{{DATASET_SUBSET}}_{{split_name}}.h5ad",
                        f"{{subset_lower}}_{{split_name}}.h5ad",
                        f"{{subset_lower}}/{{subset_lower}}_{{split_name}}.h5ad"]:
            try:
                path = hf_hub_download(repo_id=DATASET_PATH, filename=pattern, repo_type="dataset")
                split_data[split_name] = ad.read_h5ad(path)
                print(f"  Loaded {{split_name}}: {{split_data[split_name].shape[0]}} cells")
                break
            except Exception:
                continue

    if not split_data:
        raise FileNotFoundError(f"Could not download h5ad files for {{DATASET_SUBSET}} from {{DATASET_PATH}}")

    # Concatenate all splits
    for name, adata in split_data.items():
        adata.obs["_split"] = name
    adata_all = ad.concat(list(split_data.values()), join="outer")

    # Extract expression matrix
    X_dense = adata_all.X.toarray() if hasattr(adata_all.X, "toarray") else np.array(adata_all.X)
    y = adata_all.obs[TARGET_COLUMN].values
    splits = adata_all.obs["_split"].values

    print(f"  Total: {{len(y)}} cells, {{X_dense.shape[1]}} genes")

    # Preprocessing: log1p + HVG selection + scaling
    N_HVG = {n_hvg}
    X_log = np.log1p(X_dense)
    gene_vars = np.var(X_log, axis=0)
    top_idx = np.argsort(gene_vars)[-N_HVG:]
    X = X_log[:, top_idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"  After HVG selection + scaling: {{X.shape[1]}} features")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Use predefined splits from h5ad files
    train_mask = splits == "train"
    val_mask = splits == "valid"
    test_mask = splits == "test"

    X_train, y_train = X[train_mask], y_encoded[train_mask]
    X_val, y_val = X[val_mask], y_encoded[val_mask]
    X_test, y_test = X[test_mask], y_encoded[test_mask]

    print(f"  Train: {{len(X_train)}}, Val: {{len(X_val)}}, Test: {{len(X_test)}}")
    print(f"  Features: {{X_train.shape[1]}}")
    print(f"  Classes: {{len(label_encoder.classes_)}}")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
'''


def _build_hf_load_code(hf_repo: str, hf_subset: str, profile: DatasetProfile, preprocess_code: str) -> str:
    """Generate load_and_preprocess() for standard HuggingFace datasets.

    Handles datasets with:
    - Standard train/validation/test splits
    - Species-specific test splits (test_danio, test_fly, etc.)
    - Single split with fold_id column
    - Single split (falls back to random 70/15/15)
    """
    return f'''
def load_and_preprocess(seed: int = SEED):
    """Load dataset from HuggingFace and preprocess."""
    print(f"Loading dataset: {{DATASET_PATH}} (subset: {{DATASET_SUBSET}})")
    if DATASET_SUBSET:
        ds = load_dataset(DATASET_PATH, DATASET_SUBSET)
    else:
        ds = load_dataset(DATASET_PATH)

    # Merge all splits into one DataFrame, preserving split assignments
    split_names = list(ds.keys())
    has_multiple_splits = len(split_names) > 1

    if has_multiple_splits:
        frames = []
        for split_name in split_names:
            df_split = ds[split_name].to_pandas()
            # Normalize split names: test_* → test, validation/valid → valid
            if split_name.startswith("test"):
                df_split["_split"] = "test"
            elif split_name in ("validation", "valid", "val"):
                df_split["_split"] = "valid"
            else:
                df_split["_split"] = split_name
            frames.append(df_split)
        df = pd.concat(frames, ignore_index=True)
        print(f"  Loaded {{len(df)}} samples from {{len(split_names)}} splits: {{split_names}}")
    else:
        df = ds[split_names[0]].to_pandas()
        print(f"  Loaded {{len(df)}} samples")

    # Separate target
    y = df[TARGET_COLUMN].values
    X_df = df.drop(columns=[TARGET_COLUMN, "_split", "fold_id", "fold"], errors="ignore")

{preprocess_code}

    # Encode labels for classification
    label_encoder = None
    if TASK_TYPE in ("binary_classification", "multiclass_classification"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Use predefined splits if available, otherwise random 70/15/15
    if has_multiple_splits and "_split" in df.columns:
        train_mask = df["_split"].values == "train"
        val_mask = df["_split"].values == "valid"
        test_mask = df["_split"].values == "test"

        # Handle missing validation split
        if not val_mask.any():
            print("  No validation split found — carving 20% from train")
            train_indices = np.where(train_mask)[0]
            n_val = max(1, int(len(train_indices) * 0.2))
            rng = np.random.RandomState(seed)
            rng.shuffle(train_indices)
            val_indices = train_indices[:n_val]
            train_indices = train_indices[n_val:]
            val_mask = np.zeros(len(X), dtype=bool)
            val_mask[val_indices] = True
            train_mask = np.zeros(len(X), dtype=bool)
            train_mask[train_indices] = True

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    elif "fold_id" in df.columns:
        folds = df["fold_id"].values
        unique_folds = sorted(set(folds))
        test_fold = unique_folds[-1]
        val_fold = unique_folds[-2] if len(unique_folds) > 1 else None
        test_mask = folds == test_fold
        val_mask = folds == val_fold if val_fold is not None else np.zeros(len(X), dtype=bool)
        train_mask = ~test_mask & ~val_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=seed,
            stratify=y if TASK_TYPE in ("binary_classification", "multiclass_classification") else None,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=seed,
            stratify=y_temp if TASK_TYPE in ("binary_classification", "multiclass_classification") else None,
        )

    print(f"  Train: {{len(X_train)}}, Val: {{len(X_val)}}, Test: {{len(X_test)}}")
    print(f"  Features: {{X_train.shape[1]}}")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
'''


def _build_preprocess_code(profile: DatasetProfile, split: SplitData) -> str:
    """Generate preprocessing code based on dataset modality."""
    modality = profile.modality.value

    if modality in ("rna", "dna", "protein"):
        # Sequence data — need k-mer feature extraction
        seq_col = profile.input_columns[0] if profile.input_columns else "sequence"
        return f'''    # --- Sequence feature extraction ---
    from collections import Counter
    from itertools import product

    seq_col = "{seq_col}"
    sequences = X_df[seq_col].astype(str).values

    # Build feature matrix
    features = []
    for k in (3, 4):
        alphabet = "ACGT"
        possible_kmers = [''.join(p) for p in product(alphabet, repeat=k)]
        k_features = []
        for seq in sequences:
            kmer_counts = Counter()
            for i in range(len(seq) - k + 1):
                kmer_counts[seq[i:i+k]] += 1
            total = max(sum(kmer_counts.values()), 1)
            k_features.append([kmer_counts.get(km, 0) / total for km in possible_kmers])
        features.append(np.array(k_features))

    # Additional features: length, GC content, nucleotide composition
    lengths = np.array([len(s) for s in sequences]).reshape(-1, 1)
    gc_content = np.array([
        (s.count("G") + s.count("C")) / max(len(s), 1) for s in sequences
    ]).reshape(-1, 1)
    nuc_freqs = np.array([
        [s.count(n) / max(len(s), 1) for n in "ACGT"] for s in sequences
    ])

    X = np.hstack(features + [lengths, gc_content, nuc_freqs])

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"  Extracted {{X.shape[1]}} features (k-mer + sequence properties)")'''

    else:
        # Tabular / expression data
        return '''    # --- Tabular feature extraction ---
    # Select numeric columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    X = X_df[numeric_cols].values.astype(np.float64)

    # Handle missing values
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_means[j]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"  Using {X.shape[1]} numeric features")'''


def _generate_requirements(
    output_dir: Path, trained: TrainedModel, profile: DatasetProfile | None = None, minimal: bool = False,
) -> None:
    """Generate requirements.txt.

    minimal=False (reproduce): includes datasets, scipy, full stack
    minimal=True  (inference): just what's needed to load model + predict
    """
    import sklearn
    model_type = trained.config.model_type
    is_h5ad = profile and profile.modality.value == "cell_expression"

    if minimal:
        # Inference only — no datasets, no scipy (unless needed by model)
        reqs = [
            f"scikit-learn>={sklearn.__version__}",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
        ]
    else:
        # Full reproduction
        reqs = [
            f"scikit-learn>={sklearn.__version__}",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scipy>=1.11.0",
        ]
        if is_h5ad:
            reqs.extend(["anndata>=0.10.0", "huggingface_hub>=0.20.0"])
        else:
            reqs.append("datasets>=2.14.0")

    # Model-specific dependencies
    if model_type in ("xgboost", "embed_xgboost"):
        import xgboost
        reqs.append(f"xgboost>={xgboost.__version__}")
    elif model_type == "lightgbm":
        import lightgbm
        reqs.append(f"lightgbm>={lightgbm.__version__}")
    elif model_type in ("mlp", "bio_cnn", "ft_transformer", "embed_mlp"):
        reqs.append("torch>=2.0.0")

    # Foundation model dependencies
    if model_type in ("embed_xgboost", "embed_mlp", "aido_finetune", "concat_xgboost", "concat_mlp"):
        reqs.append("torch>=2.0.0")
        reqs.append("modelgenerator")
        reqs.append("transformers>=4.36.0")

    (output_dir / "requirements.txt").write_text("\n".join(reqs) + "\n")


def _format_params(params: dict) -> str:
    """Format a dict of hyperparameters as Python keyword arguments."""
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
