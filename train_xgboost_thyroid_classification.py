"""Reusable XGBoost thyroid-classification workflow for notebook and CLI use.

The module owns cleaning, role-based encoding, fold-wise Information Gain,
checkpointed hyperparameter search, evaluation, and artifact writing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import random
import shutil
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib_cache")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 0. Imports, constants, and global settings
# =============================================================================

DEFAULT_DATA_PATH = Path(__file__).with_name("Thyroid_Diff.csv")
DEFAULT_TARGET_COL = "Recurred"
DEFAULT_CV_SPLITS = 10
DEFAULT_SEED = 42
DEFAULT_KEEP_RUNS = 1
DEFAULT_DEVICE = "cpu"
DEFAULT_TREE_METHOD = "auto"
DEFAULT_SELECTION_METRIC = "accuracy"
METRIC_DISPLAY_DECIMALS = 4
DEFAULT_IG_TOP_K = ["all", *range(10, 21)]
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_METADATA_NAME = "grid_search_checkpoint.json"
GRID_SEARCH_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "roc_auc",
)
FINAL_REPORT_METRICS = (
    *GRID_SEARCH_METRICS,
    "mcc",
    "pr_auc",
    "brier_score",
)
OPTIONAL_XGBOOST_PARAMS = (
    "base_score",
    "gamma",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
    "scale_pos_weight",
    "max_bin",
)
NUMERIC_FEATURES = ("Age",)
BINARY_FEATURE_MAPPINGS = {
    "Gender": {"F": 0, "M": 1},
    "Smoking": {"No": 0, "Yes": 1},
    "Hx Smoking": {"No": 0, "Yes": 1},
    "Hx Radiothreapy": {"No": 0, "Yes": 1},
    "Focality": {"Uni-Focal": 0, "Multi-Focal": 1},
}
ORDINAL_FEATURE_MAPPINGS = {
    "Risk": {"Low": 0, "Intermediate": 1, "High": 2},
    "T": {"T1a": 0, "T1b": 1, "T2": 2, "T3a": 3, "T3b": 4, "T4a": 5, "T4b": 6},
    "N": {"N0": 0, "N1a": 1, "N1b": 2},
    "M": {"M0": 0, "M1": 1},
    "Stage": {"I": 0, "II": 1, "III": 2, "IVA": 3, "IVB": 4},
    "Response": {
        "Excellent": 0,
        "Indeterminate": 1,
        "Biochemical Incomplete": 2,
        "Structural Incomplete": 3,
    },
}
NOMINAL_FEATURES = (
    "Thyroid Function",
    "Physical Examination",
    "Adenopathy",
    "Pathology",
)
NOMINAL_FEATURE_CATEGORIES = {
    "Thyroid Function": (
        "Clinical Hyperthyroidism",
        "Clinical Hypothyroidism",
        "Euthyroid",
        "Subclinical Hyperthyroidism",
        "Subclinical Hypothyroidism",
    ),
    "Physical Examination": (
        "Diffuse goiter",
        "Multinodular goiter",
        "Normal",
        "Single nodular goiter-left",
        "Single nodular goiter-right",
    ),
    "Adenopathy": (
        "Bilateral",
        "Extensive",
        "Left",
        "No",
        "Posterior",
        "Right",
    ),
    "Pathology": (
        "Follicular",
        "Hurthel cell",
        "Micropapillary",
        "Papillary",
    ),
}


# =============================================================================
# 1. Runtime bootstrap and warning handling
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*CUDA path could not be detected.*",
    category=UserWarning,
)


# =============================================================================
# 2. CLI configuration and validation
# =============================================================================

def parse_top_k(value: str) -> int | str:
    normalized = str(value).strip().lower()
    if normalized == "all":
        return "all"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ig-top-k values must be positive integers or 'all'.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--ig-top-k values must be positive integers or 'all'.")
    return parsed


def parse_args() -> argparse.Namespace:
    """Build the command-line interface for the classification workflow."""
    parser = argparse.ArgumentParser(
        description=(
            "Thyroid cancer recurrence classification with XGBoost, "
            "Information Gain feature selection, and grid search."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument(
        "--exclude-features",
        nargs="*",
        default=[],
        help=(
            "Optional feature columns to exclude before modeling. Useful for "
            "sensitivity checks when a variable may not be available at "
            "prediction time."
        ),
    )
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument("--positive-class", default="Yes")
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "f1", "precision", "recall", "roc_auc"],
        default=DEFAULT_SELECTION_METRIC,
        help="Metric used to choose the best checkpointed grid-search combination.",
    )
    parser.add_argument("--ig-top-k", nargs="+", type=parse_top_k, default=list(DEFAULT_IG_TOP_K))
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--max-depths", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--n-estimators", nargs="+", type=int, default=[100, 200, 300, 400, 500])
    parser.add_argument("--subsamples", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--colsample-bytree", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--base-score", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--min-child-weight", type=float, default=None)
    parser.add_argument("--reg-alpha", type=float, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--scale-pos-weight", type=float, default=None)
    parser.add_argument("--max-bin", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default=DEFAULT_DEVICE,
        help=(
            "Compute device. 'auto' keeps the public XGBoost default, 'cpu' "
            "forces CPU execution, and 'gpu' uses XGBoost's CUDA device."
        ),
    )
    parser.add_argument(
        "--tree-method",
        choices=["auto", "exact", "approx", "hist"],
        default=DEFAULT_TREE_METHOD,
        help=(
            "XGBoost tree construction method. GPU mode requires 'hist'. "
            "CPU mode can use auto, exact, approx, or hist."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--search-workers",
        type=int,
        default=-1,
        help=(
            "Number of independent grid combinations evaluated in parallel. "
            "Use -1 for every logical CPU thread. Parallel search assigns one "
            "XGBoost thread to each worker to avoid CPU oversubscription."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional fixed output folder name inside --output-dir. Use this "
            "for checkpoint/resume runs."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print checkpoint progress every N completed grid combinations.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--keep-runs", type=int, default=DEFAULT_KEEP_RUNS)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cv_splits < 2:
        raise ValueError("--cv-splits must be at least 2.")
    if getattr(args, "selection_metric", DEFAULT_SELECTION_METRIC) not in {"accuracy", "f1", "precision", "recall", "roc_auc"}:
        raise ValueError("--selection-metric must be one of: accuracy, f1, precision, recall, roc_auc.")
    if any(value <= 0 for value in args.learning_rates):
        raise ValueError("--learning-rates must contain positive values.")
    if any(value < 1 for value in args.max_depths):
        raise ValueError("--max-depths must contain positive values.")
    if any(value < 1 for value in args.n_estimators):
        raise ValueError("--n-estimators must contain positive values.")
    if any(value <= 0 or value > 1 for value in args.subsamples):
        raise ValueError("--subsamples values must be in the range (0, 1].")
    if any(value <= 0 or value > 1 for value in args.colsample_bytree):
        raise ValueError("--colsample-bytree values must be in the range (0, 1].")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs must not be 0. Use -1 for all available CPU threads.")
    if getattr(args, "search_workers", 1) == 0 or getattr(args, "search_workers", 1) < -1:
        raise ValueError("--search-workers must be -1 or a positive integer.")
    if getattr(args, "progress_every", 25) < 1:
        raise ValueError("--progress-every must be at least 1.")
    if getattr(args, "device", DEFAULT_DEVICE) not in {"auto", "cpu", "gpu"}:
        raise ValueError("--device must be one of: auto, cpu, gpu.")
    if getattr(args, "tree_method", DEFAULT_TREE_METHOD) not in {"auto", "exact", "approx", "hist"}:
        raise ValueError("--tree-method must be one of: auto, exact, approx, hist.")
    if getattr(args, "device", DEFAULT_DEVICE) == "gpu" and getattr(args, "tree_method", DEFAULT_TREE_METHOD) != "hist":
        raise ValueError(
            "device='gpu' requires tree_method='hist'. Use device='cpu' for "
            "manual checks with tree_method='exact'."
        )
    if getattr(args, "device", DEFAULT_DEVICE) == "gpu" and getattr(
        args, "search_workers", 1
    ) not in {-1, 1}:
        raise ValueError(
            "GPU mode accepts search_workers=-1 (automatic) or 1 so "
            "independent models do not compete for the same GPU memory."
        )
    fixed_params = xgboost_fixed_model_params(args)
    if "max_bin" in fixed_params and int(fixed_params["max_bin"]) < 2:
        raise ValueError("max_bin must be at least 2.")
    if args.keep_runs < 1:
        raise ValueError("--keep-runs must be at least 1.")
    if args.target_col in set(args.exclude_features):
        raise ValueError("--exclude-features must not include the target column.")


# =============================================================================
# 3. Reproducibility and compute-device helpers
# =============================================================================

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def xgboost_device_params(
    device: str = DEFAULT_DEVICE,
    tree_method: str = DEFAULT_TREE_METHOD,
) -> dict[str, str]:
    requested_device = str(device).lower()
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be one of: auto, cpu, gpu.")
    selected_tree_method = str(tree_method).lower()
    if selected_tree_method not in {"auto", "exact", "approx", "hist"}:
        raise ValueError("tree_method must be one of: auto, exact, approx, hist.")

    params = {"tree_method": selected_tree_method}
    if requested_device == "gpu":
        if selected_tree_method != "hist":
            raise ValueError("device='gpu' requires tree_method='hist'.")
        params["device"] = "cuda"
        return params
    if requested_device == "cpu":
        params["device"] = "cpu"
        return params
    return params


def effective_xgboost_device(device: str = DEFAULT_DEVICE) -> str:
    requested_device = str(device).lower()
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be one of: auto, cpu, gpu.")
    if requested_device == "auto":
        return "cpu"
    return requested_device


@lru_cache(maxsize=1)
def query_nvidia_gpus() -> tuple[dict[str, str], ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return tuple()

    gpus: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mb": parts[3],
            }
        )
    return tuple(gpus)


def query_cpu_info() -> dict[str, str]:
    processor = platform.processor().strip()
    machine = platform.machine().strip()
    system = platform.system().strip()

    generic_processor = (
        not processor
        or "family" in processor.lower()
        or processor.upper() in {"AMD64", "X86_64"}
    )
    if generic_processor and system.lower() == "windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                registry_processor, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            processor = str(registry_processor).strip()
        except (ImportError, FileNotFoundError, OSError):
            processor = ""

    generic_processor = (
        not processor
        or "family" in processor.lower()
        or processor.upper() in {"AMD64", "X86_64"}
    )
    if generic_processor and system.lower() == "windows":
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            processor = completed.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            processor = ""

    return {
        "processor": processor or "unknown",
        "machine": machine or "unknown",
        "system": system or "unknown",
    }


def xgboost_device_info(
    device: str = DEFAULT_DEVICE,
    tree_method: str = DEFAULT_TREE_METHOD,
) -> dict[str, object]:
    requested_device = str(device).lower()
    selected_tree_method = str(tree_method).lower()
    nvidia_gpus = query_nvidia_gpus() if requested_device == "gpu" else tuple()
    effective_device = effective_xgboost_device(requested_device)
    params = xgboost_device_params(requested_device, selected_tree_method)
    training_device = "cuda:0" if effective_device == "gpu" else "cpu"
    array_backend = (
        "cupy_gpu_arrays_for_fit_and_predict"
        if effective_device == "gpu"
        else "numpy_cpu_arrays_for_fit_and_predict"
    )
    return {
        "requested_device": requested_device,
        "effective_device": effective_device,
        "xgboost_device_parameter": params.get("device", "xgboost_default_cpu"),
        "xgboost_tree_method_parameter": params["tree_method"],
        "xgboost_training_device": training_device,
        "array_backend": array_backend,
        "cpu_info": query_cpu_info() if effective_device == "cpu" else {},
        "nvidia_smi_available": bool(nvidia_gpus),
        "nvidia_smi_checked": requested_device == "gpu",
        "cuda_discrete_gpu_required": requested_device == "gpu",
        "nvidia_gpus": nvidia_gpus,
        "note": (
            "GPU training is requested only when device='gpu'. GPU mode checks "
            "NVIDIA/CUDA availability through nvidia-smi and uses CuPy arrays "
            "for fit and predict. CPU and auto modes use NumPy arrays and do "
            "not inspect local GPU hardware. Tree method is selected separately "
            "through the tree_method setting."
        ),
    }


def format_xgboost_device_report(device_info: dict[str, object]) -> str:
    lines = [
        "XGBoost runtime:",
        f"- requested_device: {device_info['requested_device']}",
        f"- effective_device: {device_info['effective_device']}",
        f"- xgboost_device_parameter: {device_info['xgboost_device_parameter']}",
        f"- xgboost_tree_method_parameter: {device_info['xgboost_tree_method_parameter']}",
        f"- xgboost_training_device: {device_info['xgboost_training_device']}",
        f"- array_backend: {device_info['array_backend']}",
    ]
    nvidia_gpus = device_info.get("nvidia_gpus", [])
    if nvidia_gpus:
        lines.append("- NVIDIA GPUs from nvidia-smi:")
        for gpu in nvidia_gpus:
            lines.append(
                "  "
                f"GPU {gpu['index']}: {gpu['name']} | "
                f"driver {gpu['driver_version']} | "
                f"memory_total {gpu['memory_total_mb']} MB"
            )
    elif device_info.get("nvidia_smi_checked"):
        lines.append("- nvidia-smi: not available or no NVIDIA GPU was reported")
    if device_info["effective_device"] == "cpu":
        cpu_info = device_info.get("cpu_info", {})
        lines.append("- CPU runtime:")
        lines.append(f"  processor: {cpu_info.get('processor', 'unknown')}")
        lines.append(f"  machine: {cpu_info.get('machine', 'unknown')}")
        lines.append(f"  system: {cpu_info.get('system', 'unknown')}")
    if device_info["requested_device"] == "gpu" and not device_info["nvidia_smi_available"]:
        lines.append(
            "- gpu_runtime_check: NVIDIA/CUDA GPU was requested, but nvidia-smi did not report one"
        )
    lines.append(f"- note: {device_info['note']}")
    return "\n".join(lines)


# =============================================================================
# 4. Data loading, audit, and preprocessing
# =============================================================================

def load_dataset(data_path: Path, target_col: str) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file was not found: {data_path}")
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the CSV.")
    ordered_columns = [column for column in df.columns if column != target_col] + [target_col]
    return df.loc[:, ordered_columns].copy()


def clean_classification_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows_before = len(df)
    duplicate_rows = int(df.duplicated().sum())
    missing_rows = int(df.isna().any(axis=1).sum())

    cleaned = df.drop_duplicates(keep="first").dropna(axis=0, how="any").reset_index(drop=True)
    stats = {
        "rows_before": rows_before,
        "duplicate_rows_removed": duplicate_rows,
        "missing_rows_before_cleaning": missing_rows,
        "rows_after": len(cleaned),
        "total_rows_removed": rows_before - len(cleaned),
    }
    return cleaned, stats


def raw_data_audit(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = [
        {"item": "rows", "value": len(df)},
        {"item": "columns", "value": df.shape[1]},
        {"item": "target_column", "value": target_col},
        {"item": "feature_columns", "value": df.shape[1] - 1},
        {"item": "duplicate_rows", "value": int(df.duplicated().sum())},
        {"item": "total_missing_values", "value": int(df.isna().sum().sum())},
    ]
    for label, count in df[target_col].value_counts(dropna=False).items():
        rows.append({"item": f"target_count_{label}", "value": int(count)})
    return pd.DataFrame(rows)


def preprocessing_summary_table(stats: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "rows_before_preprocessing", "value": stats["rows_before"]},
            {"item": "duplicate_rows_removed_keep_first", "value": stats["duplicate_rows_removed"]},
            {"item": "missing_rows_before_cleaning", "value": stats["missing_rows_before_cleaning"]},
            {"item": "rows_after_preprocessing", "value": stats["rows_after"]},
            {"item": "total_rows_removed", "value": stats["total_rows_removed"]},
        ]
    )


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    exclude_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    exclude_features = exclude_features or []
    missing_excluded = [col for col in exclude_features if col not in df.columns]
    if missing_excluded:
        raise ValueError(f"Excluded feature columns were not found: {missing_excluded}")
    x = df.drop(columns=[target_col, *exclude_features]).copy()
    y = df[target_col].copy()
    if y.isna().any():
        raise ValueError("The target column contains missing values.")
    return x, y


def encode_target(y: pd.Series, positive_class: str) -> tuple[np.ndarray, dict[str, object]]:
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str))
    classes = list(encoder.classes_)
    if positive_class in classes:
        positive_label = int(encoder.transform([positive_class])[0])
    elif len(classes) == 2:
        positive_label = 1
    else:
        positive_label = None
    metadata = {
        "classes": classes,
        "positive_class": positive_class if positive_class in classes else None,
        "positive_label": positive_label,
        "label_mapping": {label: int(encoder.transform([label])[0]) for label in classes},
        "encoder": encoder,
    }
    return encoded, metadata


def normalize_category_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def feature_role(feature: str, series: pd.Series | None = None) -> str:
    if feature in BINARY_FEATURE_MAPPINGS:
        return "binary"
    if feature in ORDINAL_FEATURE_MAPPINGS:
        return "ordinal"
    if feature in NOMINAL_FEATURES:
        return "nominal"
    if feature in NUMERIC_FEATURES:
        return "numeric"
    if series is not None and pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "nominal"


def feature_role_table(x: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in x.columns:
        role = feature_role(col, x[col])
        if role == "binary":
            encoding = "manual binary mapping"
        elif role == "ordinal":
            encoding = "manual ordinal mapping"
        elif role == "nominal":
            encoding = "one-hot encoding"
        else:
            encoding = "numeric passthrough"
        rows.append(
            {
                "feature": col,
                "role": role,
                "model_encoding": encoding,
                "unique_values": int(x[col].nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)


def map_ordered_feature(series: pd.Series, mapping: dict[str, int], feature: str) -> np.ndarray:
    normalized = normalize_category_series(series)
    mapped = normalized.map(mapping)
    unknown_values = sorted(normalized[mapped.isna()].dropna().unique().tolist())
    if unknown_values:
        raise ValueError(
            f"Feature '{feature}' contains values not defined in the mapping: {unknown_values}"
        )
    return mapped.to_numpy(dtype=float).reshape(-1, 1)


def encoded_feature_role(feature: str) -> str:
    if feature in NUMERIC_FEATURES:
        return "numeric"
    if feature in BINARY_FEATURE_MAPPINGS:
        return "binary_encoded"
    if feature in ORDINAL_FEATURE_MAPPINGS:
        return "ordinal_encoded"
    for original_feature in NOMINAL_FEATURES:
        if feature.startswith(f"{original_feature}_"):
            return "one_hot_encoded"
    return "numeric" if pd.api.types.is_numeric_dtype(feature) else "encoded"


def encode_feature_dataframe(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded_columns: dict[str, np.ndarray] = {}
    role_rows: list[dict[str, object]] = []

    for col in x.columns:
        role = feature_role(col, x[col])
        if role == "numeric":
            encoded_columns[col] = pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float)
            role_rows.append(
                {
                    "source_feature": col,
                    "encoded_feature": col,
                    "source_role": role,
                    "encoded_role": "numeric",
                    "encoding": "numeric passthrough",
                }
            )
        elif role == "binary":
            encoded_columns[col] = map_ordered_feature(x[col], BINARY_FEATURE_MAPPINGS[col], col).reshape(-1)
            role_rows.append(
                {
                    "source_feature": col,
                    "encoded_feature": col,
                    "source_role": role,
                    "encoded_role": "binary_encoded",
                    "encoding": "manual binary mapping",
                }
            )
        elif role == "ordinal":
            encoded_columns[col] = map_ordered_feature(x[col], ORDINAL_FEATURE_MAPPINGS[col], col).reshape(-1)
            role_rows.append(
                {
                    "source_feature": col,
                    "encoded_feature": col,
                    "source_role": role,
                    "encoded_role": "ordinal_encoded",
                    "encoding": "manual ordinal mapping",
                }
            )
        else:
            categories = NOMINAL_FEATURE_CATEGORIES[col]
            normalized = normalize_category_series(x[col])
            unknown_values = sorted(set(normalized.dropna()) - set(categories))
            if unknown_values:
                raise ValueError(
                    f"Feature '{col}' contains values not defined in the nominal category list: {unknown_values}"
                )
            for category in categories:
                encoded_name = f"{col}_{category}"
                encoded_columns[encoded_name] = normalized.eq(category).astype(int).to_numpy()
                role_rows.append(
                    {
                        "source_feature": col,
                        "encoded_feature": encoded_name,
                        "source_role": role,
                        "encoded_role": "one_hot_encoded",
                        "encoding": "manual one-hot encoding",
                    }
                )

    encoded_df = pd.DataFrame(encoded_columns, index=x.index)
    role_df = pd.DataFrame(role_rows)
    return encoded_df, role_df


def build_encoded_modeling_dataframe(
    encoded_x: pd.DataFrame,
    y_encoded: np.ndarray,
    target_col: str,
) -> pd.DataFrame:
    if len(encoded_x) != len(y_encoded):
        raise ValueError("Encoded features and target must have the same number of rows.")
    modeling_df = encoded_x.copy()
    modeling_df[target_col] = np.asarray(y_encoded, dtype=int)
    return modeling_df


# =============================================================================
# 5. Feature construction and model-data preparation
# =============================================================================

def compute_information_gain_table(
    x: pd.DataFrame,
    y_encoded: np.ndarray,
    random_state: int = DEFAULT_SEED,
) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer

    del random_state

    target_entropy = entropy_from_labels(y_encoded)
    rows: list[dict[str, object]] = []
    for col in x.columns:
        series = x[col]
        role = encoded_feature_role(col)
        if pd.api.types.is_numeric_dtype(series):
            imputer = SimpleImputer(strategy="median")
            encoded_values = imputer.fit_transform(series.to_frame()).reshape(-1)
        else:
            imputer = SimpleImputer(strategy="most_frequent")
            encoded_values = imputer.fit_transform(series.to_frame()).astype(str).reshape(-1)
        if role == "numeric":
            encoding_for_ig = "encoded numeric values"
        elif role == "binary_encoded":
            encoding_for_ig = "encoded binary values"
        elif role == "ordinal_encoded":
            encoding_for_ig = "encoded ordinal values"
        elif role == "one_hot_encoded":
            encoding_for_ig = "encoded one-hot dummy values"
        else:
            encoding_for_ig = "encoded values"

        conditional = conditional_entropy(y_encoded, encoded_values)
        rows.append(
            {
                "feature": col,
                "role": role,
                "information_gain": target_entropy - conditional,
                "target_entropy": target_entropy,
                "conditional_entropy": conditional,
                "feature_value_count": int(pd.Series(encoded_values).nunique(dropna=False)),
                "ig_encoding": encoding_for_ig,
                "formula": "H(Y) - H(Y|X)",
            }
        )

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["information_gain", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def entropy_from_labels(labels: np.ndarray | pd.Series) -> float:
    values = pd.Series(labels)
    if values.empty:
        return 0.0
    probabilities = values.value_counts(dropna=False, normalize=True).to_numpy(dtype=float)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def conditional_entropy(target: np.ndarray | pd.Series, feature_values: np.ndarray | pd.Series) -> float:
    frame = pd.DataFrame({"target": target, "feature": feature_values})
    if frame.empty:
        return 0.0
    total = len(frame)
    weighted_entropy = 0.0
    for _, group in frame.groupby("feature", dropna=False):
        weight = len(group) / total
        weighted_entropy += weight * entropy_from_labels(group["target"])
    return float(weighted_entropy)


# =============================================================================
# 6. Metrics and result formatting
# =============================================================================

def selection_metric_key(selection_metric: str = DEFAULT_SELECTION_METRIC) -> str:
    if selection_metric == "f1":
        return "f1_score"
    return selection_metric


def classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    positive_label: int | None,
    class_labels: list[str],
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    average = "binary" if len(class_labels) == 2 and positive_label is not None else "weighted"
    kwargs = {"zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = positive_label
    recall_value = float(recall_score(y_true, y_pred, average=average, **kwargs))
    scores = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, **kwargs)),
        "recall": recall_value,
        "f1_score": float(f1_score(y_true, y_pred, average=average, **kwargs)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if len(class_labels) == 2 and positive_label is not None:
        negative_label = [label for label in np.unique(y_true) if label != positive_label]
        if negative_label:
            labels = [negative_label[0], positive_label]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
            denominator = tn + fp
            scores["specificity"] = float(tn / denominator) if denominator else np.nan
        else:
            scores["specificity"] = np.nan
    else:
        scores["specificity"] = np.nan
    if len(class_labels) == 2 and y_proba is not None:
        positive_proba = y_proba[:, positive_label if positive_label is not None else 1]
        positive_target = (np.asarray(y_true) == positive_label).astype(int)
        scores["roc_auc"] = float(roc_auc_score(positive_target, positive_proba))
        scores["pr_auc"] = float(average_precision_score(positive_target, positive_proba))
        scores["brier_score"] = float(brier_score_loss(positive_target, positive_proba))
    else:
        scores["roc_auc"] = np.nan
        scores["pr_auc"] = np.nan
        scores["brier_score"] = np.nan
    return scores


# =============================================================================
# 7. Model building, validation, and search
# =============================================================================

def build_xgboost_classifier(
    num_classes: int,
    random_state: int,
    n_jobs: int,
    **params: Any,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Run: pip install -r requirements.txt") from exc

    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    model_params: dict[str, Any] = {
        "objective": objective,
        "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    model_params.update(params)
    return XGBClassifier(**model_params)


def require_cupy_for_gpu_mode():
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "device='gpu' requires CuPy so XGBoost fit and predict can use "
            "GPU arrays end to end. Install the project requirements with: "
            "pip install -r requirements.txt"
        ) from exc

    try:
        gpu_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "device='gpu' was selected, but CuPy could not access a CUDA GPU. "
            "Use device='cpu' or fix the local NVIDIA/CUDA setup."
        ) from exc

    if gpu_count < 1:
        raise RuntimeError(
            "device='gpu' was selected, but CuPy reported zero CUDA GPUs. "
            "Use device='cpu' on this machine."
        )

    try:
        probe = cp.asarray([0.0], dtype=cp.float32).astype(cp.float64)
        cp.cuda.Stream.null.synchronize()
        del probe
    except Exception as exc:
        raise RuntimeError(
            "CuPy detected a CUDA GPU, but its CUDA runtime/header components "
            "are incomplete. Reinstall the project requirements so "
            "cupy-cuda12x[ctk] is installed, or use device='cpu'."
        ) from exc
    return cp


def prepare_xgboost_classification_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    device: str,
) -> tuple[Any, Any, Any, str]:
    effective_device = effective_xgboost_device(device)
    if effective_device == "gpu":
        if not query_nvidia_gpus():
            raise RuntimeError(
                "device='gpu' was selected, but nvidia-smi did not report an "
                "NVIDIA GPU. This project supports NVIDIA/CUDA GPU mode only; "
                "use device='cpu' on AMD, Intel, or integrated GPUs."
            )
        cp = require_cupy_for_gpu_mode()
        return (
            cp.asarray(x_train),
            cp.asarray(y_train.reshape(-1)),
            cp.asarray(x_test),
            "cupy_gpu_arrays",
        )

    return x_train, y_train.reshape(-1), x_test, "numpy_cpu_arrays"


def prediction_to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "get"):
        return np.asarray(values.get())
    return np.asarray(values)


def predict_xgboost_classifier(model: Any, x_values: Any) -> np.ndarray:
    return prediction_to_numpy(model.predict(x_values))


def predict_xgboost_classifier_proba(model: Any, x_values: Any) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    return prediction_to_numpy(model.predict_proba(x_values))


def get_xgboost_booster_device(model: Any) -> str:
    try:
        config = json.loads(model.get_booster().save_config())
        return str(
            config.get("learner", {})
            .get("generic_param", {})
            .get("device", "unknown")
        )
    except Exception:
        return "unknown"


def build_param_grid(
    args: argparse.Namespace,
    top_k_values: list[int | str] | None = None,
) -> dict[str, list[object]]:
    return {
        "ig_top_k": list(args.ig_top_k if top_k_values is None else top_k_values),
        "learning_rate": args.learning_rates,
        "max_depth": args.max_depths,
        "n_estimators": args.n_estimators,
        "subsample": args.subsamples,
        "colsample_bytree": args.colsample_bytree,
    }


def top_k_sort_value(value: object) -> tuple[int, int]:
    if str(value).lower() == "all":
        return (0, 0)
    return (1, int(value))


def build_best_by_feature_count_table(
    results_df: pd.DataFrame,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    scoring_name = selection_metric_key(selection_metric)
    metric_col = f"mean_cv_{scoring_name}"
    if metric_col not in results_df.columns:
        raise ValueError(f"results_df must contain {metric_col}.")

    rows: list[pd.Series] = []
    for _, group in results_df.groupby("ig_top_k", sort=False):
        best_row = group.sort_values(
            [metric_col, "mean_cv_f1_score", "mean_cv_pr_auc", "mean_cv_roc_auc"],
            ascending=[False, False, False, False],
        ).iloc[0]
        rows.append(best_row)

    summary = pd.DataFrame(rows).sort_values(
        "ig_top_k",
        key=lambda series: series.map(top_k_sort_value),
    ).reset_index(drop=True)
    summary.insert(0, "feature_count_rank", np.arange(1, len(summary) + 1))
    summary = summary.rename(
        columns={
            "rank": "overall_grid_rank",
            "is_best": "is_best_overall",
        }
    )
    keep_cols = [
        "feature_count_rank",
        "ig_top_k",
        "overall_grid_rank",
        "is_best_overall",
        "mean_cv_accuracy",
        "std_cv_accuracy",
        "mean_cv_precision",
        "mean_cv_recall",
        "mean_cv_specificity",
        "mean_cv_f1_score",
        "mean_cv_mcc",
        "mean_cv_roc_auc",
        "mean_cv_pr_auc",
        "mean_cv_brier_score",
        "learning_rate",
        "max_depth",
        "n_estimators",
        "subsample",
        "colsample_bytree",
    ]
    keep_cols = [col for col in keep_cols if col in summary.columns]
    return summary[keep_cols]


def xgboost_fixed_model_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name in OPTIONAL_XGBOOST_PARAMS:
        value = getattr(args, name, None)
        if value is not None:
            params[name] = value
    return params


def xgboost_model_params(args: argparse.Namespace) -> dict[str, Any]:
    params = xgboost_fixed_model_params(args)
    params.update(
        xgboost_device_params(
            getattr(args, "device", DEFAULT_DEVICE),
            getattr(args, "tree_method", DEFAULT_TREE_METHOD),
        )
    )
    return params


def xgboost_fixed_params_for_display(args: argparse.Namespace) -> dict[str, object]:
    fixed_params = xgboost_fixed_model_params(args)
    return {
        name: fixed_params.get(name, "xgboost_default")
        for name in OPTIONAL_XGBOOST_PARAMS
    }


def make_kfold(cv_splits: int, random_state: int):
    from sklearn.model_selection import KFold

    return KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )


def build_fold_numbers(x: pd.DataFrame, y: np.ndarray, cv) -> np.ndarray:
    fold_numbers = np.empty(len(y), dtype=int)
    for fold_number, (_, test_index) in enumerate(cv.split(x, y), start=1):
        fold_numbers[test_index] = fold_number
    return fold_numbers


GRID_PARAM_COLUMNS = [
    "ig_top_k",
    "learning_rate",
    "max_depth",
    "n_estimators",
    "subsample",
    "colsample_bytree",
]


def grid_param_combinations(
    args: argparse.Namespace,
    top_k_values: list[int | str],
) -> list[dict[str, object]]:
    return [
        {
            "ig_top_k": values[0],
            "learning_rate": values[1],
            "max_depth": values[2],
            "n_estimators": values[3],
            "subsample": values[4],
            "colsample_bytree": values[5],
        }
        for values in itertools.product(
            top_k_values,
            args.learning_rates,
            args.max_depths,
            args.n_estimators,
            args.subsamples,
            args.colsample_bytree,
        )
    ]


def grid_param_key(params: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(params[column]) for column in GRID_PARAM_COLUMNS)


def existing_grid_progress(progress_path: Path) -> pd.DataFrame:
    if not progress_path.exists() or progress_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(progress_path, low_memory=False)


GRID_OUTPUT_NAMES = [
    "grid_search_results.csv",
    "best_grid_search_parameters.csv",
    "default_vs_grid_search_by_feature_count.csv",
    "best_selected_features_by_fold.csv",
    "best_selected_feature_frequency.csv",
    "best_fold_metrics.csv",
    "classification_main_summary.csv",
    "classification_supporting_summary.csv",
    "best_cv_predictions.csv",
    "best_confusion_matrix.png",
    "run_metadata.json",
]


def clear_complete_grid_outputs(output_dir: Path) -> None:
    for name in GRID_OUTPUT_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def encoded_dataset_fingerprint(x: pd.DataFrame, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    schema = {
        "columns": list(x.columns),
        "dtypes": [str(dtype) for dtype in x.dtypes],
        "shape": list(x.shape),
        "target_dtype": str(np.asarray(y).dtype),
        "target_shape": list(np.asarray(y).shape),
    }
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(x, index=True).to_numpy(dtype=np.uint64).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()


def grid_checkpoint_configuration(
    args: argparse.Namespace,
    x: pd.DataFrame,
    y: np.ndarray,
    target_metadata: dict[str, object],
    grid_top_k_values: list[int | str],
) -> dict[str, object]:
    try:
        import sklearn
        import xgboost

        library_versions = {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        }
    except ImportError:
        library_versions = {}

    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "backend_code_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_fingerprint": encoded_dataset_fingerprint(x, y),
        "encoded_columns": list(x.columns),
        "target_classes": list(target_metadata["classes"]),
        "positive_label": target_metadata["positive_label"],
        "cv_splits": int(args.cv_splits),
        "seed": int(args.seed),
        "selection_metric": str(args.selection_metric),
        "device": str(args.device),
        "tree_method": str(args.tree_method),
        "fixed_xgboost_params": xgboost_fixed_params_for_display(args),
        "grid": build_param_grid(args, grid_top_k_values),
        "library_versions": library_versions,
    }


def checkpoint_signature(configuration: dict[str, object]) -> str:
    serialized = json.dumps(configuration, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def archive_incompatible_checkpoint(output_dir: Path, reason: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_dir / f"checkpoint_archive_{timestamp}"
    suffix = 1
    while archive_dir.exists():
        archive_dir = output_dir / f"checkpoint_archive_{timestamp}_{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_files = ["grid_search_progress.csv", CHECKPOINT_METADATA_NAME, *GRID_OUTPUT_NAMES]
    for name in dict.fromkeys(checkpoint_files):
        source = output_dir / name
        if source.exists():
            shutil.move(str(source), str(archive_dir / name))
    (archive_dir / "archive_reason.txt").write_text(reason + "\n", encoding="utf-8")
    return archive_dir


def prepare_grid_checkpoint(
    args: argparse.Namespace,
    x: pd.DataFrame,
    y: np.ndarray,
    target_metadata: dict[str, object],
    output_dir: Path,
    grid_top_k_values: list[int | str],
) -> tuple[Path, dict[str, object], Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "grid_search_progress.csv"
    metadata_path = output_dir / CHECKPOINT_METADATA_NAME
    configuration = grid_checkpoint_configuration(
        args,
        x,
        y,
        target_metadata,
        grid_top_k_values,
    )
    metadata = {
        "signature": checkpoint_signature(configuration),
        "configuration": configuration,
    }
    archived_to: Path | None = None

    if progress_path.exists():
        existing_metadata: dict[str, object] | None = None
        if metadata_path.exists():
            try:
                existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_metadata = None
        existing_signature = existing_metadata.get("signature") if existing_metadata else None
        if existing_signature != metadata["signature"]:
            reason = (
                "The saved grid-search progress did not match the current encoded data, "
                "cross-validation settings, fixed XGBoost hyperparameters, grid, or library versions. "
                "It was archived instead of being reused."
            )
            archived_to = archive_incompatible_checkpoint(output_dir, reason)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return progress_path, metadata, archived_to


def ranked_grid_results(
    progress_df: pd.DataFrame,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    scoring_name = selection_metric_key(selection_metric)
    results_df = progress_df.copy()
    results_df["_feature_count_sort"] = results_df["ig_top_k"].map(
        lambda value: (
            float("inf")
            if str(value).lower() == "all"
            else int(float(value))
        )
    )
    sort_columns = [
        f"mean_cv_{scoring_name}",
        "mean_cv_f1_score",
        "mean_cv_pr_auc",
        "mean_cv_roc_auc",
        "_feature_count_sort",
        "learning_rate",
        "max_depth",
        "n_estimators",
        "subsample",
        "colsample_bytree",
    ]
    results_df = results_df.sort_values(
        sort_columns,
        ascending=[False, False, False, False, True, True, True, True, True, True],
    ).drop(columns="_feature_count_sort").reset_index(drop=True)
    results_df.insert(0, "rank", np.arange(1, len(results_df) + 1))
    results_df.insert(1, "is_best", results_df["rank"].eq(1))
    return results_df


def params_from_grid_row(row: pd.Series | dict[str, object]) -> dict[str, object]:
    data = dict(row)
    top_k = data["ig_top_k"]
    if str(top_k).lower() != "all":
        top_k = int(float(top_k))
    return {
        "ig_top_k": top_k,
        "learning_rate": float(data["learning_rate"]),
        "max_depth": int(float(data["max_depth"])),
        "n_estimators": int(float(data["n_estimators"])),
        "subsample": float(data["subsample"]),
        "colsample_bytree": float(data["colsample_bytree"]),
    }


def build_fold_information_gain_cache(
    x: pd.DataFrame,
    y: np.ndarray,
    cv,
    random_state: int = DEFAULT_SEED,
) -> list[dict[str, object]]:
    cache: list[dict[str, object]] = []
    for fold_number, (train_index, test_index) in enumerate(cv.split(x, y), start=1):
        ranking = compute_information_gain_table(
            x.iloc[train_index],
            y[train_index],
            random_state=random_state,
        )
        cache.append(
            {
                "fold": fold_number,
                "train_index": train_index,
                "test_index": test_index,
                "ranking": ranking,
                "ranked_features": ranking["feature"].tolist(),
            }
        )
    return cache


def selected_features_from_fold_cache(
    fold_cache: dict[str, object],
    top_k: int | str,
    all_features: list[str],
) -> list[str]:
    if str(top_k).lower() == "all":
        return all_features
    ranked_features = list(fold_cache["ranked_features"])
    selected_count = min(int(top_k), len(ranked_features))
    return ranked_features[:selected_count]


def top_k_cache_key(top_k: int | str) -> str:
    return "all" if str(top_k).lower() == "all" else str(int(top_k))


@lru_cache(maxsize=1)
def recommended_cpu_search_workers() -> int:
    return max(1, int(os.cpu_count() or 1))


def resolve_search_workers(
    search_workers: int,
    device: str = DEFAULT_DEVICE,
) -> int:
    if effective_xgboost_device(device) == "gpu":
        return 1
    if search_workers == -1:
        return recommended_cpu_search_workers()
    return max(1, int(search_workers))


def resolve_parallel_fit_threads(n_jobs: int, worker_count: int) -> int:
    if int(n_jobs) == -1:
        return max(
            1,
            recommended_cpu_search_workers() // max(1, int(worker_count)),
        )
    return max(1, int(n_jobs))


def build_fold_model_data_cache(
    args: argparse.Namespace,
    x: pd.DataFrame,
    y: np.ndarray,
    fold_ig_cache: list[dict[str, object]],
    top_k_values: list[int | str],
) -> dict[str, list[dict[str, object]]]:
    all_features = list(x.columns)
    unique_top_k = list(dict.fromkeys(top_k_cache_key(value) for value in top_k_values))
    cache: dict[str, list[dict[str, object]]] = {key: [] for key in unique_top_k}

    for fold_cache in fold_ig_cache:
        train_index = fold_cache["train_index"]
        test_index = fold_cache["test_index"]
        for key in unique_top_k:
            top_k: int | str = "all" if key == "all" else int(key)
            selected_features = selected_features_from_fold_cache(
                fold_cache,
                top_k,
                all_features,
            )
            x_train = np.ascontiguousarray(
                x.iloc[train_index].loc[:, selected_features].to_numpy(dtype=np.float32)
            )
            x_test = np.ascontiguousarray(
                x.iloc[test_index].loc[:, selected_features].to_numpy(dtype=np.float32)
            )
            y_train = np.ascontiguousarray(y[train_index])
            x_train_model, y_train_model, x_test_model, model_array_backend = (
                prepare_xgboost_classification_arrays(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    device=args.device,
                )
            )
            cache[key].append(
                {
                    "fold": fold_cache["fold"],
                    "train_index": train_index,
                    "test_index": test_index,
                    "selected_features": selected_features,
                    "x_train_model": x_train_model,
                    "y_train_model": y_train_model,
                    "x_test_model": x_test_model,
                    "y_test": y[test_index],
                    "model_array_backend": model_array_backend,
                }
            )
    return cache


def xgboost_tuned_params_from_grid(params: dict[str, object]) -> dict[str, object]:
    return {
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "n_estimators": int(params["n_estimators"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
    }


def selected_feature_frequency_from_cache(
    top_k: int | str,
    all_features: list[str],
    fold_ig_cache: list[dict[str, object]],
) -> tuple[list[str], pd.DataFrame]:
    feature_counts: dict[str, int] = {}
    for fold_cache in fold_ig_cache:
        selected_features = selected_features_from_fold_cache(
            fold_cache,
            top_k,
            all_features,
        )
        for feature in selected_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    frequency_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "selected_in_folds": count,
                "total_folds": len(fold_ig_cache),
                "selection_frequency_pct": (count / len(fold_ig_cache)) * 100 if fold_ig_cache else 0,
            }
            for feature, count in feature_counts.items()
        ]
    )
    if not frequency_df.empty:
        frequency_df = frequency_df.sort_values(
            ["selected_in_folds", "feature"],
            ascending=[False, True],
        ).reset_index(drop=True)
        frequency_df.insert(0, "rank", np.arange(1, len(frequency_df) + 1))
    return frequency_df["feature"].tolist(), round_metric_columns(frequency_df)


def evaluate_grid_combination(
    args: argparse.Namespace,
    params: dict[str, object],
    target_metadata: dict[str, object],
    fold_model_data_cache: dict[str, list[dict[str, object]]],
    model_n_jobs: int,
) -> dict[str, object]:
    classes = target_metadata["classes"]
    positive_label = target_metadata["positive_label"]
    model_params = xgboost_model_params(args)
    model_params.update(xgboost_tuned_params_from_grid(params))

    test_scores: list[dict[str, float]] = []
    fold_rows = fold_model_data_cache[top_k_cache_key(params["ig_top_k"])]
    for fold_data in fold_rows:
        fold_model = build_xgboost_classifier(
            num_classes=len(classes),
            random_state=args.seed,
            n_jobs=model_n_jobs,
            **model_params,
        )
        fold_model.fit(fold_data["x_train_model"], fold_data["y_train_model"])
        y_test_proba = predict_xgboost_classifier_proba(fold_model, fold_data["x_test_model"])
        y_test_pred = (
            np.argmax(y_test_proba, axis=1).astype(int)
            if y_test_proba is not None
            else predict_xgboost_classifier(fold_model, fold_data["x_test_model"]).astype(int)
        )
        test_scores.append(
            classification_scores(
                fold_data["y_test"],
                y_test_pred,
                y_test_proba,
                positive_label,
                classes,
            )
        )

    row: dict[str, object] = {column: params[column] for column in GRID_PARAM_COLUMNS}
    for metric in FINAL_REPORT_METRICS:
        test_values = np.asarray([scores[metric] for scores in test_scores], dtype=float)
        row[f"mean_cv_{metric}"] = float(np.nanmean(test_values))
        row[f"std_cv_{metric}"] = float(np.nanstd(test_values))
    row["status"] = "completed"
    return row


def evaluate_default_feature_count(
    args: argparse.Namespace,
    top_k: int | str,
    target_metadata: dict[str, object],
    fold_model_data_cache: dict[str, list[dict[str, object]]],
    model_n_jobs: int,
) -> dict[str, object]:
    classes = target_metadata["classes"]
    positive_label = target_metadata["positive_label"]
    model_params = xgboost_model_params(args)
    test_scores: list[dict[str, float]] = []

    for fold_data in fold_model_data_cache[top_k_cache_key(top_k)]:
        fold_model = build_xgboost_classifier(
            num_classes=len(classes),
            random_state=args.seed,
            n_jobs=model_n_jobs,
            **model_params,
        )
        fold_model.fit(fold_data["x_train_model"], fold_data["y_train_model"])
        y_test_proba = predict_xgboost_classifier_proba(fold_model, fold_data["x_test_model"])
        y_test_pred = (
            np.argmax(y_test_proba, axis=1).astype(int)
            if y_test_proba is not None
            else predict_xgboost_classifier(fold_model, fold_data["x_test_model"]).astype(int)
        )
        test_scores.append(
            classification_scores(
                fold_data["y_test"],
                y_test_pred,
                y_test_proba,
                positive_label,
                classes,
            )
        )

    row: dict[str, object] = {
        "ig_top_k": top_k,
        "model_type": "default_xgboost",
        "hyperparameter_setup": "xgboost_defaults",
    }
    for metric in FINAL_REPORT_METRICS:
        test_values = np.asarray([scores[metric] for scores in test_scores], dtype=float)
        row[f"mean_cv_{metric}"] = float(np.nanmean(test_values))
        row[f"std_cv_{metric}"] = float(np.nanstd(test_values))
    return row


def cross_validated_manual_predictions(
    args: argparse.Namespace,
    top_k: int | str,
    tuned_model_params: dict[str, object],
    x: pd.DataFrame,
    y: np.ndarray,
    target_metadata: dict[str, object],
    fold_ig_cache: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray | None]:
    classes = target_metadata["classes"]
    positive_label = target_metadata["positive_label"]
    all_features = list(x.columns)
    model_params = xgboost_model_params(args)
    model_params.update(tuned_model_params)

    y_cv_pred = np.empty(len(y), dtype=int)
    y_cv_proba = np.full((len(y), len(classes)), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    feature_counts: dict[str, int] = {}
    has_proba = False

    for fold_cache in fold_ig_cache:
        fold_number = int(fold_cache["fold"])
        train_index = fold_cache["train_index"]
        test_index = fold_cache["test_index"]
        selected_features = selected_features_from_fold_cache(
            fold_cache,
            top_k,
            all_features,
        )
        for feature in selected_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

        fold_model = build_xgboost_classifier(
            num_classes=len(classes),
            random_state=args.seed,
            n_jobs=args.n_jobs,
            **model_params,
        )
        x_train = x.iloc[train_index].loc[:, selected_features].to_numpy(dtype=np.float32)
        x_test = x.iloc[test_index].loc[:, selected_features].to_numpy(dtype=np.float32)
        y_train = y[train_index]
        y_test = y[test_index]
        x_train_model, y_train_model, x_test_model, model_array_backend = prepare_xgboost_classification_arrays(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            device=args.device,
        )
        fold_model.fit(x_train_model, y_train_model)
        booster_device_after_fit = get_xgboost_booster_device(fold_model)

        y_pred = predict_xgboost_classifier(fold_model, x_test_model).astype(int)
        y_proba = predict_xgboost_classifier_proba(fold_model, x_test_model)
        y_cv_pred[test_index] = y_pred
        if y_proba is not None:
            has_proba = True
            y_cv_proba[test_index, : y_proba.shape[1]] = y_proba

        scores = classification_scores(y_test, y_pred, y_proba, positive_label, classes)
        fold_rows.append(
            {
                "fold": fold_number,
                "train_samples": len(train_index),
                "test_samples": len(test_index),
                "selected_feature_count": len(selected_features),
                "selected_features": ", ".join(selected_features),
                "booster_device_after_fit": booster_device_after_fit,
                "model_array_backend": model_array_backend,
            }
        )
        metric_rows.append(
            {
                "fold": fold_number,
                "train_samples": len(train_index),
                "test_samples": len(test_index),
                "booster_device_after_fit": booster_device_after_fit,
                "model_array_backend": model_array_backend,
                **scores,
            }
        )

    total_folds = len(fold_ig_cache)
    frequency_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "selected_in_folds": count,
                "total_folds": total_folds,
                "selection_frequency_pct": (count / total_folds) * 100 if total_folds else 0,
            }
            for feature, count in feature_counts.items()
        ]
    )
    if not frequency_df.empty:
        frequency_df = frequency_df.sort_values(
            ["selected_in_folds", "feature"],
            ascending=[False, True],
        ).reset_index(drop=True)
        frequency_df.insert(0, "rank", np.arange(1, len(frequency_df) + 1))

    fold_df = pd.DataFrame(fold_rows)
    fold_metrics_df = pd.DataFrame(metric_rows)
    return fold_df, frequency_df, fold_metrics_df, y_cv_pred, y_cv_proba if has_proba else None


def mean_classification_scores_from_folds(fold_metrics_df: pd.DataFrame) -> dict[str, float]:
    metric_names = FINAL_REPORT_METRICS
    scores: dict[str, float] = {}
    for metric in metric_names:
        values = pd.to_numeric(fold_metrics_df[metric], errors="coerce").to_numpy(dtype=float)
        scores[metric] = float(np.nanmean(values))
    return scores


def ranked_default_results(
    default_df: pd.DataFrame,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    scoring_name = selection_metric_key(selection_metric)
    sort_columns = [
        f"mean_cv_{scoring_name}",
        "mean_cv_f1_score",
        "mean_cv_pr_auc",
        "mean_cv_roc_auc",
    ]
    results_df = default_df.sort_values(
        sort_columns,
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    results_df.insert(0, "default_rank", np.arange(1, len(results_df) + 1))
    results_df.insert(1, "is_best_default", results_df["default_rank"].eq(1))
    return results_df


def select_screened_feature_count(
    default_results_df: pd.DataFrame,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.Series:
    """Select one numeric IG feature count after the default-model screen."""
    scoring_name = selection_metric_key(selection_metric)
    candidates = default_results_df[
        ~default_results_df["ig_top_k"].astype(str).str.lower().eq("all")
    ].copy()
    if candidates.empty:
        raise ValueError(
            "ig_top_k must include at least one positive integer for the "
            "Information Gain feature-count experiment."
        )
    candidates["_feature_count"] = pd.to_numeric(
        candidates["ig_top_k"],
        errors="raise",
    ).astype(int)
    candidates = candidates.sort_values(
        [
            f"mean_cv_{scoring_name}",
            "mean_cv_f1_score",
            "mean_cv_pr_auc",
            "mean_cv_roc_auc",
            "_feature_count",
        ],
        ascending=[False, False, False, False, True],
    )
    return candidates.drop(columns="_feature_count").iloc[0].copy()


def run_grid_search(
    args: argparse.Namespace,
    x: pd.DataFrame,
    y: np.ndarray,
    target_metadata: dict[str, object],
    output_dir: Path,
    grid_top_k_values: list[int | str],
    fold_ig_cache: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Tune every configured feature setup using shared fold-wise IG rankings."""
    scoring_name = selection_metric_key(args.selection_metric)
    cv = make_kfold(args.cv_splits, args.seed)
    # Information Gain is fitted once per training fold and reused across the grid.
    if fold_ig_cache is None:
        fold_ig_cache = build_fold_information_gain_cache(
            x=x,
            y=y,
            cv=cv,
            random_state=args.seed,
        )
    normalized_top_k_values: list[int | str] = []
    for value in grid_top_k_values:
        normalized = "all" if str(value).lower() == "all" else int(float(value))
        if normalized not in normalized_top_k_values:
            normalized_top_k_values.append(normalized)
    numeric_top_k_values = [
        value for value in normalized_top_k_values if str(value).lower() != "all"
    ]
    if "all" not in normalized_top_k_values or not numeric_top_k_values:
        raise ValueError(
            "grid_top_k_values must contain 'all' and at least one numeric top-k."
        )

    combinations = grid_param_combinations(args, normalized_top_k_values)
    total_combinations = len(combinations)
    current_keys = {grid_param_key(params) for params in combinations}
    # A configuration signature prevents incompatible checkpoint rows from being mixed.
    progress_path, checkpoint_metadata, archived_to = prepare_grid_checkpoint(
        args=args,
        x=x,
        y=y,
        target_metadata=target_metadata,
        output_dir=output_dir,
        grid_top_k_values=normalized_top_k_values,
    )
    clear_complete_grid_outputs(output_dir)
    all_progress_df = existing_grid_progress(progress_path)
    progress_df = all_progress_df.copy()
    if not progress_df.empty:
        progress_df = progress_df[
            progress_df[GRID_PARAM_COLUMNS]
            .astype(str)
            .apply(lambda row: tuple(row[column] for column in GRID_PARAM_COLUMNS) in current_keys, axis=1)
        ].drop_duplicates(subset=GRID_PARAM_COLUMNS, keep="last").copy()
    completed_keys = {
        grid_param_key(row)
        for row in progress_df[GRID_PARAM_COLUMNS].to_dict("records")
    } if not progress_df.empty else set()

    search_workers = resolve_search_workers(
        getattr(args, "search_workers", 1),
        args.device,
    )
    model_n_jobs = resolve_parallel_fit_threads(args.n_jobs, search_workers)
    fold_model_data_cache = build_fold_model_data_cache(
        args=args,
        x=x,
        y=y,
        fold_ig_cache=fold_ig_cache,
        top_k_values=normalized_top_k_values,
    )

    if args.verbose:
        if archived_to is not None:
            print(f"\nArchived incompatible checkpoint: {archived_to}")
        print(
            "\nCheckpointed grid search"
            f"\n- completed combinations: {len(completed_keys)}/{total_combinations}"
            f"\n- progress file: {progress_path}"
            f"\n- checkpoint signature: {checkpoint_metadata['signature'][:12]}"
            f"\n- search workers: {search_workers}"
            f"\n- XGBoost threads per grid fit: {model_n_jobs}"
        )

    # Resume from completed rows and evaluate only combinations still missing.
    pending_combinations = [
        params for params in combinations if grid_param_key(params) not in completed_keys
    ]
    checkpoint_batch_size = max(1, int(args.progress_every))

    def evaluate_one(params: dict[str, object]) -> dict[str, object]:
        return evaluate_grid_combination(
            args=args,
            params=params,
            target_metadata=target_metadata,
            fold_model_data_cache=fold_model_data_cache,
            model_n_jobs=model_n_jobs,
        )

    executor = ThreadPoolExecutor(max_workers=search_workers) if search_workers > 1 else None
    progress_exists = progress_path.exists() and progress_path.stat().st_size > 0
    progress_file = progress_path.open("a", newline="", encoding="utf-8")
    writer: csv.DictWriter | None = None
    completed_this_run = 0

    def save_completed_row(row: dict[str, object], params: dict[str, object]) -> None:
        nonlocal writer, completed_this_run, progress_exists
        if writer is None:
            writer = csv.DictWriter(progress_file, fieldnames=list(row.keys()))
            if not progress_exists:
                writer.writeheader()
                progress_exists = True
        writer.writerow(row)
        progress_file.flush()
        completed_keys.add(grid_param_key(params))
        completed_this_run += 1
        if args.verbose and (
            completed_this_run == 1
            or completed_this_run % args.progress_every == 0
            or len(completed_keys) == total_combinations
        ):
            print(
                f"[{len(completed_keys)}/{total_combinations}] saved checkpoint for "
                f"ig_top_k={params['ig_top_k']}, learning_rate={params['learning_rate']}, "
                f"max_depth={params['max_depth']}, n_estimators={params['n_estimators']}, "
                f"subsample={params['subsample']}, "
                f"colsample_bytree={params['colsample_bytree']}"
            )

    try:
        for batch_start in range(0, len(pending_combinations), checkpoint_batch_size):
            batch = pending_combinations[batch_start : batch_start + checkpoint_batch_size]
            if executor is None:
                for params in batch:
                    save_completed_row(evaluate_one(params), params)
            else:
                futures = {executor.submit(evaluate_one, params): params for params in batch}
                for future in as_completed(futures):
                    params = futures[future]
                    save_completed_row(future.result(), params)
    finally:
        progress_file.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    progress_df = existing_grid_progress(progress_path)
    progress_df = progress_df[
        progress_df[GRID_PARAM_COLUMNS]
        .astype(str)
        .apply(lambda row: tuple(row[column] for column in GRID_PARAM_COLUMNS) in current_keys, axis=1)
    ].drop_duplicates(subset=GRID_PARAM_COLUMNS, keep="last").copy()
    if len(progress_df) < total_combinations:
        remaining = total_combinations - len(progress_df)
        raise RuntimeError(
            "Grid search checkpoint is incomplete. "
            f"{remaining} combinations remain. Re-run the notebook with the same "
            "run_name/output folder to continue."
        )

    results_df = ranked_grid_results(progress_df, args.selection_metric)
    round_metric_columns(results_df).to_csv(output_dir / "grid_search_results.csv", index=False)
    best_by_feature_count_df = build_best_by_feature_count_table(
        results_df,
        selection_metric=args.selection_metric,
    )
    best_params = params_from_grid_row(results_df.iloc[0])
    best = {
        "model_type": "best_grid_search",
        "selection_metric": scoring_name,
        "best_params": best_params,
        "best_cv_score": float(results_df.iloc[0][f"mean_cv_{scoring_name}"]),
        "best_by_feature_count": best_by_feature_count_df,
        "encoded_feature_names": list(x.columns),
        "fold_ig_cache": fold_ig_cache,
        "checkpoint_metadata": checkpoint_metadata,
    }
    return results_df, best


def run_default_model(
    args: argparse.Namespace,
    x: pd.DataFrame,
    y: np.ndarray,
    target_metadata: dict[str, object],
    top_k: int | str = "all",
    selected_features: list[str] | None = None,
    fold_ig_cache: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    classes = target_metadata["classes"]
    positive_label = target_metadata["positive_label"]
    scoring_name = selection_metric_key(args.selection_metric)
    cv = make_kfold(args.cv_splits, args.seed)
    if fold_ig_cache is None:
        fold_ig_cache = build_fold_information_gain_cache(
            x=x,
            y=y,
            cv=cv,
            random_state=args.seed,
        )
    (
        _,
        selected_frequency_df,
        fold_metrics_df,
        y_cv_pred,
        y_cv_proba,
    ) = cross_validated_manual_predictions(
        args=args,
        top_k=top_k,
        tuned_model_params={},
        x=x,
        y=y,
        target_metadata=target_metadata,
        fold_ig_cache=fold_ig_cache,
    )
    cv_scores = mean_classification_scores_from_folds(fold_metrics_df)
    oof_scores = classification_scores(y, y_cv_pred, y_cv_proba, positive_label, classes)
    mean_cv_score = float(cv_scores[scoring_name])
    if selected_features is None:
        if str(top_k).lower() == "all":
            selected_features = list(x.columns)
        else:
            selected_features = selected_frequency_df.head(int(top_k))["feature"].tolist()
    top_k_label = str(top_k)
    return {
        "model_type": "default_xgboost",
        "ig_top_k": top_k,
        "selection_metric": scoring_name,
        "best_params": f"xgboost_defaults_with_ig_top_k_{top_k_label}",
        "best_cv_score": mean_cv_score,
        "cv_scores": cv_scores,
        "oof_scores": oof_scores,
        "selected_features": selected_features,
        "encoded_feature_names": list(x.columns),
        "y_cv_pred": y_cv_pred,
        "y_cv_proba": y_cv_proba,
    }


def run_default_models_by_feature_count(
    args: argparse.Namespace,
    x: pd.DataFrame,
    y: np.ndarray,
    target_metadata: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    cv = make_kfold(args.cv_splits, args.seed)
    fold_ig_cache = build_fold_information_gain_cache(
        x=x,
        y=y,
        cv=cv,
        random_state=args.seed,
    )
    fold_model_data_cache = build_fold_model_data_cache(
        args=args,
        x=x,
        y=y,
        fold_ig_cache=fold_ig_cache,
        top_k_values=args.ig_top_k,
    )
    search_workers = resolve_search_workers(
        getattr(args, "search_workers", 1),
        args.device,
    )
    effective_workers = min(search_workers, max(1, len(args.ig_top_k)))
    model_n_jobs = resolve_parallel_fit_threads(args.n_jobs, effective_workers)

    def evaluate_one(top_k: int | str) -> dict[str, object]:
        return evaluate_default_feature_count(
            args=args,
            top_k=top_k,
            target_metadata=target_metadata,
            fold_model_data_cache=fold_model_data_cache,
            model_n_jobs=model_n_jobs,
        )

    if args.verbose:
        print(
            "\nDefault feature-count screening"
            f"\n- parallel workers: {effective_workers}"
            f"\n- XGBoost threads per fit: {model_n_jobs}"
        )

    if effective_workers > 1:
        rows_by_position: dict[int, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(evaluate_one, top_k): position
                for position, top_k in enumerate(args.ig_top_k)
            }
            for future in as_completed(futures):
                rows_by_position[futures[future]] = future.result()
        rows = [rows_by_position[position] for position in range(len(args.ig_top_k))]
    else:
        rows = [evaluate_one(top_k) for top_k in args.ig_top_k]
    results_df = ranked_default_results(pd.DataFrame(rows), args.selection_metric)
    screened_row = select_screened_feature_count(results_df, args.selection_metric)
    best_top_k = int(float(screened_row["ig_top_k"]))
    best_default_selected_features, selected_frequency_df = selected_feature_frequency_from_cache(
        top_k=best_top_k,
        all_features=list(x.columns),
        fold_ig_cache=fold_ig_cache,
    )
    scoring_name = selection_metric_key(args.selection_metric)
    cv_scores = {
        metric: float(screened_row[f"mean_cv_{metric}"])
        for metric in FINAL_REPORT_METRICS
    }
    best_default = {
        "model_type": "default_xgboost",
        "ig_top_k": best_top_k,
        "selection_metric": scoring_name,
        "best_params": f"xgboost_defaults_with_ig_top_k_{best_top_k}",
        "best_cv_score": float(screened_row[f"mean_cv_{scoring_name}"]),
        "cv_scores": cv_scores,
        "selected_features": best_default_selected_features,
        "selected_feature_frequency": selected_frequency_df,
        "encoded_feature_names": list(x.columns),
        "feature_count_screening_row": screened_row.to_dict(),
        "feature_count_screening_results": results_df,
        "fold_ig_cache": fold_ig_cache,
    }
    return results_df, best_default


def build_default_vs_grid_search_by_feature_count_table(
    best_by_feature_count_df: pd.DataFrame,
    default_results_df: pd.DataFrame,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    scoring_name = selection_metric_key(selection_metric)
    rows: list[dict[str, object]] = []

    for _, row in default_results_df.iterrows():
        rows.append(
            {
                "ig_top_k": row["ig_top_k"],
                "model_type": "default_xgboost",
                "hyperparameter_setup": "XGBoost defaults",
                "mean_cv_selection_score": row[f"mean_cv_{scoring_name}"],
                "mean_cv_accuracy": row["mean_cv_accuracy"],
                "std_cv_accuracy": row["std_cv_accuracy"],
                "mean_cv_precision": row["mean_cv_precision"],
                "mean_cv_recall": row["mean_cv_recall"],
                "mean_cv_specificity": row.get("mean_cv_specificity", np.nan),
                "mean_cv_f1_score": row["mean_cv_f1_score"],
                "mean_cv_roc_auc": row["mean_cv_roc_auc"],
                "mean_cv_mcc": row.get("mean_cv_mcc", np.nan),
                "mean_cv_pr_auc": row.get("mean_cv_pr_auc", np.nan),
                "mean_cv_brier_score": row.get("mean_cv_brier_score", np.nan),
                "learning_rate": "xgboost_default",
                "max_depth": "xgboost_default",
                "n_estimators": "xgboost_default",
                "subsample": "xgboost_default",
                "colsample_bytree": "xgboost_default",
            }
        )

    for _, row in best_by_feature_count_df.iterrows():
        rows.append(
            {
                "ig_top_k": row["ig_top_k"],
                "model_type": "best_grid_search",
                "hyperparameter_setup": "Best grid-search combination for this feature count",
                "mean_cv_selection_score": row[f"mean_cv_{scoring_name}"],
                "mean_cv_accuracy": row["mean_cv_accuracy"],
                "std_cv_accuracy": row["std_cv_accuracy"],
                "mean_cv_precision": row["mean_cv_precision"],
                "mean_cv_recall": row["mean_cv_recall"],
                "mean_cv_specificity": row.get("mean_cv_specificity", np.nan),
                "mean_cv_f1_score": row["mean_cv_f1_score"],
                "mean_cv_roc_auc": row["mean_cv_roc_auc"],
                "mean_cv_mcc": row.get("mean_cv_mcc", np.nan),
                "mean_cv_pr_auc": row.get("mean_cv_pr_auc", np.nan),
                "mean_cv_brier_score": row.get("mean_cv_brier_score", np.nan),
                "learning_rate": row["learning_rate"],
                "max_depth": row["max_depth"],
                "n_estimators": row["n_estimators"],
                "subsample": row["subsample"],
                "colsample_bytree": row["colsample_bytree"],
            }
        )

    comparison = pd.DataFrame(rows)
    ordered_groups: list[pd.DataFrame] = []
    for _, group in comparison.groupby("ig_top_k", sort=False):
        group = group.sort_values(
            [
                "mean_cv_selection_score",
                "mean_cv_f1_score",
                "mean_cv_pr_auc",
                "mean_cv_roc_auc",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        group.insert(0, "feature_count_model_rank", np.arange(1, len(group) + 1))
        group.insert(1, "is_best_for_feature_count", group["feature_count_model_rank"].eq(1))
        ordered_groups.append(group)

    result = pd.concat(ordered_groups, ignore_index=True)
    result["_ig_sort_group"] = result["ig_top_k"].map(lambda value: top_k_sort_value(value)[0])
    result["_ig_sort_value"] = result["ig_top_k"].map(lambda value: top_k_sort_value(value)[1])
    result = result.sort_values(
        ["_ig_sort_group", "_ig_sort_value", "feature_count_model_rank"],
        ascending=[True, True, True],
    ).drop(columns=["_ig_sort_group", "_ig_sort_value"]).reset_index(drop=True)
    return result


def finalize_classification_results(
    args: argparse.Namespace,
    x: pd.DataFrame,
    y: np.ndarray,
    target_metadata: dict[str, object],
    best: dict[str, object],
    default_results_df: pd.DataFrame,
) -> dict[str, object]:
    """Build comparable summaries and out-of-fold results for final reporting."""
    def is_all_features(value: object) -> bool:
        return str(value).lower() == "all"

    def feature_count_value(value: object) -> int:
        return len(x.columns) if is_all_features(value) else int(float(value))

    def feature_setup_label(value: object) -> str:
        return "All encoded features" if is_all_features(value) else f"Top {int(float(value))} IG features"

    def model_setup_label(row: pd.Series) -> str:
        if row["model_type"] == "default_xgboost":
            return "Untuned XGBoost"
        return (
            f"lr={row['learning_rate']}, depth={int(float(row['max_depth']))}, "
            f"n={int(float(row['n_estimators']))}, subsample={row['subsample']}, "
            f"colsample={row['colsample_bytree']}"
        )

    def select_one_winner(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            raise ValueError("No candidate rows were available for a required summary group.")
        work = frame.copy()
        work["_feature_count"] = work["ig_top_k"].map(feature_count_value)
        work = work.sort_values(
            [
                "mean_cv_accuracy",
                "mean_cv_f1_score",
                "mean_cv_pr_auc",
                "mean_cv_roc_auc",
                "_feature_count",
            ],
            ascending=[False, False, False, False, True],
        )
        return work.drop(columns="_feature_count").iloc[0].copy()

    fold_ig_cache = best.get("fold_ig_cache")
    if fold_ig_cache is None:
        fold_ig_cache = build_fold_information_gain_cache(
            x=x,
            y=y,
            cv=make_kfold(args.cv_splits, args.seed),
            random_state=args.seed,
        )

    def recompute_grid_row(row: pd.Series) -> tuple[pd.Series, dict[str, object]]:
        updated_row = row.copy()
        params = params_from_grid_row(updated_row)
        selected_by_fold_df, selected_frequency_df, fold_metrics_df, y_cv_pred, y_cv_proba = (
            cross_validated_manual_predictions(
                args=args,
                top_k=params["ig_top_k"],
                tuned_model_params=xgboost_tuned_params_from_grid(params),
                x=x,
                y=y,
                target_metadata=target_metadata,
                fold_ig_cache=fold_ig_cache,
            )
        )
        mean_scores = mean_classification_scores_from_folds(fold_metrics_df)
        for metric, value in mean_scores.items():
            updated_row[f"mean_cv_{metric}"] = value
        if is_all_features(params["ig_top_k"]):
            selected_features = list(x.columns)
        else:
            selected_features = selected_frequency_df.head(int(params["ig_top_k"]))["feature"].tolist()
        return updated_row, {
            "params": params,
            "mean_scores": mean_scores,
            "oof_scores": classification_scores(
                y,
                y_cv_pred,
                y_cv_proba,
                target_metadata["positive_label"],
                target_metadata["classes"],
            ),
            "selected_features": selected_features,
            "selected_features_by_fold": selected_by_fold_df,
            "selected_feature_frequency": selected_frequency_df,
            "fold_metrics": fold_metrics_df,
            "y_cv_pred": y_cv_pred,
            "y_cv_proba": y_cv_proba,
        }

    comparison_by_feature_count_df = build_default_vs_grid_search_by_feature_count_table(
        best_by_feature_count_df=best["best_by_feature_count"],
        default_results_df=default_results_df,
        selection_metric=args.selection_metric,
    )

    default_rows = comparison_by_feature_count_df["model_type"].eq("default_xgboost")
    grid_rows = comparison_by_feature_count_df["model_type"].eq("best_grid_search")
    all_rows = comparison_by_feature_count_df["ig_top_k"].map(is_all_features)

    default_all_row = select_one_winner(comparison_by_feature_count_df[default_rows & all_rows])
    grid_all_row = select_one_winner(comparison_by_feature_count_df[grid_rows & all_rows])
    numeric_rows = ~all_rows
    default_best_ig_row = select_one_winner(
        comparison_by_feature_count_df[default_rows & numeric_rows]
    )
    final_grid_row = select_one_winner(
        comparison_by_feature_count_df[grid_rows & numeric_rows]
    )
    final_grid_row, final_grid_evaluation = recompute_grid_row(final_grid_row)

    summary_items = [
        ("Default XGBoost + all features", default_all_row),
        ("Best grid search + all features", grid_all_row),
        ("Default XGBoost + best IG feature count", default_best_ig_row),
        ("Best grid search + best IG feature count", final_grid_row),
    ]

    best.update(
        {
            "best_params": final_grid_evaluation["params"],
            "best_cv_score": float(final_grid_row[f"mean_cv_{selection_metric_key(args.selection_metric)}"]),
            "cv_scores": final_grid_evaluation["mean_scores"],
            "oof_scores": final_grid_evaluation["oof_scores"],
            "selected_features": final_grid_evaluation["selected_features"],
            "selected_features_by_fold": final_grid_evaluation["selected_features_by_fold"],
            "selected_feature_frequency": final_grid_evaluation["selected_feature_frequency"],
            "fold_metrics": final_grid_evaluation["fold_metrics"],
            "y_cv_pred": final_grid_evaluation["y_cv_pred"],
            "y_cv_proba": final_grid_evaluation["y_cv_proba"],
        }
    )

    main_summary_df = pd.DataFrame(
        [
            {
                "Result": label,
                "Feature setup": feature_setup_label(row["ig_top_k"]),
                "Accuracy": row["mean_cv_accuracy"],
                "Precision": row["mean_cv_precision"],
                "Recall": row["mean_cv_recall"],
                "Specificity": row["mean_cv_specificity"],
                "F1-score": row["mean_cv_f1_score"],
            }
            for label, row in summary_items
        ]
    )
    supporting_summary_df = pd.DataFrame(
        [
            {
                "Result": label,
                "Feature setup": feature_setup_label(row["ig_top_k"]),
                "MCC": row.get("mean_cv_mcc", np.nan),
                "ROC-AUC": row["mean_cv_roc_auc"],
                "PR-AUC": row.get("mean_cv_pr_auc", np.nan),
                "Brier score": row.get("mean_cv_brier_score", np.nan),
                "Setup": model_setup_label(row),
            }
            for label, row in summary_items
        ]
    )

    return {
        "best": best,
        "default_vs_grid_by_feature_count": round_metric_columns(comparison_by_feature_count_df),
        "main_summary": round_metric_columns(main_summary_df),
        "supporting_summary": round_metric_columns(supporting_summary_df),
    }


def build_prediction_table(
    x: pd.DataFrame,
    y: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    target_metadata: dict[str, object],
    fold_numbers: np.ndarray | None = None,
) -> pd.DataFrame:
    encoder = target_metadata["encoder"]
    actual_label = encoder.inverse_transform(y)
    predicted_label = encoder.inverse_transform(y_pred)
    values: dict[str, object] = {
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "is_correct": actual_label == predicted_label,
    }
    if fold_numbers is not None:
        values = {"fold": fold_numbers, **values}
    result = pd.DataFrame(values, index=x.index)
    if y_proba is not None:
        classes = target_metadata["classes"]
        for class_index, class_name in enumerate(classes):
            result[f"probability_{class_name}"] = y_proba[:, class_index]
    return round_metric_columns(result.reset_index(names="source_row_index"))


# =============================================================================
# 8. Output, plotting, and artifact management
# =============================================================================

def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    output_path: Path,
    show_plot: bool = False,
) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show_plot:
        plt.show()
    plt.close(fig)


def format_decimal(value: float, decimals: int = METRIC_DISPLAY_DECIMALS) -> str:
    """Format a displayed number with at most ``decimals`` decimal places."""
    rounded_value = round(float(value), decimals)
    if rounded_value == 0:
        rounded_value = 0.0
    return f"{rounded_value:.{decimals}f}".rstrip("0").rstrip(".")


def round_metric_columns(frame: pd.DataFrame, decimals: int = METRIC_DISPLAY_DECIMALS) -> pd.DataFrame:
    rounded = frame.copy()
    metric_tokens = (
        "score",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc",
        "mcc",
        "brier",
        "gain",
        "entropy",
        "probability",
        "std",
    )
    for col in rounded.columns:
        if any(token in col.lower() for token in metric_tokens):
            converted = pd.to_numeric(rounded[col], errors="coerce")
            if converted.notna().sum() == rounded[col].notna().sum():
                rounded[col] = converted.round(decimals)
    return rounded


def cleanup_old_output_runs(
    output_root: Path,
    current_output_dir: Path,
    keep_runs: int = DEFAULT_KEEP_RUNS,
) -> list[Path]:
    output_root = output_root.resolve()
    current_output_dir = current_output_dir.resolve()
    if not output_root.exists():
        return []
    run_dirs = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and (
            path.name.startswith("xgboost_classification_run_")
            or path.name.startswith("notebook_xgboost_classification_run_")
        )
    ]
    completed = [
        path
        for path in run_dirs
        if (path / "grid_search_results.csv").exists()
        or (path / "best_cv_predictions.csv").exists()
    ]
    completed = sorted(completed, key=lambda path: path.stat().st_mtime, reverse=True)
    keep = {current_output_dir}
    for path in completed:
        if len(keep) >= keep_runs:
            break
        keep.add(path.resolve())

    removed: list[Path] = []
    for path in run_dirs:
        resolved = path.resolve()
        if output_root not in resolved.parents:
            continue
        if resolved in keep:
            continue
        for child in sorted(resolved.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        resolved.rmdir()
        removed.append(path)
    return removed


# =============================================================================
# 9. Main workflow
# =============================================================================

def main() -> None:
    """Run the command-line workflow from raw CSV data to saved artifacts."""
    args = parse_args()
    validate_args(args)
    set_reproducible_seed(args.seed)

    raw_df = load_dataset(args.data, args.target_col)
    raw_audit_df = raw_data_audit(raw_df, args.target_col)
    df, preprocessing_stats = clean_classification_dataset(raw_df)
    preprocessing_summary_df = preprocessing_summary_table(preprocessing_stats)
    x, y_raw = split_features_target(df, args.target_col, args.exclude_features)
    y_encoded, target_metadata = encode_target(y_raw, args.positive_class)
    feature_roles_df = feature_role_table(x)
    encoded_x, encoded_feature_roles_df = encode_feature_dataframe(x)
    encoded_modeling_df = build_encoded_modeling_dataframe(encoded_x, y_encoded, args.target_col)
    device_info = xgboost_device_info(args.device, args.tree_method)

    print("\nData summary")
    print(raw_audit_df.to_string(index=False, float_format=format_decimal))
    print("\nPreprocessing summary")
    print(preprocessing_summary_df.to_string(index=False, float_format=format_decimal))
    print(f"Samples: {len(encoded_x)} | Encoded features: {encoded_x.shape[1]} | K-Fold splits: {args.cv_splits}")
    print(f"Target mapping: {target_metadata['label_mapping']}")
    print(f"Main grid-search metric: {selection_metric_key(args.selection_metric)}")
    hyperparameter_combinations = int(
        np.prod(
            [
                len(args.learning_rates),
                len(args.max_depths),
                len(args.n_estimators),
                len(args.subsamples),
                len(args.colsample_bytree),
            ]
        )
    )
    screening_fits = len(args.ig_top_k) * args.cv_splits
    grid_combinations = len(args.ig_top_k) * hyperparameter_combinations
    grid_fits = grid_combinations * args.cv_splits
    final_oof_fits = args.cv_splits
    print(f"Feature-count screening fits: {screening_fits}")
    print(f"Grid combinations: {grid_combinations}")
    print(f"Grid-search fits: {grid_fits}")
    print(f"Final out-of-fold prediction fits: {final_oof_fits}")
    print(f"Total planned model fits: {screening_fits + grid_fits + final_oof_fits}")
    print(format_xgboost_device_report(device_info))

    if args.prepare_only:
        print("\n--prepare-only is active; XGBoost training was not run.")
        return

    if getattr(args, "run_name", None):
        output_dir = args.output_dir / str(args.run_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = args.output_dir / f"xgboost_classification_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_roles_df.to_csv(output_dir / "feature_encoding_roles.csv", index=False)
    encoded_modeling_df.to_csv(output_dir / "encoded_classification_dataset.csv", index=False)
    encoded_feature_roles_df.to_csv(output_dir / "encoded_feature_encoding_roles.csv", index=False)
    default_results_df, default = run_default_models_by_feature_count(
        args=args,
        x=encoded_x,
        y=y_encoded,
        target_metadata=target_metadata,
    )
    default_screened_top_k = int(default["ig_top_k"])
    grid_top_k_values: list[int | str] = list(args.ig_top_k)
    results_df, best = run_grid_search(
        args=args,
        x=encoded_x,
        y=y_encoded,
        target_metadata=target_metadata,
        output_dir=output_dir,
        grid_top_k_values=grid_top_k_values,
        fold_ig_cache=default.get("fold_ig_cache"),
    )

    finalized = finalize_classification_results(
        args=args,
        x=encoded_x,
        y=y_encoded,
        target_metadata=target_metadata,
        best=best,
        default_results_df=default_results_df,
    )
    best = finalized["best"]
    default_vs_grid_by_feature_count_df = finalized["default_vs_grid_by_feature_count"]
    classification_main_summary_df = finalized["main_summary"]
    classification_supporting_summary_df = finalized["supporting_summary"]

    default_vs_grid_by_feature_count_df.to_csv(
        output_dir / "default_vs_grid_search_by_feature_count.csv",
        index=False,
    )
    classification_main_summary_df.to_csv(output_dir / "classification_main_summary.csv", index=False)
    classification_supporting_summary_df.to_csv(
        output_dir / "classification_supporting_summary.csv",
        index=False,
    )
    pd.DataFrame([best["best_params"]]).to_csv(
        output_dir / "best_grid_search_parameters.csv",
        index=False,
    )
    best["selected_features_by_fold"].to_csv(
        output_dir / "best_selected_features_by_fold.csv",
        index=False,
    )
    round_metric_columns(best["selected_feature_frequency"]).to_csv(
        output_dir / "best_selected_feature_frequency.csv",
        index=False,
    )
    round_metric_columns(best["fold_metrics"]).to_csv(
        output_dir / "best_fold_metrics.csv",
        index=False,
    )

    fold_numbers = build_fold_numbers(
        encoded_x,
        y_encoded,
        make_kfold(args.cv_splits, args.seed),
    )
    predictions_df = build_prediction_table(
        x=encoded_x,
        y=y_encoded,
        y_pred=best["y_cv_pred"],
        y_proba=best["y_cv_proba"],
        target_metadata=target_metadata,
        fold_numbers=fold_numbers,
    )
    predictions_df.to_csv(output_dir / "best_cv_predictions.csv", index=False)
    save_confusion_matrix_plot(
        y_true=y_encoded,
        y_pred=best["y_cv_pred"],
        class_labels=target_metadata["classes"],
        output_path=output_dir / "best_confusion_matrix.png",
        show_plot=args.show_plot,
    )

    metadata = {
        "data_path": str(args.data),
        "target_col": args.target_col,
        "exclude_features": args.exclude_features,
        "cv_splits": args.cv_splits,
        "positive_class": args.positive_class,
        "selection_metric": args.selection_metric,
        "preprocessing_stats": preprocessing_stats,
        "feature_encoding_roles": feature_roles_df.to_dict("records"),
        "encoded_feature_encoding_roles": encoded_feature_roles_df.to_dict("records"),
        "encoded_feature_count": int(encoded_x.shape[1]),
        "seed": args.seed,
        "search_workers": getattr(args, "search_workers", 1),
        "effective_search_workers": resolve_search_workers(
            getattr(args, "search_workers", 1),
            args.device,
        ),
        "run_name": getattr(args, "run_name", None),
        "progress_every": getattr(args, "progress_every", 25),
        "grid_search_progress_path": str(output_dir / "grid_search_progress.csv"),
        "compute_device": device_info,
        "feature_count_screening_options": list(args.ig_top_k),
        "default_screened_ig_top_k": default_screened_top_k,
        "selected_grid_ig_top_k": best["best_params"].get("ig_top_k"),
        "grid_feature_setups": grid_top_k_values,
        "param_grid": build_param_grid(args, grid_top_k_values),
        "feature_count_screening_fits": screening_fits,
        "grid_search_combinations": grid_combinations,
        "grid_search_fits": grid_fits,
        "final_oof_prediction_fits": final_oof_fits,
        "total_planned_model_fits": screening_fits + grid_fits + final_oof_fits,
        "fixed_xgboost_params": xgboost_fixed_params_for_display(args),
        "tree_method": args.tree_method,
        "target_metadata": {
            key: value
            for key, value in target_metadata.items()
            if key != "encoder"
        },
        "best_params": best["best_params"],
        "selected_features": best["selected_features"],
        "best_default_ig_top_k": default.get("ig_top_k", "all"),
        "checkpoint_signature": best.get("checkpoint_metadata", {}).get("signature"),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    removed_runs = cleanup_old_output_runs(args.output_dir, output_dir, args.keep_runs)
    print("\nAccuracy-focused model summary")
    print(classification_main_summary_df.to_string(index=False, float_format=format_decimal))
    print("\nSupporting classification metrics")
    print(classification_supporting_summary_df.to_string(index=False, float_format=format_decimal))
    print(f"\nOutput folder: {output_dir}")
    print(f"Removed old output folders: {[str(path) for path in removed_runs]}")


if __name__ == "__main__":
    main()
