from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_CANDIDATES = (
    PROJECT_ROOT / "data/prl/metadata/prl_dataset_clean.xlsx",
    PROJECT_ROOT / "data/prl/metadata/prl_dataset_final.xlsx",
    PROJECT_ROOT / "data/prl/metadata/prl_dataset_mapped.xlsx",
    PROJECT_ROOT / "data/prl/metadata/weather_data.csv",
)
DEFAULT_EMBEDDING_CANDIDATES = (
    PROJECT_ROOT / "data/prl/features/dinov2_vitb14_embeddings.csv",
    PROJECT_ROOT / "data/prl_features/all_embeddings.csv",
    PROJECT_ROOT / "data/merged/merged_dataset.csv",
)

REQUIRED_METADATA_COLUMNS = (
    "image_name",
    "date",
    "time",
    "temperature_C",
    "relative_humidity_%",
    "solar_intensity_Wm2",
    "visibility_km",
)
WEATHER_COLUMNS = (
    "temperature_C",
    "relative_humidity_%",
    "solar_intensity_Wm2",
    "hour",
)


class ProjectDataError(RuntimeError):
    """Raised when project data fail a required validation."""


def project_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def first_existing(candidates: Iterable[Path]) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ProjectDataError(f"Unsupported tabular file: {path}")


def _normalise_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "temperature": "temperature_C",
        "humidity": "relative_humidity_%",
        "solar_intensity": "solar_intensity_Wm2",
        "solar": "solar_intensity_Wm2",
        "visibility": "visibility_km",
        "distance": "distance_m",
    }
    df = df.rename(columns={column: aliases.get(column, column) for column in df.columns})
    if "date" in df:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = parsed.dt.strftime("%Y-%m-%d")
    if "time" in df:
        parsed_time = pd.to_datetime(df["time"].astype(str), format="mixed", errors="coerce")
        df["time"] = parsed_time.dt.strftime("%H:%M:%S")
        if "hour" not in df:
            df["hour"] = parsed_time.dt.hour
    if "day" not in df and "date" in df:
        df["day"] = pd.to_datetime(df["date"], errors="coerce").dt.day
    return df


def load_prl_metadata(path: str | os.PathLike[str] | None = None) -> tuple[pd.DataFrame, Path]:
    resolved = project_path(path) if path else first_existing(DEFAULT_METADATA_CANDIDATES)
    if resolved is None or not resolved.exists():
        raise ProjectDataError(
            "No PRL metadata file found. Expected one of: "
            + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in DEFAULT_METADATA_CANDIDATES)
        )
    df = _normalise_metadata_columns(read_table(resolved))
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in df.columns]
    if missing:
        raise ProjectDataError(f"Metadata file {resolved} is missing columns: {missing}")
    return df, resolved


def feature_columns(df: pd.DataFrame) -> list[str]:
    prefixed = [column for column in df.columns if column.startswith("dino_")]
    if prefixed:
        return sorted(prefixed, key=lambda value: int(value.split("_")[-1]))
    prefixed = [column for column in df.columns if column.startswith("feature_")]
    if prefixed:
        return sorted(prefixed, key=lambda value: int(value.split("_")[-1]))
    numeric_names = [column for column in df.columns if str(column).isdigit()]
    return sorted(numeric_names, key=lambda value: int(value))


def load_prl_embeddings(path: str | os.PathLike[str] | None = None) -> tuple[pd.DataFrame, Path, list[str]]:
    resolved = project_path(path) if path else first_existing(DEFAULT_EMBEDDING_CANDIDATES)
    if resolved is None or not resolved.exists():
        raise ProjectDataError(
            "No PRL embedding file found. Run src/prl/02_extract_prl_features.py first."
        )
    df = pd.read_csv(resolved)
    if "image_name" not in df.columns:
        raise ProjectDataError(f"Embedding file {resolved} has no image_name column")
    columns = feature_columns(df)
    if not columns:
        raise ProjectDataError(f"No DINOv2 feature columns found in {resolved}")
    renamed = {column: f"dino_{index:03d}" for index, column in enumerate(columns, start=1)}
    df = df.rename(columns=renamed)
    columns = [renamed[column] for column in columns]
    return df[["image_name", *columns]], resolved, columns


def load_prepared_prl(path: str | os.PathLike[str] = "data/prl/processed/prl_fusion_dataset.csv") -> tuple[pd.DataFrame, list[str]]:
    resolved = project_path(path)
    if not resolved.exists():
        raise ProjectDataError(f"Prepared PRL dataset not found: {resolved}")
    df = pd.read_csv(resolved)
    columns = feature_columns(df)
    if not columns:
        raise ProjectDataError(f"Prepared dataset has no DINOv2 columns: {resolved}")
    return df, columns


def normalise_image_name(name: Any) -> str:
    return str(name).strip().replace("\\", "/").split("/")[-1].lower()


def merge_metadata_embeddings(metadata: pd.DataFrame, embeddings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = metadata.copy()
    right = embeddings.copy()
    left["_image_key"] = left["image_name"].map(normalise_image_name)
    right["_image_key"] = right["image_name"].map(normalise_image_name)

    exact = left.merge(right.drop(columns="image_name"), on="_image_key", how="inner")
    if len(exact) == len(left) == len(right):
        merged = exact.drop(columns="_image_key")
        return merged, {"strategy": "normalised_filename", "matched_rows": len(merged)}

    # The project contains both GOPRxxxx and PRL_000x naming schemes. When both
    # files have the same verified row count, a stable row-order mapping is safer
    # than silently dropping all rows. The mapping is explicitly recorded.
    if len(left) == len(right) and left["_image_key"].is_unique and right["_image_key"].is_unique:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        feature_only = right.drop(columns=["image_name", "_image_key"])
        merged = pd.concat([left.drop(columns="_image_key"), feature_only], axis=1)
        return merged, {
            "strategy": "verified_row_order",
            "matched_rows": len(merged),
            "warning": "Filenames differed; rows were paired by stable order after uniqueness/count checks.",
        }

    missing_metadata = sorted(set(left["_image_key"]) - set(right["_image_key"]))[:20]
    missing_embeddings = sorted(set(right["_image_key"]) - set(left["_image_key"]))[:20]
    raise ProjectDataError(
        "Could not align metadata and embeddings. "
        f"metadata={len(left)}, embeddings={len(right)}, "
        f"missing_embedding_examples={missing_metadata}, "
        f"missing_metadata_examples={missing_embeddings}"
    )


def validate_finite(matrix: np.ndarray, name: str) -> None:
    if not np.isfinite(matrix).all():
        bad = int((~np.isfinite(matrix)).sum())
        raise ProjectDataError(f"{name} contains {bad} NaN/inf values")


def calculate_metrics(y_true: Sequence[float], y_pred: Sequence[float], ceiling: float | None = None) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Metric shape mismatch: {true.shape} != {pred.shape}")
    errors = pred - true
    unique_labels = np.unique(true)
    per_label_mae = [mean_absolute_error(true[true == label], pred[true == label]) for label in unique_labels]
    if ceiling is None:
        ceiling = float(np.nanmax(true))
    low_mask = true < ceiling
    r2 = float(r2_score(true, pred)) if np.unique(true).size > 1 else math.nan
    return {
        "n": int(len(true)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "r2": r2,
        "median_absolute_error": float(np.median(np.abs(errors))),
        "mean_bias": float(np.mean(errors)),
        "macro_mae": float(np.mean(per_label_mae)),
        "low_visibility_mae": float(mean_absolute_error(true[low_mask], pred[low_mask])) if low_mask.any() else math.nan,
        "ceiling_label_mae": float(mean_absolute_error(true[~low_mask], pred[~low_mask])) if (~low_mask).any() else math.nan,
        "within_1km": float(np.mean(np.abs(errors) <= 1.0)),
        "within_2km": float(np.mean(np.abs(errors) <= 2.0)),
    }


def per_label_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> pd.DataFrame:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    rows: list[dict[str, float]] = []
    for label in np.unique(true):
        mask = true == label
        rows.append(
            {
                "visibility_km": float(label),
                "n": int(mask.sum()),
                "mae": float(mean_absolute_error(true[mask], pred[mask])),
                "rmse": float(mean_squared_error(true[mask], pred[mask]) ** 0.5),
                "mean_prediction": float(np.mean(pred[mask])),
                "bias": float(np.mean(pred[mask] - true[mask])),
            }
        )
    return pd.DataFrame(rows)


def inverse_frequency_weights(y: Sequence[float], max_ratio: float = 10.0) -> np.ndarray:
    values = pd.Series(np.asarray(y, dtype=float))
    counts = values.value_counts()
    weights = values.map(lambda value: len(values) / (len(counts) * counts[value])).to_numpy(dtype=float)
    weights /= np.mean(weights)
    return np.clip(weights, 1.0 / max_ratio, max_ratio)


def group_bootstrap_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[Any],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    groups_array = np.asarray(groups)
    unique_groups = np.unique(groups_array)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in ("mae", "rmse", "macro_mae")}
    group_indices = {group: np.flatnonzero(groups_array == group) for group in unique_groups}
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in chosen])
        metric = calculate_metrics(true[indices], pred[indices])
        for key in samples:
            samples[key].append(metric[key])
    result: dict[str, dict[str, float]] = {}
    point = calculate_metrics(true, pred)
    for key, values in samples.items():
        array = np.asarray(values, dtype=float)
        result[key] = {
            "estimate": float(point[key]),
            "ci95_low": float(np.nanpercentile(array, 2.5)),
            "ci95_high": float(np.nanpercentile(array, 97.5)),
        }
    return result


def set_reproducible_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def save_json(payload: Any, path: str | os.PathLike[str]) -> Path:
    resolved = ensure_parent(project_path(path))
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    return resolved


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def dump_bundle(bundle: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    resolved = ensure_parent(project_path(path))
    joblib.dump(bundle, resolved)
    return resolved


def load_bundle(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = project_path(path)
    if not resolved.exists():
        raise ProjectDataError(f"Model bundle not found: {resolved}")
    bundle = joblib.load(resolved)
    if not isinstance(bundle, dict) or "model_type" not in bundle:
        raise ProjectDataError(f"Invalid PRL model bundle: {resolved}")
    return bundle


@dataclass(frozen=True)
class CandidateResult:
    name: str
    predictions: np.ndarray
    metrics: dict[str, float]
    details: dict[str, Any]
