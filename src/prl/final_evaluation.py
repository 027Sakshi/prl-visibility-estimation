from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.prl.optimized_estimators import OptimizedSVRBundle, engineer_weather


PREDICTION_MIN = 4.0
PREDICTION_MAX = 10.0


@dataclass(frozen=True)
class SVRCandidate:
    feature_mode: str
    pca_components: int
    C: float
    epsilon: float

    @property
    def name(self) -> str:
        return (
            f"{self.feature_mode}_svr_pc{self.pca_components}"
            f"_C{self.C}_e{self.epsilon}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_mode": self.feature_mode,
            "pca_components": self.pca_components,
            "C": self.C,
            "epsilon": self.epsilon,
        }


def locked_svr_grid() -> list[SVRCandidate]:
    """Small grid frozen before final nested evaluation."""
    candidates: list[SVRCandidate] = []
    for components in (2, 3):
        for c_value in (10.0, 30.0):
            for epsilon in (0.25, 0.50, 0.75):
                candidates.append(SVRCandidate("image", components, c_value, epsilon))
    for components in (2, 3):
        for c_value in (3.0, 10.0):
            for epsilon in (0.25, 0.50):
                candidates.append(SVRCandidate("fusion", components, c_value, epsilon))
    return candidates


def balanced_regime_mae(y_true: np.ndarray, y_pred: np.ndarray, ceiling: float = 10.0) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    low = true < ceiling
    if not low.any() or not (~low).any():
        return float("nan")
    return float(
        0.5
        * (
            mean_absolute_error(true[low], pred[low])
            + mean_absolute_error(true[~low], pred[~low])
        )
    )


def date_macro_mae(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    group_array = np.asarray(groups).astype(str)
    values = [
        mean_absolute_error(true[group_array == group], pred[group_array == group])
        for group in np.unique(group_array)
    ]
    return float(np.mean(values))


def final_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
    ceiling: float = 10.0,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    errors = pred - true
    low = true < ceiling
    labels = np.unique(true)
    per_label = [mean_absolute_error(true[true == value], pred[true == value]) for value in labels]
    metrics = {
        "n": int(len(true)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "r2": float(r2_score(true, pred)) if labels.size > 1 else float("nan"),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "mean_bias": float(np.mean(errors)),
        "macro_mae": float(np.mean(per_label)),
        "low_visibility_mae": float(mean_absolute_error(true[low], pred[low])) if low.any() else float("nan"),
        "ceiling_label_mae": float(mean_absolute_error(true[~low], pred[~low])) if (~low).any() else float("nan"),
        "within_1km": float(np.mean(np.abs(errors) <= 1.0)),
        "within_2km": float(np.mean(np.abs(errors) <= 2.0)),
        "balanced_regime_mae": balanced_regime_mae(true, pred, ceiling=ceiling),
    }
    if groups is not None:
        metrics["date_macro_mae"] = date_macro_mae(true, pred, np.asarray(groups))
    return metrics


def _safe_components(requested: int, train_rows: int, feature_width: int) -> int:
    components = min(int(requested), int(train_rows) - 1, int(feature_width))
    if components < 1:
        raise ValueError("Not enough rows to fit PCA")
    return components


def _fit_preprocessors(
    image_train: np.ndarray,
    weather_train_raw: np.ndarray,
    max_components: int,
    seed: int,
) -> tuple[StandardScaler, PCA, StandardScaler, np.ndarray, np.ndarray]:
    image_scaler = StandardScaler().fit(image_train)
    scaled = image_scaler.transform(image_train)
    pca = PCA(
        n_components=_safe_components(max_components, len(image_train), image_train.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    ).fit(scaled)
    weather_scaler = StandardScaler().fit(engineer_weather(weather_train_raw))
    return (
        image_scaler,
        pca,
        weather_scaler,
        pca.transform(scaled),
        weather_scaler.transform(engineer_weather(weather_train_raw)),
    )


def _candidate_features(
    candidate: SVRCandidate,
    image_pc: np.ndarray,
    weather_scaled: np.ndarray,
) -> np.ndarray:
    image_block = image_pc[:, : candidate.pca_components]
    if candidate.feature_mode == "image":
        return image_block
    if candidate.feature_mode == "fusion":
        return np.hstack([image_block, weather_scaled])
    raise ValueError(f"Unsupported feature mode: {candidate.feature_mode}")


def predict_candidate(
    candidate: SVRCandidate,
    image_train: np.ndarray,
    weather_train_raw: np.ndarray,
    y_train: np.ndarray,
    image_test: np.ndarray,
    weather_test_raw: np.ndarray,
    seed: int = 42,
    prediction_min: float = PREDICTION_MIN,
    prediction_max: float = PREDICTION_MAX,
) -> np.ndarray:
    image_scaler, pca, weather_scaler, train_pc, train_weather = _fit_preprocessors(
        image_train, weather_train_raw, candidate.pca_components, seed
    )
    test_pc = pca.transform(image_scaler.transform(image_test))
    test_weather = weather_scaler.transform(engineer_weather(weather_test_raw))
    train_features = _candidate_features(candidate, train_pc, train_weather)
    test_features = _candidate_features(candidate, test_pc, test_weather)
    model = SVR(kernel="rbf", gamma="scale", C=candidate.C, epsilon=candidate.epsilon)
    model.fit(train_features, y_train)
    return np.clip(model.predict(test_features), prediction_min, prediction_max)


def inner_grouped_predictions(
    candidate: SVRCandidate,
    image: np.ndarray,
    weather_raw: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    group_array = np.asarray(groups).astype(str)
    unique_groups = np.unique(group_array)
    splits = min(int(n_splits), len(unique_groups))
    if splits < 2:
        raise ValueError("At least two acquisition-date groups are required")
    predictions = np.full(len(target), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=splits)
    for train_index, validation_index in splitter.split(image, target, group_array):
        predictions[validation_index] = predict_candidate(
            candidate,
            image[train_index],
            weather_raw[train_index],
            target[train_index],
            image[validation_index],
            weather_raw[validation_index],
            seed=seed,
        )
    if np.isnan(predictions).any():
        raise RuntimeError("Inner grouped predictions did not cover all rows")
    return predictions


def select_candidate(
    candidates: Iterable[SVRCandidate],
    image: np.ndarray,
    weather_raw: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    selection_metric: str,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[SVRCandidate, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        prediction = inner_grouped_predictions(
            candidate,
            image,
            weather_raw,
            target,
            groups,
            n_splits=n_splits,
            seed=seed,
        )
        metrics = final_metrics(target, prediction, groups)
        rows.append({**candidate.as_dict(), **metrics})
    frame = pd.DataFrame(rows)
    if selection_metric not in frame.columns:
        raise ValueError(f"Unknown selection metric: {selection_metric}")
    ranked = frame.sort_values(
        [selection_metric, "mae", "rmse", "pca_components", "C", "epsilon"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_name = str(ranked.iloc[0]["name"])
    by_name = {candidate.name: candidate for candidate in candidates}
    return by_name[selected_name], frame


def fit_full_svr_bundle(
    candidate: SVRCandidate,
    image: np.ndarray,
    weather_raw: np.ndarray,
    target: np.ndarray,
    image_columns: list[str],
    raw_weather_columns: list[str],
    seed: int = 42,
    prediction_min: float = PREDICTION_MIN,
    prediction_max: float = PREDICTION_MAX,
) -> OptimizedSVRBundle:
    image_scaler, pca, weather_scaler, image_pc, weather_scaled = _fit_preprocessors(
        image, weather_raw, candidate.pca_components, seed
    )
    features = _candidate_features(candidate, image_pc, weather_scaled)
    model = SVR(kernel="rbf", gamma="scale", C=candidate.C, epsilon=candidate.epsilon)
    model.fit(features, target)
    return OptimizedSVRBundle(
        model_type="final_nested_pca_svr",
        model=model,
        feature_mode=candidate.feature_mode,
        image_scaler=image_scaler,
        pca=pca,
        weather_scaler=weather_scaler if candidate.feature_mode == "fusion" else None,
        image_columns=image_columns,
        raw_weather_columns=raw_weather_columns,
        pca_components=candidate.pca_components,
        C=candidate.C,
        epsilon=candidate.epsilon,
        prediction_min=prediction_min,
        prediction_max=prediction_max,
    )
