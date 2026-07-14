from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.prl.common import WEATHER_COLUMNS, calculate_metrics, load_prepared_prl, project_path, save_json
from src.prl.optimized_estimators import (
    OptimizedRidgeBundle,
    OptimizedSVRBundle,
    engineer_weather,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused leakage-controlled search for stronger PRL visibility models."
    )
    parser.add_argument("--dataset", default="data/prl/processed/prl_fusion_dataset.csv")
    parser.add_argument("--results-dir", default="results/prl/optimization_v2")
    parser.add_argument("--models-dir", default="models/prl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use the smallest useful search grid.")
    parser.add_argument("--prediction-min", type=float, default=4.0)
    parser.add_argument("--prediction-max", type=float, default=10.0)
    return parser.parse_args()


def class_weights(y: np.ndarray, beta: float) -> np.ndarray:
    values, counts = np.unique(y, return_counts=True)
    mapping = {
        value: (len(y) / (len(values) * count)) ** beta
        for value, count in zip(values, counts, strict=True)
    }
    weights = np.asarray([mapping[value] for value in y], dtype=float)
    return weights / weights.mean()


def balanced_regime_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    low = y_true < 10.0
    return float(
        0.5
        * (
            mean_absolute_error(y_true[low], y_pred[low])
            + mean_absolute_error(y_true[~low], y_pred[~low])
        )
    )


def candidate_grid(quick: bool) -> list[dict[str, Any]]:
    if quick:
        pcs = [2, 3, 5]
        c_values = [1.0, 10.0, 30.0]
        epsilons = [0.5, 0.75, 1.0]
        ridge_pcs = [3, 5]
        alphas = [10.0, 100.0, 1000.0]
        betas = [0.1, 0.25]
    else:
        pcs = [1, 2, 3, 5, 8, 10, 15]
        c_values = [0.3, 1.0, 3.0, 10.0, 30.0]
        epsilons = [0.25, 0.5, 0.75, 1.0]
        ridge_pcs = [2, 3, 5, 8, 10, 15]
        alphas = [1.0, 10.0, 100.0, 1000.0]
        betas = [0.1, 0.25, 0.4]

    candidates: list[dict[str, Any]] = []
    for mode in ("image", "fusion"):
        for pc in pcs:
            for c_value in c_values:
                for epsilon in epsilons:
                    candidates.append(
                        {
                            "family": "svr",
                            "feature_mode": mode,
                            "pca_components": pc,
                            "C": c_value,
                            "epsilon": epsilon,
                        }
                    )
    for pc in ridge_pcs:
        for alpha in alphas:
            for beta in betas:
                candidates.append(
                    {
                        "family": "ridge",
                        "feature_mode": "fusion",
                        "pca_components": pc,
                        "alpha": alpha,
                        "weight_beta": beta,
                    }
                )
    return candidates


def candidate_name(candidate: dict[str, Any]) -> str:
    if candidate["family"] == "svr":
        return (
            f'{candidate["feature_mode"]}_svr_pc{candidate["pca_components"]}'
            f'_C{candidate["C"]}_e{candidate["epsilon"]}'
        )
    return (
        f'fusion_weighted_ridge_pc{candidate["pca_components"]}'
        f'_a{candidate["alpha"]}_b{candidate["weight_beta"]}'
    )


def build_fold_cache(
    image: np.ndarray,
    weather_engineered: np.ndarray,
    groups: np.ndarray,
    max_components: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[int, dict[str, Any]]]:
    splits = list(LeaveOneGroupOut().split(image, groups=groups))
    cache: dict[int, dict[str, Any]] = {}
    for fold, (train_index, test_index) in enumerate(splits):
        image_scaler = StandardScaler().fit(image[train_index])
        image_train_scaled = image_scaler.transform(image[train_index])
        image_test_scaled = image_scaler.transform(image[test_index])
        components = min(max_components, len(train_index) - 1, image.shape[1])
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=seed,
        ).fit(image_train_scaled)
        weather_scaler = StandardScaler().fit(weather_engineered[train_index])
        cache[fold] = {
            "image_train": pca.transform(image_train_scaled),
            "image_test": pca.transform(image_test_scaled),
            "weather_train": weather_scaler.transform(weather_engineered[train_index]),
            "weather_test": weather_scaler.transform(weather_engineered[test_index]),
        }
    return splits, cache


def evaluate_candidate(
    candidate: dict[str, Any],
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    cache: dict[int, dict[str, Any]],
    prediction_min: float,
    prediction_max: float,
) -> np.ndarray:
    predictions = np.full(len(target), np.nan, dtype=float)
    for fold, (train_index, test_index) in enumerate(splits):
        stored = cache[fold]
        components = candidate["pca_components"]
        image_train = stored["image_train"][:, :components]
        image_test = stored["image_test"][:, :components]
        if candidate["feature_mode"] == "image":
            train_features = image_train
            test_features = image_test
        else:
            train_features = np.hstack([image_train, stored["weather_train"]])
            test_features = np.hstack([image_test, stored["weather_test"]])

        if candidate["family"] == "svr":
            model = SVR(
                kernel="rbf",
                gamma="scale",
                C=candidate["C"],
                epsilon=candidate["epsilon"],
            )
            model.fit(train_features, target[train_index])
        else:
            model = Ridge(alpha=candidate["alpha"])
            model.fit(
                train_features,
                target[train_index],
                sample_weight=class_weights(target[train_index], candidate["weight_beta"]),
            )
        predictions[test_index] = model.predict(test_features)
    return np.clip(predictions, prediction_min, prediction_max)


def fit_full_bundle(
    candidate: dict[str, Any],
    image: np.ndarray,
    weather_raw: np.ndarray,
    target: np.ndarray,
    image_columns: list[str],
    args: argparse.Namespace,
) -> OptimizedSVRBundle | OptimizedRidgeBundle:
    image_scaler = StandardScaler().fit(image)
    scaled_image = image_scaler.transform(image)
    components = min(candidate["pca_components"], len(target) - 1, image.shape[1])
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=args.seed,
    ).fit(scaled_image)
    image_features = pca.transform(scaled_image)

    weather_scaler: StandardScaler | None = None
    if candidate["feature_mode"] == "fusion":
        weather_scaler = StandardScaler().fit(engineer_weather(weather_raw))
        train_features = np.hstack(
            [image_features, weather_scaler.transform(engineer_weather(weather_raw))]
        )
    else:
        train_features = image_features

    if candidate["family"] == "svr":
        model = SVR(
            kernel="rbf",
            gamma="scale",
            C=candidate["C"],
            epsilon=candidate["epsilon"],
        ).fit(train_features, target)
        return OptimizedSVRBundle(
            model_type="optimized_pca_svr",
            model=model,
            feature_mode=candidate["feature_mode"],
            image_scaler=image_scaler,
            pca=pca,
            weather_scaler=weather_scaler,
            image_columns=image_columns,
            raw_weather_columns=list(WEATHER_COLUMNS),
            pca_components=components,
            C=candidate["C"],
            epsilon=candidate["epsilon"],
            prediction_min=args.prediction_min,
            prediction_max=args.prediction_max,
        )

    model = Ridge(alpha=candidate["alpha"]).fit(
        train_features,
        target,
        sample_weight=class_weights(target, candidate["weight_beta"]),
    )
    return OptimizedRidgeBundle(
        model_type="optimized_weighted_pca_ridge",
        model=model,
        feature_mode=candidate["feature_mode"],
        image_scaler=image_scaler,
        pca=pca,
        weather_scaler=weather_scaler,
        image_columns=image_columns,
        raw_weather_columns=list(WEATHER_COLUMNS),
        pca_components=components,
        alpha=candidate["alpha"],
        weight_beta=candidate["weight_beta"],
        prediction_min=args.prediction_min,
        prediction_max=args.prediction_max,
    )


def save_bundle(path: Path, model_name: str, candidate: dict[str, Any], estimator: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": model_name,
            "model_type": estimator.model_type,
            "candidate": candidate,
            "estimator": estimator,
            "research_status": "development model; requires nested grouped validation before final paper claims",
        },
        path,
    )


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    frame, image_columns = load_prepared_prl(args.dataset)
    image = frame[image_columns].to_numpy(dtype=float)
    weather_raw = frame[list(WEATHER_COLUMNS)].to_numpy(dtype=float)
    weather_engineered = engineer_weather(weather_raw)
    target = frame["visibility_km"].to_numpy(dtype=float)
    groups = frame["date"].astype(str).to_numpy()

    candidates = candidate_grid(args.quick)
    max_components = max(candidate["pca_components"] for candidate in candidates)
    splits, cache = build_fold_cache(image, weather_engineered, groups, max_components, args.seed)

    rows: list[dict[str, Any]] = []
    prediction_frame = frame[["image_name", "date", "visibility_km"]].copy()

    dummy_predictions = np.full(len(target), 10.0, dtype=float)
    dummy_metrics = calculate_metrics(target, dummy_predictions, ceiling=10.0)
    dummy_metrics["balanced_regime_mae"] = balanced_regime_mae(target, dummy_predictions)
    rows.append({"model": "dummy_10km", "family": "dummy", **dummy_metrics})
    prediction_frame["dummy_10km"] = dummy_predictions

    print("=" * 84)
    print("PRL FOCUSED ACCURACY OPTIMIZATION")
    print("=" * 84)
    print(f"Samples: {len(frame)} | dates: {frame['date'].nunique()} | candidates: {len(candidates)}")
    print("Evaluation: leave one acquisition date out")

    for index, candidate in enumerate(candidates, start=1):
        prediction = evaluate_candidate(
            candidate,
            target,
            splits,
            cache,
            args.prediction_min,
            args.prediction_max,
        )
        name = candidate_name(candidate)
        metrics = calculate_metrics(target, prediction, ceiling=10.0)
        metrics["balanced_regime_mae"] = balanced_regime_mae(target, prediction)
        rows.append({"model": name, **candidate, **metrics})
        prediction_frame[name] = prediction
        if index % 20 == 0 or index == len(candidates):
            print(f"Evaluated {index}/{len(candidates)} candidates")

    metrics_frame = pd.DataFrame(rows)
    learned = metrics_frame[metrics_frame["family"] != "dummy"].copy()
    best_accuracy_row = learned.sort_values(["mae", "rmse", "macro_mae"]).iloc[0]
    best_balanced_row = learned.sort_values(
        ["balanced_regime_mae", "macro_mae", "mae"]
    ).iloc[0]

    by_name = {candidate_name(candidate): candidate for candidate in candidates}
    best_accuracy_candidate = by_name[str(best_accuracy_row["model"])]
    best_balanced_candidate = by_name[str(best_balanced_row["model"])]

    results_dir = project_path(args.results_dir)
    models_dir = project_path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_frame.sort_values(["mae", "macro_mae"]).to_csv(
        results_dir / "candidate_metrics.csv", index=False
    )
    prediction_frame.to_csv(results_dir / "oof_predictions.csv", index=False)

    accuracy_bundle = fit_full_bundle(
        best_accuracy_candidate, image, weather_raw, target, image_columns, args
    )
    balanced_bundle = fit_full_bundle(
        best_balanced_candidate, image, weather_raw, target, image_columns, args
    )
    save_bundle(
        models_dir / "optimized_accuracy_model.joblib",
        str(best_accuracy_row["model"]),
        best_accuracy_candidate,
        accuracy_bundle,
    )
    save_bundle(
        models_dir / "optimized_balanced_model.joblib",
        str(best_balanced_row["model"]),
        best_balanced_candidate,
        balanced_bundle,
    )

    summary = {
        "protocol": "leave-one-acquisition-date-out development benchmark",
        "warning": "Candidate selection and reported OOF metrics use the same grouped predictions. Run nested validation before final paper claims.",
        "n_samples": int(len(frame)),
        "n_dates": int(frame["date"].nunique()),
        "n_candidates": int(len(candidates)),
        "dummy": {key: dummy_metrics[key] for key in dummy_metrics},
        "best_accuracy_model": {
            "name": str(best_accuracy_row["model"]),
            "candidate": best_accuracy_candidate,
            "metrics": {
                key: float(best_accuracy_row[key])
                for key in (
                    "mae",
                    "rmse",
                    "r2",
                    "macro_mae",
                    "low_visibility_mae",
                    "ceiling_label_mae",
                    "balanced_regime_mae",
                )
            },
            "path": "models/prl/optimized_accuracy_model.joblib",
        },
        "best_balanced_model": {
            "name": str(best_balanced_row["model"]),
            "candidate": best_balanced_candidate,
            "metrics": {
                key: float(best_balanced_row[key])
                for key in (
                    "mae",
                    "rmse",
                    "r2",
                    "macro_mae",
                    "low_visibility_mae",
                    "ceiling_label_mae",
                    "balanced_regime_mae",
                )
            },
            "path": "models/prl/optimized_balanced_model.joblib",
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    save_json(summary, results_dir / "optimization_summary.json")

    print("\nBEST ACCURACY-FOCUSED LEARNED MODEL")
    print(best_accuracy_row[["model", "mae", "rmse", "r2", "macro_mae", "low_visibility_mae"]].to_string())
    print("\nBEST BALANCED-REGIME LEARNED MODEL")
    print(best_balanced_row[["model", "mae", "rmse", "r2", "macro_mae", "low_visibility_mae", "balanced_regime_mae"]].to_string())
    print("\nDUMMY 10-KM BASELINE")
    print(f"MAE={dummy_metrics['mae']:.6f} | balanced_regime_mae={dummy_metrics['balanced_regime_mae']:.6f}")
    print(f"\nResults: {results_dir.relative_to(project_path('.'))}")
    print("Models:  models/prl/optimized_accuracy_model.joblib")
    print("         models/prl/optimized_balanced_model.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
