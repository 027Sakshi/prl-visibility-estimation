from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.prl.common import (
    PROJECT_ROOT,
    WEATHER_COLUMNS,
    calculate_metrics,
    dump_bundle,
    load_prepared_prl,
    project_path,
    save_json,
    set_reproducible_seed,
)
from src.prl.estimators import LocalRidgeBundle, TransferRidgeBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and train leakage-controlled PRL visibility models with source-to-target adaptation."
    )
    parser.add_argument("--dataset", default="data/prl/processed/prl_fusion_dataset.csv")
    parser.add_argument("--results-dir", default="results/prl/training")
    parser.add_argument("--models-dir", default="models/prl")
    parser.add_argument("--pca-components", type=int, default=15)
    parser.add_argument("--ridge-alpha-weather", type=float, default=10.0)
    parser.add_argument("--ridge-alpha-image", type=float, default=100.0)
    parser.add_argument("--ridge-alpha-fusion", type=float, default=100.0)
    parser.add_argument("--transfer-alpha", type=float, default=100.0)
    parser.add_argument("--target-weight-factor", type=float, default=3.0)
    parser.add_argument("--xgb-estimators", type=int, default=50)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--selection-metric", choices=("mae", "rmse", "macro_mae", "low_visibility_mae"), default="macro_mae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-min", type=float, default=0.0)
    parser.add_argument("--prediction-max", type=float, default=20.0)
    return parser.parse_args()


def make_xgb(args: argparse.Namespace) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.xgb_estimators,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=args.seed,
        n_jobs=1,
        verbosity=0,
    )


def fit_local_components(
    image_train: np.ndarray,
    weather_train: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[StandardScaler, PCA, StandardScaler, np.ndarray, np.ndarray]:
    image_scaler = StandardScaler().fit(image_train)
    scaled_image = image_scaler.transform(image_train)
    components = min(n_components, scaled_image.shape[0] - 1, scaled_image.shape[1])
    if components < 1:
        raise ValueError("Not enough training samples for PCA")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed).fit(scaled_image)
    weather_scaler = StandardScaler().fit(weather_train)
    return image_scaler, pca, weather_scaler, pca.transform(scaled_image), weather_scaler.transform(weather_train)


def source_transfer_assets(n_components: int, seed: int) -> dict[str, Any] | None:
    image_path = PROJECT_ROOT / "data/processed/X_image.npy"
    target_path = PROJECT_ROOT / "data/processed/y.npy"
    metadata_path = PROJECT_ROOT / "data/processed/feature_metadata.csv"
    if not (image_path.exists() and target_path.exists() and metadata_path.exists()):
        return None
    source_image = np.load(image_path, mmap_mode="r")
    source_target = np.load(target_path).astype(float)
    source_metadata = pd.read_csv(metadata_path)
    source_weather = source_metadata[["temperature", "humidity", "solar"]].to_numpy(dtype=float)
    if len(source_image) != len(source_target) or len(source_target) != len(source_weather):
        raise ValueError("SkyFinder source arrays have inconsistent row counts")

    components = min(n_components, source_image.shape[0] - 1, source_image.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed).fit(source_image)
    source_pc = pca.transform(source_image)
    pc_scaler = StandardScaler().fit(source_pc)
    source_weather_scaler = StandardScaler().fit(source_weather)
    source_features = np.hstack(
        [
            pc_scaler.transform(source_pc),
            source_weather_scaler.transform(source_weather),
            np.zeros((len(source_target), 1), dtype=float),
        ]
    )
    return {
        "image": source_image,
        "target": source_target,
        "weather": source_weather,
        "pca": pca,
        "pc_scaler": pc_scaler,
        "weather_scaler": source_weather_scaler,
        "features": source_features,
    }


def clip_prediction(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), args.prediction_min, args.prediction_max)


def evaluate_models(
    frame: pd.DataFrame,
    image_columns: list[str],
    args: argparse.Namespace,
    transfer: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    image = frame[image_columns].to_numpy(dtype=float)
    weather = frame[list(WEATHER_COLUMNS)].to_numpy(dtype=float)
    target = frame["visibility_km"].to_numpy(dtype=float)
    groups = frame["date"].astype(str).to_numpy()

    candidate_names = [
        "dummy_median",
        "weather_ridge",
        "image_pca_ridge",
        "fusion_pca_ridge",
    ]
    if not args.skip_xgboost:
        candidate_names.append("fusion_pca_xgboost")
    if transfer is not None:
        candidate_names.append("transfer_weighted_ridge")
    predictions = {name: np.full(len(target), np.nan, dtype=float) for name in candidate_names}
    fold_rows: list[dict[str, Any]] = []

    logo = LeaveOneGroupOut()
    for fold, (train_index, test_index) in enumerate(logo.split(image, target, groups), start=1):
        fold_start = time.perf_counter()
        train_image, test_image = image[train_index], image[test_index]
        train_weather, test_weather = weather[train_index], weather[test_index]
        train_target, test_target = target[train_index], target[test_index]

        dummy = DummyRegressor(strategy="median").fit(train_weather, train_target)
        predictions["dummy_median"][test_index] = clip_prediction(dummy.predict(test_weather), args)

        weather_scaler_simple = StandardScaler().fit(train_weather)
        weather_model = Ridge(alpha=args.ridge_alpha_weather).fit(
            weather_scaler_simple.transform(train_weather), train_target
        )
        predictions["weather_ridge"][test_index] = clip_prediction(
            weather_model.predict(weather_scaler_simple.transform(test_weather)), args
        )

        image_scaler, pca, weather_scaler, train_pc, train_weather_scaled = fit_local_components(
            train_image, train_weather, args.pca_components, args.seed
        )
        test_pc = pca.transform(image_scaler.transform(test_image))
        test_weather_scaled = weather_scaler.transform(test_weather)

        image_model = Ridge(alpha=args.ridge_alpha_image).fit(train_pc, train_target)
        predictions["image_pca_ridge"][test_index] = clip_prediction(image_model.predict(test_pc), args)

        train_fusion = np.hstack([train_pc, train_weather_scaled])
        test_fusion = np.hstack([test_pc, test_weather_scaled])
        fusion_model = Ridge(alpha=args.ridge_alpha_fusion).fit(train_fusion, train_target)
        predictions["fusion_pca_ridge"][test_index] = clip_prediction(fusion_model.predict(test_fusion), args)

        if "fusion_pca_xgboost" in predictions:
            xgb_model = make_xgb(args).fit(train_fusion, train_target)
            predictions["fusion_pca_xgboost"][test_index] = clip_prediction(xgb_model.predict(test_fusion), args)

        if transfer is not None:
            target_weather_scaler = StandardScaler().fit(train_weather[:, :3])
            target_train_features = np.hstack(
                [
                    transfer["pc_scaler"].transform(transfer["pca"].transform(train_image)),
                    target_weather_scaler.transform(train_weather[:, :3]),
                    np.ones((len(train_index), 1), dtype=float),
                ]
            )
            target_test_features = np.hstack(
                [
                    transfer["pc_scaler"].transform(transfer["pca"].transform(test_image)),
                    target_weather_scaler.transform(test_weather[:, :3]),
                    np.ones((len(test_index), 1), dtype=float),
                ]
            )
            combined_features = np.vstack([transfer["features"], target_train_features])
            combined_target = np.concatenate([transfer["target"], train_target])
            target_row_weight = args.target_weight_factor * len(transfer["target"]) / len(train_target)
            sample_weight = np.concatenate(
                [
                    np.ones(len(transfer["target"]), dtype=float),
                    np.full(len(train_target), target_row_weight, dtype=float),
                ]
            )
            transfer_model = Ridge(alpha=args.transfer_alpha).fit(
                combined_features, combined_target, sample_weight=sample_weight
            )
            predictions["transfer_weighted_ridge"][test_index] = clip_prediction(
                transfer_model.predict(target_test_features), args
            )

        fold_seconds = time.perf_counter() - fold_start
        for name in candidate_names:
            fold_prediction = predictions[name][test_index]
            metrics = calculate_metrics(test_target, fold_prediction)
            fold_rows.append(
                {
                    "fold": fold,
                    "held_out_date": groups[test_index][0],
                    "model": name,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "seconds_for_all_models": fold_seconds,
                    **metrics,
                }
            )
        print(f"Fold {fold:02d}/{len(np.unique(groups))}: held out {groups[test_index][0]} ({len(test_index)} samples)")

    if any(np.isnan(values).any() for values in predictions.values()):
        missing = {name: int(np.isnan(values).sum()) for name, values in predictions.items()}
        raise RuntimeError(f"OOF predictions are incomplete: {missing}")

    prediction_frame = frame[["image_name", "date", "time", "visibility_km"]].copy()
    prediction_frame = prediction_frame.rename(columns={"visibility_km": "actual_visibility_km"})
    metric_rows: list[dict[str, Any]] = []
    for name, values in predictions.items():
        prediction_frame[name] = values
        metric_rows.append({"model": name, **calculate_metrics(target, values)})
    metrics_frame = pd.DataFrame(metric_rows).sort_values([args.selection_metric, "mae"], kind="stable")
    fold_frame = pd.DataFrame(fold_rows)
    details = {
        "evaluation": "Leave-one-date-out cross-validation",
        "groups": int(len(np.unique(groups))),
        "selection_metric": args.selection_metric,
        "prediction_range": [args.prediction_min, args.prediction_max],
        "pca_components": args.pca_components,
        "transfer_available": transfer is not None,
        "target_weight_factor": args.target_weight_factor if transfer is not None else None,
        "important_interpretation": (
            "The median dummy is a mandatory reference. Because most PRL labels equal 10 km, it may win ordinary MAE while having no predictive skill. "
            "Learned-model selection therefore excludes the dummy and defaults to macro MAE across visibility labels."
        ),
    }
    return prediction_frame, metrics_frame, {"fold_metrics": fold_frame, **details}


def fit_final_models(
    frame: pd.DataFrame,
    image_columns: list[str],
    args: argparse.Namespace,
    transfer: dict[str, Any] | None,
    models_dir: Path,
) -> dict[str, Any]:
    image = frame[image_columns].to_numpy(dtype=float)
    weather = frame[list(WEATHER_COLUMNS)].to_numpy(dtype=float)
    target = frame["visibility_km"].to_numpy(dtype=float)
    models_dir.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, Any] = {}

    weather_scaler = StandardScaler().fit(weather)
    weather_model = Ridge(alpha=args.ridge_alpha_weather).fit(weather_scaler.transform(weather), target)
    bundles["weather_ridge"] = LocalRidgeBundle(
        model_type="local_ridge",
        model=weather_model,
        weather_scaler=weather_scaler,
        feature_mode="weather",
        image_columns=image_columns,
        weather_columns=list(WEATHER_COLUMNS),
        prediction_min=args.prediction_min,
        prediction_max=args.prediction_max,
    )

    image_scaler, pca, fusion_weather_scaler, image_pc, weather_scaled = fit_local_components(
        image, weather, args.pca_components, args.seed
    )
    image_model = Ridge(alpha=args.ridge_alpha_image).fit(image_pc, target)
    bundles["image_pca_ridge"] = LocalRidgeBundle(
        model_type="local_ridge",
        model=image_model,
        image_scaler=image_scaler,
        pca=pca,
        feature_mode="image",
        image_columns=image_columns,
        weather_columns=list(WEATHER_COLUMNS),
        prediction_min=args.prediction_min,
        prediction_max=args.prediction_max,
    )
    fusion_matrix = np.hstack([image_pc, weather_scaled])
    fusion_model = Ridge(alpha=args.ridge_alpha_fusion).fit(fusion_matrix, target)
    bundles["fusion_pca_ridge"] = LocalRidgeBundle(
        model_type="local_ridge",
        model=fusion_model,
        image_scaler=image_scaler,
        pca=pca,
        weather_scaler=fusion_weather_scaler,
        feature_mode="fusion",
        image_columns=image_columns,
        weather_columns=list(WEATHER_COLUMNS),
        prediction_min=args.prediction_min,
        prediction_max=args.prediction_max,
    )

    if not args.skip_xgboost:
        xgb_model = make_xgb(args).fit(fusion_matrix, target)
        bundles["fusion_pca_xgboost"] = LocalRidgeBundle(
            model_type="local_xgboost",
            model=xgb_model,
            image_scaler=image_scaler,
            pca=pca,
            weather_scaler=fusion_weather_scaler,
            feature_mode="fusion",
            image_columns=image_columns,
            weather_columns=list(WEATHER_COLUMNS),
            prediction_min=args.prediction_min,
            prediction_max=args.prediction_max,
        )
        xgb_model.save_model(models_dir / "fusion_pca_xgboost.json")

    if transfer is not None:
        target_weather_scaler = StandardScaler().fit(weather[:, :3])
        target_features = np.hstack(
            [
                transfer["pc_scaler"].transform(transfer["pca"].transform(image)),
                target_weather_scaler.transform(weather[:, :3]),
                np.ones((len(target), 1), dtype=float),
            ]
        )
        combined_features = np.vstack([transfer["features"], target_features])
        combined_target = np.concatenate([transfer["target"], target])
        target_row_weight = args.target_weight_factor * len(transfer["target"]) / len(target)
        sample_weight = np.concatenate(
            [
                np.ones(len(transfer["target"]), dtype=float),
                np.full(len(target), target_row_weight, dtype=float),
            ]
        )
        model = Ridge(alpha=args.transfer_alpha).fit(combined_features, combined_target, sample_weight=sample_weight)
        bundles["transfer_weighted_ridge"] = TransferRidgeBundle(
            model_type="transfer_weighted_ridge",
            model=model,
            source_pca=transfer["pca"],
            source_pc_scaler=transfer["pc_scaler"],
            target_weather_scaler=target_weather_scaler,
            image_columns=image_columns,
            weather_columns=list(WEATHER_COLUMNS[:3]),
            target_weight_factor=args.target_weight_factor,
            alpha=args.transfer_alpha,
            prediction_min=args.prediction_min,
            prediction_max=args.prediction_max,
        )

    for name, estimator in bundles.items():
        joblib.dump({"model_type": estimator.model_type, "estimator": estimator, "model_name": name}, models_dir / f"{name}.joblib")
    return bundles


def main() -> int:
    args = parse_args()
    set_reproducible_seed(args.seed)
    results_dir = project_path(args.results_dir)
    models_dir = project_path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    frame, image_columns = load_prepared_prl(args.dataset)
    transfer = source_transfer_assets(args.pca_components, args.seed)

    print("=" * 78)
    print("PRL LEAKAGE-CONTROLLED MODEL DEVELOPMENT")
    print("=" * 78)
    print(f"Samples: {len(frame)} | dates: {frame['date'].nunique()} | DINOv2 dimensions: {len(image_columns)}")
    print(f"Source transfer assets: {'available' if transfer is not None else 'not available'}")
    print("Evaluation: leave one acquisition date out at a time")

    predictions, metrics, details = evaluate_models(frame, image_columns, args, transfer)
    predictions.to_csv(results_dir / "oof_predictions.csv", index=False)
    metrics.to_csv(results_dir / "model_metrics.csv", index=False)
    details["fold_metrics"].to_csv(results_dir / "fold_metrics.csv", index=False)

    learned = metrics[metrics["model"] != "dummy_median"].copy()
    selected_name = learned.sort_values([args.selection_metric, "mae"], kind="stable").iloc[0]["model"]
    bundles = fit_final_models(frame, image_columns, args, transfer, models_dir)
    if selected_name not in bundles:
        raise RuntimeError(f"Selected model {selected_name} was not fitted")
    selected_bundle = {
        "model_type": bundles[selected_name].model_type,
        "estimator": bundles[selected_name],
        "model_name": selected_name,
        "selection_metric": args.selection_metric,
        "selected_cv_metrics": metrics.set_index("model").loc[selected_name].to_dict(),
    }
    dump_bundle(selected_bundle, models_dir / "prl_visibility_model.joblib")

    dummy_row = metrics.set_index("model").loc["dummy_median"]
    selected_row = metrics.set_index("model").loc[selected_name]
    summary = {
        **{key: value for key, value in details.items() if key != "fold_metrics"},
        "selected_model": selected_name,
        "selected_metrics": selected_row.to_dict(),
        "dummy_metrics": dummy_row.to_dict(),
        "dummy_wins_standard_mae": bool(dummy_row["mae"] < selected_row["mae"]),
        "selection_note": (
            "The production/research model is selected among learned models only. The dummy remains the honest standard-MAE reference caused by the 10-km label ceiling/imbalance."
        ),
        "artifacts": {
            "selected_bundle": "models/prl/prl_visibility_model.joblib",
            "all_bundles": [f"models/prl/{name}.joblib" for name in bundles],
            "oof_predictions": "results/prl/training/oof_predictions.csv",
            "metrics": "results/prl/training/model_metrics.csv",
        },
    }
    save_json(summary, results_dir / "training_summary.json")

    print("\nMODEL METRICS (OOF, GROUPED BY DATE)")
    print(metrics[["model", "mae", "rmse", "r2", "macro_mae", "low_visibility_mae"]].to_string(index=False))
    print(f"\nSelected learned model: {selected_name} by {args.selection_metric}")
    if summary["dummy_wins_standard_mae"]:
        print(
            "Important: the constant median baseline has lower ordinary MAE because 105/127 labels are 10 km. "
            "This must be reported rather than hidden."
        )
    print(f"Selected bundle: {(models_dir / 'prl_visibility_model.joblib').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        raise SystemExit(main())
