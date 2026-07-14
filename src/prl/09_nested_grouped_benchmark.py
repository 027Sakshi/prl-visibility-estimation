from __future__ import annotations

import argparse
import json
import time
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.prl.common import WEATHER_COLUMNS, load_prepared_prl, project_path, save_json
from src.prl.final_evaluation import (
    PREDICTION_MAX,
    PREDICTION_MIN,
    final_metrics,
    fit_full_svr_bundle,
    locked_svr_grid,
    predict_candidate,
    select_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final nested acquisition-date-grouped benchmark for PRL visibility estimation."
    )
    parser.add_argument("--dataset", default="data/prl/processed/prl_fusion_dataset.csv")
    parser.add_argument("--results-dir", default="results/prl/final_nested")
    parser.add_argument("--models-dir", default="models/prl")
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-min", type=float, default=PREDICTION_MIN)
    parser.add_argument("--prediction-max", type=float, default=PREDICTION_MAX)
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip the fixed fusion XGBoost comparison if XGBoost is unavailable.",
    )
    return parser.parse_args()


def make_xgb(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=50,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def clip(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), args.prediction_min, args.prediction_max)


def fit_fixed_baselines(
    train_image: np.ndarray,
    test_image: np.ndarray,
    train_weather: np.ndarray,
    test_weather: np.ndarray,
    train_target: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {
        "dummy_10km": np.full(len(test_image), 10.0, dtype=float),
    }

    weather_scaler = StandardScaler().fit(train_weather)
    weather_ridge = Ridge(alpha=10.0).fit(weather_scaler.transform(train_weather), train_target)
    predictions["weather_ridge"] = clip(
        weather_ridge.predict(weather_scaler.transform(test_weather)), args
    )

    image_scaler = StandardScaler().fit(train_image)
    train_scaled = image_scaler.transform(train_image)
    test_scaled = image_scaler.transform(test_image)
    components = min(15, len(train_image) - 1, train_image.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=args.seed).fit(
        train_scaled
    )
    train_pc = pca.transform(train_scaled)
    test_pc = pca.transform(test_scaled)

    image_ridge = Ridge(alpha=100.0).fit(train_pc, train_target)
    predictions["image_pca_ridge"] = clip(image_ridge.predict(test_pc), args)

    fusion_train = np.hstack([train_pc, weather_scaler.transform(train_weather)])
    fusion_test = np.hstack([test_pc, weather_scaler.transform(test_weather)])
    fusion_ridge = Ridge(alpha=100.0).fit(fusion_train, train_target)
    predictions["fusion_pca_ridge"] = clip(fusion_ridge.predict(fusion_test), args)

    if not args.skip_xgboost:
        fusion_xgb = make_xgb(args.seed).fit(fusion_train, train_target)
        predictions["fusion_pca_xgboost"] = clip(fusion_xgb.predict(fusion_test), args)
    return predictions


def add_transfer_reference(
    prediction_frame: pd.DataFrame,
    dataset_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    path = project_path("results/prl/training/oof_predictions.csv")
    if not path.exists():
        return prediction_frame, None
    legacy = pd.read_csv(path)
    required = {"image_name", "date", "actual_visibility_km", "transfer_weighted_ridge"}
    if not required.issubset(legacy.columns) or len(legacy) != len(dataset_frame):
        return prediction_frame, None
    left = dataset_frame[["image_name", "date", "visibility_km"]].reset_index(drop=True)
    right = legacy[["image_name", "date", "actual_visibility_km", "transfer_weighted_ridge"]].reset_index(drop=True)
    aligned = (
        left["image_name"].astype(str).equals(right["image_name"].astype(str))
        and left["date"].astype(str).equals(right["date"].astype(str))
        and np.allclose(left["visibility_km"].to_numpy(float), right["actual_visibility_km"].to_numpy(float))
    )
    if not aligned:
        return prediction_frame, None
    prediction_frame["transfer_weighted_ridge_reference"] = right[
        "transfer_weighted_ridge"
    ].to_numpy(float)
    return prediction_frame, "Imported from the previously completed fixed leave-one-date-out baseline."


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    frame, image_columns = load_prepared_prl(args.dataset)
    image = frame[image_columns].to_numpy(dtype=float)
    weather = frame[list(WEATHER_COLUMNS)].to_numpy(dtype=float)
    target = frame["visibility_km"].to_numpy(dtype=float)
    groups = frame["date"].astype(str).to_numpy()
    candidates = locked_svr_grid()

    model_names = [
        "dummy_10km",
        "weather_ridge",
        "image_pca_ridge",
        "fusion_pca_ridge",
    ]
    if not args.skip_xgboost:
        model_names.append("fusion_pca_xgboost")
    model_names.extend(["nested_svr_mae", "nested_svr_balanced"])
    predictions = {name: np.full(len(frame), np.nan, dtype=float) for name in model_names}

    selection_rows: list[dict[str, Any]] = []
    inner_score_frames: list[pd.DataFrame] = []
    logo = LeaveOneGroupOut()

    print("=" * 88)
    print("FINAL NESTED GROUPED BENCHMARK")
    print("=" * 88)
    print(f"Samples: {len(frame)} | acquisition dates: {len(np.unique(groups))}")
    print(f"Locked SVR candidates: {len(candidates)} | inner GroupKFold: {args.inner_splits}")

    for fold, (train_index, test_index) in enumerate(logo.split(image, target, groups), start=1):
        held_out_date = str(groups[test_index][0])
        fixed = fit_fixed_baselines(
            image[train_index],
            image[test_index],
            weather[train_index],
            weather[test_index],
            target[train_index],
            args,
        )
        for name, values in fixed.items():
            predictions[name][test_index] = values

        selected_mae, scores = select_candidate(
            candidates,
            image[train_index],
            weather[train_index],
            target[train_index],
            groups[train_index],
            selection_metric="mae",
            n_splits=args.inner_splits,
            seed=args.seed,
        )
        scores = scores.copy()
        scores.insert(0, "outer_held_out_date", held_out_date)
        inner_score_frames.append(scores)

        selected_balanced = candidates[0]
        balanced_ranked = scores.sort_values(
            ["balanced_regime_mae", "mae", "rmse", "name"], kind="mergesort"
        )
        selected_balanced_name = str(balanced_ranked.iloc[0]["name"])
        selected_balanced = {candidate.name: candidate for candidate in candidates}[
            selected_balanced_name
        ]

        predictions["nested_svr_mae"][test_index] = predict_candidate(
            selected_mae,
            image[train_index],
            weather[train_index],
            target[train_index],
            image[test_index],
            weather[test_index],
            seed=args.seed,
            prediction_min=args.prediction_min,
            prediction_max=args.prediction_max,
        )
        predictions["nested_svr_balanced"][test_index] = predict_candidate(
            selected_balanced,
            image[train_index],
            weather[train_index],
            target[train_index],
            image[test_index],
            weather[test_index],
            seed=args.seed,
            prediction_min=args.prediction_min,
            prediction_max=args.prediction_max,
        )

        mae_row = scores.loc[scores["name"] == selected_mae.name].iloc[0]
        balanced_row = scores.loc[scores["name"] == selected_balanced.name].iloc[0]
        selection_rows.extend(
            [
                {
                    "outer_fold": fold,
                    "outer_held_out_date": held_out_date,
                    "selection_objective": "mae",
                    **selected_mae.as_dict(),
                    "inner_mae": float(mae_row["mae"]),
                    "inner_balanced_regime_mae": float(mae_row["balanced_regime_mae"]),
                },
                {
                    "outer_fold": fold,
                    "outer_held_out_date": held_out_date,
                    "selection_objective": "balanced_regime_mae",
                    **selected_balanced.as_dict(),
                    "inner_mae": float(balanced_row["mae"]),
                    "inner_balanced_regime_mae": float(balanced_row["balanced_regime_mae"]),
                },
            ]
        )
        print(
            f"Fold {fold:02d}/23 | held out {held_out_date} | "
            f"MAE-select={selected_mae.name} | balanced-select={selected_balanced.name}"
        )

    for name, values in predictions.items():
        if np.isnan(values).any():
            raise RuntimeError(f"Missing outer predictions for {name}")

    prediction_frame = frame[["image_name", "date", "time", "visibility_km"]].copy()
    prediction_frame = prediction_frame.rename(columns={"visibility_km": "actual_visibility_km"})
    prediction_frame = pd.concat(
        [prediction_frame.reset_index(drop=True), pd.DataFrame(predictions)], axis=1
    )
    prediction_frame, transfer_note = add_transfer_reference(prediction_frame, frame)

    metrics_rows: list[dict[str, Any]] = []
    non_model_columns = {"image_name", "date", "time", "actual_visibility_km"}
    for name in [column for column in prediction_frame.columns if column not in non_model_columns]:
        metrics_rows.append(
            {
                "model": name,
                **final_metrics(target, prediction_frame[name].to_numpy(float), groups),
            }
        )
    metrics_frame = pd.DataFrame(metrics_rows).sort_values(["mae", "rmse"])

    results_dir = project_path(args.results_dir)
    models_dir = project_path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(results_dir / "nested_oof_predictions.csv", index=False)
    metrics_frame.to_csv(results_dir / "nested_model_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(results_dir / "outer_fold_selections.csv", index=False)
    pd.concat(inner_score_frames, ignore_index=True).to_csv(
        results_dir / "inner_candidate_scores.csv", index=False
    )

    final_mae_candidate, full_scores = select_candidate(
        candidates,
        image,
        weather,
        target,
        groups,
        selection_metric="mae",
        n_splits=args.inner_splits,
        seed=args.seed,
    )
    balanced_name = str(
        full_scores.sort_values(
            ["balanced_regime_mae", "mae", "rmse", "name"], kind="mergesort"
        ).iloc[0]["name"]
    )
    final_balanced_candidate = {candidate.name: candidate for candidate in candidates}[balanced_name]
    full_scores.to_csv(results_dir / "full_data_inner_candidate_scores.csv", index=False)

    for objective, candidate, filename in (
        ("mae", final_mae_candidate, "final_nested_mae_model.joblib"),
        ("balanced_regime_mae", final_balanced_candidate, "final_nested_balanced_model.joblib"),
    ):
        estimator = fit_full_svr_bundle(
            candidate,
            image,
            weather,
            target,
            image_columns,
            list(WEATHER_COLUMNS),
            seed=args.seed,
            prediction_min=args.prediction_min,
            prediction_max=args.prediction_max,
        )
        joblib.dump(
            {
                "model_name": f"final_{objective}_selected_{candidate.name}",
                "model_type": estimator.model_type,
                "selection_objective": objective,
                "candidate": candidate.as_dict(),
                "estimator": estimator,
                "research_status": "Deployable fit on all PRL observations; paper performance comes from nested outer OOF predictions.",
            },
            models_dir / filename,
        )

    summary = {
        "protocol": {
            "outer": "LeaveOneGroupOut by acquisition date",
            "inner": f"GroupKFold(n_splits={min(args.inner_splits, len(np.unique(groups)) - 1)}) on outer-training dates",
            "selection_objectives": ["mae", "balanced_regime_mae"],
            "preprocessing": "Image scaling, PCA, weather engineering/scaling, and SVR fitting repeated inside every inner and outer training split.",
        },
        "n_samples": int(len(frame)),
        "n_dates": int(len(np.unique(groups))),
        "locked_candidate_count": int(len(candidates)),
        "locked_candidates": [candidate.as_dict() for candidate in candidates],
        "final_full_data_mae_candidate": final_mae_candidate.as_dict(),
        "final_full_data_balanced_candidate": final_balanced_candidate.as_dict(),
        "transfer_reference_note": transfer_note,
        "runtime_seconds": float(time.perf_counter() - started),
        "result_files": {
            "predictions": str(results_dir / "nested_oof_predictions.csv"),
            "metrics": str(results_dir / "nested_model_metrics.csv"),
            "outer_selections": str(results_dir / "outer_fold_selections.csv"),
        },
    }
    save_json(summary, results_dir / "nested_benchmark_summary.json")

    print("\nFINAL NESTED OUTER-FOLD METRICS")
    print(
        metrics_frame[
            [
                "model",
                "mae",
                "rmse",
                "r2",
                "low_visibility_mae",
                "ceiling_label_mae",
                "balanced_regime_mae",
                "date_macro_mae",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
