from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.prl.common import project_path, save_json
from src.prl.final_evaluation import final_metrics


METRICS = (
    "mae",
    "rmse",
    "r2",
    "mean_bias",
    "low_visibility_mae",
    "ceiling_label_mae",
    "balanced_regime_mae",
    "date_macro_mae",
    "within_1km",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster-bootstrap uncertainty and paired comparisons for final nested PRL results."
    )
    parser.add_argument(
        "--predictions",
        default="results/prl/final_nested/nested_oof_predictions.csv",
    )
    parser.add_argument("--results-dir", default="results/prl/final_nested/evaluation")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def model_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"image_name", "date", "time", "actual_visibility_km"}
    return [column for column in frame.columns if column not in excluded]


def bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    unique = np.unique(groups)
    chosen = rng.choice(unique, size=len(unique), replace=True)
    indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
    return indices, [str(group) for group in chosen]


def bootstrap_date_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    chosen_groups: list[str],
) -> float:
    values = []
    for group in chosen_groups:
        mask = groups == group
        values.append(float(np.mean(np.abs(y_pred[mask] - y_true[mask]))))
    return float(np.mean(values))


def build_bootstrap_samples(
    groups: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> list[tuple[np.ndarray, list[str]]]:
    rng = np.random.default_rng(seed)
    return [bootstrap_indices(groups, rng) for _ in range(n_bootstrap)]


def bootstrap_metric_values(
    frame: pd.DataFrame,
    model: str,
    samples: list[tuple[np.ndarray, list[str]]],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    true = frame["actual_visibility_km"].to_numpy(float)
    pred = frame[model].to_numpy(float)
    groups = frame["date"].astype(str).to_numpy()
    point = final_metrics(true, pred, groups)
    values = {metric: np.full(len(samples), np.nan, dtype=float) for metric in METRICS}
    for iteration, (indices, chosen_groups) in enumerate(samples):
        metric = final_metrics(true[indices], pred[indices])
        for name in METRICS:
            if name == "date_macro_mae":
                values[name][iteration] = bootstrap_date_macro(
                    true, pred, groups, chosen_groups
                )
            else:
                values[name][iteration] = metric.get(name, np.nan)
    return point, values


def ci_rows(model: str, point: dict[str, float], values: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        array = values[metric]
        rows.append(
            {
                "model": model,
                "metric": metric,
                "estimate": float(point[metric]),
                "ci95_low": float(np.nanpercentile(array, 2.5)),
                "ci95_high": float(np.nanpercentile(array, 97.5)),
            }
        )
    return rows


def per_date_rows(frame: pd.DataFrame, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, date_frame in frame.groupby("date", sort=True):
        true = date_frame["actual_visibility_km"].to_numpy(float)
        for model in models:
            pred = date_frame[model].to_numpy(float)
            rows.append(
                {
                    "date": str(date),
                    "model": model,
                    "n": int(len(date_frame)),
                    "n_low_visibility": int(np.sum(true < 10.0)),
                    "mae": float(np.mean(np.abs(pred - true))),
                    "rmse": float(np.mean((pred - true) ** 2) ** 0.5),
                    "mean_bias": float(np.mean(pred - true)),
                }
            )
    return rows


def make_plots(
    frame: pd.DataFrame,
    ci_frame: pd.DataFrame,
    per_date: pd.DataFrame,
    output_dir: Path,
) -> None:
    mae = ci_frame[ci_frame["metric"] == "mae"].sort_values("estimate")
    x = np.arange(len(mae))
    lower = mae["estimate"].to_numpy() - mae["ci95_low"].to_numpy()
    upper = mae["ci95_high"].to_numpy() - mae["estimate"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, mae["estimate"].to_numpy())
    ax.errorbar(x, mae["estimate"].to_numpy(), yerr=np.vstack([lower, upper]), fmt="none", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(mae["model"], rotation=35, ha="right")
    ax.set_ylabel("MAE (km)")
    ax.set_title("Final nested grouped performance with date-cluster 95% intervals")
    fig.tight_layout()
    fig.savefig(output_dir / "final_model_mae_ci.png", dpi=200)
    plt.close(fig)

    true = frame["actual_visibility_km"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in [name for name in ("nested_svr_mae", "nested_svr_balanced") if name in frame]:
        ax.scatter(true, frame[model].to_numpy(float), label=model, alpha=0.75)
    ax.plot([4, 10], [4, 10], linestyle="--")
    ax.set_xlim(3.8, 10.2)
    ax.set_ylim(3.8, 10.2)
    ax.set_xlabel("Observed visibility (km)")
    ax.set_ylabel("Predicted visibility (km)")
    ax.set_title("Nested outer-fold predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_observed_vs_predicted.png", dpi=200)
    plt.close(fig)

    pivot = per_date[per_date["model"].isin(["dummy_10km", "nested_svr_mae", "nested_svr_balanced"])].pivot(
        index="date", columns="model", values="mae"
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MAE (km)")
    ax.set_title("MAE by held-out acquisition date")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output_dir / "final_mae_by_date.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(project_path(args.predictions))
    models = model_columns(frame)
    if "dummy_10km" not in models:
        raise ValueError("The final prediction file must contain dummy_10km")

    ci_data: list[dict[str, Any]] = []
    all_bootstrap: dict[str, dict[str, np.ndarray]] = {}
    point_metrics: dict[str, dict[str, float]] = {}
    groups = frame["date"].astype(str).to_numpy()
    shared_samples = build_bootstrap_samples(groups, args.bootstrap, args.seed)
    for model in models:
        point, values = bootstrap_metric_values(frame, model, shared_samples)
        point_metrics[model] = point
        all_bootstrap[model] = values
        ci_data.extend(ci_rows(model, point, values))
        print(f"Bootstrapped {model}")

    paired_rows: list[dict[str, Any]] = []
    dummy_values = all_bootstrap["dummy_10km"]
    for model in models:
        if model == "dummy_10km":
            continue
        for metric in METRICS:
            difference = all_bootstrap[model][metric] - dummy_values[metric]
            estimate = point_metrics[model][metric] - point_metrics["dummy_10km"][metric]
            low = float(np.nanpercentile(difference, 2.5))
            high = float(np.nanpercentile(difference, 97.5))
            paired_rows.append(
                {
                    "model": model,
                    "reference": "dummy_10km",
                    "metric": metric,
                    "difference_model_minus_dummy": float(estimate),
                    "ci95_low": low,
                    "ci95_high": high,
                    "interval_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "direction": "lower_is_better" if metric not in {"r2", "within_1km"} else "higher_is_better",
                }
            )

    ci_frame = pd.DataFrame(ci_data)
    paired_frame = pd.DataFrame(paired_rows)
    per_date = pd.DataFrame(per_date_rows(frame, models))
    output_dir = project_path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ci_frame.to_csv(output_dir / "metric_confidence_intervals.csv", index=False)
    paired_frame.to_csv(output_dir / "paired_vs_dummy.csv", index=False)
    per_date.to_csv(output_dir / "per_date_metrics.csv", index=False)
    make_plots(frame, ci_frame, per_date, output_dir)

    summary = {
        "bootstrap_unit": "acquisition date",
        "paired_resampling": "All models use the same sampled acquisition dates in each bootstrap replicate.",
        "n_bootstrap": int(args.bootstrap),
        "n_dates": int(frame["date"].nunique()),
        "n_samples": int(len(frame)),
        "models": models,
        "point_metrics": point_metrics,
    }
    save_json(summary, output_dir / "nested_evaluation_summary.json")

    display = ci_frame[ci_frame["metric"].isin(["mae", "rmse", "balanced_regime_mae", "date_macro_mae"])]
    print("\nDATE-CLUSTER BOOTSTRAP RESULTS")
    print(display.to_string(index=False))
    print(f"\nSaved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
