from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.prl.common import (
    PROJECT_ROOT,
    calculate_metrics,
    group_bootstrap_metrics,
    per_label_metrics,
    project_path,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PRL out-of-fold predictions and generate figures.")
    parser.add_argument("--predictions", default="results/prl/training/oof_predictions.csv")
    parser.add_argument("--training-summary", default="results/prl/training/training_summary.json")
    parser.add_argument("--output-dir", default="results/prl/evaluation")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def paired_group_bootstrap_difference(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample = np.concatenate([indices[group] for group in chosen])
        mae_a = np.mean(np.abs(prediction_a[sample] - actual[sample]))
        mae_b = np.mean(np.abs(prediction_b[sample] - actual[sample]))
        differences.append(float(mae_a - mae_b))
    values = np.asarray(differences)
    return {
        "mae_difference_a_minus_b": float(np.mean(np.abs(prediction_a - actual)) - np.mean(np.abs(prediction_b - actual))),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "probability_a_better_than_b": float(np.mean(values < 0)),
    }


def save_figures(frame: pd.DataFrame, selected: str, output_dir: Path) -> list[str]:
    actual = frame["actual_visibility_km"].to_numpy(dtype=float)
    predicted = frame[selected].to_numpy(dtype=float)
    residual = predicted - actual
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(actual, predicted, alpha=0.75)
    low = min(actual.min(), predicted.min()) - 0.5
    high = max(actual.max(), predicted.max()) + 0.5
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Observed visibility (km)")
    ax.set_ylabel("Out-of-fold predicted visibility (km)")
    ax.set_title(f"Observed vs predicted: {selected}")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "observed_vs_predicted.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path.relative_to(PROJECT_ROOT)))

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(predicted, residual, alpha=0.75)
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted visibility (km)")
    ax.set_ylabel("Residual: predicted - observed (km)")
    ax.set_title(f"Residual analysis: {selected}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "residual_plot.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path.relative_to(PROJECT_ROOT)))

    counts = frame["actual_visibility_km"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar([str(value) for value in counts.index], counts.values)
    ax.set_xlabel("Visibility label (km)")
    ax.set_ylabel("Number of samples")
    ax.set_title("PRL visibility-label distribution")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    path = output_dir / "label_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path.relative_to(PROJECT_ROOT)))

    date_mae = frame.assign(abs_error=np.abs(residual)).groupby("date", as_index=False)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(date_mae["date"], date_mae["abs_error"])
    ax.set_xlabel("Held-out acquisition date")
    ax.set_ylabel("MAE (km)")
    ax.set_title(f"Leave-one-date-out error: {selected}")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    path = output_dir / "mae_by_date.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path.relative_to(PROJECT_ROOT)))

    return paths


def main() -> int:
    args = parse_args()
    prediction_path = project_path(args.predictions)
    summary_path = project_path(args.training_summary)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(prediction_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        training_summary = json.load(handle)
    selected = training_summary["selected_model"]
    if selected not in frame.columns:
        raise ValueError(f"Selected model column {selected!r} is absent from {prediction_path}")

    identity_columns = {"image_name", "date", "time", "actual_visibility_km"}
    models = [column for column in frame.columns if column not in identity_columns]
    actual = frame["actual_visibility_km"].to_numpy(dtype=float)
    groups = frame["date"].astype(str).to_numpy()

    metrics_rows: list[dict] = []
    confidence: dict[str, dict] = {}
    for model in models:
        prediction = frame[model].to_numpy(dtype=float)
        metrics_rows.append({"model": model, **calculate_metrics(actual, prediction)})
        confidence[model] = group_bootstrap_metrics(
            actual, prediction, groups, n_bootstrap=args.bootstrap, seed=args.seed
        )
    metrics = pd.DataFrame(metrics_rows).sort_values(["macro_mae", "mae"], kind="stable")
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    save_json(confidence, output_dir / "bootstrap_confidence_intervals.json")

    selected_per_label = per_label_metrics(actual, frame[selected].to_numpy(dtype=float))
    selected_per_label.to_csv(output_dir / "selected_model_per_label_metrics.csv", index=False)

    date_rows = []
    for date, date_frame in frame.groupby("date", sort=True):
        values = calculate_metrics(
            date_frame["actual_visibility_km"], date_frame[selected]
        )
        date_rows.append({"date": date, **values})
    pd.DataFrame(date_rows).to_csv(output_dir / "selected_model_per_date_metrics.csv", index=False)

    comparison = paired_group_bootstrap_difference(
        actual,
        frame[selected].to_numpy(dtype=float),
        frame["dummy_median"].to_numpy(dtype=float),
        groups,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    save_json(comparison, output_dir / "selected_vs_dummy_bootstrap.json")
    figures = save_figures(frame, selected, output_dir)

    selected_metrics = metrics.set_index("model").loc[selected].to_dict()
    dummy_metrics = metrics.set_index("model").loc["dummy_median"].to_dict()
    report = {
        "selected_model": selected,
        "selected_metrics": selected_metrics,
        "dummy_metrics": dummy_metrics,
        "selected_confidence_intervals": confidence[selected],
        "selected_vs_dummy": comparison,
        "figures": figures,
        "evaluation_protocol": "Every prediction is out-of-fold under leave-one-acquisition-date-out cross-validation.",
        "interpretation": [
            "Ordinary MAE is dominated by the 10 km majority label.",
            "Macro MAE gives equal importance to the 4, 5, 9 and 10 km labels.",
            "Negative or low R² is possible even when MAE is small because the target variance is very small and strongly imbalanced.",
            "The dummy baseline must remain in all paper tables.",
        ],
    }
    save_json(report, output_dir / "evaluation_summary.json")

    markdown = [
        "# PRL Evaluation Summary",
        "",
        f"**Selected learned model:** `{selected}`",
        "",
        "## Protocol",
        "",
        "Leave-one-acquisition-date-out cross-validation was used so that near-duplicate images from the same day never appear in both training and test partitions.",
        "",
        "## Main metrics",
        "",
        metrics[["model", "mae", "rmse", "r2", "macro_mae", "low_visibility_mae", "within_1km"]].to_markdown(index=False),
        "",
        "## Critical interpretation",
        "",
        "The median dummy predicts 10 km for every sample and can outperform learned models on ordinary MAE because 105 of 127 labels equal 10 km. This is a dataset limitation, not evidence that a constant model is scientifically useful.",
        "",
        "The selected learned model is chosen by macro MAE, which weights each observed visibility level equally. All conclusions should include uncertainty intervals and the dummy comparison.",
    ]
    (output_dir / "evaluation_summary.md").write_text("\n".join(markdown), encoding="utf-8")

    print("=" * 72)
    print("PRL EVALUATION COMPLETE")
    print("=" * 72)
    print(metrics[["model", "mae", "rmse", "r2", "macro_mae", "low_visibility_mae"]].to_string(index=False))
    print(f"\nSelected model: {selected}")
    print(f"Output: {output_dir.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
