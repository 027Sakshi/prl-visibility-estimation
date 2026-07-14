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

from src.prl.common import PROJECT_ROOT, project_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready PRL tables, figures and a Markdown results draft.")
    parser.add_argument("--dataset", default="data/prl/processed/prl_fusion_dataset.csv")
    parser.add_argument("--training-metrics", default="results/prl/training/model_metrics.csv")
    parser.add_argument("--training-summary", default="results/prl/training/training_summary.json")
    parser.add_argument("--evaluation-dir", default="results/prl/evaluation")
    parser.add_argument("--output-dir", default="results/prl/paper")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_comparison_plot(metrics: pd.DataFrame, output_dir: Path) -> str:
    ordered = metrics.sort_values("macro_mae", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    positions = np.arange(len(ordered))
    width = 0.38
    ax.barh(positions - width / 2, ordered["mae"], height=width, label="MAE")
    ax.barh(positions + width / 2, ordered["macro_mae"], height=width, label="Macro MAE")
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlabel("Error (km; lower is better)")
    ax.set_title("PRL model comparison under leave-one-date-out evaluation")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "paper_model_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT))


def main() -> int:
    args = parse_args()
    dataset_path = project_path(args.dataset)
    metrics_path = project_path(args.training_metrics)
    training_summary_path = project_path(args.training_summary)
    evaluation_dir = project_path(args.evaluation_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(dataset_path)
    metrics = pd.read_csv(metrics_path)
    training_summary = load_json(training_summary_path)
    evaluation_summary = load_json(evaluation_dir / "evaluation_summary.json")
    confidence = load_json(evaluation_dir / "bootstrap_confidence_intervals.json")
    per_label = pd.read_csv(evaluation_dir / "selected_model_per_label_metrics.csv")
    per_date = pd.read_csv(evaluation_dir / "selected_model_per_date_metrics.csv")

    selected = training_summary["selected_model"]
    labels = data["visibility_km"].value_counts().sort_index()
    dataset_summary = pd.DataFrame(
        [
            {"statistic": "Samples", "value": len(data)},
            {"statistic": "Acquisition dates", "value": data["date"].nunique()},
            {"statistic": "DINOv2 dimensions", "value": len([c for c in data if c.startswith("dino_")])},
            {"statistic": "Visibility minimum (km)", "value": data["visibility_km"].min()},
            {"statistic": "Visibility maximum (km)", "value": data["visibility_km"].max()},
            {"statistic": "10-km majority fraction", "value": float((data["visibility_km"] == 10).mean())},
            {"statistic": "Temperature mean (°C)", "value": data["temperature_C"].mean()},
            {"statistic": "Humidity mean (%)", "value": data["relative_humidity_%"].mean()},
            {"statistic": "Solar mean (W/m²)", "value": data["solar_intensity_Wm2"].mean()},
        ]
    )
    dataset_summary.to_csv(output_dir / "table_dataset_summary.csv", index=False)

    label_table = labels.rename_axis("visibility_km").reset_index(name="n")
    label_table["fraction"] = label_table["n"] / len(data)
    label_table.to_csv(output_dir / "table_label_distribution.csv", index=False)

    selected_ci = confidence[selected]
    paper_metrics = metrics.copy()
    paper_metrics["is_selected_learned_model"] = paper_metrics["model"].eq(selected)
    for metric_name in ("mae", "rmse", "macro_mae"):
        paper_metrics[f"{metric_name}_ci95_low"] = np.nan
        paper_metrics[f"{metric_name}_ci95_high"] = np.nan
        for model in paper_metrics["model"]:
            values = confidence[model][metric_name]
            paper_metrics.loc[paper_metrics["model"] == model, f"{metric_name}_ci95_low"] = values["ci95_low"]
            paper_metrics.loc[paper_metrics["model"] == model, f"{metric_name}_ci95_high"] = values["ci95_high"]
    paper_metrics.to_csv(output_dir / "table_model_comparison.csv", index=False)
    per_label.to_csv(output_dir / "table_selected_model_per_label.csv", index=False)
    per_date.to_csv(output_dir / "table_selected_model_per_date.csv", index=False)

    figure = model_comparison_plot(metrics, output_dir)

    selected_row = metrics.set_index("model").loc[selected]
    dummy_row = metrics.set_index("model").loc["dummy_median"]
    source_results_path = PROJECT_ROOT / "results/comparison/comparison_metrics.csv"
    source_results = pd.read_csv(source_results_path) if source_results_path.exists() else None

    limitations = [
        "Only 127 labelled samples are available from one camera and one location.",
        "The target is highly imbalanced: 105 of 127 observations are labelled 10 km.",
        "Many images are temporally adjacent; evaluation therefore groups by acquisition date.",
        "The 10-km label may behave as a measurement ceiling. This must be verified from the label-generation protocol before treating it as an exact continuous value.",
        "SkyFinder solar radiation and PRL solar intensity are not directly interchangeable; the transfer pipeline uses separate domain scalers rather than the archived global scaler.",
        "A negative R² can coexist with a small MAE when target variance is low and a constant predictor is strong.",
    ]

    lines = [
        "# PRL Visibility Estimation — Results Draft",
        "",
        "## Dataset",
        "",
        f"The local PRL dataset contained **{len(data)} observations** collected over **{data['date'].nunique()} acquisition dates**. Each observation combined a 768-dimensional DINOv2 embedding with temperature, relative humidity, instantaneous solar intensity and acquisition hour.",
        "",
        f"The label distribution was strongly imbalanced: **{int(labels.get(10, 0))}/{len(data)} ({(labels.get(10, 0)/len(data)):.1%})** observations were labelled 10 km. The remaining labels were {', '.join(f'{value:g} km: {count}' for value, count in labels.items() if value != 10)}.",
        "",
        "## Evaluation protocol",
        "",
        "All model-development results were generated with leave-one-acquisition-date-out cross-validation. PCA and all target-domain scalers were fitted inside each training fold. This prevents images captured on the same date from leaking into both training and testing data.",
        "",
        "## Model comparison",
        "",
        paper_metrics[["model", "mae", "rmse", "r2", "macro_mae", "low_visibility_mae", "within_1km"]].to_markdown(index=False),
        "",
        f"The selected learned model was **{selected}**, selected by macro MAE rather than ordinary MAE. It achieved MAE **{selected_row['mae']:.3f} km**, RMSE **{selected_row['rmse']:.3f} km**, macro MAE **{selected_row['macro_mae']:.3f} km**, and R² **{selected_row['r2']:.3f}**.",
        "",
        f"The mandatory median dummy achieved lower ordinary MAE (**{dummy_row['mae']:.3f} km**) by predicting 10 km for every observation. This result demonstrates the severity of the label imbalance and prevents an inflated claim of predictive performance.",
        "",
        f"The selected model's group-bootstrap MAE interval was **[{selected_ci['mae']['ci95_low']:.3f}, {selected_ci['mae']['ci95_high']:.3f}] km** and its macro-MAE interval was **[{selected_ci['macro_mae']['ci95_low']:.3f}, {selected_ci['macro_mae']['ci95_high']:.3f}] km**.",
        "",
        "## Per-label behaviour",
        "",
        per_label.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The experiment does not support a claim that the learned model beats a constant predictor on overall MAE. Its value is instead as a leakage-controlled domain-adaptation benchmark that gives greater attention to minority visibility states. The next accuracy gain is more likely to come from collecting diverse low-visibility observations and clarifying whether 10 km is a censored upper bound than from further hyperparameter search.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in limitations],
        "",
        "## Generated figure",
        "",
        f"- `{figure}`",
    ]
    if source_results is not None:
        lines.extend(
            [
                "",
                "## Archived SkyFinder pretraining benchmark",
                "",
                source_results.to_markdown(index=False),
                "",
                "These SkyFinder hold-out results are not directly comparable with the PRL leave-one-date-out results because the datasets, label distributions and evaluation partitions differ.",
            ]
        )

    report_path = output_dir / "paper_results_draft.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "selected_model": selected,
        "tables": [
            "table_dataset_summary.csv",
            "table_label_distribution.csv",
            "table_model_comparison.csv",
            "table_selected_model_per_label.csv",
            "table_selected_model_per_date.csv",
        ],
        "figures": [figure],
        "draft": str(report_path.relative_to(PROJECT_ROOT)),
        "limitations": limitations,
    }
    save_json(manifest, output_dir / "paper_results_manifest.json")

    print("=" * 72)
    print("PAPER RESULTS GENERATED")
    print("=" * 72)
    print(f"Selected model: {selected}")
    print(f"Draft: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Tables/figures: {output_dir.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
