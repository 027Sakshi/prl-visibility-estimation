from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import pandas as pd

from src.prl.common import project_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate locked final paper tables from nested PRL results.")
    parser.add_argument("--nested-dir", default="results/prl/final_nested")
    parser.add_argument("--output-dir", default="results/prl/final_paper")
    return parser.parse_args()


def format_ci(row: pd.Series, digits: int = 3) -> str:
    return f"{row['estimate']:.{digits}f} [{row['ci95_low']:.{digits}f}, {row['ci95_high']:.{digits}f}]"


def main() -> int:
    args = parse_args()
    nested_dir = project_path(args.nested_dir)
    evaluation_dir = nested_dir / "evaluation"
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(nested_dir / "nested_model_metrics.csv")
    ci = pd.read_csv(evaluation_dir / "metric_confidence_intervals.csv")
    paired = pd.read_csv(evaluation_dir / "paired_vs_dummy.csv")
    per_date = pd.read_csv(evaluation_dir / "per_date_metrics.csv")
    selections = pd.read_csv(nested_dir / "outer_fold_selections.csv")
    full_scores = pd.read_csv(nested_dir / "full_data_inner_candidate_scores.csv")
    with (nested_dir / "nested_benchmark_summary.json").open("r", encoding="utf-8") as handle:
        benchmark_summary = json.load(handle)

    wide_ci = ci.pivot(index="model", columns="metric", values=["estimate", "ci95_low", "ci95_high"])
    wide_ci.columns = [f"{metric}_{stat}" for stat, metric in wide_ci.columns]
    table = metrics.merge(wide_ci.reset_index(), on="model", how="left")
    for metric in ("mae", "rmse", "balanced_regime_mae", "date_macro_mae"):
        subset = ci[ci["metric"] == metric].set_index("model")
        table[f"{metric}_95ci"] = table["model"].map(
            lambda model: format_ci(subset.loc[model]) if model in subset.index else ""
        )
    table = table.sort_values(["mae", "rmse"])
    table.to_csv(output_dir / "table_final_model_comparison.csv", index=False)
    paired.to_csv(output_dir / "table_final_paired_vs_dummy.csv", index=False)
    per_date.to_csv(output_dir / "table_final_per_date.csv", index=False)
    selections.to_csv(output_dir / "table_outer_fold_hyperparameter_selections.csv", index=False)
    full_scores.sort_values(["mae", "balanced_regime_mae"]).to_csv(
        output_dir / "table_full_data_inner_selection.csv", index=False
    )

    for filename in (
        "final_model_mae_ci.png",
        "final_observed_vs_predicted.png",
        "final_mae_by_date.png",
    ):
        source = evaluation_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    dummy = table.loc[table["model"] == "dummy_10km"].iloc[0]
    learned = table[table["model"] != "dummy_10km"].copy()
    best_mae = learned.sort_values(["mae", "rmse"]).iloc[0]
    best_balanced = learned.sort_values(["balanced_regime_mae", "mae"]).iloc[0]
    best_rmse = learned.sort_values(["rmse", "mae"]).iloc[0]

    mae_pair = paired[(paired["model"] == best_mae["model"]) & (paired["metric"] == "mae")]
    balanced_pair = paired[
        (paired["model"] == best_balanced["model"])
        & (paired["metric"] == "balanced_regime_mae")
    ]

    def paired_sentence(row_frame: pd.DataFrame, label: str) -> str:
        if row_frame.empty:
            return f"No paired bootstrap comparison was available for {label}."
        row = row_frame.iloc[0]
        conclusion = "excluded" if bool(row["interval_excludes_zero"]) else "included"
        return (
            f"The paired date-cluster bootstrap difference was {row['difference_model_minus_dummy']:.3f} km "
            f"(95% CI {row['ci95_low']:.3f} to {row['ci95_high']:.3f}); the interval {conclusion} zero."
        )

    draft = f"""# Final Locked PRL Results Draft

## Evaluation protocol

Final performance was estimated with nested acquisition-date-grouped validation over {benchmark_summary['n_samples']} images from {benchmark_summary['n_dates']} acquisition dates. The outer loop left one complete date out. Within each outer-training set, GroupKFold selected among the predeclared PCA-SVR configurations. Image scaling, PCA, engineered-weather scaling when applicable, and SVR fitting were repeated inside each training split. Confidence intervals were obtained by resampling acquisition dates as clusters.

## Primary sample-weighted result

The best learned model by nested outer-fold MAE was **{best_mae['model']}**, with MAE {best_mae['mae']:.3f} km, RMSE {best_mae['rmse']:.3f} km, and R² {best_mae['r2']:.3f}. The constant 10 km reference achieved MAE {dummy['mae']:.3f} km and RMSE {dummy['rmse']:.3f} km. {paired_sentence(mae_pair, 'MAE')}

## Reduced-visibility and regime-balanced result

The best learned model by balanced-regime MAE was **{best_balanced['model']}**, with balanced-regime MAE {best_balanced['balanced_regime_mae']:.3f} km, low-visibility MAE {best_balanced['low_visibility_mae']:.3f} km, and 10 km-label MAE {best_balanced['ceiling_label_mae']:.3f} km. The constant reference had balanced-regime MAE {dummy['balanced_regime_mae']:.3f} km and low-visibility MAE {dummy['low_visibility_mae']:.3f} km. {paired_sentence(balanced_pair, 'balanced-regime MAE')}

## Additional result

The lowest learned-model RMSE was obtained by **{best_rmse['model']}** at {best_rmse['rmse']:.3f} km. Date-macro MAE, which weights each acquisition date equally, was {best_mae['date_macro_mae']:.3f} km for the best-MAE learned model and {dummy['date_macro_mae']:.3f} km for the constant reference.

## Interpretation constraint

The dataset is small, single-location, and strongly concentrated at exactly 10 km visibility. Claims should therefore distinguish sample-weighted overall error from performance on reduced-visibility observations. The nested outer-fold predictions and date-cluster intervals in this directory are the only values intended for the abstract, final results, discussion, and conclusion. Development-search values from `results/prl/optimization_v2` must not be presented as final test performance.
"""
    (output_dir / "final_results_draft.md").write_text(draft, encoding="utf-8")

    manifest = {
        "status": "final_locked_nested_results",
        "source_directory": str(nested_dir),
        "primary_metric": "mae",
        "secondary_metrics": [
            "rmse",
            "r2",
            "low_visibility_mae",
            "ceiling_label_mae",
            "balanced_regime_mae",
            "date_macro_mae",
            "within_1km",
            "mean_bias",
        ],
        "best_learned_by_mae": best_mae["model"],
        "best_learned_by_balanced_regime_mae": best_balanced["model"],
        "paper_tables": [
            "table_final_model_comparison.csv",
            "table_final_paired_vs_dummy.csv",
            "table_final_per_date.csv",
            "table_outer_fold_hyperparameter_selections.csv",
            "table_full_data_inner_selection.csv",
        ],
    }
    save_json(manifest, output_dir / "final_results_manifest.json")

    print("=" * 84)
    print("FINAL PAPER RESULTS GENERATED")
    print("=" * 84)
    print(table[["model", "mae", "rmse", "r2", "balanced_regime_mae", "date_macro_mae"]].to_string(index=False))
    print(f"\nDraft: {output_dir / 'final_results_draft.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
