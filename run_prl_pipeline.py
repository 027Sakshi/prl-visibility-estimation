from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete PRL visibility adaptation pipeline.")
    parser.add_argument("--extract-features", action="store_true", help="Re-extract PRL embeddings from data/prl_images.")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print("\n" + "=" * 88)
    print("RUNNING:", " ".join(command))
    print("=" * 88)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    if args.extract_features:
        extraction_args = ["--backend", "torchhub"]
        if args.force_features:
            extraction_args.append("--force")
        run("src/prl/02_extract_prl_features.py", *extraction_args)

    verification_args = []
    if args.require_images:
        verification_args.append("--require-images")
    run("src/prl/01_verify_prl_dataset.py", *verification_args)
    run("src/prl/03_prepare_prl_dataset.py")

    training_args = []
    if args.skip_xgboost:
        training_args.append("--skip-xgboost")
    run("src/prl/04_finetune_fusion.py", *training_args)
    run("src/prl/05_evaluate_prl.py", "--bootstrap", str(args.bootstrap))
    run("src/prl/07_generate_paper_results.py")

    print("\n" + "=" * 88)
    print("PRL PIPELINE COMPLETED")
    print("Selected model: models/prl/prl_visibility_model.joblib")
    print("Metrics:        results/prl/evaluation/evaluation_metrics.csv")
    print("Paper draft:    results/prl/paper/paper_results_draft.md")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
