from __future__ import annotations

import argparse
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import numpy as np
import pandas as pd

from src.prl.common import (
    PROJECT_ROOT,
    WEATHER_COLUMNS,
    feature_columns,
    load_prl_embeddings,
    load_prl_metadata,
    merge_metadata_embeddings,
    project_path,
    save_json,
    validate_finite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PRL metadata and DINOv2 embeddings without leakage.")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--output-dir", default="data/prl/processed")
    parser.add_argument("--expected-dimension", type=int, default=768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, metadata_path = load_prl_metadata(args.metadata)
    embeddings, embedding_path, embedding_columns = load_prl_embeddings(args.embeddings)
    if len(embedding_columns) != args.expected_dimension:
        raise ValueError(
            f"Expected {args.expected_dimension} DINOv2 features, found {len(embedding_columns)} in {embedding_path}"
        )

    merged, merge_info = merge_metadata_embeddings(metadata, embeddings)
    dino_columns = feature_columns(merged)
    required = [*WEATHER_COLUMNS, "visibility_km", "date", "image_name"]
    missing = [column for column in required if column not in merged.columns]
    if missing:
        raise ValueError(f"Prepared dataset is missing required columns: {missing}")

    numeric_columns = [*WEATHER_COLUMNS, "visibility_km", *dino_columns]
    merged[numeric_columns] = merged[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if merged[numeric_columns].isna().any().any():
        bad = merged[numeric_columns].isna().sum()
        raise ValueError(f"Numeric conversion introduced missing values:\n{bad[bad > 0]}")

    merged["date"] = pd.to_datetime(merged["date"], errors="raise").dt.strftime("%Y-%m-%d")
    merged = merged.sort_values(["date", "time", "image_name"], kind="stable").reset_index(drop=True)

    image_matrix = merged[dino_columns].to_numpy(dtype=np.float32)
    weather_matrix = merged[list(WEATHER_COLUMNS)].to_numpy(dtype=np.float32)
    target = merged["visibility_km"].to_numpy(dtype=np.float32)
    groups = merged["date"].astype(str).to_numpy(dtype=str)
    image_names = merged["image_name"].astype(str).to_numpy(dtype=str)
    validate_finite(image_matrix, "PRL image features")
    validate_finite(weather_matrix, "PRL weather features")
    validate_finite(target, "PRL target")

    csv_path = output_dir / "prl_fusion_dataset.csv"
    merged.to_csv(csv_path, index=False)
    np.save(output_dir / "X_image.npy", image_matrix)
    np.save(output_dir / "X_weather_raw.npy", weather_matrix)
    np.save(output_dir / "y.npy", target)
    np.save(output_dir / "groups_date.npy", groups)
    np.save(output_dir / "image_names.npy", image_names)

    source_summary = None
    source_metadata_path = PROJECT_ROOT / "data/processed/feature_metadata.csv"
    if source_metadata_path.exists():
        source = pd.read_csv(source_metadata_path)
        source_summary = {
            "temperature": source["temperature"].describe().to_dict() if "temperature" in source else None,
            "humidity": source["humidity"].describe().to_dict() if "humidity" in source else None,
            "solar": source["solar"].describe().to_dict() if "solar" in source else None,
            "note": "Source and target weather are intentionally kept raw. Scaling is fitted inside each CV training fold to avoid leakage and to handle the incompatible solar domains.",
        }

    manifest = {
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
        "embedding_path": str(embedding_path.relative_to(PROJECT_ROOT)),
        "output_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "merge": merge_info,
        "rows": int(len(merged)),
        "dates": int(merged["date"].nunique()),
        "embedding_dimension": int(len(dino_columns)),
        "weather_columns": list(WEATHER_COLUMNS),
        "target_column": "visibility_km",
        "label_counts": {str(key): int(value) for key, value in merged["visibility_km"].value_counts().sort_index().items()},
        "image_shape": list(image_matrix.shape),
        "weather_shape": list(weather_matrix.shape),
        "target_shape": list(target.shape),
        "source_weather_summary": source_summary,
        "leakage_controls": [
            "No target scaling is fitted in this preparation step.",
            "PCA and scalers are fitted only on training folds during evaluation.",
            "Date is retained as a grouping variable for leave-one-date-out evaluation.",
        ],
    }
    save_json(manifest, output_dir / "dataset_manifest.json")

    print("=" * 72)
    print("PRL FUSION DATASET PREPARED")
    print("=" * 72)
    print(f"Rows:       {len(merged)}")
    print(f"Dates:      {merged['date'].nunique()}")
    print(f"DINOv2:     {image_matrix.shape}")
    print(f"Weather:    {weather_matrix.shape} (raw, not globally scaled)")
    print(f"Target:     {target.shape}")
    print(f"Merge:      {merge_info['strategy']}")
    print(f"Output:     {csv_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
