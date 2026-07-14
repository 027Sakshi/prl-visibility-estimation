from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.prl.common import (
    PROJECT_ROOT,
    ProjectDataError,
    feature_columns,
    first_existing,
    load_prl_embeddings,
    load_prl_metadata,
    normalise_image_name,
    project_path,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PRL metadata, images and DINOv2 embeddings.")
    parser.add_argument("--metadata", default=None, help="Metadata CSV/XLSX path relative to project root.")
    parser.add_argument("--embeddings", default=None, help="Embedding CSV path relative to project root.")
    parser.add_argument("--images", default="data/prl_images", help="PRL image directory.")
    parser.add_argument("--report-dir", default="results/prl/verification")
    parser.add_argument("--require-images", action="store_true", help="Fail when the image directory is absent/empty.")
    parser.add_argument("--expected-dimension", type=int, default=768)
    return parser.parse_args()


def hash_names(values: pd.Series) -> str:
    joined = "\n".join(sorted(values.map(normalise_image_name)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def inspect_images(image_dir: Path) -> dict:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not image_dir.exists():
        return {"exists": False, "count": 0, "files": [], "invalid": []}
    files = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    invalid: list[dict[str, str]] = []
    dimensions: list[tuple[int, int]] = []
    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions.append(image.size)
        except Exception as exc:  # pragma: no cover - depends on local image corruption
            invalid.append({"file": str(path.relative_to(PROJECT_ROOT)), "error": str(exc)})
    unique_dimensions = sorted({f"{width}x{height}" for width, height in dimensions})
    return {
        "exists": True,
        "count": len(files),
        "files": [path.name for path in files],
        "invalid": invalid,
        "unique_dimensions": unique_dimensions,
    }


def main() -> int:
    args = parse_args()
    report_dir = project_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata, metadata_path = load_prl_metadata(args.metadata)
    embeddings, embedding_path, dino_columns = load_prl_embeddings(args.embeddings)
    image_info = inspect_images(project_path(args.images))

    numeric_columns = [
        "temperature_C",
        "relative_humidity_%",
        "solar_intensity_Wm2",
        "visibility_km",
        "latitude",
        "longitude",
        "distance_m",
        "hour",
    ]
    numeric_columns = [column for column in numeric_columns if column in metadata.columns]
    numeric_frame = metadata[numeric_columns].apply(pd.to_numeric, errors="coerce")

    metadata_keys = metadata["image_name"].map(normalise_image_name)
    embedding_keys = embeddings["image_name"].map(normalise_image_name)
    image_keys = pd.Series([normalise_image_name(name) for name in image_info.get("files", [])], dtype="object")

    checks: list[dict[str, object]] = []

    def add_check(name: str, passed: bool, details: object, severity: str = "error") -> None:
        checks.append({"check": name, "passed": bool(passed), "severity": severity, "details": details})

    add_check("metadata_rows", len(metadata) > 0, len(metadata))
    add_check("metadata_unique_image_names", metadata_keys.is_unique, int(metadata_keys.duplicated().sum()))
    add_check("metadata_missing_values", int(metadata.isna().sum().sum()) == 0, metadata.isna().sum().to_dict())
    add_check("metadata_numeric_parse", int(numeric_frame.isna().sum().sum()) == 0, numeric_frame.isna().sum().to_dict())
    add_check("embedding_rows", len(embeddings) > 0, len(embeddings))
    add_check("embedding_unique_image_names", embedding_keys.is_unique, int(embedding_keys.duplicated().sum()))
    add_check("embedding_dimension", len(dino_columns) == args.expected_dimension, len(dino_columns))
    embedding_matrix = embeddings[dino_columns].to_numpy(dtype=np.float32)
    add_check("embedding_finite", np.isfinite(embedding_matrix).all(), int((~np.isfinite(embedding_matrix)).sum()))
    add_check("metadata_embedding_count", len(metadata) == len(embeddings), {"metadata": len(metadata), "embeddings": len(embeddings)})

    exact_name_match = set(metadata_keys) == set(embedding_keys)
    row_order_fallback_possible = len(metadata) == len(embeddings) and metadata_keys.is_unique and embedding_keys.is_unique
    add_check(
        "metadata_embedding_names",
        exact_name_match or row_order_fallback_possible,
        {
            "exact_match": exact_name_match,
            "row_order_fallback_possible": row_order_fallback_possible,
            "metadata_hash": hash_names(metadata["image_name"]),
            "embedding_hash": hash_names(embeddings["image_name"]),
        },
        severity="warning" if not exact_name_match and row_order_fallback_possible else "error",
    )

    if image_info["exists"] and image_info["count"] > 0:
        add_check("images_readable", len(image_info["invalid"]) == 0, image_info["invalid"])
        add_check("image_count_matches_metadata", image_info["count"] == len(metadata), {"images": image_info["count"], "metadata": len(metadata)})
        names_match = set(image_keys) in (set(metadata_keys), set(embedding_keys))
        add_check(
            "image_names_match_table",
            names_match or image_info["count"] == len(metadata),
            {"exact_match": names_match, "same_count": image_info["count"] == len(metadata)},
            severity="warning" if not names_match and image_info["count"] == len(metadata) else "error",
        )
    else:
        add_check("images_available", not args.require_images, image_info, severity="error" if args.require_images else "warning")

    label_counts = metadata["visibility_km"].value_counts(dropna=False).sort_index()
    majority_fraction = float(label_counts.max() / len(metadata))
    add_check(
        "label_imbalance",
        majority_fraction < 0.80,
        {"counts": {str(key): int(value) for key, value in label_counts.items()}, "majority_fraction": majority_fraction},
        severity="warning",
    )
    add_check(
        "visibility_nonnegative",
        bool((numeric_frame["visibility_km"] >= 0).all()),
        {"min": float(numeric_frame["visibility_km"].min()), "max": float(numeric_frame["visibility_km"].max())},
    )
    add_check(
        "humidity_range",
        bool(numeric_frame["relative_humidity_%"].between(0, 100).all()),
        {"min": float(numeric_frame["relative_humidity_%"].min()), "max": float(numeric_frame["relative_humidity_%"].max())},
    )

    source_metadata_path = PROJECT_ROOT / "data/processed/feature_metadata.csv"
    domain_warnings: list[str] = []
    if source_metadata_path.exists():
        source = pd.read_csv(source_metadata_path)
        if "solar" in source.columns:
            source_solar = pd.to_numeric(source["solar"], errors="coerce")
            target_solar = numeric_frame["solar_intensity_Wm2"]
            ratio = float(target_solar.median() / source_solar.median()) if source_solar.median() else float("inf")
            if ratio > 10:
                domain_warnings.append(
                    "Solar features are in incompatible numerical/physical domains: SkyFinder appears to use daily NASA POWER energy while PRL uses instantaneous W/m². Do not reuse the SkyFinder weather scaler directly."
                )
            add_check(
                "source_target_solar_scale",
                ratio <= 10,
                {
                    "source_median": float(source_solar.median()),
                    "target_median": float(target_solar.median()),
                    "median_ratio": ratio,
                },
                severity="warning",
            )

    report = {
        "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
        "embedding_path": str(embedding_path.relative_to(PROJECT_ROOT)),
        "image_directory": str(project_path(args.images).relative_to(PROJECT_ROOT)),
        "rows": len(metadata),
        "embedding_dimension": len(dino_columns),
        "dates": int(metadata["date"].nunique()),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "majority_fraction": majority_fraction,
        "numeric_summary": numeric_frame.describe().to_dict(),
        "embedding_summary": {
            "mean": float(embedding_matrix.mean()),
            "std": float(embedding_matrix.std()),
            "min": float(embedding_matrix.min()),
            "max": float(embedding_matrix.max()),
            "mean_l2_norm": float(np.linalg.norm(embedding_matrix, axis=1).mean()),
        },
        "images": image_info,
        "domain_warnings": domain_warnings,
        "checks": checks,
    }
    save_json(report, report_dir / "verification_report.json")
    pd.DataFrame(checks).to_csv(report_dir / "verification_checks.csv", index=False)

    failures = [item for item in checks if not item["passed"] and item["severity"] == "error"]
    warnings = [item for item in checks if not item["passed"] and item["severity"] == "warning"]

    print("=" * 72)
    print("PRL DATASET VERIFICATION")
    print("=" * 72)
    print(f"Metadata:   {metadata_path.relative_to(PROJECT_ROOT)} ({len(metadata)} rows)")
    print(f"Embeddings: {embedding_path.relative_to(PROJECT_ROOT)} ({len(embeddings)} rows, {len(dino_columns)} features)")
    print(f"Images:     {image_info['count']} found in {args.images}")
    print(f"Dates:      {metadata['date'].nunique()}")
    print(f"Labels:     {dict(label_counts)}")
    print(f"Errors:     {len(failures)}")
    print(f"Warnings:   {len(warnings)}")
    for item in warnings:
        print(f"WARNING - {item['check']}: {item['details']}")
    for item in failures:
        print(f"ERROR   - {item['check']}: {item['details']}")
    print(f"Report:     {report_dir.relative_to(PROJECT_ROOT) / 'verification_report.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
