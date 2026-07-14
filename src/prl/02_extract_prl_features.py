from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.prl.common import PROJECT_ROOT, project_path, save_json


SOURCE_COMPATIBLE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageFolderDataset(Dataset):
    def __init__(self, files: list[Path], transform: Any) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "image_name": path.name}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 768-D DINOv2 embeddings for PRL images.")
    parser.add_argument("--images", default="data/prl_images")
    parser.add_argument("--output", default="data/prl/features/dinov2_vitb14_embeddings.csv")
    parser.add_argument("--output-npy", default="data/prl/features/dinov2_vitb14_embeddings.npy")
    parser.add_argument("--metadata-output", default="data/prl/features/dinov2_vitb14_extraction.json")
    parser.add_argument("--backend", choices=("torchhub", "huggingface"), default="torchhub")
    parser.add_argument("--model", default=None, help="Override model identifier.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_files(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def file_manifest_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def extract_torchhub(files: list[Path], device: torch.device, batch_size: int, workers: int, model_name: str) -> tuple[np.ndarray, list[str]]:
    dataset = ImageFolderDataset(files, SOURCE_COMPATIBLE_TRANSFORM)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    names: list[str] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="DINOv2", unit="batch"):
            output = model(batch["image"].to(device, non_blocking=True))
            if isinstance(output, dict):
                output = output.get("x_norm_clstoken", output.get("last_hidden_state"))
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"Unexpected DINOv2 output type: {type(output).__name__}")
            if output.ndim == 3:
                output = output[:, 0, :]
            chunks.append(output.detach().cpu().float().numpy())
            names.extend(batch["image_name"])
    return np.concatenate(chunks, axis=0), names


def extract_huggingface(files: list[Path], device: torch.device, batch_size: int, model_name: str) -> tuple[np.ndarray, list[str]]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    chunks: list[np.ndarray] = []
    names: list[str] = []
    for start in tqdm(range(0, len(files), batch_size), desc="DINOv2", unit="batch"):
        batch_files = files[start : start + batch_size]
        images = []
        for path in batch_files:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state[:, 0, :]
        chunks.append(output.detach().cpu().float().numpy())
        names.extend(path.name for path in batch_files)
    return np.concatenate(chunks, axis=0), names


def main() -> int:
    args = parse_args()
    image_dir = project_path(args.images)
    output_csv = project_path(args.output)
    output_npy = project_path(args.output_npy)
    metadata_output = project_path(args.metadata_output)

    if output_csv.exists() and not args.force:
        print(f"Embedding file already exists: {output_csv.relative_to(PROJECT_ROOT)}")
        print("Use --force to re-extract. For transfer validity, re-extraction with --backend torchhub is recommended.")
        return 0
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    files = image_files(image_dir)
    if not files:
        raise FileNotFoundError(f"No supported images found under: {image_dir}")

    device = choose_device(args.device)
    batch_size = args.batch_size or (64 if device.type == "cuda" else 8)
    if args.backend == "torchhub":
        model_name = args.model or "dinov2_vitb14"
        features, names = extract_torchhub(files, device, batch_size, args.workers, model_name)
        preprocessing = "Resize((224,224)) + ImageNet normalization; matches the archived SkyFinder extraction script"
    else:
        model_name = args.model or "facebook/dinov2-base"
        features, names = extract_huggingface(files, device, batch_size, model_name)
        preprocessing = "Hugging Face AutoImageProcessor"

    if features.ndim != 2 or features.shape[0] != len(files):
        raise RuntimeError(f"Invalid extracted shape: {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("Extracted embeddings contain NaN or infinity")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    features = features.astype(np.float32, copy=False)
    np.save(output_npy, features)
    columns = [f"dino_{index:03d}" for index in range(1, features.shape[1] + 1)]
    frame = pd.DataFrame(features, columns=columns)
    frame.insert(0, "image_name", names)
    frame.to_csv(output_csv, index=False)

    metadata = {
        "backend": args.backend,
        "model": model_name,
        "preprocessing": preprocessing,
        "device": str(device),
        "batch_size": batch_size,
        "image_directory": str(image_dir.relative_to(PROJECT_ROOT)),
        "image_count": len(files),
        "embedding_dimension": int(features.shape[1]),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "manifest_sha256": file_manifest_hash(files),
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "mean_l2_norm": float(np.linalg.norm(features, axis=1).mean()),
        "output_csv": str(output_csv.relative_to(PROJECT_ROOT)),
        "output_npy": str(output_npy.relative_to(PROJECT_ROOT)),
    }
    save_json(metadata, metadata_output)

    print("=" * 72)
    print("PRL DINOv2 FEATURE EXTRACTION COMPLETE")
    print("=" * 72)
    print(f"Images:     {len(files)}")
    print(f"Model:      {model_name} ({args.backend})")
    print(f"Device:     {device}")
    print(f"Shape:      {features.shape}")
    print(f"CSV:        {output_csv.relative_to(PROJECT_ROOT)}")
    print(f"NumPy:      {output_npy.relative_to(PROJECT_ROOT)}")
    print(f"Metadata:   {metadata_output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
