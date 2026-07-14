from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.prl.common import PROJECT_ROOT, feature_columns, load_bundle, project_path
from src.prl.estimators import predict_from_bundle


SOURCE_COMPATIBLE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PRL visibility for one image/weather observation.")
    parser.add_argument("--model", default="models/prl/prl_visibility_model.joblib")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a JPG/PNG image.")
    source.add_argument("--embedding-file", help="CSV containing image_name plus DINOv2 columns.")
    parser.add_argument("--image-name", help="Required with --embedding-file when it contains multiple rows.")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature in degrees Celsius.")
    parser.add_argument("--humidity", type=float, required=True, help="Relative humidity in percent.")
    parser.add_argument("--solar", type=float, required=True, help="Instantaneous PRL solar intensity in W/m².")
    parser.add_argument("--hour", type=float, required=True, help="Local hour in 24-hour format.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", default=None, help="Optional JSON output file.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_image(path: Path, device: torch.device) -> np.ndarray:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(device).eval()
    with Image.open(path) as image:
        tensor = SOURCE_COMPATIBLE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
    if isinstance(output, dict):
        output = output.get("x_norm_clstoken", output.get("last_hidden_state"))
    if output.ndim == 3:
        output = output[:, 0, :]
    return output.detach().cpu().float().numpy()


def load_embedding(path: Path, image_name: str | None) -> tuple[np.ndarray, str]:
    frame = pd.read_csv(path)
    columns = feature_columns(frame)
    if not columns:
        raise ValueError(f"No DINOv2 columns found in {path}")
    if image_name:
        if "image_name" not in frame:
            raise ValueError("--image-name was supplied but embedding CSV has no image_name column")
        match = frame[frame["image_name"].astype(str).str.lower() == image_name.lower()]
        if len(match) != 1:
            raise ValueError(f"Expected exactly one row for {image_name!r}; found {len(match)}")
        row = match.iloc[[0]]
        resolved_name = str(match.iloc[0]["image_name"])
    else:
        if len(frame) != 1:
            raise ValueError("Embedding CSV has multiple rows; pass --image-name")
        row = frame.iloc[[0]]
        resolved_name = str(row.iloc[0].get("image_name", path.name))
    return row[columns].to_numpy(dtype=float), resolved_name


def main() -> int:
    args = parse_args()
    bundle = load_bundle(args.model)

    if args.image:
        image_path = project_path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        device = choose_device(args.device)
        embedding = extract_image(image_path, device)
        source_name = image_path.name
        embedding_source = "image_extraction_torchhub_dinov2_vitb14"
    else:
        embedding_path = project_path(args.embedding_file)
        embedding, source_name = load_embedding(embedding_path, args.image_name)
        embedding_source = str(embedding_path.relative_to(PROJECT_ROOT))

    if embedding.shape[1] != 768:
        raise ValueError(f"Expected a 768-dimensional embedding, found {embedding.shape}")
    weather = np.array([[args.temperature, args.humidity, args.solar, args.hour]], dtype=float)
    prediction = float(predict_from_bundle(bundle, embedding, weather)[0])

    payload = {
        "image_name": source_name,
        "predicted_visibility_km": prediction,
        "model_name": bundle.get("model_name"),
        "model_type": bundle.get("model_type"),
        "embedding_source": embedding_source,
        "inputs": {
            "temperature_C": args.temperature,
            "relative_humidity_%": args.humidity,
            "solar_intensity_Wm2": args.solar,
            "hour": args.hour,
        },
        "warning": "This model is a research prototype trained on a small, imbalanced, single-location dataset. Do not use it for safety-critical decisions.",
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        output_path = project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
