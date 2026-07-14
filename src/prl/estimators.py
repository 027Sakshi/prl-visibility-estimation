from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class LocalRidgeBundle:
    model_type: str
    model: Ridge
    image_scaler: StandardScaler | None = None
    pca: PCA | None = None
    weather_scaler: StandardScaler | None = None
    feature_mode: str = "fusion"
    image_columns: Sequence[str] = ()
    weather_columns: Sequence[str] = ()
    prediction_min: float = 0.0
    prediction_max: float = 20.0

    def transform(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.feature_mode in {"image", "fusion"}:
            if self.image_scaler is None or self.pca is None:
                raise RuntimeError("Image preprocessing is missing")
            blocks.append(self.pca.transform(self.image_scaler.transform(image)))
        if self.feature_mode in {"weather", "fusion"}:
            if self.weather_scaler is None:
                raise RuntimeError("Weather preprocessing is missing")
            blocks.append(self.weather_scaler.transform(weather))
        return np.hstack(blocks)

    def predict(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        values = self.model.predict(self.transform(image, weather))
        return np.clip(values, self.prediction_min, self.prediction_max)


@dataclass
class TransferRidgeBundle:
    model_type: str
    model: Ridge
    source_pca: PCA
    source_pc_scaler: StandardScaler
    target_weather_scaler: StandardScaler
    image_columns: Sequence[str]
    weather_columns: Sequence[str]
    target_weight_factor: float
    alpha: float
    prediction_min: float = 0.0
    prediction_max: float = 20.0

    def transform_target(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        z_image = self.source_pc_scaler.transform(self.source_pca.transform(image))
        weather_width = len(self.weather_columns)
        if weather.shape[1] < weather_width:
            raise ValueError(f"Expected at least {weather_width} weather columns, found {weather.shape[1]}")
        z_weather = self.target_weather_scaler.transform(weather[:, :weather_width])
        domain = np.ones((len(image), 1), dtype=float)
        return np.hstack([z_image, z_weather, domain])

    def predict(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        values = self.model.predict(self.transform_target(image, weather))
        return np.clip(values, self.prediction_min, self.prediction_max)


def predict_from_bundle(bundle: dict[str, Any] | LocalRidgeBundle | TransferRidgeBundle, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
    if isinstance(bundle, (LocalRidgeBundle, TransferRidgeBundle)):
        return bundle.predict(image, weather)
    payload = bundle.get("estimator") if isinstance(bundle, dict) else None
    if isinstance(payload, (LocalRidgeBundle, TransferRidgeBundle)):
        return payload.predict(image, weather)
    raise TypeError(f"Unsupported model bundle: {type(bundle).__name__}")
