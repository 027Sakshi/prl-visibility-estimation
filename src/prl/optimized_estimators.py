from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge


def engineer_weather(weather: np.ndarray) -> np.ndarray:
    """Create inference-safe physical/time features from [T, RH, solar, hour]."""
    raw = np.asarray(weather, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(f"Expected weather matrix with at least four columns, found {raw.shape}")
    temperature = raw[:, 0]
    humidity = np.clip(raw[:, 1], 1e-6, 100.0)
    solar = raw[:, 2]
    hour = raw[:, 3]

    a = 17.625
    b = 243.04
    gamma = np.log(humidity / 100.0) + (a * temperature) / (b + temperature)
    dew_point = (b * gamma) / (a - gamma)
    vapour_pressure = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
    vpd = vapour_pressure * (1.0 - humidity / 100.0)

    return np.column_stack(
        [
            temperature,
            humidity,
            solar,
            hour,
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            dew_point,
            temperature - dew_point,
            vpd,
            temperature * humidity / 100.0,
            solar * humidity / 100.0,
            solar * (1.0 - humidity / 100.0),
        ]
    )


@dataclass
class OptimizedSVRBundle:
    model_type: str
    model: SVR
    feature_mode: str
    image_scaler: StandardScaler | None
    pca: PCA | None
    weather_scaler: StandardScaler | None
    image_columns: Sequence[str]
    raw_weather_columns: Sequence[str]
    pca_components: int
    C: float
    epsilon: float
    prediction_min: float = 4.0
    prediction_max: float = 10.0

    def transform(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.feature_mode in {"image", "fusion"}:
            if self.image_scaler is None or self.pca is None:
                raise RuntimeError("Image preprocessing is missing")
            blocks.append(self.pca.transform(self.image_scaler.transform(np.asarray(image, dtype=float))))
        if self.feature_mode in {"weather", "fusion"}:
            if self.weather_scaler is None:
                raise RuntimeError("Weather preprocessing is missing")
            blocks.append(self.weather_scaler.transform(engineer_weather(weather)))
        if not blocks:
            raise RuntimeError(f"Unsupported feature mode: {self.feature_mode}")
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def predict(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(self.transform(image, weather))
        return np.clip(np.asarray(prediction, dtype=float), self.prediction_min, self.prediction_max)


@dataclass
class OptimizedRidgeBundle:
    model_type: str
    model: Ridge
    feature_mode: str
    image_scaler: StandardScaler | None
    pca: PCA | None
    weather_scaler: StandardScaler | None
    image_columns: Sequence[str]
    raw_weather_columns: Sequence[str]
    pca_components: int
    alpha: float
    weight_beta: float
    prediction_min: float = 4.0
    prediction_max: float = 10.0

    def transform(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.feature_mode in {"image", "fusion"}:
            if self.image_scaler is None or self.pca is None:
                raise RuntimeError("Image preprocessing is missing")
            blocks.append(self.pca.transform(self.image_scaler.transform(np.asarray(image, dtype=float))))
        if self.feature_mode in {"weather", "fusion"}:
            if self.weather_scaler is None:
                raise RuntimeError("Weather preprocessing is missing")
            blocks.append(self.weather_scaler.transform(engineer_weather(weather)))
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def predict(self, image: np.ndarray, weather: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(self.transform(image, weather))
        return np.clip(np.asarray(prediction, dtype=float), self.prediction_min, self.prediction_max)
