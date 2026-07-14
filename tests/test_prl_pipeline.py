from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prl.common import WEATHER_COLUMNS, load_bundle, load_prepared_prl, load_prl_embeddings, load_prl_metadata
from src.prl.estimators import predict_from_bundle


class TestPRLPipeline(unittest.TestCase):
    def test_metadata_and_embeddings_align_in_size(self) -> None:
        metadata, _ = load_prl_metadata()
        embeddings, _, columns = load_prl_embeddings()
        self.assertEqual(len(metadata), 127)
        self.assertEqual(len(embeddings), 127)
        self.assertEqual(len(columns), 768)

    def test_prepared_dataset_shapes(self) -> None:
        frame, image_columns = load_prepared_prl()
        self.assertEqual(frame.shape[0], 127)
        self.assertEqual(len(image_columns), 768)
        self.assertEqual(frame["date"].nunique(), 23)
        self.assertTrue(set(WEATHER_COLUMNS).issubset(frame.columns))
        self.assertFalse(frame[[*WEATHER_COLUMNS, "visibility_km", *image_columns]].isna().any().any())

    def test_selected_bundle_predicts_finite_value(self) -> None:
        frame, image_columns = load_prepared_prl()
        bundle = load_bundle("models/prl/prl_visibility_model.joblib")
        image = frame.loc[[0], image_columns].to_numpy(dtype=float)
        weather = frame.loc[[0], list(WEATHER_COLUMNS)].to_numpy(dtype=float)
        prediction = predict_from_bundle(bundle, image, weather)
        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction[0]), 0.0)
        self.assertLessEqual(float(prediction[0]), 20.0)

    def test_oof_predictions_cover_every_sample(self) -> None:
        predictions = pd.read_csv(ROOT / "results/prl/training/oof_predictions.csv")
        self.assertEqual(len(predictions), 127)
        model_columns = [c for c in predictions.columns if c not in {"image_name", "date", "time", "actual_visibility_km"}]
        self.assertGreaterEqual(len(model_columns), 5)
        self.assertFalse(predictions[model_columns].isna().any().any())


if __name__ == "__main__":
    unittest.main()
