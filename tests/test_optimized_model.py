from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.prl.optimized_estimators import engineer_weather


class TestOptimizedModelUtilities(unittest.TestCase):
    def test_engineered_weather_is_finite(self) -> None:
        raw = np.array([[31.1, 82.0, 610.0, 14.0]], dtype=float)
        engineered = engineer_weather(raw)
        self.assertEqual(engineered.shape, (1, 12))
        self.assertTrue(np.isfinite(engineered).all())


if __name__ == "__main__":
    unittest.main()
