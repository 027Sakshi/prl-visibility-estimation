from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.prl.final_evaluation import (
    balanced_regime_mae,
    final_metrics,
    locked_svr_grid,
)


class TestFinalEvaluationUtilities(unittest.TestCase):
    def test_locked_grid_is_small_and_unique(self) -> None:
        grid = locked_svr_grid()
        self.assertEqual(len(grid), 20)
        self.assertEqual(len({candidate.name for candidate in grid}), 20)

    def test_final_metrics_include_regime_and_date_metrics(self) -> None:
        true = np.array([10.0, 10.0, 5.0, 9.0])
        pred = np.array([9.8, 9.9, 6.0, 8.5])
        groups = np.array(["a", "a", "b", "c"])
        metrics = final_metrics(true, pred, groups)
        self.assertTrue(np.isfinite(metrics["balanced_regime_mae"]))
        self.assertTrue(np.isfinite(metrics["date_macro_mae"]))
        self.assertAlmostEqual(
            metrics["balanced_regime_mae"], balanced_regime_mae(true, pred)
        )


if __name__ == "__main__":
    unittest.main()
