# PRL Accuracy Optimization Run Guide

Copy `src/prl/optimized_estimators.py`, `src/prl/08_optimize_accuracy.py`, and
`tests/test_optimized_model.py` into the existing project.

Quick focused search:

```powershell
python src/prl/08_optimize_accuracy.py --quick
```

Full focused search:

```powershell
python src/prl/08_optimize_accuracy.py
```

The script does not overwrite the existing baseline model. It creates:

- `models/prl/optimized_accuracy_model.joblib`
- `models/prl/optimized_balanced_model.joblib`
- `results/prl/optimization_v2/candidate_metrics.csv`
- `results/prl/optimization_v2/oof_predictions.csv`
- `results/prl/optimization_v2/optimization_summary.json`

Prediction example:

```powershell
python src/prl/06_predict.py `
  --model models/prl/optimized_accuracy_model.joblib `
  --image data/prl_images/PRL_0001.jpg `
  --temperature 31.1 `
  --humidity 82 `
  --solar 610 `
  --hour 14 `
  --device cpu
```

These are development-selection results. Nested grouped validation is still required before
using the selected metrics as final paper estimates.
