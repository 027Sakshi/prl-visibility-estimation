# PRL Visibility Project â€” Completion Run Guide

This module completes the planned PRL stages while preserving the existing SkyFinder work.

## 1. Open the correct folder in VS Code

Open the project root, the folder that contains `data/`, `src/`, `models/`, `results/`, and `run_prl_pipeline.py`.

Use Python 3.11. In the VS Code terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Put the PRL images in the expected folder

Place all 127 original images under:

```text
data/prl_images/
```

The supplied archive contains the PRL metadata and legacy embeddings, but not the image files. The best-validity run should re-extract the embeddings because the archived PRL embeddings were produced through a different DINOv2 loading/preprocessing path from the SkyFinder embeddings.

## 3. Recommended complete run

```powershell
python run_prl_pipeline.py --extract-features --force-features --require-images
```

The first DINOv2 run downloads model weights and therefore needs internet access. GPU is used automatically when available.

For a CPU test using the existing embeddings:

```powershell
python run_prl_pipeline.py
```

This test is runnable immediately, but the result must retain the embedding-backend compatibility warning.

## 4. Run stages separately

```powershell
python src/prl/01_verify_prl_dataset.py
python src/prl/02_extract_prl_features.py --backend torchhub --force
python src/prl/03_prepare_prl_dataset.py
python src/prl/04_finetune_fusion.py
python src/prl/05_evaluate_prl.py --bootstrap 1000
python src/prl/07_generate_paper_results.py
```

To omit the XGBoost benchmark during a quick run:

```powershell
python src/prl/04_finetune_fusion.py --skip-xgboost
```

## 5. Make a prediction

Using an existing embedding row:

```powershell
python src/prl/06_predict.py `
  --embedding-file data/prl/features/dinov2_vitb14_embeddings.csv `
  --image-name PRL_0001.jpg `
  --temperature 31.1 `
  --humidity 82 `
  --solar 610 `
  --hour 14
```

Using a new image:

```powershell
python src/prl/06_predict.py `
  --image path/to/new_image.jpg `
  --temperature 31.1 `
  --humidity 82 `
  --solar 610 `
  --hour 14
```

## 6. Important outputs

```text
results/prl/verification/verification_report.json
results/prl/training/oof_predictions.csv
results/prl/training/model_metrics.csv
results/prl/evaluation/evaluation_metrics.csv
results/prl/evaluation/evaluation_summary.md
results/prl/paper/paper_results_draft.md
models/prl/prl_visibility_model.joblib
```

## 7. Accuracy and research rules built into the scripts

- Images from the same acquisition date never appear in both training and test partitions.
- PCA and PRL scalers are fitted only inside training folds.
- SkyFinder and PRL solar variables are scaled separately because their units/domains are incompatible.
- The constant median predictor is always reported.
- Ordinary MAE, macro MAE, low-visibility MAE, RMSE, RÂ², bias, and tolerance accuracy are reported.
- Confidence intervals use acquisition-date bootstrap resampling.
- The selected learned model is chosen by macro MAE, not by hiding the stronger constant baseline on ordinary MAE.

## 8. Current result from the supplied legacy embeddings

The leakage-controlled run selects `transfer_weighted_ridge` by macro MAE. However, the constant 10-km median predictor has the lowest ordinary MAE because 105 of the 127 labels equal 10 km. This is a central dataset finding and must remain in the final paper.

