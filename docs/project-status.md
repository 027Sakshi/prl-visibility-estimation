# PRL Visibility Project — Completion Report

## Implemented stages

1. `src/prl/01_verify_prl_dataset.py`
2. `src/prl/02_extract_prl_features.py`
3. `src/prl/03_prepare_prl_dataset.py`
4. `src/prl/04_finetune_fusion.py`
5. `src/prl/05_evaluate_prl.py`
6. `src/prl/06_predict.py`
7. `src/prl/07_generate_paper_results.py`
8. `run_prl_pipeline.py`

## Validation completed

- 127 metadata rows and 127 embedding rows
- 768 DINOv2 features per image
- 23 acquisition dates
- No missing numeric values
- No duplicate image identifiers
- Four automated integration tests pass
- Prediction CLI returns a finite prediction from the selected model bundle

## Corrected methodological problems

- Replaced random row splitting with leave-one-acquisition-date-out evaluation.
- Fit PCA and target scalers inside each training fold.
- Avoided applying the SkyFinder solar scaler directly to PRL solar values.
- Added a constant median baseline to expose label-imbalance effects.
- Added ordinary MAE, macro MAE, low-visibility MAE, RMSE, R², bias, tolerance accuracy, per-label metrics and date-bootstrap confidence intervals.
- Preserved XGBoost as a benchmark while adding lower-variance regularized models for the 127-sample setting.
- Added weighted source-to-target adaptation using SkyFinder DINOv2/PCA features and separately scaled source/target weather domains.

## Current leakage-controlled result

The selected learned model is `transfer_weighted_ridge`, chosen by macro MAE:

- MAE: 1.033 km
- RMSE: 1.640 km
- R²: -0.306
- Macro MAE: 2.685 km
- Low-visibility MAE: 2.672 km

The XGBoost PCA fusion benchmark achieved:

- MAE: 0.875 km
- RMSE: 1.476 km
- R²: -0.057
- Macro MAE: 2.822 km

The constant median baseline achieved MAE 0.528 km because 105 of 127 labels are exactly 10 km. It has no learned predictive skill, but its lower ordinary MAE must be reported.

## Required rerun for strongest validity

The archive contains legacy PRL embeddings but not the original PRL images. The legacy embeddings were generated through a different DINOv2 loading/preprocessing route from the SkyFinder embeddings. Put the 127 images in `data/prl_images/` and run:

```powershell
python run_prl_pipeline.py --extract-features --force-features --require-images
```

This re-extracts PRL features with the same `dinov2_vitb14` torch-hub model and preprocessing used for the archived SkyFinder features.

## Highest-impact data improvement

Collect substantially more 4–9 km observations. With 82.7% of labels at 10 km and only one 4-km observation, further hyperparameter tuning cannot establish robust low-visibility accuracy. Also confirm whether 10 km is an exact measurement or an upper-censored label such as “10 km or more.”
