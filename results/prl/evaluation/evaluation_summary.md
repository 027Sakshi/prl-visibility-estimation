# PRL Evaluation Summary

**Selected learned model:** `transfer_weighted_ridge`

## Protocol

Leave-one-acquisition-date-out cross-validation was used so that near-duplicate images from the same day never appear in both training and test partitions.

## Main metrics

| model                   |      mae |    rmse |         r2 |   macro_mae |   low_visibility_mae |   within_1km |
|:------------------------|---------:|--------:|-----------:|------------:|---------------------:|-------------:|
| transfer_weighted_ridge | 1.03683  | 1.62748 | -0.285605  |     2.69013 |              2.64368 |     0.661417 |
| fusion_pca_xgboost      | 0.892731 | 1.48665 | -0.0727343 |     2.79402 |              2.51026 |     0.874016 |
| weather_ridge           | 0.968529 | 1.52042 | -0.122031  |     2.81861 |              2.60857 |     0.889764 |
| image_pca_ridge         | 1.008    | 1.54794 | -0.16302   |     2.84001 |              2.48854 |     0.740157 |
| fusion_pca_ridge        | 1.01315  | 1.55962 | -0.180637  |     2.85008 |              2.50518 |     0.740157 |
| dummy_median            | 0.527559 | 1.52924 | -0.135089  |     3       |              3.04545 |     0.913386 |

## Critical interpretation

The median dummy predicts 10 km for every sample and can outperform learned models on ordinary MAE because 105 of 127 labels equal 10 km. This is a dataset limitation, not evidence that a constant model is scientifically useful.

The selected learned model is chosen by macro MAE, which weights each observed visibility level equally. All conclusions should include uncertainty intervals and the dummy comparison.