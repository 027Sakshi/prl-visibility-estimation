# PRL Visibility Estimation — Results Draft

## Dataset

The local PRL dataset contained **127 observations** collected over **23 acquisition dates**. Each observation combined a 768-dimensional DINOv2 embedding with temperature, relative humidity, instantaneous solar intensity and acquisition hour.

The label distribution was strongly imbalanced: **105/127 (82.7%)** observations were labelled 10 km. The remaining labels were 4 km: 1, 5 km: 10, 9 km: 11.

## Evaluation protocol

All model-development results were generated with leave-one-acquisition-date-out cross-validation. PCA and all target-domain scalers were fitted inside each training fold. This prevents images captured on the same date from leaking into both training and testing data.

## Model comparison

| model                   |      mae |    rmse |         r2 |   macro_mae |   low_visibility_mae |   within_1km |
|:------------------------|---------:|--------:|-----------:|------------:|---------------------:|-------------:|
| transfer_weighted_ridge | 1.03683  | 1.62748 | -0.285605  |     2.69013 |              2.64368 |     0.661417 |
| fusion_pca_xgboost      | 0.892731 | 1.48665 | -0.0727343 |     2.79402 |              2.51026 |     0.874016 |
| weather_ridge           | 0.968529 | 1.52042 | -0.122031  |     2.81861 |              2.60857 |     0.889764 |
| image_pca_ridge         | 1.008    | 1.54794 | -0.16302   |     2.84001 |              2.48854 |     0.740157 |
| fusion_pca_ridge        | 1.01315  | 1.55962 | -0.180637  |     2.85008 |              2.50518 |     0.740157 |
| dummy_median            | 0.527559 | 1.52924 | -0.135089  |     3       |              3.04545 |     0.913386 |

The selected learned model was **transfer_weighted_ridge**, selected by macro MAE rather than ordinary MAE. It achieved MAE **1.037 km**, RMSE **1.627 km**, macro MAE **2.690 km**, and R² **-0.286**.

The mandatory median dummy achieved lower ordinary MAE (**0.528 km**) by predicting 10 km for every observation. This result demonstrates the severity of the label imbalance and prevents an inflated claim of predictive performance.

The selected model's group-bootstrap MAE interval was **[0.710, 1.506] km** and its macro-MAE interval was **[1.811, 3.318] km**.

## Per-label behaviour

|   visibility_km |   n |      mae |     rmse |   mean_prediction |       bias |
|----------------:|----:|---------:|---------:|------------------:|-----------:|
|               4 |   1 | 4.77958  | 4.77958  |           8.77958 |  4.77958   |
|               5 |  10 | 4.70726  | 4.73819  |           9.70726 |  4.70726   |
|               9 |  11 | 0.573519 | 0.666988 |           8.92086 | -0.0791395 |
|              10 | 105 | 0.700157 | 0.895176 |           9.48907 | -0.510926  |

## Interpretation

The experiment does not support a claim that the learned model beats a constant predictor on overall MAE. Its value is instead as a leakage-controlled domain-adaptation benchmark that gives greater attention to minority visibility states. The next accuracy gain is more likely to come from collecting diverse low-visibility observations and clarifying whether 10 km is a censored upper bound than from further hyperparameter search.

## Limitations

- Only 127 labelled samples are available from one camera and one location.
- The target is highly imbalanced: 105 of 127 observations are labelled 10 km.
- Many images are temporally adjacent; evaluation therefore groups by acquisition date.
- The 10-km label may behave as a measurement ceiling. This must be verified from the label-generation protocol before treating it as an exact continuous value.
- SkyFinder solar radiation and PRL solar intensity are not directly interchangeable; the transfer pipeline uses separate domain scalers rather than the archived global scaler.
- A negative R² can coexist with a small MAE when target variance is low and a constant predictor is strong.

## Generated figure

- `results\prl\paper\paper_model_comparison.png`

## Archived SkyFinder pretraining benchmark

| Model   |     MAE |    RMSE |       R2 |
|:--------|--------:|--------:|---------:|
| Fusion  | 1.54986 | 4.10235 | 0.604022 |
| Image   | 2.00517 | 4.77627 | 0.463237 |
| Weather | 2.071   | 4.92797 | 0.428598 |

These SkyFinder hold-out results are not directly comparable with the PRL leave-one-date-out results because the datasets, label distributions and evaluation partitions differ.
