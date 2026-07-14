# Final Locked PRL Results Draft

## Evaluation protocol

Final performance was estimated with nested acquisition-date-grouped validation over 127 images from 23 acquisition dates. The outer loop left one complete date out. Within each outer-training set, GroupKFold selected among the predeclared PCA-SVR configurations. Image scaling, PCA, engineered-weather scaling when applicable, and SVR fitting were repeated inside each training split. Confidence intervals were obtained by resampling acquisition dates as clusters.

## Primary sample-weighted result

The best learned model by nested outer-fold MAE was **nested_svr_mae**, with MAE 0.651 km, RMSE 1.454 km, and R² -0.026. The constant 10 km reference achieved MAE 0.528 km and RMSE 1.529 km. The paired date-cluster bootstrap difference was 0.124 km (95% CI 0.051 to 0.191); the interval excluded zero.

## Reduced-visibility and regime-balanced result

The best learned model by balanced-regime MAE was **nested_svr_balanced**, with balanced-regime MAE 1.426 km, low-visibility MAE 2.491 km, and 10 km-label MAE 0.362 km. The constant reference had balanced-regime MAE 1.523 km and low-visibility MAE 3.045 km. The paired date-cluster bootstrap difference was -0.096 km (95% CI -0.170 to -0.012); the interval excluded zero.

## Additional result

The lowest learned-model RMSE was obtained by **nested_svr_balanced** at 1.422 km. Date-macro MAE, which weights each acquisition date equally, was 0.872 km for the best-MAE learned model and 0.768 km for the constant reference.

## Interpretation constraint

The dataset is small, single-location, and strongly concentrated at exactly 10 km visibility. Claims should therefore distinguish sample-weighted overall error from performance on reduced-visibility observations. The nested outer-fold predictions and date-cluster intervals in this directory are the only values intended for the abstract, final results, discussion, and conclusion. Development-search values from `results/prl/optimization_v2` must not be presented as final test performance.
