# PRL Visibility Estimation

Research pipeline for estimating atmospheric visibility from sky images and meteorological variables. The project uses frozen DINOv2 image embeddings, weather features, transfer learning from SkyFinder, and leakage-controlled regression benchmarks on the PRL dataset.

## Current status

The complete PRL pipeline is operational:

1. Dataset verification
2. DINOv2 ViT-B/14 feature extraction
3. PRL fusion-dataset preparation
4. Leave-one-acquisition-date-out model development
5. Evaluation and bootstrap uncertainty analysis
6. Single-image inference
7. Paper tables and figures

All integration tests currently pass.

## Current PRL benchmark

The evaluation uses 127 images collected over 23 acquisition dates. The target distribution is strongly imbalanced: 105 observations have visibility exactly equal to 10 km.

| Model | MAE (km) | RMSE (km) | R² | Macro MAE (km) |
|---|---:|---:|---:|---:|
| Transfer-weighted Ridge | 1.037 | 1.627 | -0.286 | **2.690** |
| Fusion PCA XGBoost | **0.893** | **1.487** | **-0.073** | 2.794 |
| Weather Ridge | 0.969 | 1.520 | -0.122 | 2.819 |
| Image PCA Ridge | 1.008 | 1.548 | -0.163 | 2.840 |
| Fusion PCA Ridge | 1.013 | 1.560 | -0.181 | 2.850 |
| Median baseline | 0.528 | 1.529 | -0.135 | 3.000 |

The constant baseline has the lowest ordinary MAE because of the target imbalance. This limitation is reported explicitly and is not treated as evidence of useful predictive skill.

## Repository layout

```text
configs/                 Pipeline configuration
src/prl/                 Current PRL pipeline
src/                     SkyFinder preparation and baseline modules
tests/                   Integration tests
experiments/legacy/      Earlier exploratory experiment entry points
docs/                    Run guide, status notes, and archived documents
data/README.md            Local data instructions; datasets are not committed
models/README.md          Model-generation instructions; binaries are not committed
results/                  Aggregate metrics, reports, and paper figures
run_prl_pipeline.py       End-to-end PRL entry point
requirements-prl.txt      Direct project dependencies
```

## Environment

Python 3.11 is recommended.

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-prl.txt
```

## Data setup

The repository intentionally excludes raw images, embeddings, processed matrices, and model binaries. See [`data/README.md`](data/README.md) for the expected local paths.

## Run the pipeline

For a full run that extracts fresh DINOv2 features from the original PRL images:

```powershell
python run_prl_pipeline.py --extract-features --force-features --require-images
```

To reuse local embeddings that have already been generated:

```powershell
python run_prl_pipeline.py
```

## Run the tests

The tests expect the local dataset and generated model/results artifacts to exist.

```powershell
python -m unittest tests.test_prl_pipeline -v
```

## Single-image inference

```powershell
python src/prl/06_predict.py `
  --image data/prl_images/PRL_0001.jpg `
  --temperature 31.1 `
  --humidity 82 `
  --solar 610 `
  --hour 14 `
  --device cpu `
  --output results/prl/predictions/PRL_0001_prediction.json
```

## Research limitations

- Only 127 PRL images and 23 acquisition dates are currently available.
- The labels are highly imbalanced toward 10 km.
- The 4 km class contains only one observation.
- Current learned models do not beat the median baseline on ordinary MAE.
- The saved model is a research prototype and must not be used for safety-critical decisions.

## Next research stage

The next locked experiment cycle will add nested grouped model selection, physically engineered weather features, PLS regression, horizon-region embeddings, and a secondary reduced-visibility classification benchmark.
