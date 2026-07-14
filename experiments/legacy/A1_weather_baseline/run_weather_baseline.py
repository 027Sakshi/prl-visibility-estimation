from src.evaluation.cross_validation import run_cv
import pandas as pd

from src.models.weather_model import create_weather_model

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("A1 - WEATHER BASELINE")
print("=" * 60)

print("\nLoading Dataset...")

df = pd.read_excel(
    "data/prl/metadata/prl_dataset_clean.xlsx"
)

print("Dataset Loaded Successfully!")

# ==========================================================
# SELECT FEATURES
# ==========================================================

X = df[
    [
        "temperature_C",
        "relative_humidity_%",
        "solar_intensity_Wm2",
        "hour"
    ]
]

y = df["visibility_km"]

print("\nFeature Matrix Shape:", X.shape)

print("Target Shape:", y.shape)

# ==========================================================
# CREATE MODEL
# ==========================================================

print("\nCreating Weather Model...")

model = create_weather_model()

print(model)

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nStarting 5-Fold Cross Validation...\n")

results = run_cv(
    model,
    X,
    y
)

print("\n" + "=" * 60)
print("A1 RESULTS")
print("=" * 60)

print(results)
import os

results_df = pd.DataFrame([results])

os.makedirs(
    "results/A1",
    exist_ok=True
)

results_df.to_csv(
    "results/A1/weather_results.csv",
    index=False
)

print("\nResults saved successfully!")