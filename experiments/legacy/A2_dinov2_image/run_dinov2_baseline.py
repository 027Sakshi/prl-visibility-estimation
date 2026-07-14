import os
import pandas as pd

from src.models.xgboost_model import create_xgb_model
from src.evaluation.cross_validation import run_cv

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("A2 - DINOv2 IMAGE BASELINE")
print("=" * 60)

print("\nLoading Merged Dataset...")

df = pd.read_csv(
    "data/merged/merged_dataset.csv"
)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# ==========================================================
# SELECT DINO FEATURES
# ==========================================================

feature_columns = [
    col
    for col in df.columns
    if col.startswith("dino_")
]

X = df[feature_columns]

y = df["visibility_km"]

print("\nImage Feature Matrix:", X.shape)
print("Target Shape:", y.shape)

# ==========================================================
# CREATE MODEL
# ==========================================================

print("\nCreating XGBoost Model...")

model = create_xgb_model()

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

# ==========================================================
# RESULTS
# ==========================================================

print("\n" + "=" * 60)
print("A2 RESULTS")
print("=" * 60)

print(results)

# ==========================================================
# SAVE RESULTS
# ==========================================================

os.makedirs(
    "results/A2",
    exist_ok=True
)

results_df = pd.DataFrame([results])

results_df.to_csv(
    "results/A2/image_results.csv",
    index=False
)

print("\nResults saved successfully!")