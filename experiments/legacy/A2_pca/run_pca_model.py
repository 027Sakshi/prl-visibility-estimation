import os
import pandas as pd

from src.models.pca_model import create_pca_model
from src.evaluation.cross_validation import run_cv

print("=" * 60)
print("A2.1 PCA MODEL")
print("=" * 60)

print()

print("Loading Dataset...")

df = pd.read_csv(
    "data/prl_pca/pca_features.csv"
)

print("Dataset Loaded!")

print(df.shape)

# ==========================================================
# SELECT PCA FEATURES
# ==========================================================

feature_columns = [
    col
    for col in df.columns
    if col.startswith("pca_")
]

X = df[feature_columns]

# Load visibility labels
metadata = pd.read_excel(
    "data/prl/metadata/prl_dataset_clean.xlsx"
)

y = metadata["visibility_km"]

print("\nPCA Feature Matrix:", X.shape)
print("Target Shape:", y.shape)

# ==========================================================
# CREATE MODEL
# ==========================================================

print("\nCreating PCA Model...")

model = create_pca_model()

print(model)

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nRunning Cross Validation...\n")

results = run_cv(
    model,
    X,
    y
)

# ==========================================================
# RESULTS
# ==========================================================

print("\n" + "=" * 60)
print("A2.1 PCA RESULTS")
print("=" * 60)

print(results)

# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df = pd.DataFrame([results])

os.makedirs(
    "results/A2_PCA",
    exist_ok=True
)

results_df.to_csv(
    "results/A2_PCA/pca_results.csv",
    index=False
)

print("\nResults Saved Successfully!")

