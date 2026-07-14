import os
import pandas as pd

print("=" * 60)
print("MERGING WEATHER + PCA FEATURES")
print("=" * 60)

# -------------------------
# Load Metadata
# -------------------------

metadata = pd.read_excel(
    "data/prl/metadata/prl_dataset_clean.xlsx"
)

print("\nMetadata Loaded!")
print(metadata.shape)

# -------------------------
# Load PCA Features
# -------------------------

pca = pd.read_csv(
    "data/prl_pca/pca_features.csv"
)

print("\nPCA Features Loaded!")
print(pca.shape)

# -------------------------
# Merge
# -------------------------

merged = pd.merge(

    metadata,

    pca,

    on="image_name",

    how="inner"

)

print("\nMerged Successfully!")

print(merged.shape)

# -------------------------
# Save
# -------------------------

os.makedirs(
    "data/merged_pca",
    exist_ok=True
)

merged.to_csv(

    "data/merged_pca/merged_pca_dataset.csv",

    index=False

)

print("\nDataset Saved Successfully!")