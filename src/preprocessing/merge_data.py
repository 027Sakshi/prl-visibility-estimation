import os
import pandas as pd

# ======================================================
# PATHS
# ======================================================

METADATA_FILE = "data/prl/metadata/prl_dataset_clean.xlsx"

FEATURE_FILE = "data/prl_features/all_embeddings.csv"

OUTPUT_FILE = "data/merged/merged_dataset.csv"

# ======================================================

print("=" * 60)
print("MERGING DATASETS")
print("=" * 60)

# ======================================================

print("\nLoading Metadata...")

metadata = pd.read_excel(METADATA_FILE)

print("Rows :", len(metadata))

print("\nLoading DINO Features...")

features = pd.read_csv(FEATURE_FILE)

print("Rows :", len(features))

# ======================================================

print("\nMerging...")

merged = pd.merge(

    metadata,

    features,

    on="image_name",

    how="inner"

)

print("\nMerge Complete!")

print("Rows :", len(merged))

print("Columns :", len(merged.columns))

# ======================================================

os.makedirs(

    "data/merged",

    exist_ok=True

)

merged.to_csv(

    OUTPUT_FILE,

    index=False

)

print("\nSaved Successfully!")

print(OUTPUT_FILE)