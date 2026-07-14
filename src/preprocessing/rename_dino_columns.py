import os
import pandas as pd

# ======================================================
# PATHS
# ======================================================

INPUT_FILE = "data/merged/merged_dataset.csv"

OUTPUT_FILE = "data/merged/merged_dataset.csv"

# ======================================================

print("=" * 60)
print("RENAMING DINO FEATURES")
print("=" * 60)

# ======================================================

print("\nLoading Dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset Loaded!")

print("Shape:", df.shape)

# ======================================================
# FIND FEATURE COLUMNS
# ======================================================

feature_columns = [

    col

    for col in df.columns

    if col.startswith("feature_")

]

print(f"\nFound {len(feature_columns)} DINO Features")

# ======================================================
# CREATE NEW NAMES
# ======================================================

rename_dict = {}

for i, col in enumerate(feature_columns, start=1):

    rename_dict[col] = f"dino_{i:03d}"

# ======================================================
# RENAME
# ======================================================

df.rename(

    columns=rename_dict,

    inplace=True

)

# ======================================================
# SAVE
# ======================================================

df.to_csv(

    OUTPUT_FILE,

    index=False

)

print("\nRenaming Complete!")

print("\nFirst 15 Columns:\n")

print(df.columns[:15])

print("\nLast 10 Columns:\n")

print(df.columns[-10:])