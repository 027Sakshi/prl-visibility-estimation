import os
import pandas as pd

print("=" * 70)
print("CREATING UNIQUE SOLAR REQUESTS")
print("=" * 70)

# =====================================================
# INPUT
# =====================================================

CAMERA_ID = input(
    "\nEnter Camera ID : "
).strip()

INPUT_PATH = (
    f"data/external/skyfinder/final/"
    f"camera{CAMERA_ID}_prl.csv"
)
OUTPUT_FOLDER = "data/external/skyfinder/final"

OUTPUT_FILE = "solar_requests_{CAMERA_ID}.csv"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset Loaded!")

print("Rows :", len(df))

# =====================================================
# KEEP REQUIRED COLUMNS
# =====================================================

solar_df = df[
    [
        "date",
        "latitude",
        "longitude"
    ]
]

# =====================================================
# REMOVE DUPLICATES
# =====================================================

solar_df = solar_df.drop_duplicates()

solar_df = solar_df.sort_values(
    by="date"
)

solar_df = solar_df.reset_index(
    drop=True
)

print("\nUnique Requests :", len(solar_df))

# =====================================================
# SAVE
# =====================================================

save_path = os.path.join(
    OUTPUT_FOLDER,
    OUTPUT_FILE
)

solar_df.to_csv(
    save_path,
    index=False
)

print("\nSaved Successfully!")

print(save_path)

print("\nFirst Five Requests\n")

print(solar_df.head())

print("\nDone!")