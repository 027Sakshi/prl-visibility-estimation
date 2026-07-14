import os
import pandas as pd

print("=" * 70)
print("MERGING SOLAR RADIATION")
print("=" * 70)

# ==========================================================
# PATHS
# ==========================================================

DATASET_PATH = "data/external/skyfinder/final/camera{CAMERA_ID}_prl.csv"

SOLAR_PATH = "data/external/skyfinder/final/solar_data_{CAMERA_ID}.csv"

OUTPUT_FOLDER = "data/external/skyfinder/final"

OUTPUT_FILE = "camera{CAMERA_ID}_final.csv"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading SkyFinder Dataset...")

dataset = pd.read_csv(DATASET_PATH)

print("Loaded!")

print("Shape :", dataset.shape)

print("\nLoading Solar Dataset...")

solar = pd.read_csv(SOLAR_PATH)

print("Loaded!")

print("Shape :", solar.shape)

# ==========================================================
# REMOVE PLACEHOLDER COLUMN IF EXISTS
# ==========================================================

if "solar_intensity" in dataset.columns:

    print("\nRemoving Empty Placeholder Column...")

    dataset = dataset.drop(columns=["solar_intensity"])

# ==========================================================
# MERGE
# ==========================================================

print("\nMerging Solar Radiation...")

merged = pd.merge(

    dataset,

    solar,

    on="date",

    how="left"

)

print("Merged Successfully!")

print("Merged Shape :", merged.shape)

# ==========================================================
# REORDER COLUMNS
# ==========================================================

column_order = [

    "image_name",

    "date",

    "day",

    "time",

    "hour",

    "temperature",

    "relative_humidity",

    "solar_intensity",

    "visibility_km",

    "latitude",

    "longitude",

    "distance_m"

]

merged = merged[column_order]

# ==========================================================
# SAVE
# ==========================================================

save_path = os.path.join(

    OUTPUT_FOLDER,

    OUTPUT_FILE

)

merged.to_csv(

    save_path,

    index=False

)

print("\n" + "=" * 70)
print("FINAL DATASET CREATED")
print("=" * 70)

print("\nSaved To :")

print(save_path)

print("\nFinal Shape :")

print(merged.shape)

print("\nColumns :")

for col in merged.columns:

    print(col)

print("\nMissing Solar Values :")

print(

    merged["solar_intensity"].isna().sum()

)

print("\nFirst Five Rows\n")

print(

    merged.head()

)

print("\nDone!")