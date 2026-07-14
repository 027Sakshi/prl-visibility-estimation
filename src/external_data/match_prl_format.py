import os
import pandas as pd

print("="*70)
print("MATCHING SKYFINDER DATASET TO PRL FORMAT")
print("="*70)

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------

CAMERA_ID = input(
    "\nEnter Camera ID : "
).strip()

INPUT_PATH = (
    f"data/external/skyfinder/final/"
    f"camera{CAMERA_ID}_dataset.csv"
)
OUTPUT_PATH = (
    f"data/external/skyfinder/final/"
    f"camera{CAMERA_ID}_prl.csv"
)
# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

print("\nLoading Dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset Loaded!")
print(df.shape)

# ----------------------------------------------------
# CREATE DAY COLUMN
# ----------------------------------------------------

df["day"] = pd.to_datetime(df["date"]).dt.day

# ----------------------------------------------------
# CREATE HOUR COLUMN
# ----------------------------------------------------

df["hour"] = pd.to_datetime(df["time"]).dt.hour

# ----------------------------------------------------
# CREATE PLACEHOLDER COLUMNS
# ----------------------------------------------------

# Solar radiation will be added later
df["solar_intensity"] = None

# Distance is fixed placeholder
df["distance_m"] = 0

# ----------------------------------------------------
# RENAME COLUMNS
# ----------------------------------------------------

df = df.rename(columns={
    "humidity":"relative_humidity",
    "visibility":"visibility_km",
    "Latitude":"latitude",
    "Longitude":"longitude"
})

# ----------------------------------------------------
# KEEP ONLY PRL COLUMNS
# ----------------------------------------------------

df = df[
[
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
]

# ----------------------------------------------------
# SAVE
# ----------------------------------------------------

os.makedirs(
    "data/external/skyfinder/final",
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved Successfully!")

print("\nOutput:")
print(OUTPUT_PATH)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())