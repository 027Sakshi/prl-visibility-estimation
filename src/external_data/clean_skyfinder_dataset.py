import os
import numpy as np
import pandas as pd

print("=" * 70)
print("CLEANING SKYFINDER DATASET")
print("=" * 70)

# =====================================================
# PATHS
# =====================================================

INPUT_PATH = "data/external/skyfinder/metadata/complete_table_with_mcr.csv"

OUTPUT_FOLDER = "data/external/skyfinder/final"

OUTPUT_PATH = os.path.join(
    OUTPUT_FOLDER,
    "skyfinder_clean.csv"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading Dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset Loaded!")
print("Original Shape :", df.shape)

# =====================================================
# REPLACE INVALID VALUES
# =====================================================

print("\nReplacing -9999 with NaN...")

df.replace(
    -9999,
    np.nan,
    inplace=True
)

# =====================================================
# KEEP IMPORTANT COLUMNS
# =====================================================

required_columns = [

    "Filename",

    "CamId",

    "Latitude",

    "Longitude",

    "Year",

    "Month",

    "Day",

    "Hour",

    "Min",

    "TempM",

    "Hum",

    "VisM",

    "PressureM",

    "WspdM",

    "Conds",

    "Fog",

    "Rain",

    "Snow",

    "daylight",

    "night"

]

df = df[required_columns]

print("\nColumns Selected!")

# =====================================================
# REMOVE MISSING VALUES
# =====================================================

print("\nRemoving Missing Values...")

before = len(df)

df = df.dropna()

after = len(df)

print("Rows Removed :", before - after)

print("Remaining Rows :", after)

# =====================================================
# REMOVE DUPLICATES
# =====================================================

print("\nRemoving Duplicate Rows...")

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Duplicates Removed :", before - after)

# =====================================================
# KEEP DAYTIME IMAGES
# =====================================================

print("\nFiltering Daytime Images...")

print("\nUnique daylight values:")

print(sorted(df["daylight"].unique())[:20])

# Keep strong daylight images
df = df[
    df["daylight"] >= 0.5
]

print("Remaining Images :", len(df))

# =====================================================
# CREATE DATE COLUMN
# =====================================================

df["date"] = (

    df["Year"].astype(int).astype(str)

    + "-"

    + df["Month"].astype(int).astype(str).str.zfill(2)

    + "-"

    + df["Day"].astype(int).astype(str).str.zfill(2)

)

# =====================================================
# CREATE TIME COLUMN
# =====================================================

df["time"] = (

    df["Hour"].astype(int).astype(str).str.zfill(2)

    + ":"

    + df["Min"].astype(int).astype(str).str.zfill(2)

)

# =====================================================
# RENAME COLUMNS
# =====================================================

df = df.rename(columns={

    "Filename": "image_name",

    "TempM": "temperature",

    "Hum": "humidity",

    "VisM": "visibility",

    "PressureM": "pressure",

    "WspdM": "wind_speed"

})

# =====================================================
# REMOVE OLD COLUMNS
# =====================================================

df = df.drop(columns=[

    "Year",

    "Month",

    "Day",

    "Hour",

    "Min"

])

# =====================================================
# SORT
# =====================================================

df = df.sort_values(

    by=[

        "CamId",

        "date",

        "time"

    ]

).reset_index(drop=True)

# =====================================================
# SAVE
# =====================================================

df.to_csv(

    OUTPUT_PATH,

    index=False

)

print("\nDataset Saved Successfully!")

print("Saved To :")

print(OUTPUT_PATH)

print("\nFinal Shape :")

print(df.shape)

print("\nFirst Five Rows\n")

print(df.head())