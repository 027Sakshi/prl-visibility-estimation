import os
import pandas as pd

print("=" * 70)
print("MERGING IMAGES WITH METADATA")
print("=" * 70)

# ======================================================
# PATHS
# ======================================================

CSV_PATH = "data/external/skyfinder/final/skyfinder_clean.csv"

CAMERA_ID = input(
    "\nEnter Camera ID : "
).strip()

IMAGE_FOLDER = f"data/external/skyfinder/images/{CAMERA_ID}"
OUTPUT_FOLDER = "data/external/skyfinder/final"

OUTPUT_FILE = f"camera{CAMERA_ID}_dataset.csv"
os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ======================================================
# LOAD CSV
# ======================================================

print("\nLoading Metadata...")

df = pd.read_csv(CSV_PATH)

print("Rows :", len(df))

# ======================================================
# KEEP CAMERA 10066 ONLY
# ======================================================

df = df[df["CamId"] == int(CAMERA_ID)]

print("Rows for Camera :", len(df))

# ======================================================
# GET IMAGE LIST
# ======================================================

print("\nScanning Image Folder...")

images = set(

    os.listdir(
        IMAGE_FOLDER
    )

)

print("Downloaded Images :", len(images))

# ======================================================
# KEEP EXISTING IMAGES ONLY
# ======================================================

df = df[
    df["image_name"].isin(images)
]

print("Matching Images :", len(df))

# ======================================================
# CREATE IMAGE PATH
# ======================================================

df["image_path"] = df["image_name"].apply(

    lambda x:

    os.path.join(
        IMAGE_FOLDER,
        x
    )

)

# ======================================================
# REORDER COLUMNS
# ======================================================

columns = [

    "image_name",

    "image_path",

    "CamId",

    "temperature",

    "humidity",

    "visibility",

    "pressure",

    "wind_speed",

    "Latitude",

    "Longitude",

    "date",

    "time",

    "Conds",

    "Fog",

    "Rain",

    "Snow"

]

df = df[
    columns
]

# ======================================================
# SAVE
# ======================================================

save_path = os.path.join(

    OUTPUT_FOLDER,

    OUTPUT_FILE

)

df.to_csv(

    save_path,

    index=False

)

print("\nDataset Saved!")

print(save_path)

print("\nFinal Shape")

print(df.shape)

print("\nFirst Five Rows")

print(df.head())