"""
==============================================================
VERIFY MASTER DATASET
PRL Visibility Estimation Project
==============================================================
"""

import os
import pandas as pd

# ==============================================================
# PATHS
# ==============================================================

DATASET_PATH = "data/external/skyfinder/final/skyfinder_master_dataset.csv"

IMAGE_FOLDER = "data/external/skyfinder/images"

REPORT_FOLDER = "results/external"

REPORT_FILE = os.path.join(
    REPORT_FOLDER,
    "master_dataset_report.txt"
)

# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("VERIFYING SKYFINDER MASTER DATASET")
print("=" * 70)

# ==============================================================
# LOAD DATASET
# ==============================================================

print("\nLoading Dataset...")

df = pd.read_csv(DATASET_PATH)

print("Done!")

print("\nDataset Shape")

print(df.shape)

# ==============================================================
# BASIC INFORMATION
# ==============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nColumns\n")

for col in df.columns:
    print(col)

# ==============================================================
# MISSING VALUES
# ==============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

# ==============================================================
# DUPLICATES
# ==============================================================

print("\n" + "=" * 70)
print("DUPLICATE IMAGES")
print("=" * 70)

duplicates = df.duplicated(subset=["image_name"]).sum()

print("Duplicate Images :", duplicates)

# ==============================================================
# IMAGE VERIFICATION
# ==============================================================

print("\n" + "=" * 70)
print("VERIFYING IMAGE FILES")
print("=" * 70)

missing_images = []

for image in df["image_name"]:

    found = False

    for folder in os.listdir(IMAGE_FOLDER):

        path = os.path.join(
            IMAGE_FOLDER,
            folder,
            image
        )

        if os.path.exists(path):
            found = True
            break

    if not found:
        missing_images.append(image)

print("Existing Images :", len(df) - len(missing_images))
print("Missing Images  :", len(missing_images))

# ==============================================================
# TEMPERATURE
# ==============================================================

print("\n" + "=" * 70)
print("TEMPERATURE")
print("=" * 70)

print(df["temperature"].describe())

# ==============================================================
# HUMIDITY
# ==============================================================

print("\n" + "=" * 70)
print("RELATIVE HUMIDITY")
print("=" * 70)

print(df["relative_humidity"].describe())

# ==============================================================
# VISIBILITY
# ==============================================================

print("\n" + "=" * 70)
print("VISIBILITY")
print("=" * 70)

print(df["visibility_km"].describe())

# ==============================================================
# SOLAR RADIATION
# ==============================================================

print("\n" + "=" * 70)
print("SOLAR RADIATION")
print("=" * 70)

print(df["solar_intensity"].describe())

print("\nMissing Solar Values :")

print(df["solar_intensity"].isna().sum())

# ==============================================================
# LOCATION INFORMATION
# ==============================================================

print("\n" + "=" * 70)
print("CAMERA LOCATIONS")
print("=" * 70)

locations = df[
    ["latitude", "longitude"]
].drop_duplicates()

print(locations)

print("\nTotal Locations :", len(locations))

# ==============================================================
# SAVE REPORT
# ==============================================================

os.makedirs(REPORT_FOLDER, exist_ok=True)

with open(REPORT_FILE, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("MASTER DATASET REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Rows : {len(df)}\n")
    f.write(f"Columns : {len(df.columns)}\n")
    f.write(f"Duplicate Images : {duplicates}\n")
    f.write(f"Missing Images : {len(missing_images)}\n")
    f.write(
        f"Missing Solar Values : "
        f"{df['solar_intensity'].isna().sum()}\n"
    )

print("\nReport Saved!")

print(REPORT_FILE)

# ==============================================================
# FINAL STATUS
# ==============================================================

print("\n" + "=" * 70)

if (
    duplicates == 0
    and len(missing_images) == 0
    and df["solar_intensity"].isna().sum() == 0
):

    print("DATASET VERIFIED SUCCESSFULLY")

else:

    print("DATASET HAS ISSUES")

print("=" * 70)