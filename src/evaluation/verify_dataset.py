import os
import pandas as pd

# ======================================================
# PATHS
# ======================================================

IMAGE_FOLDER = "data/prl_images"

DATASET_FILE = "data/prl/metadata/prl_dataset_clean.xlsx"

# ======================================================

print("=" * 60)
print("PRL DATASET VERIFICATION")
print("=" * 60)

# ------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------

df = pd.read_excel(DATASET_FILE)

print("\nDataset Loaded Successfully")

print(f"Rows : {len(df)}")
print(f"Columns : {len(df.columns)}")

# ------------------------------------------------------
# IMAGE COUNT
# ------------------------------------------------------

images = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"\nImages Found : {len(images)}")

# ------------------------------------------------------
# CHECK 1
# IMAGE COUNT
# ------------------------------------------------------

print("\n[1] Image Count Check")

if len(images) == len(df):

    print("PASS")

else:

    print("FAIL")
    print(f"Dataset Rows : {len(df)}")
    print(f"Images : {len(images)}")

# ------------------------------------------------------
# CHECK 2
# DUPLICATE IMAGE NAMES
# ------------------------------------------------------

print("\n[2] Duplicate Image Names")

duplicates = df["image_name"].duplicated().sum()

if duplicates == 0:

    print("PASS")

else:

    print("FAIL")
    print("Duplicates :", duplicates)

# ------------------------------------------------------
# CHECK 3
# MISSING VALUES
# ------------------------------------------------------

print("\n[3] Missing Values")

missing = df.isnull().sum()

print(missing)

# ------------------------------------------------------
# CHECK 4
# IMAGE EXISTS
# ------------------------------------------------------

print("\n[4] Image Matching")

csv_images = set(df["image_name"])

folder_images = set(images)

missing_images = csv_images - folder_images

extra_images = folder_images - csv_images

if len(missing_images) == 0:

    print("No Missing Images")

else:

    print("Missing Images")

    for img in sorted(missing_images):

        print(img)

if len(extra_images) == 0:

    print("No Extra Images")

else:

    print("Extra Images")

    for img in sorted(extra_images):

        print(img)

# ------------------------------------------------------
# CHECK 5
# TEMPERATURE
# ------------------------------------------------------

print("\n[5] Temperature")

print(df["temperature_C"].describe())

# ------------------------------------------------------
# CHECK 6
# HUMIDITY
# ------------------------------------------------------

print("\n[6] Relative Humidity")

print(df["relative_humidity_%"].describe())

# ------------------------------------------------------
# CHECK 7
# SOLAR
# ------------------------------------------------------

print("\n[7] Solar Radiation")

print(df["solar_intensity_Wm2"].describe())

# ------------------------------------------------------
# CHECK 8
# VISIBILITY
# ------------------------------------------------------

print("\n[8] Visibility")

print(df["visibility_km"].describe())

# ------------------------------------------------------
# CHECK 9
# DATE RANGE
# ------------------------------------------------------

print("\n[9] Date Range")

print("Start :", df["date"].min())

print("End   :", df["date"].max())

# ------------------------------------------------------
# CHECK 10
# HOURS
# ------------------------------------------------------

print("\n[10] Hours Present")

print(sorted(df["hour"].unique()))

# ------------------------------------------------------
# CHECK 11
# IMAGES PER DAY
# ------------------------------------------------------

print("\n[11] Images Per Day")

print(df.groupby("date").size())

# ------------------------------------------------------
# CHECK 12
# DATA TYPES
# ------------------------------------------------------

print("\n[12] Data Types")

print(df.dtypes)

print("\nVerification Complete")
report = []

report.append(f"Total Images : {len(images)}")
report.append(f"Dataset Rows : {len(df)}")
report.append(f"Duplicate Image Names : {duplicates}")
report.append(f"Missing Values :\n{missing}")

os.makedirs("results/verification", exist_ok=True)

with open("results/verification/dataset_report.txt", "w") as f:
    for line in report:
        f.write(str(line))
        f.write("\n\n")

print("\nVerification report saved to:")
print("results/verification/dataset_report.txt")