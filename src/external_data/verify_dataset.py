import os
import pandas as pd

print("=" * 70)
print("VERIFYING SKYFINDER DATASET")
print("=" * 70)

CSV_PATH = "data/external/skyfinder/final/camera{CAMERA_ID}_final.csv"

IMAGE_FOLDER = "data/external/skyfinder/images/{CAMERA_ID}"

df = pd.read_csv(CSV_PATH)

print("\nDataset Rows :", len(df))

missing = []

for image in df["image_name"]:

    path = os.path.join(IMAGE_FOLDER, image)

    if not os.path.exists(path):

        missing.append(image)

print("\nExisting Images :", len(df) - len(missing))

print("Missing Images :", len(missing))

if len(missing) == 0:

    print("\nDataset Verified Successfully!")

else:

    print("\nMissing Files")

    print(missing[:20])