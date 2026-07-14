import os
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

IMAGE_FOLDER = "data\prl_images"

EXCEL_FILE = "data\prl\metadata\prl_dataset_mapped.xlsx"

OUTPUT_EXCEL = "data/prl/metadata/prl_dataset_final.xlsx"

# ==========================================================

df = pd.read_excel(EXCEL_FILE)

images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if len(images) != len(df):
    raise Exception(
        f"Image count ({len(images)}) "
        f"does not match Excel rows ({len(df)})"
    )

new_names = []

for i, old_name in enumerate(images, start=1):

    extension = os.path.splitext(old_name)[1].lower()

    new_name = f"PRL_{i:04d}{extension}"

    old_path = os.path.join(
        IMAGE_FOLDER,
        old_name
    )

    new_path = os.path.join(
        IMAGE_FOLDER,
        new_name
    )

    os.rename(
        old_path,
        new_path
    )

    new_names.append(new_name)

df["image_name"] = new_names

df.to_excel(
    OUTPUT_EXCEL,
    index=False
)

print("\nDone!")
print(f"Renamed {len(new_names)} images.")
print(f"Updated Excel saved as:\n{OUTPUT_EXCEL}")