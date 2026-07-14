import os
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd

# =====================================================
# CHANGE THIS PATH TO YOUR IMAGE FOLDER
# =====================================================

IMAGE_FOLDER = r"data/prl_images"

# =====================================================

data = []

supported_extensions = (".jpg", ".jpeg", ".png")

for filename in sorted(os.listdir(IMAGE_FOLDER)):

    if not filename.lower().endswith(supported_extensions):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    date_time = "Not Found"

    try:
        image = Image.open(image_path)

        exif_data = image.getexif()

        if exif_data:

            for tag_id, value in exif_data.items():

                tag = TAGS.get(tag_id, tag_id)

                if tag == "DateTimeOriginal":
                    date_time = value
                    break

                elif tag == "DateTime":
                    date_time = value

    except Exception as e:
        print(f"Error reading {filename}: {e}")

    if date_time != "Not Found":

        try:
            date, time = date_time.split(" ")

            date = date.replace(":", "-")

        except:
            date = ""
            time = ""

    else:
        date = ""
        time = ""

    data.append({
        "image_name": filename,
        "date": date,
        "time": time
    })

df = pd.DataFrame(data)

output_file = "data/prl/metadata/image_metadata.xlsx"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

df.to_excel(output_file, index=False)

print(f"\nMetadata extracted successfully!")
print(f"Excel saved at: {output_file}")
print(f"Total Images: {len(df)}")