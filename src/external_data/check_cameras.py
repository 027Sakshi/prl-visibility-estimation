import os

print("=" * 70)
print("CHECKING SKYFINDER CAMERA FOLDERS")
print("=" * 70)

IMAGE_ROOT = "data/external/skyfinder/images"

folders = sorted(os.listdir(IMAGE_ROOT))

print("\nFound", len(folders), "camera folders\n")

total_images = 0

for folder in folders:

    path = os.path.join(IMAGE_ROOT, folder)

    if not os.path.isdir(path):
        continue

    images = [
        f for f in os.listdir(path)
        if f.lower().endswith(".jpg")
    ]

    print(f"Camera {folder:>6} : {len(images)} images")

    total_images += len(images)

print("\n" + "=" * 70)
print("TOTAL IMAGES :", total_images)
print("=" * 70)