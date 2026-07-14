import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

print("=" * 70)
print("SKYFINDER CAMERA DOWNLOADER")
print("=" * 70)

# ======================================================
# CHANGE ONLY THIS CAMERA ID
# ======================================================

CAMERA_ID = input(
    "\nEnter Camera ID : "
).strip()

# ======================================================
# URL
# ======================================================

BASE_URL = f"https://cs.valdosta.edu/~rpmihail/skyfinder/images/{CAMERA_ID}.html"

# ======================================================
# SAVE FOLDER
# ======================================================

SAVE_FOLDER = f"data/external/skyfinder/images/{CAMERA_ID}"

os.makedirs(
    SAVE_FOLDER,
    exist_ok=True
)

print("\nReading Camera Page...")

response = requests.get(BASE_URL)

if response.status_code != 200:
    print("Could not open camera page.")
    exit()

soup = BeautifulSoup(response.text, "html.parser")

links = soup.find_all("a")

image_links = []

for link in links:

    href = link.get("href")

    if href is not None and href.endswith(".jpg"):

        image_links.append(href)

print("\nTotal Images Found :", len(image_links))

downloaded = 0
skipped = 0

print("\nDownloading Images...\n")

for image in tqdm(image_links):

    image_url = BASE_URL.replace(f"{CAMERA_ID}.html", f"{CAMERA_ID}/{image}")

    save_path = os.path.join(
        SAVE_FOLDER,
        image
    )

    if os.path.exists(save_path):
        skipped += 1
        continue

    try:

        img = requests.get(image_url, timeout=20)

        if img.status_code == 200:

            with open(save_path, "wb") as f:

                f.write(img.content)

            downloaded += 1

    except Exception:
        pass

print("\n")
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print("Downloaded :", downloaded)
print("Skipped :", skipped)
print("Folder :", SAVE_FOLDER)