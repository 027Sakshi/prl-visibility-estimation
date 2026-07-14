import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

# ==========================================================
# PATHS
# ==========================================================

IMAGE_FOLDER = "data/prl_images"

OUTPUT_FILE = "data/prl_features/all_embeddings.csv"

# ==========================================================

print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

model.eval()

print("Model Loaded Successfully!")

# ==========================================================

image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"\nImages Found : {len(image_files)}")

all_features = []

# ==========================================================

for image_name in tqdm(image_files):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    embedding = (
        outputs.last_hidden_state[:,0,:]
        .squeeze()
        .numpy()
    )

    row = [image_name]

    row.extend(
        embedding
    )

    all_features.append(
        row
    )

# ==========================================================

columns = ["image_name"]

for i in range(768):

    columns.append(
        f"feature_{i+1}"
    )

df = pd.DataFrame(
    all_features,
    columns=columns
)

os.makedirs(
    "data/prl_features",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone!")
print(df.head())

print("\nShape:")

print(df.shape)