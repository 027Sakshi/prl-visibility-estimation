"""
==============================================================
DINOv2 FEATURE EXTRACTION
PRL Visibility Estimation Project

Stage 7
==============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import time
import warnings

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.feature_extraction.dataset import SkyFinderDataset
from src.feature_extraction.dinov2_extractor import (
    load_dinov2,
    DEVICE
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = (
    "data/external/skyfinder/final/"
    "skyfinder_master_dataset.csv"
)

IMAGE_FOLDER = (
    "data/external/skyfinder/images"
)

OUTPUT_FOLDER = (
    "data/processed"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

if DEVICE.type == "cuda":

    BATCH_SIZE = 64

else:

    BATCH_SIZE = 16

NUM_WORKERS = 0


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DINOv2 FEATURE EXTRACTION")
print("=" * 70)

print("\nDevice :", DEVICE)

print("Batch Size :", BATCH_SIZE)

print()


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

dataset = SkyFinderDataset(

    csv_path=DATASET_PATH,

    image_root=IMAGE_FOLDER

)
# ------------------------------------------
# TEST MODE (100 Images)
# ------------------------------------------

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=(DEVICE.type == "cuda")

)

print("\nImages :", len(dataset))

print("Total Batches :", len(loader))


# ============================================================
# LOAD MODEL
# ============================================================

print("\n")

model = load_dinov2()

print()


# ============================================================
# START TIMER
# ============================================================

start_time = time.time()


# ============================================================
# STORAGE
# ============================================================

all_features = []

image_names = []

metadata = []


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("=" * 70)
print("EXTRACTING FEATURES")
print("=" * 70)

model.eval()

with torch.inference_mode():

    for batch in tqdm(loader):

        images = batch["image"].to(DEVICE)

        features = model(images)

        features = features.cpu().numpy()

        all_features.append(features)

        image_names.extend(batch["image_name"])

        batch_size = len(batch["image_name"])

        for i in range(batch_size):

            metadata.append({

                "image_name":
                batch["image_name"][i],

                "visibility":
                float(batch["visibility"][i]),

                "temperature":
                float(batch["temperature"][i]),

                "humidity":
                float(batch["humidity"][i]),

                "solar":
                float(batch["solar"][i]),

                "latitude":
                float(batch["latitude"][i]),

                "longitude":
                float(batch["longitude"][i]),

                "date":
                batch["date"][i],

                "time":
                batch["time"][i]

            })
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
# ============================================================
# COMBINE ALL FEATURES
# ============================================================

print("\n")
print("=" * 70)
print("COMBINING FEATURES")
print("=" * 70)

all_features = np.concatenate(

    all_features,

    axis=0

)

print("Feature Matrix Shape :", all_features.shape)

print("Expected Shape       :", (len(dataset), all_features.shape[1]))

print()


# ============================================================
# VERIFY FEATURES
# ============================================================

print("=" * 70)
print("VERIFYING FEATURES")
print("=" * 70)

print()

print("Checking NaN Values...")

nan_count = np.isnan(all_features).sum()

print("NaN Values :", nan_count)

print()

print("Checking Infinite Values...")

inf_count = np.isinf(all_features).sum()

print("Infinite Values :", inf_count)

print()

print("Feature Statistics")

print("Minimum :", np.min(all_features))

print("Maximum :", np.max(all_features))

print("Mean    :", np.mean(all_features))

print("Std     :", np.std(all_features))

print()


# ============================================================
# SAVE FEATURES
# ============================================================

print("=" * 70)
print("SAVING FEATURES")
print("=" * 70)

feature_path = os.path.join(

    OUTPUT_FOLDER,

    "image_features.npy"

)
all_features = all_features.astype(np.float32)
np.save(

    feature_path,

    all_features

)

print("Saved :", feature_path)


# ============================================================
# SAVE IMAGE NAMES
# ============================================================

image_names_df = pd.DataFrame({

    "image_name": image_names

})

image_names_path = os.path.join(

    OUTPUT_FOLDER,

    "image_names.csv"

)

image_names_df.to_csv(

    image_names_path,

    index=False

)

print("Saved :", image_names_path)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(metadata)

metadata_path = os.path.join(

    OUTPUT_FOLDER,

    "feature_metadata.csv"

)

metadata_df.to_csv(

    metadata_path,

    index=False

)

print("Saved :", metadata_path)


# ============================================================
# EXTRACTION REPORT
# ============================================================

report_path = os.path.join(

    OUTPUT_FOLDER,

    "feature_extraction_report.txt"

)

elapsed = time.time() - start_time

hours = int(elapsed // 3600)

minutes = int((elapsed % 3600) // 60)

seconds = int(elapsed % 60)

with open(report_path, "w") as f:

    f.write("=" * 70 + "\n")

    f.write("FEATURE EXTRACTION REPORT\n")

    f.write("=" * 70 + "\n\n")

    f.write(f"Images              : {len(dataset)}\n")

    f.write(f"Feature Dimension   : {all_features.shape[1]}\n")

    f.write(f"Feature Shape       : {all_features.shape}\n")

    f.write(f"NaN Values          : {nan_count}\n")

    f.write(f"Infinite Values     : {inf_count}\n")

    f.write(f"Minimum             : {np.min(all_features)}\n")

    f.write(f"Maximum             : {np.max(all_features)}\n")

    f.write(f"Mean                : {np.mean(all_features)}\n")

    f.write(f"Std                 : {np.std(all_features)}\n")

    f.write(

        f"Execution Time      : "

        f"{hours}h {minutes}m {seconds}s\n"

    )

print("Saved :", report_path)

print()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print()

print("Total Images Processed :", len(dataset))

print("Feature Shape          :", all_features.shape)

print("Embedding Size         :", all_features.shape[1])

print()

print("NaN Values             :", nan_count)

print("Infinite Values        :", inf_count)

print()

print("Execution Time")

print(

    f"{hours} Hours "

    f"{minutes} Minutes "

    f"{seconds} Seconds"

)

print()

print("Generated Files")

print("--------------------------------------------")

print(feature_path)

print(image_names_path)

print(metadata_path)

print(report_path)

print()

if nan_count == 0 and inf_count == 0:

    print("STATUS : SUCCESS")

else:

    print("STATUS : WARNING - VERIFY FEATURES")

print("=" * 70)