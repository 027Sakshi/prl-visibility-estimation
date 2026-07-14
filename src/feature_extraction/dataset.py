"""
==============================================================
SkyFinder Dataset Loader
PRL Visibility Estimation Project

Loads images and metadata for DINOv2 feature extraction.
==============================================================
"""

import os
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ==============================================================
# IMAGE TRANSFORM
# ==============================================================

IMAGE_TRANSFORM = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])


# ==============================================================
# DATASET CLASS
# ==============================================================

class SkyFinderDataset(Dataset):

    """
    Dataset class for loading SkyFinder images
    together with their metadata.
    """

    # ----------------------------------------------------------

    def __init__(

        self,

        csv_path,

        image_root,

        transform=IMAGE_TRANSFORM

    ):

        print("=" * 60)
        print("Loading SkyFinder Dataset")
        print("=" * 60)

        self.df = pd.read_csv(csv_path)

        self.image_root = image_root

        self.transform = transform

        print("Images :", len(self.df))
        print("Dataset Loaded Successfully!\n")


    # ----------------------------------------------------------

    def __len__(self):

        return len(self.df)


    # ----------------------------------------------------------

    def find_image(self, image_name):

        """
        Search every camera folder
        until image is found.
        """

        for folder in os.listdir(self.image_root):

            folder_path = os.path.join(

                self.image_root,

                folder

            )

            if not os.path.isdir(folder_path):

                continue

            image_path = os.path.join(

                folder_path,

                image_name

            )

            if os.path.exists(image_path):

                return image_path

        return None


    # ----------------------------------------------------------

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_name = row["image_name"]

        image_path = self.find_image(image_name)

        if image_path is None:

            raise FileNotFoundError(

                f"Image not found : {image_name}"

            )

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        sample = {

            "image": image,

            "image_name": image_name,

            "visibility": float(row["visibility_km"]),

            "temperature": float(row["temperature"]),

            "humidity": float(row["relative_humidity"]),

            "solar": float(row["solar_intensity"]),

            "latitude": float(row["latitude"]),

            "longitude": float(row["longitude"]),

            "date": row["date"],

            "time": row["time"]

        }

        return sample


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    DATASET = SkyFinderDataset(

        csv_path="data/external/skyfinder/final/skyfinder_master_dataset.csv",

        image_root="data/external/skyfinder/images"

    )

    print("Dataset Size :", len(DATASET))

    sample = DATASET[0]

    print("\nSample Information\n")

    print("Image Shape :", sample["image"].shape)

    print("Image Name :", sample["image_name"])

    print("Visibility :", sample["visibility"])

    print("Temperature :", sample["temperature"])

    print("Humidity :", sample["humidity"])

    print("Solar :", sample["solar"])

    print("Latitude :", sample["latitude"])

    print("Longitude :", sample["longitude"])

    print("Date :", sample["date"])

    print("Time :", sample["time"])