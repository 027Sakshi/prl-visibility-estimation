"""
==============================================================
TRAINING DATA PREPARATION
PRL Visibility Estimation Project

Stage 8.1
==============================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = "data/processed/image_features.npy"

METADATA_PATH = "data/processed/feature_metadata.csv"

OUTPUT_FOLDER = "data/processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PREPARING TRAINING DATA")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading Image Features...")

image_features = np.load(FEATURES_PATH)

print("Done!")

print("Feature Shape :", image_features.shape)


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading Metadata...")

metadata = pd.read_csv(METADATA_PATH)
print("\nMetadata Columns:\n")
print(metadata.columns.tolist())

print("Done!")

print("Metadata Shape :", metadata.shape)


# ============================================================
# VERIFY
# ============================================================

print("\nVerifying Dataset...")

assert len(image_features) == len(metadata), \
    "Feature and Metadata sizes do not match!"

print("Verification Successful!")

print("\nTotal Samples :", len(metadata))


# ============================================================
# DATA SUMMARY
# ============================================================

print("\nDataset Summary")

print("-" * 50)

print("Images          :", len(metadata))

print("Feature Length  :", image_features.shape[1])

print("Weather Columns :", 3)

print("Target          : Visibility")

# ============================================================
# STEP 8.2 : CREATING IMAGE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 8.2 : CREATING IMAGE FEATURES")
print("=" * 70)

X_image = image_features.copy()

print("Done!")

print("Shape :", X_image.shape)

print()

print("First Feature Vector")

print(X_image[0][:10])

# ============================================================
# STEP 8.3 : CREATING WEATHER FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 8.3 : CREATING WEATHER FEATURES")
print("=" * 70)

weather_columns = [

    "temperature",

    "humidity",

    "solar"

]

X_weather = metadata[weather_columns].values

scaler = StandardScaler()

X_weather = scaler.fit_transform(X_weather)

joblib.dump(

    scaler,

    "data/processed/weather_scaler.pkl"

)

print("Done!")

print("Shape :", X_weather.shape)

print("Weather Scaler Saved!")

print("First Five Rows")

print(X_weather[:5])
print("\n" + "=" * 70)
print("STEP 8.4 : CREATING TARGET")
print("=" * 70)

y = metadata["visibility"].values

print("Done!")

print()

print("Shape :", y.shape)

print()

print("First Five Values")

print(y[:5])
print("\n" + "=" * 70)
print("STEP 8.5 : CREATING FUSION FEATURES")
print("=" * 70)

X_fusion = np.concatenate(

    [

        X_image,

        X_weather

    ],

    axis=1

)

print("Done!")

print()

print("Shape :", X_fusion.shape)
print("\n" + "=" * 70)
print("STEP 8.6 : SAVING")
print("=" * 70)

np.save(

    "data/processed/X_image.npy",

    X_image

)

np.save(

    "data/processed/X_weather.npy",

    X_weather

)

np.save(

    "data/processed/X_fusion.npy",

    X_fusion

)

np.save(

    "data/processed/y.npy",

    y

)

print("Done!")
