"""
============================================================
PROJECT SETUP
Creates Final Research Project Structure
============================================================
"""

import os
import shutil

print("=" * 70)
print("SETTING UP RESEARCH PROJECT")
print("=" * 70)


# ============================================================
# FOLDERS
# ============================================================

folders = [

    "models",

    "models/weather",

    "models/image",

    "models/fusion",

    "results",

    "results/weather",

    "results/image",

    "results/fusion",

    "results/comparison",

    "results/tuning",

    "results/explainability",

    "results/reports",

    "docs",

    "docs/figures",

    "docs/methodology"

]

print("\nCreating Folder Structure...\n")

for folder in folders:

    os.makedirs(folder, exist_ok=True)

    print("Created :", folder)


# ============================================================
# MOVE MODELS
# ============================================================

print("\n" + "=" * 70)
print("MOVING MODELS")
print("=" * 70)

model_files = {

    "results/models/weather_model.pkl":
    "models/weather/weather_model.pkl",

    "results/models/image_model.pkl":
    "models/image/image_model.pkl",

    "results/models/fusion_model.pkl":
    "models/fusion/fusion_model.pkl"

}

for source, destination in model_files.items():

    if os.path.exists(source):

        shutil.move(source, destination)

        print("Moved :", destination)

    else:

        print("Not Found :", source)


# ============================================================
# MOVE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("MOVING PREDICTIONS")
print("=" * 70)

prediction_files = {

    "results/metrics/weather_predictions.csv":
    "results/weather/weather_predictions.csv",

    "results/metrics/image_predictions.csv":
    "results/image/image_predictions.csv",

    "results/metrics/fusion_predictions.csv":
    "results/fusion/fusion_predictions.csv"

}

for source, destination in prediction_files.items():

    if os.path.exists(source):

        shutil.move(source, destination)

        print("Moved :", destination)

    else:

        print("Not Found :", source)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PROJECT STRUCTURE READY")
print("=" * 70)

print("\nResearch Project Initialized Successfully!")