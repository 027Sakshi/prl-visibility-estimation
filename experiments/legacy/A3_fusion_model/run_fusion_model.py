import os
import pandas as pd

from src.models.fusion_model import create_fusion_model

from src.evaluation.cross_validation import run_cv

print("=" * 60)
print("A3 - FUSION MODEL")
print("=" * 60)

print()

print("Loading Dataset...")

df = pd.read_csv(
    "data/merged/merged_dataset.csv"
)

print("Dataset Loaded!")

print(df.shape)
weather_features = [

    "temperature_C",

    "relative_humidity_%",

    "solar_intensity_Wm2",

    "hour"

]
dino_features = [

    col

    for col in df.columns

    if col.startswith("dino_")

]
feature_columns = (

    weather_features

    +

    dino_features

)

X = df[
    feature_columns
]

y = df[
    "visibility_km"
]
print()

print("Creating Fusion Model...")

model = create_fusion_model()

print(model)
print()

print("Running Cross Validation...")

results = run_cv(

    model,

    X,

    y

)
print()

print("="*60)

print("A3 RESULTS")

print("="*60)

print(results)
results_df = pd.DataFrame(
    [results]
)

os.makedirs(
    "results/A3",
    exist_ok=True
)

results_df.to_csv(

    "results/A3/fusion_results.csv",

    index=False

)

print()

print("Results Saved!")
