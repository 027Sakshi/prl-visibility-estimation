import os
import pandas as pd

from xgboost import XGBRegressor

from src.evaluation.cross_validation import run_cv

print("=" * 60)
print("A3 FINAL FUSION MODEL")
print("=" * 60)

# ----------------------------------------
# LOAD DATA
# ----------------------------------------

df = pd.read_csv(
    "data/merged_pca15/merged_pca15.csv"
)

print("\nDataset Loaded!")

print(df.shape)

# ----------------------------------------
# FEATURES
# ----------------------------------------

weather_features = [

    "temperature_C",

    "relative_humidity_%",

    "solar_intensity_Wm2",

    "hour"

]

pca_features = [

    col

    for col in df.columns

    if col.startswith("pca_")

]

feature_columns = weather_features + pca_features

X = df[feature_columns]

y = df["visibility_km"]

print("\nFeature Matrix")

print(X.shape)

print("Target")

print(y.shape)

# ----------------------------------------
# MODEL
# ----------------------------------------

model = XGBRegressor(

    objective="reg:squarederror",

    random_state=42,

    n_estimators=100,

    learning_rate=0.01,

    max_depth=4,

    subsample=0.9,

    colsample_bytree=0.6,

    min_child_weight=1

)

print("\nModel")

print(model)

# ----------------------------------------
# TRAIN
# ----------------------------------------

print("\nRunning Cross Validation...\n")

results = run_cv(

    model,

    X,

    y

)

print()

print("=" * 60)

print("A3 FINAL RESULTS")

print("=" * 60)

print(results)

# ----------------------------------------
# SAVE
# ----------------------------------------

os.makedirs(

    "results/A3_Final",

    exist_ok=True

)

pd.DataFrame(

    [results]

).to_csv(

    "results/A3_Final/final_results.csv",

    index=False

)

print()

print("Results Saved!")