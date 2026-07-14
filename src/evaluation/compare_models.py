"""
============================================================
MODEL COMPARISON
PRL Visibility Estimation Research Project
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("="*70)
print("MODEL COMPARISON")
print("="*70)

# ============================================================
# PATHS
# ============================================================

weather_path = "results/weather/weather_predictions.csv"

image_path = "results/image/image_predictions.csv"

fusion_path = "results/fusion/fusion_predictions.csv"

output_folder = "results/comparison"

os.makedirs(output_folder, exist_ok=True)

# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading Prediction Files...\n")

weather = pd.read_csv(weather_path)

image = pd.read_csv(image_path)

fusion = pd.read_csv(fusion_path)

print("Weather :", weather.shape)

print("Image   :", image.shape)

print("Fusion  :", fusion.shape)

# ============================================================
# METRICS FUNCTION
# ============================================================

def evaluate(df):

    actual = df.iloc[:,0]

    predicted = df.iloc[:,1]

    mae = mean_absolute_error(actual,predicted)

    rmse = np.sqrt(mean_squared_error(actual,predicted))

    r2 = r2_score(actual,predicted)

    return mae,rmse,r2

# ============================================================
# CALCULATE METRICS
# ============================================================

weather_mae,weather_rmse,weather_r2 = evaluate(weather)

image_mae,image_rmse,image_r2 = evaluate(image)

fusion_mae,fusion_rmse,fusion_r2 = evaluate(fusion)

print("\nMetrics Calculated Successfully!")

# ============================================================
# CREATE TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model":[

        "Weather",

        "Image",

        "Fusion"

    ],

    "MAE":[

        weather_mae,

        image_mae,

        fusion_mae

    ],

    "RMSE":[

        weather_rmse,

        image_rmse,

        fusion_rmse

    ],

    "R2":[

        weather_r2,

        image_r2,

        fusion_r2

    ]

})

comparison = comparison.sort_values(

    by="R2",

    ascending=False

)

print()

print(comparison)

# ============================================================
# SAVE TABLE
# ============================================================

comparison.to_csv(

    "results/comparison/comparison_metrics.csv",

    index=False

)

print()

print("Comparison CSV Saved!")