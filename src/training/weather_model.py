"""
=========================================================
WEATHER BASELINE MODEL

Stage 9

Predict Visibility using

Temperature
Humidity
Solar Radiation
=========================================================
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import joblib
import numpy as np

from xgboost import XGBRegressor

from sklearn.model_selection import (

    train_test_split,

    KFold,

    cross_val_score

)
from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)
# =========================================================
# PATHS
# =========================================================

WEATHER_PATH = "data/processed/X_weather.npy"

TARGET_PATH = "data/processed/y.npy"

MODEL_FOLDER = "results/models"

METRIC_FOLDER = "results/metrics"

PLOT_FOLDER = "results/plots"

os.makedirs(MODEL_FOLDER, exist_ok=True)

os.makedirs(METRIC_FOLDER, exist_ok=True)

os.makedirs(PLOT_FOLDER, exist_ok=True)
# =========================================================
# LOAD DATA
# =========================================================

print("="*60)
print("LOADING DATA")
print("="*60)

X = np.load(WEATHER_PATH)

y = np.load(TARGET_PATH)

print()

print("Weather Shape")

print(X.shape)

print()

print("Target Shape")

print(y.shape)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print()

print("=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    shuffle=True

)

print()

print("Training Samples")

print(X_train.shape)

print(y_train.shape)

print()

print("Testing Samples")

print(X_test.shape)

print(y_test.shape)

# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

print()

print("=" * 60)
print("CREATING XGBOOST MODEL")
print("=" * 60)

model = XGBRegressor(

    objective="reg:squarederror",

    n_estimators=500,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1

)

print()

print("Model Created Successfully!")

print()

print(model)

# ============================================================
# TRAIN MODEL
# ============================================================

print()

print("=" * 60)
print("TRAINING MODEL")
print("=" * 60)

start_time = time.time()

model.fit(

    X_train,

    y_train

)

end_time = time.time()

training_time = end_time - start_time

print()

print("Training Completed!")

print()

print(f"Training Time : {training_time:.2f} seconds")

# ============================================================
# MAKE PREDICTIONS
# ============================================================

print()

print("=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)

# Training Predictions
y_train_pred = model.predict(X_train)

# Testing Predictions
y_test_pred = model.predict(X_test)

print()

print("Predictions Completed!")

print()

print("Training Prediction Shape")

print(y_train_pred.shape)

print()

print("Testing Prediction Shape")

print(y_test_pred.shape)

print()

print("First 10 Test Predictions")

print(y_test_pred[:10])

# ============================================================
# MODEL EVALUATION
# ============================================================

print()

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# -----------------------------
# Training Metrics
# -----------------------------

train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        y_train_pred
    )
)

train_r2 = r2_score(
    y_train,
    y_train_pred
)

# -----------------------------
# Testing Metrics
# -----------------------------

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

print()

print("TRAINING RESULTS")

print("-" * 40)

print(f"MAE  : {train_mae:.4f}")

print(f"RMSE : {train_rmse:.4f}")

print(f"R²   : {train_r2:.4f}")

print()

print("TESTING RESULTS")

print("-" * 40)

print(f"MAE  : {test_mae:.4f}")

print(f"RMSE : {test_rmse:.4f}")

print(f"R²   : {test_r2:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================

print()

print("=" * 60)
print("SAVING MODEL")
print("=" * 60)

model_path = "results/models/weather_model.pkl"

joblib.dump(
    model,
    model_path
)

print()

print("Model Saved Successfully!")

print(model_path)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

print()

print("=" * 60)
print("SAVING PREDICTIONS")
print("=" * 60)

prediction_df = pd.DataFrame({

    "Actual_Visibility": y_test,

    "Predicted_Visibility": y_test_pred

})

prediction_path = "results/metrics/weather_predictions.csv"

prediction_df.to_csv(

    prediction_path,

    index=False

)

print()

print("Predictions Saved Successfully!")

print(prediction_path)

print()

print(prediction_df.head())

# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print()

print("=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

kfold = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

# -----------------------------
# MAE
# -----------------------------

mae_scores = -cross_val_score(

    model,

    X,

    y,

    cv=kfold,

    scoring="neg_mean_absolute_error",

    n_jobs=-1

)

# -----------------------------
# RMSE
# -----------------------------

rmse_scores = np.sqrt(

    -cross_val_score(

        model,

        X,

        y,

        cv=kfold,

        scoring="neg_mean_squared_error",

        n_jobs=-1

    )

)

# -----------------------------
# R2
# -----------------------------

r2_scores = cross_val_score(

    model,

    X,

    y,

    cv=kfold,

    scoring="r2",

    n_jobs=-1

)

print()

print("MAE Scores")

print(mae_scores)

print()

print("RMSE Scores")

print(rmse_scores)

print()

print("R² Scores")

print(r2_scores)

print()

print("=" * 40)

print("AVERAGE RESULTS")

print("=" * 40)

print(f"Average MAE  : {mae_scores.mean():.4f}")

print(f"Average RMSE : {rmse_scores.mean():.4f}")

print(f"Average R²   : {r2_scores.mean():.4f}")

# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

print()

print("=" * 60)
print("GENERATING ACTUAL VS PREDICTED PLOT")
print("=" * 60)

plt.figure(figsize=(8,8))

plt.scatter(

    y_test,

    y_test_pred,

    alpha=0.6

)

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    "r--",

    linewidth=2

)

plt.xlabel("Actual Visibility (km)")

plt.ylabel("Predicted Visibility (km)")

plt.title("Weather Model : Actual vs Predicted")

plt.grid(True)

plot_path = "results/plots/weather_actual_vs_predicted.png"

plt.savefig(

    plot_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print()

print("Plot Saved Successfully!")

print(plot_path)
# ============================================================
# RESIDUAL PLOT
# ============================================================

print()

print("=" * 60)
print("GENERATING RESIDUAL PLOT")
print("=" * 60)

# Calculate residuals
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 6))

plt.scatter(

    y_test_pred,

    residuals,

    alpha=0.6

)

plt.axhline(

    y=0,

    color="red",

    linestyle="--",

    linewidth=2

)

plt.xlabel("Predicted Visibility (km)")

plt.ylabel("Residual (Actual - Predicted)")

plt.title("Weather Model : Residual Plot")

plt.grid(True)

residual_plot_path = "results/plots/weather_residual_plot.png"

plt.savefig(

    residual_plot_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print()

print("Residual Plot Saved Successfully!")

print(residual_plot_path)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()

print("=" * 60)
print("GENERATING FEATURE IMPORTANCE")
print("=" * 60)

feature_names = [

    "Temperature",

    "Humidity",

    "Solar Radiation"

]

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print()

print(importance_df)

plt.figure(figsize=(8,5))

plt.bar(

    importance_df["Feature"],

    importance_df["Importance"]

)

plt.xlabel("Weather Feature")

plt.ylabel("Importance")

plt.title("Weather Model Feature Importance")

importance_plot_path = "results/plots/weather_feature_importance.png"

plt.savefig(

    importance_plot_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print()

print("Feature Importance Plot Saved!")

print(importance_plot_path)


# ============================================================
# SAVE METRICS REPORT
# ============================================================

print()

print("=" * 60)
print("SAVING METRICS REPORT")
print("=" * 60)

report_path = "results/metrics/weather_metrics.txt"

with open(report_path, "w") as f:

    f.write("=" * 60 + "\n")

    f.write("WEATHER BASELINE MODEL REPORT\n")

    f.write("=" * 60 + "\n\n")

    f.write("TRAINING RESULTS\n")

    f.write("-----------------------------\n")

    f.write(f"MAE  : {train_mae:.4f}\n")

    f.write(f"RMSE : {train_rmse:.4f}\n")

    f.write(f"R2   : {train_r2:.4f}\n\n")

    f.write("TESTING RESULTS\n")

    f.write("-----------------------------\n")

    f.write(f"MAE  : {test_mae:.4f}\n")

    f.write(f"RMSE : {test_rmse:.4f}\n")

    f.write(f"R2   : {test_r2:.4f}\n\n")

    f.write("5-FOLD CROSS VALIDATION\n")

    f.write("-----------------------------\n")

    f.write(f"Average MAE  : {mae_scores.mean():.4f}\n")

    f.write(f"Average RMSE : {rmse_scores.mean():.4f}\n")

    f.write(f"Average R2   : {r2_scores.mean():.4f}\n\n")

    f.write("FEATURE IMPORTANCE\n")

    f.write("-----------------------------\n")

    for feature, score in zip(

        importance_df["Feature"],

        importance_df["Importance"]

    ):

        f.write(f"{feature:<20} {score:.4f}\n")

print()

print("Metrics Report Saved!")

print(report_path)


# ============================================================
# STAGE 9 COMPLETED
# ============================================================

print()

print("=" * 70)

print("STAGE 9 COMPLETED SUCCESSFULLY")

print("=" * 70)

print()

print("Generated Files")

print("---------------------------------------")

print("results/models/weather_model.pkl")

print("results/metrics/weather_metrics.txt")

print("results/metrics/weather_predictions.csv")

print("results/plots/weather_actual_vs_predicted.png")

print("results/plots/weather_residual_plot.png")

print("results/plots/weather_feature_importance.png")

print()

print("STATUS : SUCCESS")

print("=" * 70)