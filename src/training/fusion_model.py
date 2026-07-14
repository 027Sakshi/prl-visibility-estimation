"""
============================================================
FUSION MODEL
Stage 11
============================================================
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ============================================================
# PATHS
# ============================================================

FUSION_PATH = "data/processed/X_fusion.npy"

TARGET_PATH = "data/processed/y.npy"

MODEL_FOLDER = "results/models"

METRIC_FOLDER = "results/metrics"

PLOT_FOLDER = "results/plots"

os.makedirs(MODEL_FOLDER, exist_ok=True)

os.makedirs(METRIC_FOLDER, exist_ok=True)

os.makedirs(PLOT_FOLDER, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("="*60)
print("LOADING FUSION DATA")
print("="*60)

X = np.load(FUSION_PATH)

y = np.load(TARGET_PATH)

print()

print("Fusion Shape")

print(X.shape)

print()

print("Target Shape")

print(y.shape)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print()

print("="*60)
print("TRAIN TEST SPLIT")
print("="*60)

X_train,X_test,y_train,y_test=train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    shuffle=True

)

print()

print("Training")

print(X_train.shape)

print(y_train.shape)

print()

print("Testing")

print(X_test.shape)

print(y_test.shape)

# ============================================================
# CREATE MODEL
# ============================================================

print()

print("="*60)
print("CREATING FUSION MODEL")
print("="*60)

model = XGBRegressor(

    objective="reg:squarederror",

    n_estimators=200,

    learning_rate=0.05,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1

)

print()

print(model)

# ============================================================
# TRAIN MODEL
# ============================================================

print()

print("="*60)
print("TRAINING MODEL")
print("="*60)

start=time.time()

model.fit(

    X_train,

    y_train

)

end=time.time()

print()

print("Training Completed")

print()

print(f"Training Time : {end-start:.2f} seconds")

# ============================================================
# PREDICTIONS
# ============================================================

print()

print("="*60)
print("PREDICTING")
print("="*60)

y_train_pred=model.predict(X_train)

y_test_pred=model.predict(X_test)

print()

print("Prediction Completed")

print()

print(y_test_pred[:10])

# ============================================================
# EVALUATION
# ============================================================

print()

print("="*60)
print("MODEL EVALUATION")
print("="*60)

train_mae=mean_absolute_error(y_train,y_train_pred)

train_rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))

train_r2=r2_score(y_train,y_train_pred)

test_mae=mean_absolute_error(y_test,y_test_pred)

test_rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))

test_r2=r2_score(y_test,y_test_pred)

print()

print("TRAIN")

print(train_mae)

print(train_rmse)

print(train_r2)

print()

print("TEST")

print(test_mae)

print(test_rmse)

print(test_r2)

# ============================================================
# CROSS VALIDATION
# ============================================================

print()

print("="*60)
print("5 FOLD CROSS VALIDATION")
print("="*60)

kf=KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

mae=-cross_val_score(

    model,

    X,

    y,

    cv=kf,

    scoring="neg_mean_absolute_error",

    n_jobs=-1

)

rmse=np.sqrt(

    -cross_val_score(

        model,

        X,

        y,

        cv=kf,

        scoring="neg_mean_squared_error",

        n_jobs=-1

    )

)

r2=cross_val_score(

    model,

    X,

    y,

    cv=kf,

    scoring="r2",

    n_jobs=-1

)

print()

print("Average MAE :",mae.mean())

print("Average RMSE :",rmse.mean())

print("Average R2 :",r2.mean())

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    model,

    "results/models/fusion_model.pkl"

)

pd.DataFrame({

    "Actual":y_test,

    "Predicted":y_test_pred

}).to_csv(

    "results/metrics/fusion_predictions.csv",

    index=False

)

print()

print("Fusion Model Saved!")

print()

print("Stage 11 Completed!")

