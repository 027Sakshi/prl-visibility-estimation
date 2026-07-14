"""
============================================================
FUSION MODEL HYPERPARAMETER TUNING
============================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("="*70)
print("FUSION MODEL HYPERPARAMETER TUNING")
print("="*70)

# ============================================================
# PATHS
# ============================================================

X = np.load("data/processed/X_fusion.npy")

y = np.load("data/processed/y.npy")

output = "results/tuning"

os.makedirs(output,exist_ok=True)

# ============================================================
# SPLIT
# ============================================================

X_train,X_test,y_train,y_test=train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print()

print("Training Shape :",X_train.shape)

print("Testing Shape  :",X_test.shape)

# ============================================================
# MODEL
# ============================================================

model = XGBRegressor(

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1

)

# ============================================================
# SEARCH SPACE
# ============================================================

params = {

    "n_estimators":[

        100,

        200,

        300,

        500

    ],

    "max_depth":[

        4,

        5,

        6,

        8

    ],

    "learning_rate":[

        0.01,

        0.03,

        0.05,

        0.1

    ],

    "subsample":[

        0.7,

        0.8,

        1.0

    ],

    "colsample_bytree":[

        0.7,

        0.8,

        1.0

    ]

}

# ============================================================
# RANDOM SEARCH
# ============================================================

print()

print("Searching Best Parameters...\n")

search = RandomizedSearchCV(

    estimator=model,

    param_distributions=params,

    n_iter=20,

    scoring="r2",

    cv=3,

    verbose=2,

    random_state=42,

    n_jobs=-1

)

search.fit(

    X_train,

    y_train

)

best_model = search.best_estimator_

print()

print("Best Parameters")

print(search.best_params_)

# ============================================================
# EVALUATE
# ============================================================

prediction = best_model.predict(

    X_test

)

mae = mean_absolute_error(

    y_test,

    prediction

)

rmse = np.sqrt(

    mean_squared_error(

        y_test,

        prediction

    )

)

r2 = r2_score(

    y_test,

    prediction

)

print()

print("MAE :",mae)

print("RMSE :",rmse)

print("R2 :",r2)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    best_model,

    "results/tuning/best_fusion_model.pkl"

)

with open(

    "results/tuning/best_parameters.json",

    "w"

) as f:

    json.dump(

        search.best_params_,

        f,

        indent=4

    )

pd.DataFrame(

    search.cv_results_

).to_csv(

    "results/tuning/tuning_results.csv",

    index=False

)

print()

print("Best Model Saved!")

print("best_fusion_model.pkl")

print("best_parameters.json")

print("tuning_results.csv")

print()

print("STATUS : SUCCESS")