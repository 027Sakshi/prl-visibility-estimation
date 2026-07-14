import os
import joblib
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

print("=" * 70)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

features = pd.read_csv(
    "data/prl_features/all_embeddings.csv"
)

metadata = pd.read_excel(
    "data/prl/metadata/prl_dataset_clean.xlsx"
)

X = features.drop(columns=["image_name"])

y = metadata["visibility_km"]

print("\nDataset Loaded")
print("Images :", X.shape[0])
print("Original Features :", X.shape[1])

# --------------------------------------------------
# PCA (BEST FOUND = 15)
# --------------------------------------------------

print("\nApplying PCA (15 Components)...")

pca = PCA(
    n_components=15,
    random_state=42
)

X = pca.fit_transform(X)

print("Reduced Shape :", X.shape)

# --------------------------------------------------
# BASE MODEL
# --------------------------------------------------

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

# --------------------------------------------------
# PARAMETER SEARCH SPACE
# --------------------------------------------------

param_grid = {

    "n_estimators": [
        100,
        200,
        300,
        500
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
        6,
        8
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1,
        0.2
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "colsample_bytree": [
        0.6,
        0.7,
        0.8,
        1.0
    ],

    "min_child_weight": [
        1,
        2,
        3,
        5
    ]

}

# --------------------------------------------------
# RANDOM SEARCH
# --------------------------------------------------

print("\nSearching Best Parameters...\n")

search = RandomizedSearchCV(

    estimator=model,

    param_distributions=param_grid,

    n_iter=40,

    cv=5,

    scoring="neg_root_mean_squared_error",

    random_state=42,

    verbose=2,

    n_jobs=-1

)

search.fit(
    X,
    y
)

# --------------------------------------------------
# BEST MODEL
# --------------------------------------------------

best_model = search.best_estimator_

print("\n")
print("=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(search.best_params_)

print("\nBest CV Score (RMSE):")

print(-search.best_score_)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

os.makedirs(
    "models/optimized",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/optimized/xgboost_pca15.pkl"
)

# --------------------------------------------------
# SAVE PARAMETERS
# --------------------------------------------------

params = pd.DataFrame(

    [search.best_params_]

)

params["Best_RMSE"] = -search.best_score_

os.makedirs(
    "results/optimization",
    exist_ok=True
)

params.to_csv(

    "results/optimization/best_parameters.csv",

    index=False

)

print("\nBest model saved!")

print("Parameters saved!")