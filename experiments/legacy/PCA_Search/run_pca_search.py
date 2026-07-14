import os
import pandas as pd

from sklearn.decomposition import PCA

from src.models.xgboost_model import create_xgb_model
from src.evaluation.cross_validation import run_cv

print("=" * 70)
print("PCA COMPONENT SEARCH")
print("=" * 70)

# --------------------------------------------------
# LOAD EMBEDDINGS
# --------------------------------------------------

features = pd.read_csv(
    "data/prl_features/all_embeddings.csv"
)

metadata = pd.read_excel(
    "data/prl/metadata/prl_dataset_clean.xlsx"
)

X = features.drop(
    columns=["image_name"]
)

y = metadata["visibility_km"]

print("\nDataset Loaded")
print("Images :", len(X))
print("Features :", X.shape[1])

# --------------------------------------------------
# PCA VALUES TO TEST
# --------------------------------------------------

components_to_test = [

    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    75,
    90,
    100,
    110,
    120,
    127

]

results = []

# --------------------------------------------------

for n in components_to_test:

    print("\n" + "="*50)
    print(f"Testing PCA Components : {n}")
    print("="*50)

    # ----------------------------
    # PCA
    # ----------------------------

    if n == 768:

        X_pca = X.copy()

    else:

        pca = PCA(

            n_components=n,

            random_state=42

        )

        X_pca = pca.fit_transform(X)

        X_pca = pd.DataFrame(X_pca)

    # ----------------------------
    # MODEL
    # ----------------------------

    model = create_xgb_model()

    metrics = run_cv(

        model,

        X_pca,

        y

    )

    metrics["Components"] = n

    results.append(metrics)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df[

    [

        "Components",

        "Mean_MAE",

        "Mean_RMSE",

        "Mean_R2"

    ]

]

results_df = results_df.sort_values(

    by="Mean_RMSE"

)

os.makedirs(

    "results/PCA_Search",

    exist_ok=True

)

results_df.to_csv(

    "results/PCA_Search/pca_search_results.csv",

    index=False

)

print("\n")
print("="*70)
print("FINAL PCA SEARCH RESULTS")
print("="*70)

print(results_df)

print("\nSaved Successfully!")