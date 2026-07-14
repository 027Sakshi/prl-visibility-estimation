import os
import pandas as pd

from sklearn.decomposition import PCA

print("=" * 60)
print("APPLYING PCA")
print("=" * 60)

# Load embeddings

df = pd.read_csv(
    "data/prl_features/all_embeddings.csv"
)

print("\nDataset Loaded!")

print(df.shape)

# Separate image names

image_names = df["image_name"]

# Features

X = df.drop(
    columns=["image_name"]
)

print("\nOriginal Features:", X.shape)

# PCA

pca = PCA(
    n_components=100,
    random_state=42
)

X_pca = pca.fit_transform(X)

print("\nReduced Features:", X_pca.shape)

# Create dataframe

pca_df = pd.DataFrame(
    X_pca,
    columns=[
        f"pca_{i+1:03d}"
        for i in range(100)
    ]
)

pca_df.insert(
    0,
    "image_name",
    image_names
)

os.makedirs(
    "data/prl_pca",
    exist_ok=True
)

pca_df.to_csv(
    "data/prl_pca/pca_features.csv",
    index=False
)

print("\nSaved Successfully!")