import os
import pandas as pd

from sklearn.decomposition import PCA

print("=" * 60)
print("CREATING PCA-15 FUSION DATASET")
print("=" * 60)

# Load image embeddings
features = pd.read_csv(
    "data/prl_features/all_embeddings.csv"
)

# Load metadata
metadata = pd.read_excel(
    "data/prl/metadata/prl_dataset_clean.xlsx"
)

image_names = features["image_name"]

X = features.drop(columns=["image_name"])

# Apply PCA
pca = PCA(
    n_components=15,
    random_state=42
)

X_pca = pca.fit_transform(X)

# Create dataframe
pca_df = pd.DataFrame(
    X_pca,
    columns=[f"pca_{i+1:03d}" for i in range(15)]
)

pca_df.insert(
    0,
    "image_name",
    image_names
)

# Merge
merged = pd.merge(
    metadata,
    pca_df,
    on="image_name"
)

os.makedirs(
    "data/merged_pca15",
    exist_ok=True
)

merged.to_csv(
    "data/merged_pca15/merged_pca15.csv",
    index=False
)

print()
print("Dataset Created Successfully!")
print(merged.shape)