import pandas as pd
import numpy as np

feature_df = pd.read_csv(
    "data/processed/all_embeddings.csv"
)

number_of_images = len(feature_df)

print("Number of Images:")
print(number_of_images)

np.random.seed(42)

fake_visibility = np.random.uniform(
    low=1,
    high=10,
    size=number_of_images
)

label_df = pd.DataFrame({
    "visibility": fake_visibility
})

label_df.to_csv(
    "data/processed/fake_visibility.csv",
    index=False
)

print(label_df.head())