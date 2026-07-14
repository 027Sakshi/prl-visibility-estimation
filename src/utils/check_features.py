import pandas as pd

df = pd.read_csv(
    "data/processed/all_embeddings.csv"
)

print(df.shape)