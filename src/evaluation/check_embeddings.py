import pandas as pd

df = pd.read_csv(
    "data/prl_features/all_embeddings.csv"
)

print(df.shape)

print(df.isnull().sum().sum())

print(df.head())