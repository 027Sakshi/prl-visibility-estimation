import pandas as pd

df = pd.read_csv(
    "data/merged/merged_dataset.csv"
)

print("=" * 60)
print("MERGED DATASET VERIFICATION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Images:")
print(df["image_name"].duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast Columns:")
print(df.columns[-10:])