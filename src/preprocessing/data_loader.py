import pandas as pd


def load_image_features():

    return pd.read_csv(
        "data/processed/all_embeddings.csv"
    )


def load_visibility_labels():

    return pd.read_csv(
        "data/processed/fake_visibility.csv"
    )