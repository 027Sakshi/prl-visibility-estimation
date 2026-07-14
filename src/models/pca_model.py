from xgboost import XGBRegressor


def create_pca_model():

    model = XGBRegressor(

        random_state=42,

        n_estimators=200,

        learning_rate=0.05,

        max_depth=5

    )

    return model