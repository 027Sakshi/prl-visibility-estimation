from xgboost import XGBRegressor


def create_fusion_model():

    model = XGBRegressor(

        n_estimators=200,

        max_depth=6,

        learning_rate=0.05,

        random_state=42

    )

    return model