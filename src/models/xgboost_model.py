from xgboost import XGBRegressor

def create_xgb_model():

    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    return model