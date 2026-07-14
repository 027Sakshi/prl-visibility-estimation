from xgboost import XGBRegressor


def create_weather_model():
    """
    Creates the Weather Baseline Model.
    """

    model = XGBRegressor(

        n_estimators=100,

        max_depth=4,

        learning_rate=0.1,

        random_state=42

    )

    return model