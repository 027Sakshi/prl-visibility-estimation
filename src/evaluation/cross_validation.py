import numpy as np
from sklearn.model_selection import KFold

from src.evaluation.metrics import calculate_metrics


def run_cv(model, X, y, n_splits=5):

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):

        # Split data
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Evaluate
        metrics = calculate_metrics(y_test, preds)

        mae_scores.append(metrics["MAE"])
        rmse_scores.append(metrics["RMSE"])
        r2_scores.append(metrics["R2"])

        print(f"Fold {fold+1} Complete")

    return {
        "Mean_MAE": np.mean(mae_scores),
        "Mean_RMSE": np.mean(rmse_scores),
        "Mean_R2": np.mean(r2_scores),
    }