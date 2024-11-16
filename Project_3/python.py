# Note this file is not submitted for evaluation its just for referene=ce purpose.

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
def load_data(file_path, col_name="new_deaths"):
    data = pd.read_csv(file_path)
    y = data[col_name].values[:116]  # Limiting to 116 entries as in Scala code
    return y

# Random Walk Model
def random_walk_model(y, horizon):
    predictions = [y[t - horizon] for t in range(horizon, len(y))]
    return predictions

# Auto-Regressive Model
def ar_model(y, p, train_size, test_size):
    model = ARIMA(y[:train_size], order=(p, 0, 0))  # AR(p) model
    model_fit = model.fit()
    in_sample_forecast = model_fit.fittedvalues  # In-sample predictions
    test_forecast = model_fit.forecast(steps=test_size)  # Out-of-sample (TnT) predictions
    return in_sample_forecast, test_forecast

# Compute Quality of Fit Metrics
def compute_metrics(y_true, y_pred, sst=None):
    sse = np.sum((y_true - y_pred) ** 2)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r2_bar = 1 - ((1 - r2) * (len(y_true) - 1) / (len(y_true) - 2)) if len(y_true) > 2 else r2
    sde = np.std(y_true - y_pred)
    aic = len(y_true) * np.log(sse / len(y_true)) + 2 * 2  # Simplified AIC for AR(p)
    bic = len(y_true) * np.log(sse / len(y_true)) + 2 * np.log(len(y_true))  # Simplified BIC
    return {
        "rSq": r2,
        "rSqBar": r2_bar,
        "sst": sst if sst else np.var(y_true) * len(y_true),
        "sse": sse,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "sde": sde,
        "aic": aic,
        "bic": bic,
    }

# Evaluate Model
def evaluate_model(y, model_type="RW", horizon=6, p=None):
    train_size = len(y) - horizon
    test_size = horizon
    train, test = y[:train_size], y[train_size:]

    if model_type == "RW":
        test_forecast = random_walk_model(y, horizon)
        train_forecast = y[:train_size - 1]  # Random walk uses previous values
    elif model_type == "AR":
        in_sample_forecast, test_forecast = ar_model(y, p, train_size, test_size)
        train_forecast = in_sample_forecast
    else:
        raise ValueError("Unsupported model type.")

    # Compute SST for metrics
    sst = np.sum((y - np.mean(y)) ** 2)

    # Metrics for in-sample and train-test forecasts
    is_metrics = compute_metrics(train[1:], train_forecast, sst)
    tnt_metrics = compute_metrics(test, test_forecast, sst)

    return is_metrics, tnt_metrics

# Main Function
if __name__ == "__main__":
    file_path = "/content/sample_data/covid_19_weekly.csv"
    y = load_data(file_path)
    horizon = 6

    # Random Walk Model Evaluation
    print("Random Walk Model")
    rw_is_metrics, rw_tnt_metrics = evaluate_model(y, model_type="RW", horizon=horizon)
    print("In-Sample Metrics:", rw_is_metrics)
    print("Train-and-Test Metrics:", rw_tnt_metrics)

    # Auto-Regressive (AR(1)) Model Evaluation
    print("\nAuto-Regressive AR(1) Model")
    ar1_is_metrics, ar1_tnt_metrics = evaluate_model(y, model_type="AR", horizon=horizon, p=1)
    print("In-Sample Metrics:", ar1_is_metrics)
    print("Train-and-Test Metrics:", ar1_tnt_metrics)

    # Auto-Regressive (AR(2)) Model Evaluation
    print("\nAuto-Regressive AR(2) Model")
    ar2_is_metrics, ar2_tnt_metrics = evaluate_model(y, model_type="AR", horizon=horizon, p=2)
    print("In-Sample Metrics:", ar2_is_metrics)
    print("Train-and-Test Metrics:", ar2_tnt_metrics)
