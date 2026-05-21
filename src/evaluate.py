# src/evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(model, test_loader, scaler):
    model.eval()
    predictions = []
    actuals     = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            pred = model(X_batch)
            predictions.extend(pred.numpy())
            actuals.extend(Y_batch.numpy())

    predictions = np.array(predictions)
    actuals     = np.array(actuals)

    # Dénormalisation — retour en dollars
    predictions_dollars = scaler.inverse_transform(predictions)
    actuals_dollars     = scaler.inverse_transform(actuals)

    return predictions_dollars, actuals_dollars

def evaluate_returns(model, test_loader, scaler_returns, actuals_close):
    model.eval()
    predictions = []
    actuals     = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            pred = model(X_batch)
            predictions.extend(pred.numpy())
            actuals.extend(Y_batch.numpy())

    predictions = np.array(predictions)
    actuals     = np.array(actuals)

    # Dénormalisation → retour en variations %
    pred_returns = scaler_returns.inverse_transform(predictions)
    real_returns = scaler_returns.inverse_transform(actuals)

    # Directional accuracy — la métrique clé
    pred_direction = pred_returns.flatten() > 0   # prédit hausse ?
    real_direction = real_returns.flatten() > 0   # vraie hausse ?
    da = (pred_direction == real_direction).mean() * 100

    # RMSE sur les variations
    rmse = np.sqrt(mean_squared_error(real_returns, pred_returns))

    print(f"RMSE (variation)    : {rmse:.4f}")
    print(f"Directional Accuracy: {da:.1f}%")
    print(f"Référence aléatoire : 50.0%")

    return pred_returns, real_returns

def compute_metrics(predictions, actuals):
    # RMSE — erreur moyenne en dollars
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # MAE — erreur absolue moyenne en dollars
    mae = mean_absolute_error(actuals, predictions)

    # Directional accuracy — % de bonnes directions
    pred_direction = np.diff(predictions.flatten()) > 0
    real_direction = np.diff(actuals.flatten()) > 0
    da = (pred_direction == real_direction).mean() * 100

    print(f"RMSE               : {rmse:,.0f}$")
    print(f"MAE                : {mae:,.0f}$")
    print(f"Directional Accuracy: {da:.1f}%")
    print(f"Référence aléatoire : 50.0%")

    return rmse, mae, da


def plot_predictions(predictions, actuals):
    plt.figure(figsize=(14, 5))
    plt.plot(actuals,     label="Prix réel",    color="orange", linewidth=1.5)
    plt.plot(predictions, label="Prédiction",   color="blue",   linewidth=1, alpha=0.8)
    plt.title("Bitcoin — Prédiction vs Réalité")
    plt.xlabel("Jours")
    plt.ylabel("Prix USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_zoom(predictions, actuals, days=60):
    plt.figure(figsize=(14, 5))
    plt.plot(actuals[-days:],     label="Prix réel",  color="orange", linewidth=1.5)
    plt.plot(predictions[-days:], label="Prédiction", color="blue",   linewidth=1, alpha=0.8)
    plt.title(f"Zoom — {days} derniers jours")
    plt.xlabel("Jours")
    plt.ylabel("Prix USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()