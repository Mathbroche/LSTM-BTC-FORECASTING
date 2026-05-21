# main.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

from config  import *       
from dataset import create_dataloaders
from model   import BTCLSTMModel
from train   import train, plot_losses
from evaluate import evaluate, compute_metrics, plot_predictions, plot_zoom

# ================================================================
# 1. DONNÉES
# ================================================================
print("=" * 50)
print("① Chargement des données")
print("=" * 50)

df    = yf.download(TICKER, period=PERIOD)
close = df["Close"].squeeze()

print(f"Ticker  : {TICKER}")
print(f"Période : {PERIOD}")
print(f"Jours   : {len(close)}")
print(f"Min     : {close.min():.0f}$")
print(f"Max     : {close.max():.0f}$")

# ================================================================
# 2. NORMALISATION
# ================================================================
print("\n" + "=" * 50)
print("② Normalisation")
print("=" * 50)

close_values = close.values.reshape(-1, 1)
scaler       = MinMaxScaler()
close_scaled = scaler.fit_transform(close_values)

print(f"Avant : min={close_values.min():.0f}$  max={close_values.max():.0f}$")
print(f"Après : min={close_scaled.min():.4f}   max={close_scaled.max():.4f}")

# ================================================================
# 3. SÉQUENCES
# ================================================================
print("\n" + "=" * 50)
print("③ Création des séquences")
print("=" * 50)

X, Y = [], []
for i in range(len(close_scaled) - WINDOW_SIZE):
    X.append(close_scaled[i : i + WINDOW_SIZE])
    Y.append(close_scaled[i + WINDOW_SIZE])

X = np.array(X)
Y = np.array(Y)

split   = int(len(X) * TRAIN_RATIO)
X_train = X[:split] ; Y_train = Y[:split]
X_test  = X[split:] ; Y_test  = Y[split:]

print(f"Window size     : {WINDOW_SIZE} jours")
print(f"Total séquences : {len(X)}")
print(f"Train           : {len(X_train)}")
print(f"Test            : {len(X_test)}")

# ================================================================
# 4. DATALOADERS
# ================================================================
print("\n" + "=" * 50)
print("④ Création des DataLoaders")
print("=" * 50)

train_loader, test_loader = create_dataloaders(
    X_train, Y_train,
    X_test,  Y_test,
    batch_size=BATCH_SIZE
)

print(f"Batch size      : {BATCH_SIZE}")
print(f"Batches train   : {len(train_loader)}")
print(f"Batches test    : {len(test_loader)}")

# ================================================================
# 5. MODÈLE
# ================================================================
print("\n" + "=" * 50)
print("⑤ Initialisation du modèle")
print("=" * 50)

model = BTCLSTMModel(
    input_size  = INPUT_SIZE,
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Architecture    : LSTM({INPUT_SIZE}→{HIDDEN_SIZE}) + FC({HIDDEN_SIZE}→32→1)")
print(f"Paramètres      : {n_params:,}")
print(f"Dropout         : {DROPOUT}")

# ================================================================
# 6. ENTRAÎNEMENT
# ================================================================
print("\n" + "=" * 50)
print("⑥ Entraînement")
print("=" * 50)

train_losses, test_losses = train(
    model         = model,
    train_loader  = train_loader,
    test_loader   = test_loader,
    epochs        = EPOCHS,
    learning_rate = LEARNING_RATE
)

plot_losses(train_losses, test_losses)

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModèle sauvegardé → {MODEL_PATH}")

# ================================================================
# 7. ÉVALUATION
# ================================================================
print("\n" + "=" * 50)
print("⑦ Évaluation")
print("=" * 50)

predictions, actuals = evaluate(model, test_loader, scaler)
rmse, mae, da        = compute_metrics(predictions, actuals)

plot_predictions(predictions, actuals)
plot_zoom(predictions, actuals, days=60)

# ================================================================
# 8. RÉSUMÉ FINAL
# ================================================================
print("\n" + "=" * 50)
print("⑧ Résumé")
print("=" * 50)

print(f"Ticker          : {TICKER}")
print(f"Window          : {WINDOW_SIZE} jours")
print(f"Epochs          : {EPOCHS}")
print(f"RMSE            : {rmse:,.0f}$")
print(f"MAE             : {mae:,.0f}$")
print(f"DA              : {da:.1f}%")
print(f"Modèle          : {MODEL_PATH}")