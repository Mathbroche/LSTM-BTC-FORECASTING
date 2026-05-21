# src/train.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def train(model, train_loader, test_loader, epochs=50, learning_rate=0.001):

    # Loss et optimiseur
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Historique des losses
    train_losses = []
    test_losses  = []

    for epoch in range(epochs):

        # ── MODE ENTRAÎNEMENT ──────────────────────────
        model.train()
        batch_losses = []

        for X_batch, Y_batch in train_loader:

            # 1. Forward pass
            predictions = model(X_batch)

            # 2. Calcul de la loss
            loss = criterion(predictions, Y_batch)

            # 3. Reset des gradients
            optimizer.zero_grad()

            # 4. Backward pass
            loss.backward()

            # 5. Mise à jour des poids
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # ── MODE ÉVALUATION ───────────────────────────
        model.eval()
        batch_losses_test = []

        with torch.no_grad():   # pas de gradient en évaluation
            for X_batch, Y_batch in test_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, Y_batch)
                batch_losses_test.append(loss.item())

        test_loss = np.mean(batch_losses_test)
        test_losses.append(test_loss)

        # Affiche toutes les 10 époques
        if (epoch + 1) % 10 == 0:
            print(f"Époque {epoch+1:>3}/{epochs} "
                  f"| Train Loss : {train_loss:.6f} "
                  f"| Test Loss  : {test_loss:.6f}")

    return train_losses, test_losses


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(12, 4))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(test_losses,  label="Test Loss",  color="orange")
    plt.title("Évolution de la Loss")
    plt.xlabel("Époque")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()