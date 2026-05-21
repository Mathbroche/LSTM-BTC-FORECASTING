# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BTCDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # Convertit numpy → tenseurs PyTorch float32
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        # Nombre total d'exemples
        return len(self.X)

    def __getitem__(self, idx):
        # Retourne un exemple (séquence, cible)
        return self.X[idx], self.Y[idx]


def create_dataloaders(X_train, Y_train, X_test, Y_test, batch_size=32):
    train_dataset = BTCDataset(X_train, Y_train)
    test_dataset  = BTCDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False      # False car série temporelle — l'ordre compte
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader