# src/model.py

import torch
import torch.nn as nn

class BTCLSTMModel(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(BTCLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Couche LSTM
        self.lstm = nn.LSTM(
            input_size  = input_size,   # 1 feature par jour (Close)
            hidden_size = hidden_size,  # 64 valeurs dans l'état caché
            num_layers  = num_layers,   # 2 couches LSTM empilées
            dropout     = dropout,      # régularisation entre les couches
            batch_first = True          # (batch, séquence, features)
        )

        # Couches fully connected
        self.fc1    = nn.Linear(hidden_size, 32)
        self.relu   = nn.ReLU()
        self.fc2    = nn.Linear(32, 1)

    def forward(self, x):
        # x shape : (batch_size, 30, 1)

        # Initialise l'état caché à zéro
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Passe dans le LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out shape : (batch_size, 30, 64)
        # on prend seulement le dernier timestep
        out = out[:, -1, :]
        # out shape : (batch_size, 64)

        # Passe dans les fully connected
        out = self.fc1(out)   # (batch_size, 32)
        out = self.relu(out)  # (batch_size, 32)
        out = self.fc2(out)   # (batch_size, 1)

        return out