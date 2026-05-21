# BTC-LSTM Forecasting

Prédiction du prix Bitcoin via un LSTM entraîné from scratch en PyTorch.

## Résultats
| Métrique | Valeur |
|----------|--------|
| RMSE     | 3 020$ |
| MAE      | 2 387$ |
| DA       | 48.7%  |

## Ce que j'ai appris
- Pipeline ML complet from scratch
- Pourquoi un bon RMSE ≠ bon modèle
- Marchés efficients et limites de la prédiction

## Installation
```bash
git clone https://github.com/Mathbroche/LSTM-BTC-FORECASTING.git
cd LSTM-BTC-FORECASTING
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Architecture
LSTM(1→64, 2 couches) + Linear(64→32) + ReLU + Linear(32→1)
