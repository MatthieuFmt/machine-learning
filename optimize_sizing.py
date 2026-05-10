import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta

# === PARTIE RÉUTILISABLE : fonction de backtest ===
def run_backtest(ANNEE_TEST, weight_func, SEUIL_CONFIANCE=45.0):
    """
    weight_func(proba_max) -> float (poids entre 0 et 2 conseillé)
    """
    # Chargement identique...
    preds = pd.read_csv(f'./results/Predictions_{ANNEE_TEST}_TripleBarrier.csv', index_col='Time', parse_dates=True)
    prices = pd.read_csv('./cleaned-data/EURUSD_H1_cleaned.csv', index_col='Time', parse_dates=True)
    dataset_ml = pd.read_csv('./ready-data/EURUSD_Master_ML_Ready.csv', index_col='Time', parse_dates=True)
    X_cols = [c for c in dataset_ml.columns if c not in ['Target', 'Spread']]
    
    df = preds.join(prices[['High','Low','Close']], how='inner')
    df = df.join(dataset_ml[X_cols], how='inner')
    
    TP_PIPS=20; SL_PIPS=10; WINDOW=24; PIP_SIZE=0.0001
    df['proba_max'] = df[['Confiance_Hausse_%','Confiance_Neutre_%','Confiance_Baisse_%']].max(axis=1)/100
    df['Signal'] = 0
    df['Weight'] = 0.0
    
    mask_long = (df['Prediction_Modele']==1) & (df['Confiance_Hausse_%']/100 >= SEUIL_CONFIANCE/100)
    mask_short = (df['Prediction_Modele']==-1) & (df['Confiance_Baisse_%']/100 >= SEUIL_CONFIANCE/100)
    df.loc[mask_long,'Signal'] = 1
    df.loc[mask_short,'Signal'] = -1
    
    # Application de la fonction de poids
    df.loc[df['Signal']!=0, 'Weight'] = weight_func(df.loc[df['Signal']!=0, 'proba_max'])
    
    # Simulation... (identique, je résume pour ne pas tout recopier)
    # ...
    # Retourne total_pips, max_drawdown, nb_trades, win_rate, etc.
    return {'profit_net': total_pips, 'dd': max_drawdown, 'trades': nb_trades, 'win_rate': win_rate}

# === Définition de plusieurs fonctions candidates ===
def weight_linear(proba, seuil=0.45):
    excess = (proba - seuil) / 0.15
    excess = np.clip(excess, 0, 1)
    return 0.5 + excess

def weight_linear_v2(proba, seuil=0.45):
    # Pente plus douce, entre 0.8 et 1.2
    return 0.8 + 0.8 * np.clip((proba - seuil)/0.10, 0, 1)

def weight_exp(proba, seuil=0.45):
    # Accélération pour hautes confiances
    z = (proba - seuil) / 0.15
    z = np.clip(z, 0, 1)
    return 0.5 + z**2

def weight_step(proba, seuil=0.45):
    # Trois paliers : 0.5, 1.0, 1.5
    cond1 = proba < 0.50
    cond2 = (proba >= 0.50) & (proba < 0.55)
    cond3 = proba >= 0.55
    weights = np.zeros_like(proba)
    weights[cond1] = 0.5
    weights[cond2] = 1.0
    weights[cond3] = 1.5
    return weights

candidates = {
    'linear_0.5-1.5': lambda p: weight_linear(p),
    'linear_0.8-1.2': lambda p: weight_linear_v2(p),
    'exp_0.5-1.5': lambda p: weight_exp(p),
    'step': lambda p: weight_step(p)
}

# Phase 1 : Évaluation sur 2024
ANNEE_VAL = 2024
results = {}
for name, func in candidates.items():
    res = run_backtest(ANNEE_VAL, func)
    results[name] = res
    print(f"{name}: Profit={res['profit_net']:.0f}pips, DD={res['dd']:.1f}, WR={res['win_rate']:.1f}%")

best_name = max(results, key=lambda n: results[n]['profit_net'])
print(f"\nMeilleure fonction sur {ANNEE_VAL} : {best_name}")

# Phase 2 : Application sur 2025
ANNEE_TEST = 2025
best_func = candidates[best_name]
final_res = run_backtest(ANNEE_TEST, best_func)
print(f"\n=== RÉSULTAT FINAL SUR 2025 AVEC {best_name} ===")
print(f"Profit Net : {final_res['profit_net']:.1f} pips")
print(f"Max Drawdown : {final_res['dd']:.1f} pips")