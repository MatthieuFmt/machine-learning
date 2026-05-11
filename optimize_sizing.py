"""Sélection de la fonction de sizing en split 3-étages strict (audit I5) :

  Train ≤ TRAIN_END_YEAR (avec embargo PURGE_HOURS) → 3_model_training.py
  Sélection de la pondération sur VAL_YEAR
  Validation finale sur TEST_YEAR (jamais vu par le modèle ni par le sélecteur)

Les prédictions VAL_YEAR et TEST_YEAR doivent provenir du même run de
3_model_training.py — sinon le split est compromis.
"""
import random

import numpy as np

from backtest_utils import compute_metrics, load_backtest_inputs, simulate_trades
from config import RANDOM_SEED, TEST_YEAR, VAL_YEAR

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def run_backtest(annee, weight_func):
    df = load_backtest_inputs(annee)
    if df is None:
        return None
    trades_df, n_signaux, _ = simulate_trades(df, weight_func)
    metrics = compute_metrics(trades_df, annee=annee, df=df)
    metrics['n_signaux'] = n_signaux
    return metrics


# === Fonctions de poids candidates ===
def weight_linear(proba, seuil=0.45):
    excess = (proba - seuil) / 0.15
    excess = np.clip(excess, 0, 1)
    return 0.5 + excess


def weight_linear_v2(proba, seuil=0.45):
    # Pente plus douce, entre 0.8 et 1.2
    return 0.8 + 0.8 * np.clip((proba - seuil) / 0.10, 0, 1)


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
    'linear_0.5-1.5': weight_linear,
    'linear_0.8-1.2': weight_linear_v2,
    'exp_0.5-1.5': weight_exp,
    'step': weight_step,
}

# Phase 1 : Sélection sur VAL_YEAR
print(f"=== Phase 1 : sélection de la fonction de poids sur {VAL_YEAR} ===")
results = {}
for name, func in candidates.items():
    res = run_backtest(VAL_YEAR, func)
    if res is None:
        print(f"⚠️  Fichier manquant pour {VAL_YEAR}, on saute {name}.")
        continue
    results[name] = res
    print(f"{name}: Profit={res['profit_net']:.0f}pips, DD={res['dd']:.1f}, WR={res['win_rate']:.1f}%")

if results:
    best_name = max(results, key=lambda n: results[n]['profit_net'])
    print(f"\nMeilleure fonction sur {VAL_YEAR} : {best_name}")

    # Phase 2 : Validation finale sur TEST_YEAR
    print(f"\n=== Phase 2 : validation finale sur {TEST_YEAR} avec {best_name} ===")
    best_func = candidates[best_name]
    final_res = run_backtest(TEST_YEAR, best_func)
    if final_res is None:
        print(f"⚠️  Fichier manquant pour {TEST_YEAR}, impossible de finaliser.")
    else:
        print(f"Profit Net  : {final_res['profit_net']:.1f} pips ({final_res['total_return_pct']:+.2f}%)")
        print(f"Max Drawdown: {final_res['dd']:.1f} pips ({final_res['max_dd_pct']:.2f}%)")
        print(f"Win Rate    : {final_res['win_rate']:.1f}%")
        print(f"Trades      : {final_res['trades']} (signaux={final_res['n_signaux']})")
        print(f"Sharpe      : {final_res['sharpe']:.2f}")
        print(f"Buy & Hold  : {final_res['bh_pips']:+.1f} pips ({final_res['bh_return_pct']:+.2f}%)")
        print(f"Alpha       : {final_res['alpha_pips']:+.1f} pips ({final_res['alpha_return_pct']:+.2f}%)")
