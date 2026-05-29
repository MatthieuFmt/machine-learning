import random

import numpy as np
import pandas as pd

from backtest_utils import (
    compute_metrics,
    load_backtest_inputs,
    save_report_md,
    save_trades_detailed,
    simulate_trades,
)
from config import RANDOM_SEED

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Sizing linéaire borné [0.8, 1.2] en fonction de proba_max (centré à 0.45).
WEIGHT_FUNC = lambda proba: np.clip(0.8 + 0.4 * ((proba - 0.45) / 0.10), 0.8, 1.2)
ANNEES = [2022, 2023, 2024, 2025]

print("Validation multi-années du sizing linéaire 0.8-1.2\n")
results = []
for an in ANNEES:
    df = load_backtest_inputs(an)
    if df is None:
        print(f"⚠️  Fichier manquant pour {an}, on le saute.")
        continue
    trades_df, n_signaux = simulate_trades(df, WEIGHT_FUNC)
    res = compute_metrics(trades_df, annee=an, df=df)
    res['n_signaux'] = n_signaux
    results.append(res)
    save_trades_detailed(trades_df, an, df=df)
    save_report_md(res, an, n_signaux=n_signaux)
    print(
        f"{an} | Profit={res['profit_net']:8.1f}p ({res['total_return_pct']:+5.1f}%) | "
        f"DD={res['dd']:6.1f}p | WR={res['win_rate']:5.1f}% | "
        f"Trades={res['trades']:4d} (sig={n_signaux:4d}) | "
        f"Sharpe={res['sharpe']:.2f} | "
        f"B&H={res['bh_pips']:+7.1f}p | Alpha={res['alpha_pips']:+7.1f}p"
    )

if results:
    df_res = pd.DataFrame(results).set_index('annee')
    print("\n=== SYNTHÈSE ===")
    print(df_res.to_string())
    total_profit = df_res['profit_net'].sum()
    print(f"\nProfit total sur les années testées : {total_profit:.0f} pips")
