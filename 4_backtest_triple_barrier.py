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
from config import (
    RANDOM_SEED,
    SEUIL_CONFIANCE,
    TP_PIPS,
    SL_PIPS,
    TP_SL_VARIANTS,
    SEUILS_ALTERNATIFS,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Priorité 3 : sizing centré sur SEUIL_CONFIANCE (0.38 au lieu de 0.45)
WEIGHT_FUNC = lambda proba: np.clip(0.8 + 0.4 * ((proba - SEUIL_CONFIANCE) / 0.10), 0.8, 1.2)
ANNEES = [2022, 2023, 2024, 2025]


def run_backtest_for_config(annees, tp, sl, seuil, label=""):
    """Exécute un backtest complet pour une configuration TP/SL/seuil donnée.

    Retourne un DataFrame de synthèse et imprime les résultats.
    """
    config_tag = f"TP={tp:.0f}p/SL={sl:.0f}p/seuil={seuil:.2f}"
    if label:
        config_tag = f"{label} ({config_tag})"
    print(f"\n{'='*60}")
    print(f"🔧 Configuration : {config_tag}")
    print(f"{'='*60}")

    # Sizing centré sur le seuil courant
    wf = lambda proba: np.clip(0.8 + 0.4 * ((proba - seuil) / 0.10), 0.8, 1.2)

    results = []
    for an in annees:
        df = load_backtest_inputs(an)
        if df is None:
            print(f"⚠️  Fichier manquant pour {an}, on le saute.")
            continue
        trades_df, n_signaux, n_filtres = simulate_trades(
            df, wf, tp_pips=tp, sl_pips=sl, seuil_confiance=seuil,
        )
        res = compute_metrics(trades_df, annee=an, df=df)
        res['n_signaux'] = n_signaux
        res['filtres'] = n_filtres
        results.append(res)
        save_trades_detailed(trades_df, an, df=df)

        # Afficher stats filtres
        filtre_str = ", ".join(f"{k}={v}" for k, v in n_filtres.items() if v > 0)
        print(
            f"{an} | Profit={res['profit_net']:8.1f}p ({res['total_return_pct']:+5.1f}%) | "
            f"DD={res['dd']:6.1f}p | WR={res['win_rate']:5.1f}% | "
            f"Trades={res['trades']:4d} (sig={n_signaux:4d}) | "
            f"Sharpe={res['sharpe']:.2f} | Strade={res['sharpe_per_trade']:.2f} | "
            f"B&H={res['bh_pips']:+7.1f}p | Alpha={res['alpha_pips']:+7.1f}p"
            f"{' | Filtres: '+filtre_str if filtre_str else ''}"
        )

    if results:
        df_res = pd.DataFrame(results).set_index('annee')
        print(f"\n=== SYNTHÈSE {config_tag} ===")
        cols_show = ['profit_net', 'win_rate', 'trades', 'n_signaux', 'sharpe', 'alpha_pips']
        print(df_res[cols_show].to_string())
        total_profit = df_res['profit_net'].sum()
        print(f"Profit total sur les années testées : {total_profit:.0f} pips")

    return results


# ================================================================
# PHASE 1 : Baseline avec la configuration principale (TP/SL config)
# ================================================================
print("🚀 PHASE 1 : Configuration principale")
results_baseline = run_backtest_for_config(
    ANNEES, TP_PIPS, SL_PIPS, SEUIL_CONFIANCE, label="baseline"
)

# Sauvegarde des rapports pour la baseline
for res in results_baseline:
    an = res['annee']
    n_signaux = res.get('n_signaux')
    save_report_md(res, an, n_signaux=n_signaux)

# ================================================================
# PHASE 2 : Tests des variantes TP/SL (Priorité 5)
# ================================================================
print("\n\n🚀 PHASE 2 : Tests des variantes TP/SL (Priorité 5)")
all_config_results = {}

for label, (tp, sl) in TP_SL_VARIANTS.items():
    if label == 'baseline':
        continue  # déjà fait en phase 1
    results_cfg = run_backtest_for_config(
        ANNEES, tp, sl, SEUIL_CONFIANCE, label=label
    )
    all_config_results[label] = results_cfg
    # Sauvegarde dans un sous-dossier par variante
    for res in results_cfg:
        an = res['annee']
        n_signaux = res.get('n_signaux')
        save_report_md(res, an, version=label, n_signaux=n_signaux)

# ================================================================
# PHASE 3 : Comparatif final
# ================================================================
print("\n\n" + "=" * 80)
print("📊 COMPARATIF MULTI-CONFIGURATIONS")
print("=" * 80)

# Construire un tableau comparatif sur 2025 (année test)
print("\nSur 2025 (OOS) :")
print(f"{'Configuration':<30} {'Profit':>10} {'WR':>8} {'Trades':>8} {'Sharpe':>8} {'Alpha':>10}")
print("-" * 75)

# Baseline
for res in results_baseline:
    if res['annee'] == 2025:
        print(f"{'baseline':<30} {res['profit_net']:+9.1f}p {res['win_rate']:7.1f}% {res['trades']:7d} {res['sharpe']:7.2f} {res['alpha_pips']:+9.1f}p")

for label, results_cfg in all_config_results.items():
    for res in results_cfg:
        if res['annee'] == 2025:
            print(f"{label:<30} {res['profit_net']:+9.1f}p {res['win_rate']:7.1f}% {res['trades']:7d} {res['sharpe']:7.2f} {res['alpha_pips']:+9.1f}p")

print("\n✅ Backtest multi-configurations terminé.")
print("   Rapports sauvegardés dans predictions/ et predictions/<variante>/")
