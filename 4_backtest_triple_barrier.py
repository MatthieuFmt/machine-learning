import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('./results', exist_ok=True)

# --- PARAMÈTRE GLOBAL ---
ANNEE_TEST = 2025

# 1. Chargement des données
filepath = f'./results/Predictions_{ANNEE_TEST}_TripleBarrier.csv'
preds = pd.read_csv(filepath, index_col='Time', parse_dates=True)
prices = pd.read_csv('./cleaned-data/EURUSD_H1_cleaned.csv', index_col='Time', parse_dates=True)

# Charger le dataset maître pour récupérer les features et les colonnes
dataset_ml = pd.read_csv('./ready-data/EURUSD_Master_ML_Ready.csv', index_col='Time', parse_dates=True)
X_cols = [c for c in dataset_ml.columns if c not in ['Target', 'Spread']]

# Fusion pour avoir les High/Low futurs ET les features
df = preds.join(prices[['High', 'Low', 'Close']], how='inner')
df = df.join(dataset_ml[X_cols], how='inner')

# 2. Paramètres
TP_PIPS = 20.0
SL_PIPS = 10.0
SEUIL_CONFIANCE = 45.0      # en pourcentage
WINDOW = 24
PIP_SIZE = 0.0001

# 3. Génération des signaux + poids dynamique
df['proba_max'] = df[['Confiance_Hausse_%', 'Confiance_Neutre_%', 'Confiance_Baisse_%']].max(axis=1) / 100.0
df['Signal'] = 0
df['Weight'] = 0.0

mask_long = (df['Prediction_Modele'] == 1) & (df['Confiance_Hausse_%'] / 100 >= SEUIL_CONFIANCE / 100)
mask_short = (df['Prediction_Modele'] == -1) & (df['Confiance_Baisse_%'] / 100 >= SEUIL_CONFIANCE / 100)

df.loc[mask_long, 'Signal'] = 1
df.loc[mask_short, 'Signal'] = -1

# Poids entre 0.5 et 1.5, proportionnel à l'excédent de confiance au-dessus du seuil
excess = (df['proba_max'] - SEUIL_CONFIANCE / 100) / 0.15
excess = excess.clip(0, 1)
df.loc[df['Signal'] != 0, 'Weight'] = 0.5 + excess

# 4. Simulation Stateful
dates = df.index
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
signals = df['Signal'].values
weights = df['Weight'].values
spreads = df['Spread'].values

features_arr = df[X_cols].values
proba_hausse_arr = df['Confiance_Hausse_%'].values / 100.0
proba_neutre_arr = df['Confiance_Neutre_%'].values / 100.0
proba_baisse_arr = df['Confiance_Baisse_%'].values / 100.0

trade_records = []
i = 0

print(f"Simulation Stateful en cours pour {ANNEE_TEST} (Recherche des exits exacts)...")

while i < len(df):
    if signals[i] != 0:
        entry_time = dates[i]
        entry_price = closes[i]
        signal = signals[i]
        spread_cost = spreads[i] / 10.0
        weight = weights[i]

        entry_features = dict(zip(X_cols, features_arr[i]))
        entry_probas = {
            'proba_hausse': proba_hausse_arr[i],
            'proba_neutre': proba_neutre_arr[i],
            'proba_baisse': proba_baisse_arr[i]
        }

        if signal == 1:
            tp_price = entry_price + (TP_PIPS * PIP_SIZE)
            sl_price = entry_price - (SL_PIPS * PIP_SIZE)
        else:
            tp_price = entry_price - (TP_PIPS * PIP_SIZE)
            sl_price = entry_price + (SL_PIPS * PIP_SIZE)

        pips_bruts = -SL_PIPS - spread_cost   # par défaut perte max
        result_type = 'loss_sl'

        for j in range(1, WINDOW + 1):
            if i + j >= len(df):
                i = len(df)
                break

            curr_high = highs[i + j]
            curr_low = lows[i + j]

            if signal == 1:
                if curr_low <= sl_price:
                    pips_bruts = -SL_PIPS - spread_cost
                    result_type = 'loss_sl'
                    i = i + j
                    break
                elif curr_high >= tp_price:
                    pips_bruts = TP_PIPS - spread_cost
                    result_type = 'win'
                    i = i + j
                    break
            else:
                if curr_high >= sl_price:
                    pips_bruts = -SL_PIPS - spread_cost
                    result_type = 'loss_sl'
                    i = i + j
                    break
                elif curr_low <= tp_price:
                    pips_bruts = TP_PIPS - spread_cost
                    result_type = 'win'
                    i = i + j
                    break
        else:
            i = i + WINDOW
            result_type = 'loss_timeout'

        # Résultat pondéré
        pips_ponderes = pips_bruts * weight

        trade_records.append({
            'Time': entry_time,
            'Signal': signal,
            'Pips_Nets': pips_ponderes,        # pour les métriques de performance
            'Pips_Bruts': pips_bruts,          # pour analyse
            'Weight': weight,
            'result': result_type,
            **entry_features,
            **entry_probas
        })
        continue

    i += 1

# 5. Calcul des métriques finales
trades_df = pd.DataFrame(trade_records)

if not trades_df.empty:
    trades_df.set_index('Time', inplace=True)
    trades_df.to_csv(f'./results/Trades_Detailed_{ANNEE_TEST}.csv')
    print(f"💾 Trades détaillés sauvegardés dans ./results/Trades_Detailed_{ANNEE_TEST}.csv")

    nb_trades = len(trades_df)
    # Win rate basé sur les pips bruts pour rester comparable à l'ancienne version
    trades_gagnants = trades_df[trades_df['Pips_Bruts'] > 0]
    win_rate = (len(trades_gagnants) / nb_trades) * 100

    total_pips = trades_df['Pips_Nets'].sum()          # somme des pips pondérés
    expectancy = total_pips / nb_trades

    trades_df['Cumulative_Pips'] = trades_df['Pips_Nets'].cumsum()
    trades_df['High_Water_Mark'] = trades_df['Cumulative_Pips'].cummax()
    trades_df['Drawdown'] = trades_df['Cumulative_Pips'] - trades_df['High_Water_Mark']
    max_drawdown = trades_df['Drawdown'].min()

    # Ratio de Sharpe approximatif (quotidien) – optionnel, mais utile pour comparer
    daily_pnl = trades_df.resample('D')['Pips_Nets'].sum().dropna()
    sharpe_ratio = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if len(daily_pnl) > 1 and daily_pnl.std() != 0 else 0

    print("\n" + "="*60)
    print(f" 📊 BACKTEST FINANCIER - STATEFUL ({ANNEE_TEST}) 📊")
    print("="*60)
    print(f"Stratégie            : TP {TP_PIPS} pips | SL {SL_PIPS} pips")
    print(f"Filtre Confiance     : > {SEUIL_CONFIANCE}%")
    print(f"Position sizing      : dynamique (0.5x → 1.5x)")
    print(f"Nombre de Trades     : {nb_trades}")
    print(f"Taux de réussite     : {win_rate:.2f}% (sur pips bruts)")
    print(f"Espérance / trade    : {expectancy:.2f} pips (pondéré)")
    print(f"Max Drawdown         : {max_drawdown:.1f} Pips")
    print(f"RÉSULTAT NET         : {total_pips:.1f} Pips (pondéré)")
    print(f"Ratio de Sharpe      : {sharpe_ratio:.2f}")
    print("="*60)

    # Graphique
    plt.figure(figsize=(12, 6))
    plt.plot(trades_df.index, trades_df['Cumulative_Pips'], color='blue', label='Capital (Pips pondérés)')
    plt.fill_between(trades_df.index, trades_df['Cumulative_Pips'], trades_df['High_Water_Mark'],
                     color='red', alpha=0.3, label='Drawdown')
    plt.title(f"Courbe d'équité - Sizing dynamique - {nb_trades} Trades (Seuil {SEUIL_CONFIANCE}%) - {ANNEE_TEST}")
    plt.ylabel('Pips Cumulés')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./results/Equity_Curve_Dynamic_{ANNEE_TEST}.png')

    # Rapport markdown
    md_content = f"""# 📈 Rapport de Performance Réel (Stateful)
**Version :** Triple Barrière V3 + Sizing dynamique
**Période :** {ANNEE_TEST}

---

## 📊 Statistiques Globales
| Métrique | Valeur |
| :--- | :--- |
| **Nombre de Trades** | {nb_trades} |
| **Taux de Réussite** | {win_rate:.2f}% (brut) |
| **Résultat Net (pondéré)** | **{total_pips:.1f} Pips** |
| **Max Drawdown** | {max_drawdown:.1f} Pips |
| **Espérance / trade** | {expectancy:.2f} pips (pondéré) |
| **Ratio de Sharpe** | {sharpe_ratio:.2f} |

---
*Généré par le backtester Stateful - Zéro Pyramidation autorisée.*
"""
    with open(f'./results/Rapport_Performance_Dynamic_{ANNEE_TEST}.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
else:
    print(f"Aucun trade pris avec un seuil de confiance de {SEUIL_CONFIANCE}% sur l'année {ANNEE_TEST}.")