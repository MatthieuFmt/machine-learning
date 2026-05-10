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

# MODIF : charger le dataset maître pour récupérer les features et les colonnes
dataset_ml = pd.read_csv('./ready-data/EURUSD_Master_ML_Ready.csv', index_col='Time', parse_dates=True)
X_cols = [c for c in dataset_ml.columns if c not in ['Target', 'Spread']]

# Fusion pour avoir les High/Low futurs ET les features
df = preds.join(prices[['High', 'Low', 'Close']], how='inner')
df = df.join(dataset_ml[X_cols], how='inner')   # MODIF : ajout des features

# 2. Paramètres
TP_PIPS = 20.0
SL_PIPS = 10.0
SEUIL_CONFIANCE = 45.0  
WINDOW = 24
PIP_SIZE = 0.0001

# 3. Génération des signaux
df['Signal'] = 0
df.loc[(df['Prediction_Modele'] == 1) & (df['Confiance_Hausse_%'] >= SEUIL_CONFIANCE), 'Signal'] = 1
df.loc[(df['Prediction_Modele'] == -1) & (df['Confiance_Baisse_%'] >= SEUIL_CONFIANCE), 'Signal'] = -1

# 4. Simulation Stateful (Interdiction de pyramider)
dates = df.index
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
signals = df['Signal'].values
spreads = df['Spread'].values
# MODIF : on convertit aussi les features et probas en arrays pour accès rapide
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

        # MODIF : capturer les features et probas à l'instant d'entrée
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
            
        pips_nets = -SL_PIPS - spread_cost 
        result_type = 'loss_sl'  # par défaut, sauf si timeout ou win
        
        for j in range(1, WINDOW + 1):
            if i + j >= len(df):
                i = len(df)
                break
                
            curr_high = highs[i + j]
            curr_low = lows[i + j]
            
            if signal == 1: 
                if curr_low <= sl_price:
                    pips_nets = -SL_PIPS - spread_cost
                    result_type = 'loss_sl'
                    i = i + j 
                    break
                elif curr_high >= tp_price:
                    pips_nets = TP_PIPS - spread_cost
                    result_type = 'win'
                    i = i + j
                    break
            else: 
                if curr_high >= sl_price:
                    pips_nets = -SL_PIPS - spread_cost
                    result_type = 'loss_sl'
                    i = i + j
                    break
                elif curr_low <= tp_price:
                    pips_nets = TP_PIPS - spread_cost
                    result_type = 'win'
                    i = i + j
                    break
        else:
            # timeout
            i = i + WINDOW
            result_type = 'loss_timeout'
            # pips_nets reste à -SL_PIPS (pire cas) – tu peux aussi calculer le P&L réel à la bougie i+WINDOW
            # Pour l'analyse on garde la valeur actuelle

        # MODIF : enregistrement complet
        trade_records.append({
            'Time': entry_time,
            'Signal': signal,
            'Pips_Nets': pips_nets,
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
    # MODIF : Sauvegarde détaillée pour l'analyse des pertes
    trades_df.to_csv(f'./results/Trades_Detailed_{ANNEE_TEST}.csv')
    print(f"💾 Trades détaillés sauvegardés dans ./results/Trades_Detailed_{ANNEE_TEST}.csv")
    
    nb_trades = len(trades_df)
    trades_gagnants = trades_df[trades_df['Pips_Nets'] > 0]
    win_rate = (len(trades_gagnants) / nb_trades) * 100
    total_pips = trades_df['Pips_Nets'].sum()
    expectancy = total_pips / nb_trades
    
    trades_df['Cumulative_Pips'] = trades_df['Pips_Nets'].cumsum()
    trades_df['High_Water_Mark'] = trades_df['Cumulative_Pips'].cummax()
    trades_df['Drawdown'] = trades_df['Cumulative_Pips'] - trades_df['High_Water_Mark']
    max_drawdown = trades_df['Drawdown'].min() 
    
    print("\n" + "="*55)
    print(f" 📊 BACKTEST FINANCIER - STATEFUL ({ANNEE_TEST}) 📊")
    print("="*55)
    print(f"Stratégie          : TP {TP_PIPS} pips | SL {SL_PIPS} pips")
    print(f"Filtre Confiance   : > {SEUIL_CONFIANCE}%")
    print(f"Nombre de Trades   : {nb_trades}")
    print(f"Taux de réussite   : {win_rate:.2f}%")
    print(f"Espérance / trade  : {expectancy:.2f} pips")
    print(f"Max Drawdown       : {max_drawdown:.1f} Pips")
    print(f"RÉSULTAT NET       : {total_pips:.1f} Pips")
    print("="*55)

    plt.figure(figsize=(12, 6))
    plt.plot(trades_df.index, trades_df['Cumulative_Pips'], color='blue', label='Capital (Pips)')
    plt.fill_between(trades_df.index, trades_df['Cumulative_Pips'], trades_df['High_Water_Mark'], color='red', alpha=0.3, label='Drawdown')
    plt.title(f"Courbe d'équité Réelle - {nb_trades} Trades (Seuil {SEUIL_CONFIANCE}%) - {ANNEE_TEST}")
    plt.ylabel('Pips Cumulés')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./results/Equity_Curve_Stateful_{ANNEE_TEST}.png')

    md_content = f"""# 📈 Rapport de Performance Réel (Stateful)
**Version :** Triple Barrière V3 (Gestion de position stricte)
**Période :** {ANNEE_TEST}

---

## 📊 Statistiques Globales
| Métrique | Valeur |
| :--- | :--- |
| **Nombre de Trades** | {nb_trades} |
| **Taux de Réussite (Win Rate)** | {win_rate:.2f}% |
| **Résultat Net** | **{total_pips:.1f} Pips** |
| **Max Drawdown** | {max_drawdown:.1f} Pips |
| **Espérance par trade** | {expectancy:.2f} pips / trade |

---
*Généré par le backtester Stateful - Zéro Pyramidation autorisée.*
"""
    with open(f'./results/Rapport_Performance_Stateful_{ANNEE_TEST}.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
else:
    print(f"Aucun trade pris avec un seuil de confiance de {SEUIL_CONFIANCE}% sur l'année {ANNEE_TEST}.")