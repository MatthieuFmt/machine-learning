import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('./results', exist_ok=True)

# 1. Chargement des données (Prédictions + Prix réels pour simuler le déroulement)
preds = pd.read_csv('./results/Predictions_2026_TripleBarrier.csv', index_col='Time', parse_dates=True)
prices = pd.read_csv('./cleaned-data/EURUSD_H1_cleaned.csv', index_col='Time', parse_dates=True)

# Fusion pour avoir les High/Low futurs
df = preds.join(prices[['High', 'Low', 'Close']], how='inner')

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

trade_records = []
i = 0

print("Simulation Stateful en cours (Recherche des exits exacts)...")

while i < len(df):
    if signals[i] != 0:
        entry_time = dates[i]
        entry_price = closes[i]
        signal = signals[i]
        spread_cost = spreads[i] / 10.0
        
        # Calcul des niveaux de prix stricts
        if signal == 1:
            tp_price = entry_price + (TP_PIPS * PIP_SIZE)
            sl_price = entry_price - (SL_PIPS * PIP_SIZE)
        else:
            tp_price = entry_price - (TP_PIPS * PIP_SIZE)
            sl_price = entry_price + (SL_PIPS * PIP_SIZE)
            
        pips_nets = -SL_PIPS - spread_cost # Valeur par défaut pessimiste (Timeout)
        
        # Parcours des bougies futures pour trouver la sortie
        for j in range(1, WINDOW + 1):
            if i + j >= len(df):
                i = len(df) # Fin du dataset
                break
                
            curr_high = highs[i + j]
            curr_low = lows[i + j]
            
            if signal == 1: # Achat
                if curr_low <= sl_price:
                    pips_nets = -SL_PIPS - spread_cost
                    i = i + j # Le trade se ferme ici, on avance l'index global
                    break
                elif curr_high >= tp_price:
                    pips_nets = TP_PIPS - spread_cost
                    i = i + j
                    break
            else: # Vente
                if curr_high >= sl_price:
                    pips_nets = -SL_PIPS - spread_cost
                    i = i + j
                    break
                elif curr_low <= tp_price:
                    pips_nets = TP_PIPS - spread_cost
                    i = i + j
                    break
        else:
            # Si la boucle for se termine sans break (Timeout des 24h)
            i = i + WINDOW
            
        trade_records.append({'Time': entry_time, 'Pips_Nets': pips_nets})
        continue # On passe au i suivant (qui a déjà été avancé)
        
    i += 1

# 5. Calcul des métriques finales
trades_df = pd.DataFrame(trade_records)

if not trades_df.empty:
    trades_df.set_index('Time', inplace=True)
    nb_trades = len(trades_df)
    trades_gagnants = trades_df[trades_df['Pips_Nets'] > 0]
    win_rate = (len(trades_gagnants) / nb_trades) * 100
    total_pips = trades_df['Pips_Nets'].sum()
    expectancy = total_pips / nb_trades
    
    trades_df['Cumulative_Pips'] = trades_df['Pips_Nets'].cumsum()
    trades_df['High_Water_Mark'] = trades_df['Cumulative_Pips'].cummax()
    trades_df['Drawdown'] = trades_df['Cumulative_Pips'] - trades_df['High_Water_Mark']
    max_drawdown = trades_df['Drawdown'].min() 
    
    # Affichage
    print("\n" + "="*55)
    print(" 📊 BACKTEST FINANCIER - STATEFUL (Zéro Pyramidation) 📊")
    print("="*55)
    print(f"Stratégie          : TP {TP_PIPS} pips | SL {SL_PIPS} pips")
    print(f"Filtre Confiance   : > {SEUIL_CONFIANCE}%")
    print(f"Nombre de Trades   : {nb_trades}")
    print(f"Taux de réussite   : {win_rate:.2f}%")
    print(f"Espérance / trade  : {expectancy:.2f} pips")
    print(f"Max Drawdown       : {max_drawdown:.1f} Pips")
    print(f"RÉSULTAT NET       : {total_pips:.1f} Pips")
    print("="*55)

    # Graphique
    plt.figure(figsize=(12, 6))
    plt.plot(trades_df.index, trades_df['Cumulative_Pips'], color='blue', label='Capital (Pips)')
    plt.fill_between(trades_df.index, trades_df['Cumulative_Pips'], trades_df['High_Water_Mark'], color='red', alpha=0.3, label='Drawdown')
    plt.title(f"Courbe d'équité Réelle - {nb_trades} Trades (Seuil {SEUIL_CONFIANCE}%)")
    plt.ylabel('Pips Cumulés')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./results/Equity_Curve_Stateful.png')

    # Markdown
    md_content = f"""# 📈 Rapport de Performance Réel (Stateful)
**Version :** Triple Barrière V3 (Gestion de position stricte)
**Période :** 2026

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
    with open('./results/Rapport_Performance_Stateful.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
else:
    print(f"Aucun trade pris avec un seuil de confiance de {SEUIL_CONFIANCE}%.")