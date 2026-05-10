import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('./results', exist_ok=True)

# ======================= FONCTIONS DE POIDS =======================
def identity(proba, seuil=0.45):
    """Poids fixe = 1 (pas de sizing)"""
    return np.ones_like(proba)

def linear_0_5_1_5(proba, seuil=0.45):
    excess = (proba - seuil) / 0.15
    excess = np.clip(excess, 0, 1)
    return 0.5 + excess

def linear_0_8_1_2(proba, seuil=0.45):
    excess = (proba - seuil) / 0.10
    excess = np.clip(excess, 0, 1)
    return 0.8 + 0.4 * excess

def exp_0_5_1_5(proba, seuil=0.45):
    z = (proba - seuil) / 0.15
    z = np.clip(z, 0, 1)
    return 0.5 + z ** 2

def step(proba, seuil=0.45):
    w = np.zeros_like(proba)
    w[proba < 0.50] = 0.5
    w[(proba >= 0.50) & (proba < 0.55)] = 1.0
    w[proba >= 0.55] = 1.5
    return w

WEIGHT_FUNCTIONS = {
    'Fixe 1.0': identity,
    'Linéaire 0.5-1.5': linear_0_5_1_5,
    'Linéaire 0.8-1.2': linear_0_8_1_2,
    'Exponentiel 0.5-1.5': exp_0_5_1_5,
    'Paliers 0.5/1.0/1.5': step
}

# ======================= FONCTION DE BACKTEST =======================
def backtest_year(annee, weight_func, seuil_conf=45.0):
    file_pred = f'./results/Predictions_{annee}_TripleBarrier.csv'
    if not os.path.exists(file_pred):
        raise FileNotFoundError(f"Fichier de prédictions manquant : {file_pred}")
    
    preds = pd.read_csv(file_pred, index_col='Time', parse_dates=True)
    prices = pd.read_csv('./cleaned-data/EURUSD_H1_cleaned.csv', index_col='Time', parse_dates=True)
    dataset_ml = pd.read_csv('./ready-data/EURUSD_Master_ML_Ready.csv', index_col='Time', parse_dates=True)
    X_cols = [c for c in dataset_ml.columns if c not in ['Target', 'Spread']]

    df = preds.join(prices[['High','Low','Close']], how='inner')
    df = df.join(dataset_ml[X_cols], how='inner')

    TP_PIPS=20.0; SL_PIPS=10.0; WINDOW=24; PIP_SIZE=0.0001
    df['proba_max'] = df[['Confiance_Hausse_%','Confiance_Neutre_%','Confiance_Baisse_%']].max(axis=1)/100
    df['Signal'] = 0
    df['Weight'] = 0.0

    mask_long = (df['Prediction_Modele']==1) & (df['Confiance_Hausse_%']/100 >= seuil_conf/100)
    mask_short = (df['Prediction_Modele']==-1) & (df['Confiance_Baisse_%']/100 >= seuil_conf/100)
    df.loc[mask_long,'Signal'] = 1
    df.loc[mask_short,'Signal'] = -1

    # Appliquer la fonction de poids
    signal_mask = df['Signal'] != 0
    df.loc[signal_mask, 'Weight'] = weight_func(df.loc[signal_mask, 'proba_max'])

    # Simulation stateful
    dates = df.index
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    signals = df['Signal'].values
    weights = df['Weight'].values
    spreads = df['Spread'].values

    trade_records = []
    i = 0
    while i < len(df):
        if signals[i] != 0:
            entry_time = dates[i]
            entry_price = closes[i]
            signal = signals[i]
            spread_cost = spreads[i] / 10.0
            weight = weights[i]

            if signal == 1:
                tp = entry_price + TP_PIPS * PIP_SIZE
                sl = entry_price - SL_PIPS * PIP_SIZE
            else:
                tp = entry_price - TP_PIPS * PIP_SIZE
                sl = entry_price + SL_PIPS * PIP_SIZE

            pips_brut = -SL_PIPS - spread_cost
            result_type = 'loss_sl'

            for j in range(1, WINDOW+1):
                if i+j >= len(df):
                    i = len(df)
                    break
                curr_high = highs[i+j]
                curr_low = lows[i+j]
                if signal == 1:
                    if curr_low <= sl:
                        pips_brut = -SL_PIPS - spread_cost
                        result_type = 'loss_sl'
                        i = i+j
                        break
                    elif curr_high >= tp:
                        pips_brut = TP_PIPS - spread_cost
                        result_type = 'win'
                        i = i+j
                        break
                else:
                    if curr_high >= sl:
                        pips_brut = -SL_PIPS - spread_cost
                        result_type = 'loss_sl'
                        i = i+j
                        break
                    elif curr_low <= tp:
                        pips_brut = TP_PIPS - spread_cost
                        result_type = 'win'
                        i = i+j
                        break
            else:
                i = i + WINDOW
                result_type = 'loss_timeout'

            pips_pond = pips_brut * weight
            trade_records.append({
                'Time': entry_time,
                'Pips_Nets': pips_pond,
                'Pips_Bruts': pips_brut,
                'Weight': weight,
                'result': result_type
            })
            continue
        i += 1

    trades_df = pd.DataFrame(trade_records)
    if trades_df.empty:
        return {'profit_net': 0, 'dd': 0, 'trades': 0, 'win_rate': 0, 'sharpe': 0}
    
    trades_df.set_index('Time', inplace=True)
    win_rate = (trades_df['Pips_Bruts'] > 0).mean() * 100
    total_pips = trades_df['Pips_Nets'].sum()
    trades_df['Cum'] = trades_df['Pips_Nets'].cumsum()
    trades_df['DD'] = trades_df['Cum'] - trades_df['Cum'].cummax()
    max_dd = trades_df['DD'].min()
    daily = trades_df.resample('D')['Pips_Nets'].sum().dropna()
    if len(daily) > 1 and daily.std() != 0:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        'profit_net': total_pips,
        'dd': max_dd,
        'trades': len(trades_df),
        'win_rate': win_rate,
        'sharpe': sharpe
    }

# ======================= OPTIMISATION SUR 2024 =======================
ANNEE_VAL = 2024
print(f"Optimisation des fonctions de sizing sur {ANNEE_VAL}...\n")
results_val = {}
for name, func in WEIGHT_FUNCTIONS.items():
    res = backtest_year(ANNEE_VAL, func)
    results_val[name] = res
    print(f"{name:25s} | Profit={res['profit_net']:8.1f} pips | DD={res['dd']:6.1f} pips | WR={res['win_rate']:5.1f}% | Trades={res['trades']}")

best_name = max(results_val, key=lambda x: results_val[x]['profit_net'])
print(f"\n>>> Meilleure fonction sur {ANNEE_VAL} : {best_name} (Profit={results_val[best_name]['profit_net']:.1f} pips)")

# ======================= APPLICATION SUR 2025 =======================
ANNEE_TEST = 2025
print(f"\nApplication sur {ANNEE_TEST} avec '{best_name}'...")
best_func = WEIGHT_FUNCTIONS[best_name]
final_res = backtest_year(ANNEE_TEST, best_func)

print("\n" + "="*60)
print(f" 📊 RÉSULTAT FINAL SUR {ANNEE_TEST} – Sizing '{best_name}'")
print("="*60)
print(f"Stratégie            : TP 20 | SL 10 | Filtre >45%")
print(f"Fonction de sizing   : {best_name}")
print(f"Nombre de Trades     : {final_res['trades']}")
print(f"Taux de réussite     : {final_res['win_rate']:.2f}% (brut)")
print(f"RÉSULTAT NET         : {final_res['profit_net']:.1f} Pips (pondéré)")
print(f"Max Drawdown         : {final_res['dd']:.1f} Pips")
print(f"Ratio de Sharpe      : {final_res['sharpe']:.2f}")
print("="*60)
