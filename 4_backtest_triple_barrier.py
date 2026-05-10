import pandas as pd
import numpy as np
import os

# ---------------------------- CONFIG ----------------------------
WEIGHT_FUNC = lambda proba: np.clip(0.8 + 0.4 * ((proba - 0.45) / 0.10), 0.8, 1.2)
SEUIL_CONFIANCE = 45.0
TP_PIPS, SL_PIPS, WINDOW, PIP_SIZE = 20.0, 10.0, 24, 0.0001
ANNEES = [2022, 2023, 2024, 2025]

# ---------------------------- BACKTEST ----------------------------
def backtest_year(annee):
    preds = pd.read_csv(f'./results/Predictions_{annee}_TripleBarrier.csv',
                        index_col='Time', parse_dates=True)
    prices = pd.read_csv('./cleaned-data/EURUSD_H1_cleaned.csv',
                         index_col='Time', parse_dates=True)
    ml = pd.read_csv('./ready-data/EURUSD_Master_ML_Ready.csv',
                     index_col='Time', parse_dates=True)
    X_cols = [c for c in ml.columns if c not in ['Target', 'Spread']]

    df = preds.join(prices[['High','Low','Close']], how='inner')
    df = df.join(ml[X_cols], how='inner')

    df['proba_max'] = df[['Confiance_Hausse_%','Confiance_Neutre_%','Confiance_Baisse_%']].max(axis=1) / 100
    df['Signal'] = 0
    df['Weight'] = 0.0

    mask_long = (df['Prediction_Modele'] == 1) & (df['Confiance_Hausse_%'] / 100 >= SEUIL_CONFIANCE / 100)
    mask_short = (df['Prediction_Modele'] == -1) & (df['Confiance_Baisse_%'] / 100 >= SEUIL_CONFIANCE / 100)
    df.loc[mask_long, 'Signal'] = 1
    df.loc[mask_short, 'Signal'] = -1

    signal_mask = df['Signal'] != 0
    df.loc[signal_mask, 'Weight'] = WEIGHT_FUNC(df.loc[signal_mask, 'proba_max'])

    dates = df.index
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    signals = df['Signal'].values
    weights = df['Weight'].values
    spreads = df['Spread'].values

    trades = []
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
                curr_high, curr_low = highs[i+j], lows[i+j]
                if signal == 1:
                    if curr_low <= sl:
                        i += j
                        break
                    elif curr_high >= tp:
                        pips_brut = TP_PIPS - spread_cost
                        result_type = 'win'
                        i += j
                        break
                else:
                    if curr_high >= sl:
                        i += j
                        break
                    elif curr_low <= tp:
                        pips_brut = TP_PIPS - spread_cost
                        result_type = 'win'
                        i += j
                        break
            else:
                i += WINDOW
                result_type = 'loss_timeout'

            trades.append({
                'Time': entry_time,
                'Pips_Nets': pips_brut * weight,
                'Pips_Bruts': pips_brut,
                'Weight': weight,
                'result': result_type
            })
            continue
        i += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {'annee': annee, 'profit_net': 0, 'dd': 0, 'trades': 0, 'win_rate': 0, 'sharpe': 0}
    trades_df.set_index('Time', inplace=True)
    win_rate = (trades_df['Pips_Bruts'] > 0).mean() * 100
    profit_net = trades_df['Pips_Nets'].sum()
    trades_df['Cum'] = trades_df['Pips_Nets'].cumsum()
    trades_df['DD'] = trades_df['Cum'] - trades_df['Cum'].cummax()
    max_dd = trades_df['DD'].min()
    daily = trades_df.resample('D')['Pips_Nets'].sum().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if (len(daily) > 1 and daily.std() != 0) else 0
    return {'annee': annee, 'profit_net': profit_net, 'dd': max_dd,
            'trades': len(trades_df), 'win_rate': win_rate, 'sharpe': sharpe}

# ---------------------------- EXÉCUTION ----------------------------
print("Validation multi-années du sizing linéaire 0.8-1.2\n")
results = []
for an in ANNEES:
    if not os.path.exists(f'./results/Predictions_{an}_TripleBarrier.csv'):
        print(f"⚠️  Fichier manquant pour {an}, on le saute.")
        continue
    res = backtest_year(an)
    results.append(res)
    print(f"{an} | Profit={res['profit_net']:8.1f} | DD={res['dd']:6.1f} | WR={res['win_rate']:5.1f}% | Trades={res['trades']:4d} | Sharpe={res['sharpe']:.2f}")

# Synthèse
if results:
    df_res = pd.DataFrame(results).set_index('annee')
    print("\n=== SYNTHÈSE ===")
    print(df_res.to_string())
    total_profit = df_res['profit_net'].sum()
    print(f"\nProfit total sur les années testées : {total_profit:.0f} pips")