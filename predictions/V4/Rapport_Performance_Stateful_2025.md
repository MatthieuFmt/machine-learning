import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def apply_triple_barrier(df, tp_pips=20, sl_pips=10, window=24, pip_size=0.0001):
    print(f"🎯 Calcul de la Target (TP: {tp_pips} pips | SL: {sl_pips} pips | Délai max: {window}h)...")
    targets = np.full(len(df), np.nan)
    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    for i in range(len(df) - window):
        entry_price = closes[i]
        long_tp, long_sl = entry_price + tp_dist, entry_price - sl_dist
        short_tp, short_sl = entry_price - tp_dist, entry_price + sl_dist
        long_win, short_win = False, False
        
        for j in range(1, window + 1):
            if lows[i + j] <= long_sl: break
            elif highs[i + j] >= long_tp:
                long_win = True
                break
                
        for j in range(1, window + 1):
            if highs[i + j] >= short_sl: break
            elif lows[i + j] <= short_tp:
                short_win = True
                break

        if long_win and not short_win: targets[i] = 1   
        elif short_win and not long_win: targets[i] = -1  
        else: targets[i] = 0   
            
    return targets

def calc_base_features(df, prefix=''):
    df_feat = pd.DataFrame(index=df.index)
    df_feat[f'RSI_14{prefix}'] = ta.rsi(df['Close'], length=14)
    df_feat[f'Dist_EMA_20{prefix}'] = (df['Close'] - ta.ema(df['Close'], length=20)) / df['Close']
    df_feat[f'Dist_EMA_50{prefix}'] = (df['Close'] - ta.ema(df['Close'], length=50)) / df['Close']
    return df_feat

# 1. Chargement des données EURUSD
print("Chargement des données EURUSD...")
h1 = pd.read_csv('./cleaned-data/EURUSD_H1_cleaned.csv', index_col='Time', parse_dates=True)
h4 = pd.read_csv('./cleaned-data/EURUSD_H4_cleaned.csv', index_col='Time', parse_dates=True)
d1 = pd.read_csv('./cleaned-data/EURUSD_D1_cleaned.csv', index_col='Time', parse_dates=True)

# 1.bis Chargement des données Corrélées (H1 uniquement)
print("Chargement des données Macro (XAUUSD, USDCHF)...")
xau_h1 = pd.read_csv('./cleaned-data/XAUUSD_H1_cleaned.csv', index_col='Time', parse_dates=True)
chf_h1 = pd.read_csv('./cleaned-data/USDCHF_H1_cleaned.csv', index_col='Time', parse_dates=True)

# Calcul sécurisé : chaque actif garde son propre index avant le merge
xau_feat = pd.DataFrame(index=xau_h1.index)
xau_feat['XAU_Return'] = np.log(xau_h1['Close'] / xau_h1['Close'].shift(1))

chf_feat = pd.DataFrame(index=chf_h1.index)
chf_feat['CHF_Return'] = np.log(chf_h1['Close'] / chf_h1['Close'].shift(1))

# 2. APPLICATION DE LA TARGET
h1['Target'] = apply_triple_barrier(h1, tp_pips=20, sl_pips=10, window=24)
h1.dropna(subset=['Target'], inplace=True)

# 3. Calcul des Features H1 avancées
print("Calcul des indicateurs EURUSD...")
h1['Log_Return'] = np.log(h1['Close'] / h1['Close'].shift(1))
h1['Dist_EMA_9'] = (h1['Close'] - ta.ema(h1['Close'], length=9)) / h1['Close']
h1['Dist_EMA_21'] = (h1['Close'] - ta.ema(h1['Close'], length=21)) / h1['Close']
h1['Dist_EMA_50'] = (h1['Close'] - ta.ema(h1['Close'], length=50)) / h1['Close']

h1['RSI_14'] = ta.rsi(h1['Close'], length=14)
h1['ADX_14'] = ta.adx(h1['High'], h1['Low'], h1['Close'], length=14)['ADX_14']
h1['ATR_Norm'] = ta.atr(h1['High'], h1['Low'], h1['Close'], length=14) / h1['Close']

bbands = ta.bbands(h1['Close'], length=20, std=2)
h1['BB_Width'] = (bbands.iloc[:, 2] - bbands.iloc[:, 0]) / bbands.iloc[:, 1]

h1['Hour_Sin'] = np.sin(h1.index.hour * (2. * np.pi / 24))
h1['Hour_Cos'] = np.cos(h1.index.hour * (2. * np.pi / 24))

# Extraction des features H4 et D1
feat_h4 = calc_base_features(h4, '_H4')
feat_d1 = calc_base_features(d1, '_D1')

print("Fusion de toutes les données (Merge_asof)...")
h1 = h1.sort_index().reset_index()
feat_h4 = feat_h4.sort_index().reset_index()
feat_d1 = feat_d1.sort_index().reset_index()
xau_feat = xau_feat.sort_index().reset_index()
chf_feat = chf_feat.sort_index().reset_index()

# Merge Multi-Timeframe
combined = pd.merge_asof(h1, feat_h4, on='Time', direction='backward')
combined = pd.merge_asof(combined, feat_d1, on='Time', direction='backward')

# Merge Macro (Actifs corrélés) séparément
combined = pd.merge_asof(combined, xau_feat, on='Time', direction='backward')
combined = pd.merge_asof(combined, chf_feat, on='Time', direction='backward')

combined.set_index('Time', inplace=True)
combined.dropna(inplace=True)

colonnes_finales = [
    'Target', 'Spread', 'Log_Return', 
    'Dist_EMA_9', 'Dist_EMA_21', 'Dist_EMA_50', 
    'RSI_14', 'ADX_14', 'ATR_Norm', 'BB_Width', 
    'Hour_Sin', 'Hour_Cos',
    'RSI_14_H4', 'Dist_EMA_20_H4', 'Dist_EMA_50_H4',
    'RSI_14_D1', 'Dist_EMA_20_D1', 'Dist_EMA_50_D1',
    'XAU_Return', 'CHF_Return'
]

dataset_ml = combined[colonnes_finales]

output_dir = './ready-data'
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/EURUSD_Master_ML_Ready.csv"
dataset_ml.to_csv(output_path)

print(f"✅ Dataset ML généré avec succès : {output_path}")
print(f"Répartition des Targets :\n{dataset_ml['Target'].value_counts(normalize=True).round(4) * 100}")