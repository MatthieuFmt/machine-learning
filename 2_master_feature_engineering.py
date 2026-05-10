import pandas as pd
import numpy as np
import pandas_ta as ta
import os

# --- NOUVELLE FONCTION : LA TRIPLE BARRIÈRE (Bidirectionnelle) ---
def apply_triple_barrier(df, tp_pips=20, sl_pips=10, window=24, pip_size=0.0001):
    """
    Vérifie si un Achat (1) ou une Vente (-1) atteint son TP avant son SL.
    Retourne 0 si aucun des deux ne gagne (ou si les deux perdent).
    """
    print(f"🎯 Calcul de la Target (TP: {tp_pips} pips | SL: {sl_pips} pips | Délai max: {window}h)...")
    
    # On initialise tout à NaN (Non Défini) pour pouvoir supprimer les dernières bougies
    targets = np.full(len(df), np.nan)
    
    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    # On parcourt chaque bougie (sauf les 24 dernières)
    for i in range(len(df) - window):
        entry_price = closes[i]
        
        long_tp = entry_price + tp_dist
        long_sl = entry_price - sl_dist
        
        short_tp = entry_price - tp_dist
        short_sl = entry_price + sl_dist
        
        long_win = False
        short_win = False
        
        # 1. Test de l'Achat (Long)
        for j in range(1, window + 1):
            if lows[i + j] <= long_sl:
                break # Le Stop Loss a été touché, le trade Long est mort
            elif highs[i + j] >= long_tp:
                long_win = True
                break # Le Take Profit a été touché en premier !
                
        # 2. Test de la Vente (Short)
        for j in range(1, window + 1):
            if highs[i + j] >= short_sl:
                break # Le Stop Loss a été touché, le trade Short est mort
            elif lows[i + j] <= short_tp:
                short_win = True
                break # Le Take Profit a été touché en premier !

        # 3. Assignation de la Target Finale
        if long_win and not short_win:
            targets[i] = 1   # Signal d'Achat parfait
        elif short_win and not long_win:
            targets[i] = -1  # Signal de Vente parfait
        else:
            targets[i] = 0   # Reste à l'écart (Chop, Timeout, ou grosse volatilité)
            
    return targets

def calc_base_features(df, prefix=''):
    """Calcule les indicateurs de tendance pour H4 et D1"""
    df_feat = pd.DataFrame(index=df.index)
    df_feat[f'RSI_14{prefix}'] = ta.rsi(df['Close'], length=14)
    df_feat[f'Dist_EMA_20{prefix}'] = (df['Close'] - ta.ema(df['Close'], length=20)) / df['Close']
    df_feat[f'Dist_EMA_50{prefix}'] = (df['Close'] - ta.ema(df['Close'], length=50)) / df['Close']
    return df_feat

# 1. Chargement des données
print("Chargement des données H1, H4, D1...")
h1 = pd.read_csv('./cleaned-data/EURUSD_H1_cleaned.csv', index_col='Time', parse_dates=True)
h4 = pd.read_csv('./cleaned-data/EURUSD_H4_cleaned.csv', index_col='Time', parse_dates=True)
d1 = pd.read_csv('./cleaned-data/EURUSD_D1_cleaned.csv', index_col='Time', parse_dates=True)

# 2. APPLICATION DE LA TARGET
h1['Target'] = apply_triple_barrier(h1, tp_pips=20, sl_pips=10, window=24)
# On supprime immédiatement les 24 dernières lignes qui ont une Target à NaN
h1.dropna(subset=['Target'], inplace=True)

# 3. Calcul des Features H1 avancées
print("Calcul des indicateurs...")
h1['Log_Return'] = np.log(h1['Close'] / h1['Close'].shift(1))
h1['Dist_EMA_9'] = (h1['Close'] - ta.ema(h1['Close'], length=9)) / h1['Close']
h1['Dist_EMA_21'] = (h1['Close'] - ta.ema(h1['Close'], length=21)) / h1['Close']
h1['Dist_EMA_50'] = (h1['Close'] - ta.ema(h1['Close'], length=50)) / h1['Close']

h1['RSI_14'] = ta.rsi(h1['Close'], length=14)
h1['ADX_14'] = ta.adx(h1['High'], h1['Low'], h1['Close'], length=14)['ADX_14']
h1['ATR_Norm'] = ta.atr(h1['High'], h1['Low'], h1['Close'], length=14) / h1['Close']

bbands = ta.bbands(h1['Close'], length=20, std=2)
col_upper = [col for col in bbands.columns if 'BBU' in col][0]
col_lower = [col for col in bbands.columns if 'BBL' in col][0]
col_mid = [col for col in bbands.columns if 'BBM' in col][0]
h1['BB_Width'] = (bbands[col_upper] - bbands[col_lower]) / bbands[col_mid]

h1['Hour_Sin'] = np.sin(h1.index.hour * (2. * np.pi / 24))
h1['Hour_Cos'] = np.cos(h1.index.hour * (2. * np.pi / 24))

# Extraction des features H4 et D1
feat_h4 = calc_base_features(h4, '_H4')
feat_d1 = calc_base_features(d1, '_D1')

print("Fusion des Timeframes (Merge_asof)...")
h1 = h1.sort_index().reset_index()
feat_h4 = feat_h4.sort_index().reset_index()
feat_d1 = feat_d1.sort_index().reset_index()

combined = pd.merge_asof(h1, feat_h4, on='Time', direction='backward')
combined = pd.merge_asof(combined, feat_d1, on='Time', direction='backward')

combined.set_index('Time', inplace=True)
combined.dropna(inplace=True)

colonnes_finales = [
    'Target', 'Spread', 'Log_Return', 
    'Dist_EMA_9', 'Dist_EMA_21', 'Dist_EMA_50', 
    'RSI_14', 'ADX_14', 'ATR_Norm', 'BB_Width', 
    'Hour_Sin', 'Hour_Cos',
    'RSI_14_H4', 'Dist_EMA_20_H4', 'Dist_EMA_50_H4',
    'RSI_14_D1', 'Dist_EMA_20_D1', 'Dist_EMA_50_D1'
]

dataset_ml = combined[colonnes_finales]

output_dir = './ready-data'
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/EURUSD_Master_ML_Ready.csv"
dataset_ml.to_csv(output_path)

print(f"✅ Dataset ML généré avec succès : {output_path}")
print(f"Répartition des Targets :\n{dataset_ml['Target'].value_counts(normalize=True).round(4) * 100}%")