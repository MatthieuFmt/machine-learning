import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def calc_base_features(df, prefix=''):
    """Calcule les indicateurs de tendance pour H4 et D1"""
    df_feat = pd.DataFrame(index=df.index)
    df_feat[f'RSI_14{prefix}'] = ta.rsi(df['Close'], length=14)
    df_feat[f'Dist_EMA_20{prefix}'] = (df['Close'] - ta.ema(df['Close'], length=20)) / df['Close']
    df_feat[f'Dist_EMA_50{prefix}'] = (df['Close'] - ta.ema(df['Close'], length=50)) / df['Close']
    return df_feat

print("Chargement des données...")
h1 = pd.read_csv('./cleaned-data/EURUSD_H1_cleaned.csv', index_col='Time', parse_dates=True)
h4 = pd.read_csv('./cleaned-data/EURUSD_H4_cleaned.csv', index_col='Time', parse_dates=True)
d1 = pd.read_csv('./cleaned-data/EURUSD_D1_cleaned.csv', index_col='Time', parse_dates=True)

print("Calcul des Features H1 avancées...")
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
# Tri obligatoire pour merge_asof
h1 = h1.sort_index().reset_index()
feat_h4 = feat_h4.sort_index().reset_index()
feat_d1 = feat_d1.sort_index().reset_index()

combined = pd.merge_asof(h1, feat_h4, on='Time', direction='backward')
combined = pd.merge_asof(combined, feat_d1, on='Time', direction='backward')

combined.set_index('Time', inplace=True)

# Définition de la Target sur le tableau final
combined['Target'] = (combined['Close'].shift(-1) > combined['Close']).astype(int)

# Nettoyage des NaN générés par les calculs et les décalages H4/D1
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

# Création du dossier s'il n'existe pas
output_dir = './ready-data'
os.makedirs(output_dir, exist_ok=True)

output_path = f"{output_dir}/EURUSD_Master_ML_Ready.csv"
dataset_ml.to_csv(output_path)

print(f"Dataset ML généré : {output_path}")
print(f"Lignes : {len(dataset_ml)} | Variables : {len(colonnes_finales) - 1}")