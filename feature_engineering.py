import pandas as pd
import numpy as np
import pandas_ta as ta
import os

# 1. Chargement des données
filepath = './cleaned-data/EURUSD_H1_cleaned.csv'
df = pd.read_csv(filepath, index_col='Time', parse_dates=True)

# 2. TARGET (La Cible : 1 si Hausse, 0 si Baisse au prochain H1)
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# 3. FEATURES DE PRIX ET TENDANCE (Stationnarisées)
# Retours logarithmiques (La norme en ML financier au lieu des variations simples)
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Distances relatives aux moyennes mobiles (Capture la tendance sans dépendre du prix brut)
df['Dist_EMA_9'] = (df['Close'] - ta.ema(df['Close'], length=9)) / df['Close']
df['Dist_EMA_21'] = (df['Close'] - ta.ema(df['Close'], length=21)) / df['Close']
df['Dist_EMA_50'] = (df['Close'] - ta.ema(df['Close'], length=50)) / df['Close']

# 4. FEATURES DE MOMENTUM
df['RSI_14'] = ta.rsi(df['Close'], length=14)

# Force de la tendance (ADX) - Extraction de la ligne principale
adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
df['ADX_14'] = adx['ADX_14']

# 5. FEATURES DE VOLATILITÉ
# ATR Normalisé (Volatilité exprimée en pourcentage du prix actuel)
df['ATR_Norm'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']

# Largeur des Bandes de Bollinger (Détecte les compressions/explosions de prix)
bbands = ta.bbands(df['Close'], length=20, std=2)

# Récupération dynamique des noms exacts des colonnes
col_upper = [col for col in bbands.columns if 'BBU' in col][0]
col_lower = [col for col in bbands.columns if 'BBL' in col][0]
col_mid = [col for col in bbands.columns if 'BBM' in col][0]

# Calcul de la largeur
df['BB_Width'] = (bbands[col_upper] - bbands[col_lower]) / bbands[col_mid]

# 6. CONTEXTE TEMPOREL (Cyclique)
# Transformation mathématique pour les cycles horaires et journaliers
df['Hour_Sin'] = np.sin(df.index.hour * (2. * np.pi / 24))
df['Hour_Cos'] = np.cos(df.index.hour * (2. * np.pi / 24))
df['Day_Sin'] = np.sin(df.index.dayofweek * (2. * np.pi / 5)) # 5 jours de trading
df['Day_Cos'] = np.cos(df.index.dayofweek * (2. * np.pi / 5))

# 7. NETTOYAGE FINALE
# Supprimer les lignes avec des NaN causés par les calculs (les 50 premières)
df.dropna(inplace=True)

# 8. SÉLECTION ET SAUVEGARDE
# On supprime les prix bruts (Open, High, Low, Close) car le modèle a maintenant des Features optimisées
colonnes_finales = [
    'Target', 'Spread', 'Log_Return', 
    'Dist_EMA_9', 'Dist_EMA_21', 'Dist_EMA_50', 
    'RSI_14', 'ADX_14', 'ATR_Norm', 'BB_Width', 
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]

dataset_ml = df[colonnes_finales]

output_path = './ready-data/EURUSD_H1_ML_Ready.csv'
dataset_ml.to_csv(output_path)

print(f"Dataset ML généré avec succès : {output_path}")
print(f"Nombre de variables (Features) : {len(colonnes_finales) - 1}")
print(dataset_ml.head())