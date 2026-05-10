import pandas as pd
import sys
import os

# Usage : python clean_data.py <chemin_fichier_csv>
if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = './data/EURUSD_H1.csv'

print(f"Chargement : {filepath}")

# 1. Détection du header et lecture de 7 colonnes (index 0 à 6)
with open(filepath, 'r') as f:
    first_line = f.readline().strip()

cols_to_use = [0, 1, 2, 3, 4, 5, 6]
colnames = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']

if first_line.startswith('Time'):
    # On ignore le header du fichier pour imposer nos propres noms de colonnes
    df = pd.read_csv(filepath, sep='\t', header=0, names=colnames, usecols=cols_to_use)
else:
    df = pd.read_csv(filepath, sep='\t', header=None, names=colnames, usecols=cols_to_use)

# 2. Nettoyage et formatage
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)
df.sort_index(inplace=True)

# Suppression des doublons et des valeurs manquantes
df = df[~df.index.duplicated(keep='first')]
df.dropna(inplace=True)

# 3. Sauvegarde dans /cleaned-data
output_dir = './cleaned-data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

filename = os.path.basename(filepath).replace('.csv', '_cleaned.csv')
save_path = os.path.join(output_dir, filename)

df.to_csv(save_path)

# 4. Rapport d'exécution
print(f"\nColonnes incluses : {list(df.columns)}")
print(f"Période : {df.index.min()} -> {df.index.max()}")
print(f"Fichier sauvegardé ici : {save_path}")
print(f"Nombre de bougies finales : {len(df)}")