import pandas as pd
import sys

# Usage : python clean_data.py <chemin_fichier_csv>
# Exemple : python clean_data.py ./data/EURUSD_H1.csv

if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = './data/EURUSD_H1.csv'

print(f"Chargement : {filepath}")

# 1. Détecter si la première ligne est un en-tête ou une donnée
with open(filepath, 'r') as f:
    first_line = f.readline().strip()

# Si la première ligne commence par "Time", c'est un header
if first_line.startswith('Time'):
    # Fichier avec en-tête (data/) : 7 colonnes de données, 6 noms
    # On ne garde que les 6 premières colonnes pour éviter la 7ème sans nom
    df = pd.read_csv(filepath, sep='\t', header=0, usecols=[0, 1, 2, 3, 4, 5])
else:
    # Fichier sans en-tête (data-future/) : 7 colonnes, pas de header
    df = pd.read_csv(filepath, sep='\t', header=None, usecols=[0, 1, 2, 3, 4, 5])
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

print(f"Colonnes détectées : {list(df.columns)}")
print(f"Nb lignes chargées : {len(df)}")

# 2. Convertir Time en datetime et l'utiliser comme index
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

# 3. Vérifier les valeurs manquantes (NaN)
print("\nValeurs manquantes :")
print(df.isnull().sum())

# 4. Nettoyer les données en supprimant les lignes incomplètes
before = len(df)
df.dropna(inplace=True)
print(f"Lignes supprimées (NaN) : {before - len(df)}")

# 5. Afficher les 5 premières et 5 dernières lignes
print(f"\n=== 5 premières lignes ===")
print(df.head())
print(f"\n=== 5 dernières lignes ===")
print(df.tail())
print(f"\nPériode : {df.index.min()} → {df.index.max()}")