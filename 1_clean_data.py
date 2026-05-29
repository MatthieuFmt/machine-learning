import pandas as pd
import sys
import os
import glob

# Usage : python clean_data.py <dossier_source>
# Par défaut, il ciblera le dossier ./data
if len(sys.argv) > 1:
    input_dir = sys.argv[1]
else:
    input_dir = './data'

output_dir = './cleaned-data'

# Création du dossier de sortie s'il n'existe pas
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Lister tous les fichiers CSV dans le dossier
fichiers_csv = glob.glob(os.path.join(input_dir, '*.csv'))

if not fichiers_csv:
    print(f"⚠️ Aucun fichier CSV trouvé dans '{input_dir}'.")
    sys.exit()

print(f"🔄 Détection de {len(fichiers_csv)} fichiers à nettoyer. Lancement du processus...\n")

# 2. Boucle sur chaque fichier trouvé
for filepath in fichiers_csv:
    nom_fichier = os.path.basename(filepath)
    print(f"▶️ Traitement de : {nom_fichier}")
    
    try:
        # Détection du header
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

        cols_to_use = [0, 1, 2, 3, 4, 5, 6]
        colnames = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']

        if first_line.startswith('Time'):
            df = pd.read_csv(filepath, sep='\t', header=0, names=colnames, usecols=cols_to_use)
        else:
            df = pd.read_csv(filepath, sep='\t', header=None, names=colnames, usecols=cols_to_use)

        # Nettoyage et formatage
        # errors='coerce' transforme les dates bizarres en NaT (Not a Time)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        
        # On supprime tout de suite les lignes où la date n'a pas pu être lue
        df.dropna(subset=['Time'], inplace=True)
        
        df.set_index('Time', inplace=True)
        df.sort_index(inplace=True)

        # Suppression des doublons et des autres valeurs manquantes
        df = df[~df.index.duplicated(keep='first')]
        df.dropna(inplace=True)

        # Sauvegarde
        filename_out = nom_fichier.replace('.csv', '_cleaned.csv')
        save_path = os.path.join(output_dir, filename_out)
        df.to_csv(save_path)

        print(f"   ✅ Succès : {len(df)} lignes sauvegardées ({df.index.min().date()} -> {df.index.max().date()})\n")

    except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"   ❌ Erreur avec le fichier {nom_fichier} : {type(e).__name__}: {e}\n")

print("🎉 Nettoyage de tous les fichiers terminé !")