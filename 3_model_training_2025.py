import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
from datetime import timedelta

output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

# --- PARAMÈTRE GLOBAL ---
ANNEE_TEST = 2025  

# 1. Chargement des données
filepath = './ready-data/EURUSD_Master_ML_Ready.csv'
df = pd.read_csv(filepath, index_col='Time', parse_dates=True)

# 2. Séparation chronologique stricte
split_date_start = pd.to_datetime(f'{ANNEE_TEST}-01-01')
split_date_end = pd.to_datetime(f'{ANNEE_TEST + 1}-01-01')
purge_window = timedelta(hours=24) 

train_data = df[df.index < (split_date_start - purge_window)].copy()
test_data = df[(df.index >= split_date_start) & (df.index < split_date_end)].copy()

if test_data.empty:
    raise ValueError(f"⚠️ Aucune donnée trouvée pour l'année {ANNEE_TEST}. Vérifie ton fichier CSV.")

# --- SÉCURITÉ : ORDRE DES COLONNES ---
# On définit explicitement les features pour éviter tout décalage
colonnes_a_exclure = ['Target', 'Spread']
X_cols = [c for c in df.columns if c not in colonnes_a_exclure]

X_train = train_data[X_cols]
y_train = train_data['Target']

X_test = test_data[X_cols]
y_test = test_data['Target']

print(f"📊 Taille de l'entraînement (Données avant {ANNEE_TEST}) : {len(X_train)} bougies")
print(f"📊 Taille du test ({ANNEE_TEST}) : {len(X_test)} bougies\n")

# 3. Entraînement du modèle
# J'augmente légèrement min_samples_leaf pour éviter le surapprentissage sur le bruit
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=12, 
    min_samples_leaf=5, # Sécurité supplémentaire contre l'overfitting
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced'
)

print(f"🤖 Entraînement du modèle pour l'année {ANNEE_TEST}...")
model.fit(X_train, y_train)

# 4. Importance des Features
importances = model.feature_importances_
fi_df = pd.DataFrame({
    'Indicateur': X_cols, 
    'Importance_%': np.round(importances * 100, 2)
}).sort_values(by='Importance_%', ascending=False)

print("\n=== IMPORTANCE DES INDICATEURS ===")
print(fi_df.to_string(index=False))

plt.figure(figsize=(10, 8))
plt.barh(fi_df['Indicateur'][::-1], fi_df['Importance_%'][::-1], color='skyblue')
plt.title(f"Importance des variables - Test {ANNEE_TEST}")
plt.xlabel("Impact en %")
plt.tight_layout()
plt.savefig(f'{output_dir}/Feature_Importance_{ANNEE_TEST}.png')

# 5. Prédictions et Probabilités
predictions = model.predict(X_test)
probas = model.predict_proba(X_test)

# Mapping dynamique des classes
class_map = {cls: idx for idx, cls in enumerate(model.classes_)}
proba_baisse = probas[:, class_map[-1.0]]
proba_neutre = probas[:, class_map[0.0]]
proba_hausse = probas[:, class_map[1.0]]

print(f"\n=== RÉSULTATS SUR {ANNEE_TEST} ===")
print(f"✅ Précision globale (Accuracy) : {accuracy_score(y_test, predictions):.2f}\n")
print(classification_report(y_test, predictions))

# 6. Sauvegarde pour le Backtest
results = pd.DataFrame({
    'Close_Reel_Direction': y_test,
    'Prediction_Modele': predictions,
    'Confiance_Baisse_%': np.round(proba_baisse * 100, 2),
    'Confiance_Neutre_%': np.round(proba_neutre * 100, 2), 
    'Confiance_Hausse_%': np.round(proba_hausse * 100, 2),
    'Spread': test_data['Spread']
}, index=y_test.index)

output_path = f'{output_dir}/Predictions_{ANNEE_TEST}_TripleBarrier.csv'
results.to_csv(output_path)
print(f"💾 Fichier de prédictions généré : {output_path}")