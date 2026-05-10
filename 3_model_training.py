import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
from datetime import timedelta

output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

# 1. Chargement des données
filepath = './ready-data/EURUSD_Master_ML_Ready.csv'
df = pd.read_csv(filepath, index_col='Time', parse_dates=True)

# 2. Séparation chronologique avec Purge (Correction du Data Leakage)
split_date = pd.to_datetime('2026-01-01')
purge_window = timedelta(hours=24) # La même durée que la fenêtre Triple Barrière

train_data = df[df.index < (split_date - purge_window)]
test_data = df[df.index >= split_date]

colonnes_a_exclure = ['Target', 'Spread'] 

X_train = train_data.drop(columns=colonnes_a_exclure)
y_train = train_data['Target']

X_test = test_data.drop(columns=colonnes_a_exclure)
y_test = test_data['Target']

print(f"Taille de l'entraînement : {len(X_train)} bougies")
print(f"Taille du test (2026) : {len(X_test)} bougies\n")

# 3. Entraînement du modèle
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=12, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced'
)

print("Entraînement en cours (Classification Multiclasse : -1, 0, 1)...")
model.fit(X_train, y_train)

# 4. Importance des Features
importances = model.feature_importances_
feature_names = X_train.columns

fi_df = pd.DataFrame({
    'Indicateur': feature_names, 
    'Importance_Environ_%': np.round(importances * 100, 2)
}).sort_values(by='Importance_Environ_%', ascending=False)

print("\n=== IMPORTANCE DES INDICATEURS ===")
print(fi_df.to_string(index=False))

plt.figure(figsize=(10, 8))
plt.barh(fi_df['Indicateur'][::-1], fi_df['Importance_Environ_%'][::-1])
plt.title("Importance des variables (Triple Barrière)")
plt.xlabel("Impact en % sur la décision")
plt.tight_layout()
plt.savefig(f'{output_dir}/Feature_Importance_TripleBarrier.png')

# 5. Prédictions et Probabilités
predictions = model.predict(X_test)
probas = model.predict_proba(X_test)

# Identification automatique des colonnes de probabilités
idx_short = np.where(model.classes_ == -1)[0][0]
idx_neutre = np.where(model.classes_ == 0)[0][0] # Ajout
idx_long = np.where(model.classes_ == 1)[0][0]

proba_baisse = probas[:, idx_short]
proba_neutre = probas[:, idx_neutre] # Ajout
proba_hausse = probas[:, idx_long]

print("\n=== RÉSULTATS SUR 2026 ===")
print(f"Précision globale (Accuracy) : {accuracy_score(y_test, predictions):.2f}\n")
print(classification_report(y_test, predictions))

# 6. Sauvegarde pour le Backtest
results = pd.DataFrame({
    'Close_Reel_Direction': y_test,
    'Prediction_Modele': predictions,
    'Confiance_Baisse_%': np.round(proba_baisse * 100, 2),
    'Confiance_Neutre_%': np.round(proba_neutre * 100, 2), # Ajout
    'Confiance_Hausse_%': np.round(proba_hausse * 100, 2),
    'Spread': test_data['Spread']
}, index=y_test.index)

output_path = f'{output_dir}/Predictions_2026_TripleBarrier.csv'
results.to_csv(output_path)
print(f"\nPrédictions détaillées sauvegardées dans : {output_path}")