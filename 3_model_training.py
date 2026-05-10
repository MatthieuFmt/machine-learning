import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# 1. Chargement des données
filepath = './ready-data/EURUSD_Master_ML_Ready.csv'
df = pd.read_csv(filepath, index_col='Time', parse_dates=True)

# 2. Séparation chronologique (Train / Test)
train_data = df[df.index < '2026-01-01']
test_data = df[df.index >= '2026-01-01']

colonnes_a_exclure = ['Target', 'Spread']

X_train = train_data.drop(columns=colonnes_a_exclure)
y_train = train_data['Target']

X_test = test_data.drop(columns=colonnes_a_exclure)
y_test = test_data['Target']

print(f"Taille de l'entraînement : {len(X_train)} bougies")
print(f"Taille du test (2026) : {len(X_test)} bougies\n")

# 3. Entraînement du modèle (Version Améliorée)
# Augmentation du nombre d'arbres (200) et équilibrage des classes
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=12, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced' # NOUVEAU : Empêche le biais directionnel
)

print("Entraînement en cours (cela peut prendre quelques secondes)...")
model.fit(X_train, y_train)

# 4. Importance des Features (Ce qui fait décider le modèle)
importances = model.feature_importances_
feature_names = X_train.columns

# Création d'un tableau trié
fi_df = pd.DataFrame({
    'Indicateur': feature_names, 
    'Importance_Environ_%': np.round(importances * 100, 2)
}).sort_values(by='Importance_Environ_%', ascending=False)

print("\n=== IMPORTANCE DES INDICATEURS ===")
print(fi_df.to_string(index=False))

# Optionnel : Sauvegarder un graphique de l'importance
plt.figure(figsize=(10, 6))
plt.barh(fi_df['Indicateur'][::-1], fi_df['Importance_Environ_%'][::-1])
plt.title("Importance des variables pour l'EUR/USD H1")
plt.xlabel("Impact en % sur la décision")
plt.tight_layout()
plt.savefig('./cleaned-data/Feature_Importance.png')
print("Graphique d'importance sauvegardé.\n")

# 5. Prédictions et Probabilités (NOUVEAU)
predictions = model.predict(X_test)
# predict_proba renvoie [Probabilité_Baisse, Probabilité_Hausse]
proba_hausse = model.predict_proba(X_test)[:, 1] 

print("=== RÉSULTATS SUR 2026 ===")
print(f"Précision globale (Accuracy) : {accuracy_score(y_test, predictions):.2f}\n")
print(classification_report(y_test, predictions))

# 6. Sauvegarde intelligente pour le Backtest
results = pd.DataFrame({
    'Close_Reel_Direction': y_test,
    'Prediction_Modele': predictions,
    'Confiance_Hausse_%': np.round(proba_hausse * 100, 2), # Le % de certitude
    'Spread': test_data['Spread']
}, index=y_test.index)

output_path = './cleaned-data/Predictions_2026_Baseline_H1.csv'
results.to_csv(output_path)
print(f"Prédictions détaillées (avec certitude) sauvegardées dans : {output_path}")