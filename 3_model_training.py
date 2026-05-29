"""Entraîne un modèle unique sur data ≤ TRAIN_END_YEAR (avec embargo PURGE_HOURS),
puis génère des prédictions OOS pour chaque année listée dans EVAL_YEARS.

Conforme à l'audit I5 : VAL_YEAR et TEST_YEAR proviennent du MÊME modèle qui
n'a jamais vu ces deux années en entraînement → split 3-étages strict.
"""
import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

from config import (
    DIR_RESULTS,
    EVAL_YEARS,
    FEATURES_DROPPED,
    FILE_ML_READY,
    PURGE_HOURS,
    RANDOM_SEED,
    RF_PARAMS,
    TRAIN_END_YEAR,
    VAL_YEAR,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(DIR_RESULTS, exist_ok=True)

# 1. Chargement des données
df = pd.read_csv(FILE_ML_READY, index_col='Time', parse_dates=True)

# 2. Coupe d'entraînement : tout ce qui est strictement avant (TRAIN_END_YEAR + 1) - PURGE_HOURS.
train_cutoff = pd.to_datetime(f'{TRAIN_END_YEAR + 1}-01-01') - timedelta(hours=PURGE_HOURS)
train_data = df[df.index < train_cutoff].copy()

if train_data.empty:
    raise ValueError(f"⚠️ Aucune donnée d'entraînement avant {train_cutoff}.")

colonnes_a_exclure = ['Target', 'Spread'] + list(FEATURES_DROPPED)
X_cols = [c for c in df.columns if c not in colonnes_a_exclure]

X_train = train_data[X_cols]
y_train = train_data['Target']

print(f"📊 Cutoff entraînement (purge {PURGE_HOURS}h) : {train_cutoff}")
print(f"📊 Taille entraînement : {len(X_train)} bougies")
print(f"📊 Années d'évaluation OOS : {EVAL_YEARS}")
if FEATURES_DROPPED:
    print(f"🚫 Features écartées (cf. config.FEATURES_DROPPED) : {FEATURES_DROPPED}")
print(f"✅ Features utilisées ({len(X_cols)}) : {X_cols}\n")

# 3. Entraînement (UNE SEULE FOIS, partagé par toutes les EVAL_YEARS)
model = RandomForestClassifier(**RF_PARAMS)
print(f"🤖 Entraînement du modèle (train ≤ {TRAIN_END_YEAR})...")
model.fit(X_train, y_train)

# 4. Importance des Features (impurity-based + permutation sur VAL_YEAR)
val_start = pd.to_datetime(f'{VAL_YEAR}-01-01')
val_end = pd.to_datetime(f'{VAL_YEAR + 1}-01-01')
val_slice = df[(df.index >= val_start) & (df.index < val_end)]

importances = model.feature_importances_

if val_slice.empty:
    print(f"\n⚠️ Pas de données pour VAL_YEAR={VAL_YEAR}, permutation importance ignorée.")
    fi_df = pd.DataFrame({
        'Indicateur': X_cols,
        'Impurity_%': np.round(importances * 100, 2),
    }).sort_values(by='Impurity_%', ascending=False)
else:
    print(f"\n🔁 Calcul de la permutation importance sur {VAL_YEAR} (peut prendre une minute)...")
    perm = permutation_importance(
        model, val_slice[X_cols], val_slice['Target'],
        n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1,
    )
    fi_df = pd.DataFrame({
        'Indicateur': X_cols,
        'Impurity_%': np.round(importances * 100, 2),
        'Permutation_mean': np.round(perm.importances_mean, 5),
        'Permutation_std': np.round(perm.importances_std, 5),
    }).sort_values(by='Permutation_mean', ascending=False)

print("\n=== IMPORTANCE DES INDICATEURS ===")
print(fi_df.to_string(index=False))

fi_path = f'{DIR_RESULTS}/Feature_Importance_train{TRAIN_END_YEAR}.csv'
fi_df.to_csv(fi_path, index=False)
print(f"💾 Importances sauvegardées : {fi_path}")

# 5. Prédictions OOS pour chaque EVAL_YEAR
class_map = None
for eval_year in EVAL_YEARS:
    eval_start = pd.to_datetime(f'{eval_year}-01-01')
    eval_end = pd.to_datetime(f'{eval_year + 1}-01-01')
    test_data = df[(df.index >= eval_start) & (df.index < eval_end)].copy()

    if test_data.empty:
        print(f"\n⚠️ Aucune donnée pour {eval_year}, on saute.")
        continue

    X_test = test_data[X_cols]
    y_test = test_data['Target']

    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)

    if class_map is None:
        class_map = {cls: idx for idx, cls in enumerate(model.classes_)}
    proba_baisse = probas[:, class_map[-1.0]]
    proba_neutre = probas[:, class_map[0.0]]
    proba_hausse = probas[:, class_map[1.0]]

    print(f"\n=== RÉSULTATS SUR {eval_year} ({len(X_test)} bougies) ===")
    print(f"✅ Accuracy : {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions))

    out_df = pd.DataFrame({
        'Close_Reel_Direction': y_test,
        'Prediction_Modele': predictions,
        'Confiance_Baisse_%': np.round(proba_baisse * 100, 2),
        'Confiance_Neutre_%': np.round(proba_neutre * 100, 2),
        'Confiance_Hausse_%': np.round(proba_hausse * 100, 2),
        'Spread': test_data['Spread'],
    }, index=y_test.index)

    output_path = f'{DIR_RESULTS}/Predictions_{eval_year}_TripleBarrier.csv'
    out_df.to_csv(output_path)
    print(f"💾 Prédictions sauvegardées : {output_path}")
