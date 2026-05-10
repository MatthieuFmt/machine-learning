"""Configuration centralisée du pipeline ML/trading EURUSD.

Tout magic number ou chemin partagé entre plusieurs scripts vit ici.
Les scripts importent ce qu'ils utilisent : `from config import TP_PIPS, ...`
"""

# ----- Triple barrier -----
TP_PIPS = 20.0
SL_PIPS = 10.0
WINDOW_HOURS = 24
PIP_SIZE = 0.0001

# ----- Modèle -----
RANDOM_SEED = 42
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_SEED,
}

# Embargo / purge entre train et test.
# 48h = 2 × WINDOW_HOURS, conforme à la recommandation López de Prado (audit I3) :
# l'embargo doit être au moins égal à l'horizon de la triple barrière pour éviter
# que des labels du train chevauchent temporellement le début du test.
PURGE_HOURS = 48

# ----- Backtest -----
SEUIL_CONFIANCE = 0.45  # exprimé en fraction (0.45 = 45%)
COMMISSION_PIPS = 0.0   # audit I4 — à calibrer si on intègre une commission broker

# Référence capital pour exprimer les pips en returns % (audit I2).
# Hypothèse : 1 lot = 10 000 € notionnel, 1 pip sur EURUSD ≈ 1 € sur ce notionnel.
# Sharpe(returns) = Sharpe(pips) mathématiquement (scaling linéaire), mais
# le total_return_pct devient interprétable et comparable à un benchmark.
INITIAL_CAPITAL = 10_000.0
PIP_VALUE_EUR = 1.0

# ----- Sélection de features -----
# Features écartées de l'entraînement, classées par raison du drop :
#
# Pass 1 (sur les 18 features d'origine) : permutation_importance non significative.
# Critère : mean < 2×std (~95% CI sur le run 2024).
#
# Pass 2 (après pass 1, model à 12 features) : nouvelles features suspectes
# révélées une fois le bruit de pass 1 retiré.
#
# Ces colonnes restent calculées dans le CSV ML — elles sont seulement masquées
# au moment de bâtir X_train / X_test. Vider la liste = comportement d'origine.
FEATURES_DROPPED = [
    # === Pass 1 : non significatives ===
    'Dist_EMA_9',     # mean=0.00102 std=0.00185 — std > mean
    'Dist_EMA_21',    # mean=0.00254 std=0.00198 — borderline
    'Log_Return',     # mean=0.00242 std=0.00140 — borderline
    'XAU_Return',     # mean=0.00189 std=0.00150 — borderline (macro voulu mais non significatif)
    'CHF_Return',     # mean=0.00154 std=0.00121 — borderline (macro voulu mais non significatif)
    # === Pass 2 : permutation négative (le modèle prédit MIEUX sans elles) ===
    'Dist_EMA_50_D1', # pass2: mean=-0.00077 std=0.00255
    'BB_Width',       # pass2: mean=-0.00115 std=0.00182
    # === Pass 1 + Pass 2 : encodage cyclique cassé ===
    # Hour_Sin et Hour_Cos encodent l'heure de la journée comme (sin θ, cos θ).
    # Ce couple n'a de sens que pris ensemble — drop les deux pour cohérence.
    'Hour_Cos',       # pass1: mean=0.00070 std=0.00097 — std > mean
    'Hour_Sin',       # pass2: mean=0.00026 std=0.00089 (sans Hour_Cos l'encodage est cassé)
]


# ----- Splits 3-étages (audit I5) -----
# - TRAIN_END_YEAR : dernière année incluse dans l'entraînement (cutoff = TRAIN_END_YEAR + 1 - PURGE_HOURS).
# - VAL_YEAR : année de sélection (utilisée par optimize_sizing pour choisir la meilleure
#   fonction de poids parmi les candidats).
# - TEST_YEAR : année de validation finale (le modèle ET le sizing ne l'ont jamais vue).
# Le modèle entraîné par 3_model_training.py génère des prédictions pour TOUS les
# EVAL_YEARS depuis ce même cutoff — garantit que VAL_YEAR et TEST_YEAR sont strictement OOS.
TRAIN_END_YEAR = 2023
VAL_YEAR = 2024
TEST_YEAR = 2025
EVAL_YEARS = [VAL_YEAR, TEST_YEAR]

# ----- Chemins -----
DIR_RAW = './data'
DIR_CLEAN = './cleaned-data'
DIR_READY = './ready-data'
DIR_RESULTS = './results'
DIR_PREDICTIONS = './predictions'

# ----- Fichiers -----
FILE_ML_READY = f'{DIR_READY}/EURUSD_Master_ML_Ready.csv'
FILE_EURUSD_H1_CLEAN = f'{DIR_CLEAN}/EURUSD_H1_cleaned.csv'
FILE_EURUSD_H4_CLEAN = f'{DIR_CLEAN}/EURUSD_H4_cleaned.csv'
FILE_EURUSD_D1_CLEAN = f'{DIR_CLEAN}/EURUSD_D1_cleaned.csv'
FILE_XAUUSD_H1_CLEAN = f'{DIR_CLEAN}/XAUUSD_H1_cleaned.csv'
FILE_USDCHF_H1_CLEAN = f'{DIR_CLEAN}/USDCHF_H1_cleaned.csv'
