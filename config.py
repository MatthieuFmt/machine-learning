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
# Priorité 1 : réduction overfitting — arbres moins profonds, plus nombreux, splits robustes.
# depth 12→6   : empêche d'apprendre des patterns de bruit.
# leaf 5→50    : chaque split doit couvrir au moins 50 barres.
# n 200→500    : stabilise la forêt avec plus d'arbres simples.
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'min_samples_leaf': 50,
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
# Priorité 3 : seuil abaissé de 0.45→0.38 (en 2025 proba_max≥0.45 ne couvre que 1.6% des barres).
# La plage utile identifiée par l'analyse : 0.36-0.42. 0.38 est le point médian testé sur 2024.
#
# Pass 5 (2026-05-11) : après nettoyage des features pourries, proba_max OOS
# s'effondre (4 trades 2024, 2 trades 2025 au seuil 0.38). Baissé à 0.35 pour
# tester si l'edge tient avec plus de trades (objectif : 20-30 trades/an OOS).
SEUIL_CONFIANCE = 0.35
COMMISSION_PIPS = 0.5   # audit I4 — commission broker aller-retour standard EURUSD
SLIPPAGE_PIPS = 1.0     # slippage réaliste H1 (entrée au Close, exécution non garantie)

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
# Pass 3 (Priorité 2, analyse 2025-05-10) : RSI_14_H4 et Dist_EMA_20_H4 sont
# confirmées comme bruit pur (permutation_mean < 0.004, std comparable).
# Le modèle n'utilise que les features D1 + ADX_14 de façon significative.
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
    # === Pass 3 (Priorité 2) : features H4 non discriminantes ===
    'RSI_14_H4',      # permutation_mean=0.00125 std=0.00151 — bruit pur
    'Dist_EMA_20_H4', # permutation_mean=0.00341 std=0.00253 — bruit pur
    # === Pass 4 (2026-05-11) : features de régime nuisibles au modèle ===
    # Ces features restent calculées dans le CSV — les FILTRES de régime
    # (USE_TREND_FILTER, USE_VOL_FILTER) en ont besoin — mais elles sont
    # exclues du training car leur permutation_importance est nettement
    # négative : le RF les utilise beaucoup (impurity ~11-15%) mais apprend
    # des règles contre-productives qui se retournent en OOS.
    'Dist_SMA200_D1',         # perm=-0.02240 std=0.00307 — feature de filtre uniquement
    'ATR_Norm',               # perm=-0.00282 std=0.00496 — feature de filtre uniquement
    'Volatilite_Realisee_24h',# perm=-0.00051 std=0.00436 — bruit pur (std > |mean|)
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

# ----- Filtres de régime (Priorité 4) -----
# Active/désactive chaque filtre individuellement pour tests d'ablation.
USE_TREND_FILTER = True      # Close > SMA200 → LONG only ; Close < SMA200 → SHORT only
USE_VOL_FILTER = True        # Ignore signaux si ATR_Norm > 2× médiane glissante 168h
USE_SESSION_FILTER = True    # Ignore signaux entre 22h-01h GMT (faible liquidité)

# Paramètres des filtres
# Note : la SMA200 pour le filtre de tendance est désormais calculée sur D1
# directement dans 2_master_feature_engineering.py (feature Dist_SMA200_D1).
VOL_FILTER_WINDOW = 168             # fenêtre médiane ATR_Norm (1 semaine)
VOL_FILTER_MULTIPLIER = 2.0         # seuil = multiplier × médiane
SESSION_EXCLUDE_START = 22          # heure GMT début exclusion (22h)
SESSION_EXCLUDE_END = 1             # heure GMT fin exclusion (1h le lendemain)

# ----- Variantes TP/SL à tester (Priorité 5) -----
TP_SL_VARIANTS = {
    'baseline':     (20.0, 10.0),   # actuel — ratio 2:1
    'ratio_1_1':    (20.0, 20.0),   # plus facile à gagner, WR devrait monter
    'ratio_3_1':    (30.0, 10.0),   # payoff asymétrique si features discriminantes
}

# Seuils alternatifs à tester avec le sizing optimisé
SEUILS_ALTERNATIFS = [0.36, 0.38, 0.40, 0.42]

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
