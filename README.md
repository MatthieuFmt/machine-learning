# Pipeline ML/Trading — EURUSD H1 Triple Barrier

Stratégie de trading algorithmique sur EURUSD H1 basée sur la méthode **triple barrier** (López de Prado) et un RandomForest.

## Flux d'exécution

```
1_clean_data.py
    └─► cleaned-data/
        └─► 2_master_feature_engineering.py
                └─► ready-data/EURUSD_Master_ML_Ready.csv
                    └─► 3_model_training.py
                            └─► results/Predictions_{2024,2025}_TripleBarrier.csv
                                ├─► 4_backtest_triple_barrier.py   ← résultats + rapports
                                ├─► optimize_sizing.py              ← optimisation du sizing
                                └─► 5_analyze_losses.py             ← analyse des pertes
```

## Prérequis

### Données (dossier `./data/`)

Fichiers MetaTrader 5 exportés en `.csv` (séparateur tabulation, colonnes Time/O/H/L/C/Vol/Spread) :

| Fichier attendu | Timeframe | Usage |
| :--- | :--- | :--- |
| `EURUSD_H1.csv` | H1 | Signal + backtest |
| `EURUSD_H4.csv` | H4 | Features multi-TF |
| `EURUSD_D1.csv` | D1 | Features multi-TF |
| `XAUUSD_H1.csv` | H1 | Feature macro XAU |
| `USDCHF_H1.csv` | H1 | Feature macro CHF |

### Installation

```bash
pip install -r requirements.txt
```

## Paramètres clés (`config.py`)

| Paramètre | Valeur | Description |
| :--- | :--- | :--- |
| `TP_PIPS` / `SL_PIPS` | 20 / 10 | Take profit / Stop loss |
| `WINDOW_HOURS` | 24 | Horizon maximum d'un trade |
| `PURGE_HOURS` | 48 | Embargo train/test (≥ 2 × WINDOW_HOURS) |
| `TRAIN_END_YEAR` | 2023 | Dernière année incluse dans l'entraînement |
| `VAL_YEAR` | 2024 | Validation sizing (jamais vu par le modèle) |
| `TEST_YEAR` | 2025 | Test final (jamais vu par modèle ni sizing) |
| `SEUIL_CONFIANCE` | 0.45 | Seuil de confiance pour déclencher un signal |
| `FEATURES_DROPPED` | 9 features | Features écartées (permutation importance < seuil) |

## Lancement

```bash
# 1. Nettoyer les données brutes
python 1_clean_data.py

# 2. Construire les features + labels triple barrier
python 2_master_feature_engineering.py

# 3. Entraîner le modèle + générer les prédictions OOS (2024 + 2025)
python 3_model_training.py

# 4a. Backtest avec le sizing par défaut
python 4_backtest_triple_barrier.py

# 4b. Optimiser le sizing sur VAL_YEAR, tester sur TEST_YEAR
python optimize_sizing.py

# 5. Analyser les trades perdants (nécessite 4a exécuté avant)
python 5_analyze_losses.py
```

## Structure des sorties

```
results/
├── Predictions_{2024,2025}_TripleBarrier.csv   # prédictions brutes du modèle
├── Trades_Detailed_{2024,2025}.csv             # liste des trades (pour 5_analyze_losses.py)
└── Feature_Importance_train2023.csv            # impurity + permutation importance

predictions/
└── Rapport_Performance_{annee}.md              # rapport auto-généré par backtest
```

## Limites connues

- **Sharpe élevé** : le ratio de Sharpe affiché (~11-13) est probablement surestimé. Cause suspectée : `spread_cost = spreads[i] / 10.0` dans `backtest_utils.py` — vérifier l'unité brute de la colonne `Spread` (pips × 10 ? points ?). Aucun slippage ni commission broker n'est modélisé (`COMMISSION_PIPS = 0`).
- **Pas de walk-forward** : modèle entraîné une seule fois sur ≤ 2023 ; un walk-forward sur plusieurs fenêtres glissantes renforcerait la robustesse.
- **Un seul actif** : pipeline non testé sur d'autres paires.
