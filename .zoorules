## 1. Typage
+ - `mypy --strict` doit passer (voir pyproject.toml)
+ - `ruff check` doit passer (E/F/I/N/W/UP/B/C4/SIM)
- Préférer `typing.Protocol` pour les signatures injectables (weight_func, filter)

## 2. Tests
- Commande : `python -m pytest tests/ -v --tb=short` (couvre unit + integration + acceptance)
+ - `unit/` : pas d'I/O ; `integration/` : I/O fixtures ; `acceptance/` : bout en bout
+ - Coverage seuil minimum 50% (configuré dans pyproject.toml)

## 3. Vectorisation
- Zéro `iterrows()`, `itertuples()`, `apply()` avec lambda.
+ - Exception : boucles `for` autorisées si l'algorithme est **séquentiel par nature** (stateful simulation, triple-barrier early-exit)

## 4. Anti-Data Leakage
- Triple barrier : les `window` dernières barres restent NaN (pas de label sans forward bars suffisantes)
+ - Purge `purge_hours` (≥ window) entre train et OOS (López de Prado embargo)
+ - Split 3-étages : train ≤ TRAIN_END_YEAR / VAL_YEAR / TEST_YEAR — le modèle ne voit jamais val ni test

## 5. Logging
- Utiliser `logger.exception(...)` dans tous les blocs `except` (capture le traceback automatiquement)
- Interdire `print()` dans `learning_machine_learning/` (scripts legacy `1_*` à `4_*` exemptés)

+ ## 8. Architecture du package
+ - `config/` : dataclasses immuables
+ - `core/` : logging, exceptions, types
+ - `data/` : ingestion + cleaning + validation
+ - `features/` : feature engineering, triple barrier
+ - `model/` : training, evaluation, prediction
+ - `backtest/` : simulator, filters, sizing, metrics, reporting
+ - `analysis/` : diagnostics post-backtest
+ - `pipelines/` : orchestrateurs par actif (BasePipeline + concrets)

+ ## 9. Inversion de dépendances
+ - `simulate_trades` reçoit `weight_func` et `filter_pipeline` en paramètres — JAMAIS `from config import *`
+ - Les filtres implémentent un protocole `apply(df, mask_long, mask_short)` et raisent ValueError si colonne manquante
+ - Pour dériver une config, `dataclasses.replace(config, ...)` — jamais de mutation

+ ## 10. Séparation feature-modèle vs feature-filtre
+ - Features exclues de X_train mais nécessaires aux filtres backtest : préservées via `FILTER_KEEP` (features/pipeline.py) et `_FILTER_ONLY_COLS` (model/training.py)
+ - Ex : `ATR_Norm`, `Dist_SMA200_D1` sont gardées dans `ml_data` mais retirées de `X_cols`

+ ## 11. Workflow humain
+ - Ne pas lancer `run_pipeline_v1.py` automatiquement — laisser l'utilisateur déclencher
+ - Ne pas commit sans accord explicite
+ - Documenter chaque modif dans `ml_evolution.md` AVANT de proposer la suivante