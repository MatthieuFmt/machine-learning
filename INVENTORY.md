# Inventaire du repo — 2026-05-14 (post-cleanup Prompt 02)

> Structure après nettoyage. `learning_machine_learning/` (v1) et `archive_v1/` supprimés. `learning_machine_learning_v2/` renommé `app/`.

## Dossiers de premier niveau

| Dossier | Rôle | Statut |
|---|---|---|
| `.claude/` | Configuration Claude (prompts système, règles) | À conserver |
| `.roo/` | Configuration Roo (skills, modes) | À conserver |
| `app/` | Code source principal (ex-`learning_machine_learning_v2/`) — Donchian Breakout + méta-labeling | Base active pour v3 |
| `docs/` | Rapports d'hypothèses v2 (H01–H05), roadmap v3 | À conserver — référence |
| `docs/archive_v1/` | Specs v1 obsolètes (step_*.md) — archivées | Archive — ne pas modifier |
| `predictions/` | Rapports JSON des backtests | À conserver |
| `prompts/` | 26 prompts (00_constitution à 24_post_deployment) | À conserver — contrat d'exécution |
| `scripts/` | Scripts utilitaires (inspection CSV, scraping ForexFactory) | À conserver |
| `tests/` | Suite de tests (unitaires, intégration, acceptance) | À conserver et étendre |

## Actifs disponibles (`data/raw/`)

| Actif | Timeframes | Fichier exemple |
|---|---|---|
| BTCUSD | D1, H1, H4 | BTCUSD_D1.csv |
| ETHUSD | D1, H1, H4 | ETHUSD_D1.csv |
| EURUSD | D1, H1, H4 | EURUSD_D1.csv |
| GBPUSD | D1, H1, H4 | GBPUSD_D1.csv |
| US30 | D1, H1, H4 | USA30IDXUSD_D1.csv |
| USDCHF | D1, H1, H4 | USDCHF_D1.csv |
| XAUUSD | D1, H1, H4 | XAUUSD_D1.csv |

**Format** : Tab-separated, colonnes `Time, Open, High, Low, Close, Volume` (+ `Spread` optionnelle). Découverts par [`app.data.registry.discover_assets()`](app/data/registry.py).
**Calendrier** : Jours fériés XTB dans [`app.config.calendar.XTB_HOLIDAYS`](app/config/calendar.py).

## Couche Data (`app/data/`)

| Fichier | Rôle |
|---|---|
| [`app/data/loader.py`](app/data/loader.py) | `load_asset(asset, tf)` — chargement + validation OHLCV, lecture adaptative 6/7 colonnes, gap analysis |
| [`app/data/registry.py`](app/data/registry.py) | `discover_assets()` — scan `data/raw/` sans lire le contenu des CSV |
| [`app/data/calendar_loader.py`](app/data/calendar_loader.py) | `load_calendar()` — calendrier économique macro (ForexFactory) |
| [`app/config/calendar.py`](app/config/calendar.py) | `XTB_HOLIDAYS`, `is_market_open()`, `is_normal_gap()` — jours fériés XTB par actif |
| [`app/core/exceptions.py`](app/core/exceptions.py) | `DataValidationError(PipelineError)` — exception de validation |

## Scripts `run_*.py` à la racine

| Script | Hypothèse | Description | Statut |
|---|---|---|---|
| `run_pipeline_us30.py` | H01 | RF 6 features OHLC sur US30 D1 | ❌ NO-GO (Sharpe −1.27) |
| `run_pipeline_xauusd.py` | H02 | RF sur XAUUSD H4 | ❌ NO-GO (Sharpe −2.52) |
| `run_deterministic_grid.py` | H03 | Grid search 164 backtests, stratégies déterministes multi-actif | ✅ GO (Donchian Breakout US30 D1, Sharpe +3.07) |
| `run_meta_labeling_cpcv.py` | H04 | Donchian + méta-labeling RF + CPCV | ✅ GO (Sharpe OOS +8.61, CPCV 5.79) |
| `run_walk_forward_us30.py` | H05 | Walk-forward paper trading US30 D1, Config A vs B | ✅ GO (Config B Sharpe +8.84, 12 trades/30 mois) |
| `run_v3_phase1.py` | H06–H08 | Phase 1 v3 : grid search Donchian multi-actif + 4 strats alt + portfolio equal-risk | Résultats dans `predictions/v3_phase1_results.json` |

## Rapports d'hypothèses (`docs/`)

| Fichier | Hypothèse | Verdict | Sharpe OOS |
|---|---|---|---|
| `v2_hypothesis_01.md` | H01 — RF US30 D1 | ❌ NO-GO | −1.27 |
| `v2_hypothesis_02.md` | H02 — RF XAUUSD H4 | ❌ NO-GO | −2.52 |
| `v2_hypothesis_03.md` | H03 — Grid search déterministe | ✅ GO | +3.07 |
| `v2_hypothesis_04.md` | H04 — Donchian + méta-labeling RF | ✅ GO | +8.61 (CPCV 5.79) |
| `v2_hypothesis_05.md` | H05 — Walk-forward US30 | ✅ GO | +8.84 (walk-forward) |
| `v3_roadmap.md` | Roadmap v3 (H06–H18) | — | — |
| `step_01_target_redefinition.md` | Redéfinition cible (régression forward-return) | Documentaire | — |
| `step_02_robust_validation_framework.md` | CPCV + DSR + PSR | Documentaire | — |
| `step_03_gbm_primary_classifier.md` | LightGBM/XGBoost + Optuna | Documentaire | — |
| `step_04_session_aware_features.md` | Features de session (Tokyo/Londres/NY) | Documentaire | — |
| `step_05_economic_calendar_integration.md` | Calendrier macro ForexFactory | Documentaire | — |
| `step_06_meta_labeling_calibration.md` | Calibration méta-labeling | Documentaire | — |
| `step_07_cross_asset_validation.md` | Validation cross-actif | Documentaire | — |
| `step_08_postmortem_and_v2_roadmap.md` | Post-mortem v1 → roadmap v2 | Documentaire | — |
| Fichiers `step_*_final_report.md` (×2) | Rapports finaux step 01 et 02 | Documentaire | — |
| Fichiers `step_*_implementation_spec.md` (×5) | Specs d'implémentation steps 01, 02, 04, 05, 07 | Documentaire | — |

## Code source

| Dossier | Rôle | Modules |
|---|---|---|
| `app/` (ex-v2) | Pipeline US30 D1 + Donchian + méta-labeling + stratégies | `analysis/`, `backtest/`, `config/`, `core/`, `features/`, `models/`, `pipelines/`, `strategies/`, `targets/` |

## Tests existants

| Dossier | Nombre de fichiers* | Couverture estimée |
|---|---|---|
| `tests/unit/` | 23 | Large : features, backtest, modèles, validateurs, simulateurs, filtres |
| `tests/integration/` | 1 (init seulement) | Faible |
| `tests/acceptance/` | 1 | Pipeline complet EURUSD |
| `tests/fixtures/` | — | Données de test |

*\*Fichiers `test_*.py` uniquement, hors `__init__.py`.*

## Fichiers racine notables

| Fichier | Rôle |
|---|---|
| `pyproject.toml` | Configuration projet Python (dépendances, outils) |
| `requirements.txt` | Dépendances pip |
| `ml_evolution.md` | Changelog historique des 16 itérations v1 (EURUSD H1) |
| `CLAUDE.md` | Configuration Claude |
| `prompt.txt` | Fichier de prompt utilisateur |
| `README.md` | Documentation principale |
| `.gitignore` | Règles d'exclusion git |
