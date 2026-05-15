# Journal d'exécution — Refonte v3

> Ce fichier est la mémoire vive du projet. À lire au début de chaque session, à mettre à jour à la fin.

---

## Historique v1 (résumé, archivé)

- EURUSD H1 + RandomForest sur features techniques (RSI, ADX, EMA, etc.).
- 16 itérations de tuning documentées dans [`ml_evolution.md`](ml_evolution.md).
- Verdict final : ❌ NO-GO. Sharpe ≤ 0 sur toutes les années OOS, accuracy ≈ aléatoire (0.332).
- DSR 2025 = −1.97, p(Sharpe>0) = 0.29. Biais directionnel SHORT 75–85%.
- Cause racine : RF sur indicateurs bruts ne contient aucune info prédictive forward. Cible bruitée à 36% NEUTRE.
- Code source v1 supprimé au Prompt 02. Historique conservé dans `ml_evolution.md` et `docs/archive_v1/`.

---

## Historique v2 (H01–H05)

### H01 — RF sur US30 D1 (6 features OHLC)
- **Verdict** : ❌ NO-GO
- **Sharpe OOS** : −1.27
- **Trades** : 66, WR 24.2 %
- **Leçon** : RF seul = pas de signal.

### H02 — RF sur XAUUSD H4
- **Verdict** : ❌ NO-GO
- **Sharpe OOS** : −2.52
- **Trades** : 42, WR 16.7 %
- **Leçon** : RF + TP/SL fixes inadaptés à XAUUSD H4.

### H03 — Grid search déterministe (164 backtests)
- **Verdict** : ✅ GO
- **Stratégie gagnante** : Donchian Breakout (20, 20) sur US30 D1
- **Sharpe OOS** : +3.07
- **Leçon** : L'edge se trouve par grid search systématique, pas par ML.

### H04 — Donchian + méta-labeling RF (CPCV)
- **Verdict** : ✅ GO
- **Sharpe OOS** : +8.61 (moyen CPCV 5.79, std ±10.03)
- **Leçon** : Le ML en SURCOUCHE améliore. Mais instabilité élevée.

### H05 — Walk-forward US30 (Config A vs B)
- **Verdict** : ✅ GO
- **Config B (Donchian + RF méta-labeling)** : Sharpe walk-forward +8.84
- **12 trades sur 30 mois** — peu pour valider robustesse
- **Leçon** : Walk-forward stabilise. Mais nombre de trades insuffisant (critère ≥ 30/an non atteint).

---

## Roadmap v3 cible (résumé de `docs/v3_roadmap.md`)

| Phase | Prompts | Contenu |
|---|---|---|
| Phase 0 — Nettoyage | 01, 02, 02b | Audit, cleanup, quality gates |
| Phase 1 — Data & Features | 03, 04, 05, 06 | Data layer, features harness, calendrier économique, validation framework |
| Phase 2 — Expansion univers | 07, 08, 09 | H06 Donchian multi-actif, H07 stratégies alternatives, H08 portefeuille equal-risk |
| Phase 3 — Régime & filtrage | 10, 11, 12, 13, 14 | H09 régime detector, H10-H12 méta-labeling, H11 features avancées, H12 session, H13 corrélation |
| Phase 4 — Portfolio avancé | 15, 16, 17 | H14 vol targeting, H15 TF décision, H16 timeframe stacking |
| Phase 5 — Validation finale | 18, 19 | Validation finale, H18 walk-forward continu |
| Phase 6 — Production | 20, 21, 22, 23, 24 | Signal engine, Telegram alerts, scheduler, VPS, monitoring |

**Objectif final** : Sharpe walk-forward portfolio ≥ 1.0, DSR > 0 (p < 0.05), DD < 15 %, WR > 30 %, ≥ 30 trades/an.

---

## Compteur n_trials cumulatif

| Prompt | Hypothèse | n_trials_new | n_trials_cumul | Verdict | Sharpe |
|---|---|---|---|---|---|
| baseline | v1 EURUSD H1 (16 itérations) | 16 | 16 | ❌ NO-GO | — |
| baseline | v2 H01 | 1 | 17 | ❌ NO-GO | −1.27 |
| baseline | v2 H02 | 1 | 18 | ❌ NO-GO | −2.52 |
| baseline | v2 H03 | 1 | 19 | ✅ GO | +3.07 |
| baseline | v2 H04 | 1 | 20 | ✅ GO | +8.61 |
| baseline | v2 H05 | 1 | 21 | ✅ GO | +8.84 |
| 03–06 | (phase data, pas d'hypothèse) | 0 | 21 | — | — |
| 07 | H06 (Donchian multi-actif) | 1 | 22 | 🔴 NO-GO | 6 testés, 0 GO (US30 −0.09, XAUUSD +1.46, GER30 −1.01, US500 −0.85, XAGUSD 0.00, USOIL erreur) |
| pivot | H1 (méta-labeling RF XAUUSD D1) | 1 | 23 | ❌ NO-GO | 0 trade méta — split structurel train ≤2022 non-profitable |
| pivot | H5 (RSI(2) mean-reversion US30 H1) | 1 | 24 | ❌ NO-GO | Sharpe=−0.95, DSR=−59.2, DD=92.8% |
| 08 | H07 (strats alt) | — | — | — | — |
| 09 | H08 (portfolio equal-risk) | — | — | — | — |
| 10 | H09 (régime detector) | — | — | — | — |
| 11 | H10-H12 (méta-labeling v3) | — | — | — | — |
| 12 | H11 (features avancées) | — | — | — | — |
| 13 | H12 (session features) | — | — | — | — |
| 14 | H13 (corrélation weighting) | — | — | — | — |
| 15 | H14 (vol targeting) | — | — | — | — |
| 16 | H15 (TF décision) | — | — | — | — |
| 17 | H16 (timeframe stacking) | — | — | — | — |
| 18 | Validation finale | — | — | — | — |
| 19 | H18 (walk-forward continu) | — | — | — | — |

---

## Sessions Deepseek

---

## 2026-05-14 — Prompt 01 : Audit initial

- **Statut** : ✅ Terminé
- **Fichiers créés** : `INVENTORY.md`, `JOURNAL.md`
- **Résultats clés** :
  - 0 CSV trouvés dans `data/raw/` (dossier vide, seul `economic_calendar/` présent)
  - 6 scripts `run_*.py` à la racine (H01–H05 + v3 phase1)
  - 22 rapports/docs dans `docs/` (5 hypothèses v2 + roadmap v3 + 12 specs/rapports step + 2 README)
  - 25 fichiers de test (23 unit, 1 acceptance, 1 conftest)
  - n_trials cumul initialisé à 21 (16 v1 + 5 v2)
- **Problèmes rencontrés** : Aucun CSV d'actif dans `data/raw/` — l'utilisateur devra les fournir avant le prompt 03 (data layer)
- **Hypothèses à explorer ensuite** : (traitées au prompt 02)

## 2026-05-14 — Prompt 02 : Nettoyage et restructuration

- **Statut** : ✅ Terminé
- **Fichiers/dossiers supprimés** : `learning_machine_learning/` (v1), `archive_v1/`, `results/`, `__pycache__/` (tous niveaux), `.pytest_cache/`
- **Fichiers déplacés** : 17 fichiers `docs/step_*.md` → `docs/archive_v1/`
- **Fichiers renommés** : `learning_machine_learning_v2/` → `app/` (via `git mv`)
- **Imports corrigés** : 24 fichiers `.py` (6 scripts racine + 18 internes `app/`)
- **Fichiers markdown mis à jour** : `INVENTORY.md`, `CLAUDE.md`, `README.md`, `.gitignore`, `JOURNAL.md`
- **Tests pytest** : Non exécutés (constitution règle 2 — exécution sur demande)
- **Problèmes rencontrés** : Aucun
- **Structure finale** : `app/`, `docs/`, `docs/archive_v1/`, `prompts/`, `tests/`, `scripts/`, `predictions/`, `data/raw/`

## 2026-05-14 — Prompt 02b : Quality Gates

- **Statut** : ✅ Terminé
- **Fichiers créés** : `pyproject.toml` (remplacé), `.pre-commit-config.yaml`, `requirements-dev.txt`, `Makefile`, `.github/workflows/ci.yml`
- **Fichiers enrichis** : `.gitignore` (`.env.local`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, `TEST_SET_LOCK.json`, `models/snapshots/`, `logs/`, `predictions/*.json`, `predictions/*.csv`)
- **Modules créés** : `app/testing/look_ahead_validator.py`, `app/testing/snooping_guard.py`, `app/core/retry.py`, `app/core/seeds.py`, `app/config/models.py`, `scripts/verify_no_snooping.py`
- **Tests unitaires** : 5 fichiers, 11 tests, 0 failures
- **Ruff (périmètre 02b)** : ✅ `All checks passed!`
- **Snooping guard** : ✅ `TEST_SET_LOCK.json absent : pas de scan nécessaire.`
- **make verify complet** : ⚠️ Non exécuté (231 violations ruff pré-existantes dans `app/` et `tests/` — imports `learning_machine_learning.*` résiduels de l'ère v1. Ces corrections relèvent du prompt 03+)
- **Notes** : `pre-commit install` non exécuté (nécessiterait `git init` ou un repo déjà initialisé avec hooks). Dépendances dev installées : `mypy`, `ruff`, `black`, `pre-commit`, `hypothesis`.

## 2026-05-14 — Prompt 03 : Data layer

- **Statut** : ✅ Terminé
- **Fichiers créés** : `app/config/calendar.py`, `app/data/registry.py`, `tests/unit/test_calendar.py`, `tests/unit/test_data_loader.py`
- **Fichiers modifiés** : `app/data/loader.py` (refonte complète — lecture adaptative 6/7 colonnes, gap analysis, validation OHLCV stricte)
- **Tests pytest** : ✅ 2 fichiers, 27 tests (8 calendar + 19 data_loader), 0 failures
- **Ruff** : ✅ All checks passed
- **Actifs détectés via `discover_assets()`** : `BTCUSD`, `ETHUSD`, `EURUSD`, `GBPUSD`, `US30`, `USDCHF`, `XAUUSD` (tous D1, H1, H4)
- **Problèmes rencontrés** :
  - CSV US30 D1 : 6 noms de colonnes pour 7 colonnes de données (timestamp + OHLCV + Spread) → pandas décalait tous les headers. Résolu par détection adaptative `n_headers` vs `n_data` avec `csv.reader`.
  - 326 timestamps "dupliqués" étaient un artefact du décalage de colonnes (la colonne Open était interprétée comme timestamp).
  - `timezone.utc` → `datetime.UTC` (ruff UP017) sur tout `test_calendar.py`.
  - Variable `l` → `lo` dans [`loader.py`](app/data/loader.py:144) (ruff E741).

## 2026-05-14 — Prompt 04 : Feature research harness

- **Statut** : ✅ Terminé
- **Fichiers créés** : `app/features/__init__.py` (existant, vidé), `app/features/indicators.py` (422 lignes, 18 indicateurs + `compute_all_indicators`), `app/features/research.py` (185 lignes, `rank_features`), `scripts/run_feature_research.py` (CLI), `tests/unit/test_indicators.py` (312 lignes, 61 tests), `tests/unit/test_feature_research.py` (integration mockée), `prompts/04_architecture_plan.md`
- **Tests pytest** : ✅ 61/61 passed (46 indicators + 15 research)
- **Ruff** : ✅ `All checks passed!` sur les 5 fichiers
- **Problèmes rencontrés** :
  - `pd.NA` dans `replace(0, pd.NA)` forçait un dtype `object` → `ewm()`/`rolling()` échouaient. Résolu : `replace(0, np.nan)` partout (8 occurrences).
  - `williams_r` et `cci` testés comme univariés mais sont multivariés (H, L, C) → déplacés dans `test_non_look_ahead_multivariate`.
  - `max(axis=1, skipna=True)` par défaut ignorait le NaN de `prev_close` sur la 1ère barre dans `atr()` et `adx()`. Résolu : `skipna=False`.
  - `_ohlcv_dataframe()` dans les tests créait `close` sans index DatetimeIndex → pandas alignait par index et mettait tout OHLC à NaN. Résolu : `close.index = dates` avant arithmétique.
  - `ewm()` propage les NaN indéfiniment → pattern `dropna()` + `ewm()` + `reindex()` dans `atr()` ; `mask_valid` + `ewm()` + `reindex()` dans `adx()`.
- **Notes** : `n_trials` inchangé (ce prompt n'est pas une phase d'hypothèse). Tous les indicateurs utilisent exclusivement `.shift()`, `.rolling()`, `.ewm()` — zéro boucle Python row-by-row.

## 2026-05-14 — Prompt 05 : Economic calendar

- **Statut** : ✅ Terminé
- **Fichiers créés** : `app/features/economic.py` (283 lignes, `load_calendar` + `compute_event_features`), `tests/unit/test_economic_features.py` (321 lignes, 25 tests), `prompts/05_architecture_plan.md`
- **Fichiers modifiés** : `app/features/indicators.py` (ajout paramètre `include_economic` dans `compute_all_indicators`)
- **Tests pytest** : ✅ 25/25 passed (economic) + 50/50 passed (indicators) = 75/75
- **Ruff** : ✅ `All checks passed!` sur les 3 fichiers
- **Problèmes rencontrés** :
  - `pd.date_range(..., tz="UTC")` en pandas ≥ 2.0 produit `datetime64[us, UTC]` — `.asi8` et `.values.view(np.int64)` retournent des microsecondes, pas des nanosecondes. Résolu : helper `_to_ns()` qui détecte l'unité native et normalise avec `.as_unit("ns").asi8`.
  - Les features `hours_*` étaient 1000× trop petites ; les fenêtres `event_high_within_*` couvraient 1000× trop large.
  - `filter(like="event_")` dans `test_empty_calendar` capturait `hours_to_next_event_high` → remplacé par `filter(regex="^event_")`.
  - `.astype("datetime64[ns]")` échoue sur tz-aware → abandonné au profit de `_to_ns()`.
- **Architecture** : `_event_within_window`, `_hours_since_last`, `_hours_to_next` utilisent exclusivement `np.searchsorted` — O(E × log B), zéro boucle Python row-by-row.
- **Notes** : `n_trials` inchangé. 9 features économiques : 6 booléennes `event_high_within_{1,4,24}h_{USD,EUR}` + 3 numériques `hours_since_last_{nfp,fomc}` + `hours_to_next_event_high`. Sentinelle `np.nan` pour "pas d'event". Anti-look-ahead vérifié par `test_anti_look_ahead_consistency`.

## 2026-05-14 — Prompt 06 : Validation framework

- **Statut** : ✅ Terminé
- **Fichiers créés** : `prompts/06_architecture_plan.md`, `tests/unit/test_indicators_look_ahead.py` (scan dynamique des 5 modules features)
- **Fichiers modifiés** : `app/analysis/edge_validation.py` (réécriture complète : 9 fonctions publiques + EdgeReport + v2 compat), `tests/unit/test_edge_validation.py` (25 tests), `tests/unit/test_walk_forward.py` (8 tests), `app/features/calendar.py` (fix import `learning_machine_learning` + décorateurs `@look_ahead_safe`), `app/features/regime.py` (idem), `app/features/economic.py` (`@look_ahead_safe` sur `load_calendar`), `app/features/indicators.py` (`@look_ahead_safe` sur `compute_all_indicators`), `app/features/research.py` (`@look_ahead_safe` sur `rank_features`)
- **Tests pytest** : ✅ 51/51 passed + 5 skipped (signatures multi-paramètres non testables automatiquement)
- **Ruff** : ✅ `All checks passed!`
- **Mypy** : ✅ 0 errors
- **Snooping check** : ✅ `TEST_SET_LOCK.json` absent, pas de scan nécessaire
- **Problèmes rencontrés** :
  - `_ohlcv_index` utilisait `rng.randn()` → `rng.normal()` (Generator API NumPy 1.17+)
  - `sharpe_ratio` : `std == 0.0` jamais vrai sur float → remplacé par `np.isclose(std, 0.0)`
  - `TestWalkForwardSplit` dupliqué dans `test_edge_validation.py` → renommé `TestWalkForwardEdgeCases` avec tests de garde uniquement
  - Tests `deflated_sharpe`/`probabilistic_sharpe` : `sr=5, skew=-10, kurt=50` ne rendait pas le dénominateur ≤ 0 → paramètres corrigés (`sr=2, skew=3, kurt=1.5`)
  - `test_indicators_look_ahead.py` : modules avec `learning_machine_learning` cassé → `_import_module_safe` avec fallback None + filtrage `__module__.startswith("app.features")` pour exclure les ré-exportations (`get_logger`, sklearn)
- **Architecture** : DSR Bailey & López de Prado (2014) avec constante d'Euler-Mascheroni, PSR (2012), purged k-fold avec embargo, walk-forward expanding window. Toutes les fonctions de features des 5 modules sont décorées `@look_ahead_safe`. `validate_edge` produit `EdgeReport(go, reasons, metrics)` basé sur les 5 critères de la constitution.
- **Notes** : `n_trials` inchangé (phase data, pas d'hypothèse). Toutes les features de `app/features/*.py` sont désormais protégées anti-look-ahead.

## 2026-05-14 — Prompt 07 : H06 Extension Donchian multi-actif

- **Statut** : ✅ Terminé — NO-GO, 0 actif validé sur 6 testés (+ 1 erreur USOIL, + 1 indisponible BUND)
- **Fichiers créés** : `scripts/run_h06_donchian_multi_asset.py` (370 lignes), `scripts/download_h06_missing_assets.py` (126 lignes), `docs/v3_hypothesis_06.md`, `predictions/h06_donchian_multi_asset.json`
- **Fichiers modifiés** : `app/config/instruments.py` (AssetConfig + ASSET_CONFIGS 7 actifs), `app/backtest/metrics.py` (fix import cassé)
- **Résultats clés** :
  - **6 actifs testés, 0 GO** : US30 ❌, XAUUSD ❌, GER30 ❌, US500 ❌, XAGUSD ❌, USOIL ⚠️ erreur, BUND ⚠️ indisponible
  - **US30** (N=100, M=10) : Sharpe train +0.35, val +0.58, test −0.09 — ❌ NO-GO (Sharpe −0.27, DSR −7.85, DD 362%)
  - **XAUUSD** (N=100, M=20) : Sharpe train +1.13, val 0.00, test +1.46 — ❌ NO-GO (WR 22.5% < 30%, trades/an 18.1 < 30)
  - **GER30** (N=50, M=10) : Sharpe train +0.29, val +1.86, test −1.01 — ❌ NO-GO (Sharpe −3.74, DSR −4.43, DD 4829%, trades/an 28.3)
  - **US500** (N=50, M=50) : Sharpe train +0.62, val +1.62, test −0.85 — ❌ NO-GO (Sharpe −3.60, DSR −4.81, DD 411%, trades/an 21.5)
  - **XAGUSD** (N=20, M=10) : Sharpe train 0.00, val 0.00, test 0.00 — ❌ NO-GO (WR 0.0%)
  - **USOIL** : ⚠️ Erreur — 2 barres prix ≤ 0 (WTI avril 2020), `load_asset()` rejette
  - **BUND** : ⚠️ Pas de données — yfinance bloque tous les tickers (BUND, FGBL=F, BUND.DE)
  - **Verdict** : 🔴 NO-GO — Donchian Breakout pur ne survit pas aux coûts réalistes v3. XAUUSD Sharpe 1.46 prometteur mais WR 22.5%. US30 WR 45.3% mais PnL/trade trop faible. Deux candidats méta-labeling (H10-H12).
- **Problèmes rencontrés** :
  - `ModuleNotFoundError: No module named 'app'` → corrigé par ajout `sys.path.insert(0, str(_PROJECT_ROOT))` dans le script
  - `ModuleNotFoundError: No module named 'yfinance'` → `pip install yfinance` dans .venv
  - `NameError: name 'pd' is not defined` → import pandas au niveau module dans download script
  - yfinance colonnes minuscules → `auto_adjust=False` + normalisation PascalCase
  - `load_asset()` attend TSV (`sep="\t"`) → `df.to_csv(sep="\t")` + flag `--force`
  - `app/backtest/metrics.py` importait `from learning_machine_learning.core.logging` (cassé depuis renommage) → corrigé
- **Vérifications** :
  - ruff : ✅ All checks passed
  - mypy : ✅ Success: no issues found
  - pytest : ✅ 51 passed, 5 skipped
  - snooping_check : ✅ TEST_SET_LOCK.json absent
- **Hypothèses à explorer ensuite** : Prompt 08 (H07 stratégies alternatives sur US30), Prompt 10-11 (méta-labeling pour US30 et XAUUSD)

## 2026-05-14 — Prompt 08 : H07 Stratégies trend-following alternatives

- **Statut** : ✅ Terminé — NO-GO, 0 stratégie alternative validée sur 4
- **Fichiers créés** : `scripts/run_h07_strategies_alt.py` (490 lignes), `tests/unit/test_strategy_dual_ma.py` (5 tests), `tests/unit/test_strategy_keltner.py` (6 tests), `tests/unit/test_strategy_chandelier.py` (5 tests), `tests/unit/test_strategy_parabolic.py` (5 tests), `prompts/08_architecture_plan.md`, `docs/v3_hypothesis_07.md`, `predictions/h07_strategies_alt.json`
- **Fichiers modifiés** : `app/strategies/dual_ma.py`, `app/strategies/keltner.py`, `app/strategies/chandelier.py`, `app/strategies/parabolic.py`, `app/backtest/deterministic.py`
- **Résultats clés** :
  - **4 stratégies testées, 0 GO** : Dual MA ❌, Keltner ❌, Chandelier ❌, Parabolic SAR ❌
  - **Donchian baseline** : Sharpe test −1.14, WR 48.4%, 91 trades — confirme la dégradation H06
  - **Dual MA** (fast=10, slow=50) : Sharpe train +0.79, val −0.20, test +0.36, WR 52.2%, 594 trades — seul test Sharpe positif mais DSR −12.66 (p=1.000), DD 189%
  - **Keltner** (period=20, mult=2.0) : Sharpe train +0.98, val **+3.70**, test −0.76, WR 50.7%, 75 trades — overfitting val flagrant (Sharpe +3.70 → −0.76)
  - **Chandelier** (period=44, k_atr=4.0) : Sharpe train +0.62, val +2.36, test NaN, WR 50.9%, 595 trades — PnL constant → écart-type nul → Sharpe NaN
  - **Parabolic SAR** (step=0.03, af_max=0.2) : Sharpe train +0.47, val +0.64, test −0.01, WR 49.9%, 627 trades — flat, DSR −27.52
  - **Corrélations vs Donchian** : Dual MA 0.19, Keltner 0.29, Chandelier 0.28, Parabolic 0.31 — toutes diversifiantes (ρ < 0.35)
  - **Verdict** : 🔴 NO-GO — aucune stratégie trend-following pure ne survit aux coûts réalistes v3 sur US30 D1
- **Corrections techniques** :
  - Colonnes PascalCase (conformes `load_asset()`) : `df["Close"]`, `df["High"]`, `df["Low"]`
  - `.shift(1)` anti-look-ahead sur le retour de `generate_signals()` pour les 4 stratégies
  - **Erreur de diagnostic initial** : Les stratégies avaient été converties en lowercase (`df["close"]`) mais `load_asset()` renomme en Title Case (`Close`, `High`, `Low`) après normalisation. Correction au 2ᵉ passage.
- **Problèmes rencontrés** :
  - `KeyError: 'close'` au premier run → `load_asset()` normalise en minuscules puis re-renomme en Title Case (ligne 140), les colonnes sont `Close`/`High`/`Low`, pas `close`/`high`/`low`
  - `RuntimeWarning: invalid value encountered in subtract` dans `pandas/core/nanops.py` pour Chandelier (périodes PnL constant → std=0 → division par zéro dans le calcul du Sharpe)
- **Vérifications** :
  - ruff : ✅ All checks passed (10 fichiers)
  - mypy : ✅ Success: no issues found (10 fichiers)
  - pytest : ⏳ À exécuter (Règle 2)
  - snooping_check : ✅ TEST_SET_LOCK.json absent
- **Hypothèses à explorer ensuite** : Prompt 09 (H08 combinaison naïve multi-actif equal risk), Prompt 10-11 (méta-labeling RF pour filtrer les trades Donchian)

## [PIVOT-PLAN] Post Phase-2 Pivot — 2026-05-14

**n_trials_cumul**: 22 → 27 (prévu)

### Constat
Phase 2 (H06-H08) : 0 GO sur 10 combinaisons. Coûts v3 8× v2 tuent le trend-following pur D1.

### Diagnostic
Méta-labeling RF v2 (H05, Sharpe +8.84 WF) jamais retesté avec coûts v3. Roadmap H09-H18 dépendait de H06/H07 (stratégies pures sans méta-labeling) → toutes échouées.

XAUUSD D1 : Sharpe brut +1.46, DSR +2.88 (p=0.002) → edge significatif. Seul WR (22.5%) et trades/an (18.1) bloquent.

### Plan pivot (5 hypothèses, voir docs/pivot_plan_v3.md)

| Ordre | ID | Actif | TF | Approche | Priorité | Dépend de |
|--------|-----|-------|-----|----------|----------|-----------|
| 1 | H1 | XAUUSD | D1 | Donchian + méta-labeling RF | P1 | — |
| 2 | H2 | US30 | D1 | Donchian + méta-labeling RF | P1 | H1 |
| 3 | H3 | US30/XAUUSD | H4 | Donchian + méta-labeling RF | P2 | H1,H2 |
| 4 | H4 | BTCUSD/ETHUSD | D1 | Donchian + méta-labeling RF | P3 | H1 |
| 5 | H5 | US30 | H1 | Mean-reversion RSI(2) | P4 | — |

### Règles strictes
1. Pas de stratégie pure sans méta-labeling
2. Sweep seuil méta sur TRAIN UNIQUEMENT
3. Pas de features contexte de marché dans méta-modèle
4. RF uniquement (pas GBM)
5. Split figé, test set 1×, validate_edge() systématique
6. Sharpe sur pct_change equity curve

### Prochaine étape
Exécuter H1 : méta-labeling RF sur XAUUSD D1 Donchian(N=100, M=20).

---

## [H1-NO-GO] Méta-labeling RF XAUUSD D1 — 2026-05-14

- **Statut** : ❌ NO-GO
- **Fichiers créés** : `scripts/run_h1_xauusd_meta.py`, `predictions/h1_xauusd_meta.json`
- **n_trials** : 22 → 23

### Résultats

| Période | Sharpe | WR | Trades |
|---------|--------|-----|--------|
| Train base (≤2022) | +1.03 | 1.5% | 68 |
| Val base (2023) | 0.00 | 0.0% | 4 |
| Test base (≥2024) | +2.06 | 25.8% | 31 |
| Test méta | 0.00 | 0.0% | 0 |

### Critères GO/NO-GO
- Sharpe test : 0.00 ✗ (< 1.0 requis)
- WR : 0.0% ✗ (< 30%)
- Trades/an : 0.0 ✗ (< 30)
→ **NO-GO confirmé**

### Cause racine
Train ≤2022 structurellement non-profitable pour Donchian XAUUSD : 1 win / 68 samples. Le RF ne peut rien apprendre d'un échantillon quasi-monoclasse → rejette tous les signaux en test → 0 trade.

Le split figé (train ≤2022, val=2023, test ≥2024) crée une **distribution inversée** pour XAUUSD D1 : la période rentable (test) est exclue de l'apprentissage, la période non-rentable (train) domine le méta-modèle. Ce split est viable pour US30 mais cassant pour XAUUSD.

### Leçon
Le split figé unique pour tous les actifs est un point de fragilité. Chaque actif a son propre régime de profitabilité temporelle. Une réévaluation du split par actif ou un walk-forward adaptatif est nécessaire avant de poursuivre les hypothèses D1.

---

## [H5-NO-GO] Mean-reversion RSI(2) extrême US30 H1 — 2026-05-15

- **Statut** : ❌ NO-GO
- **Fichiers créés** : `scripts/run_h5_rsi2_us30_h1.py`, `predictions/h5_rsi2_us30_h1.json`
- **n_trials** : 23 → 24

### Résultats

| Période | Sharpe | WR | Trades | PnL (pts) | Max DD (pts) | T/an |
|---------|--------|-----|--------|-----------|-------------|------|
| Train (≤2022) | +0.16 | 53.8% | 6120 | −44,255 | −44,358 | 637 |
| Val (2023) | −1.17 | 57.8% | 725 | −2,476 | −3,987 | 736 |
| Test (≥2024) | −0.95 | 55.1% | 1765 | −7,074 | −10,392 | 748 |

### Critères GO/NO-GO
- Sharpe test : −0.95 ✗ (< 1.0 requis)
- Trades/an : 748 ✓ (≥ 100)
- Max DD : 92.8% ✗ (< 20%)
→ **NO-GO confirmé**

### Cause racine
RSI(2) extrême = générateur de bruit, pas d'edge. 1765 trades, PnL moyen −4.0 pts/trade. 72% des sorties par RSI cross → le prix dérive contre la position.

### Leçon
Le mean-reversion RSI(2) extrême sur US30 H1 ne capture aucun edge directionnel. Même avec un méta-modèle RF en surcouche, la qualité du signal sous-jacent est trop faible pour être amplifiée. Les hypothèses de type mean-reversion sont à abandonner au profit d'approches trend-following (Donchian) avec méta-labeling.

## 2026-05-15 — Pivot v4 A1 : Audit simulateur (sizing + DD + Sharpe)

- **Statut** : ✅ Terminé
- **Type** : Bug fix infrastructure (0 n_trial consommé)
- **Fichiers créés** : `app/backtest/sizing.py`, `tests/unit/test_simulator_sizing.py`, `docs/simulator_audit_a1.md`
- **Fichiers modifiés** : `app/backtest/metrics.py` (mode A1 equity € + legacy préservé), `app/backtest/simulator.py` (injection sizing dans `_simulate_stateful_core` + propagation wrappers `simulate_trades` / `simulate_trades_continuous`)
- **Résultats clés** :
  - DD désormais borné [−100 %, 0 %]
  - Sizing au risque 2 % implémenté via `compute_position_size()`
  - Sharpe sur retours du capital en € (equity curve), pas en pips
  - Détection blow-up : flag `blowup_detected` dans les métriques
  - Rétrocompatibilité : `asset_cfg=None` préserve le comportement legacy
- **Tests** : 12/12 nouveaux tests + non-régression à vérifier
- **Bugs corrigés** :
  - DD calculé sur pips bruts → DD calculé sur equity €
  - Pas de sizing → sizing 200 € de risque / SL en €
  - Sharpe sur pips → Sharpe sur equity daily returns
- **Notes** : Aucune stratégie modifiée. Aucune lecture du test set 2024+. Les scripts `run_*.py` existants continuent de fonctionner (mode legacy).
- **Prochaine étape** : A2 — calibration coûts XTB réels.
