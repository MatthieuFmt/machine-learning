# Projet : Pipeline ML de Trading — EURUSD H1

**Dernière mise à jour** : 2026-05-12 (Step 01 terminée)
**Objectif** : Construire un edge statistiquement fiable (DSR > 0, p < 0.05) via ML supervisé sur séries temporelles financières.

---

## 1. Architecture Globale

```
app/
├── config/          # Dataclasses gelées (frozen) — config par domaine
│   ├── instruments.py   # InstrumentConfig (EURUSD, BTCUSD), TargetMode literal
│   ├── model.py         # ModelConfig (RF + GBM params)
│   ├── backtest.py      # BacktestConfig (TP/SL, friction, filtres)
│   ├── paths.py         # PathConfig (résolution de chemins)
│   └── registry.py      # ConfigRegistry — point d'entrée unique
├── core/            # Fondations transverses
│   ├── types.py         # Protocols (WeightFunction, SignalFilter), TypeAliases
│   ├── logging.py       # setup_logging() + get_logger()
│   └── exceptions.py    # Hiérarchie PipelineError → DataValidation, LookAhead, etc.
├── features/        # Feature engineering (zéro look-ahead)
│   ├── triple_barrier.py  # 4 fonctions cible + compute_target_series + label_distribution
│   ├── pipeline.py        # build_ml_ready() — assemble tout le DataFrame
│   ├── technical.py       # Indicateurs pandas_ta (RSI, EMA, ATR, BBands, ADX)
│   ├── regime.py          # Features de régime (volatilité, range/ATR, momentum D1)
│   ├── merger.py          # Merge multi-TF H4/D1/macro → H1 via merge_asof
│   └── macro.py           # Log-returns instruments corrélés (XAUUSD, USDCHF)
├── model/           # Entraînement & prédiction
│   ├── training.py        # train_model(), train_regressor(), walk_forward_train(), train_test_split_purge()
│   ├── prediction.py      # predict_oos() (classifieur) + predict_oos_regression()
│   ├── evaluation.py      # evaluate_model(), feature_importance_impurity/permutation
│   └── meta_labeling.py   # Méta-labeling binaire (build, train, apply filter)
├── backtest/        # Simulation stateful
│   ├── simulator.py       # _simulate_stateful_core(), simulate_trades(), simulate_trades_continuous()
│   ├── filters.py         # TrendFilter, VolFilter, FilterPipeline (protocole SignalFilter)
│   ├── metrics.py         # sharpe_ratio, sortino, max_drawdown, win_rate, expectancy, calmar
│   ├── reporting.py       # save_trades_detailed(), generate_performance_report()
│   └── sizing.py          # weight_linear(), weight_linear_v2(), weight_exp()
├── analysis/        # Diagnostics post-backtest
│   ├── diagnostics.py     # analyze_losses(), analyze_directionality(), correlation_matrix
│   └── edge_validation.py # validate_edge() — Breakeven WR, Bootstrap Sharpe, DSR, t-test
├── pipelines/       # Orchestrateurs par instrument
│   ├── base.py            # BasePipeline (ABC) — orchestration complète + walk-forward
│   └── eurusd.py          # EurUsdPipeline — charge H1/H4/D1/XAUUSD/USDCHF
tests/
├── unit/            # Tests unitaires (261 tests, 0 failures)
├── integration/     # Tests d'intégration (I/O)
├── acceptance/      # Tests bout-en-bout
└── conftest.py      # Fixtures partagées
docs/
├── README.md              # Roadmap stratégique 7 steps + métriques baseline
├── step_01_target_redefinition.md
├── step_02_robust_validation_framework.md
├── step_03_gbm_primary_classifier.md
├── ...step_07_cross_asset_validation.md
└── step_01_implementation_spec.md  # Spécification d'implémentation détaillée
```

---

## 2. Flux de Données (Pipeline Standard)

```
┌─────────┐    ┌──────────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│  Load   │───▶│ Build Features│───▶│  Train   │───▶│  Predict  │───▶│ Backtest │
│  Data   │    │ build_ml_ready│    │  Model   │    │   OOS     │    │ Stateful │
└─────────┘    └──────────────┘    └──────────┘    └───────────┘    └──────────┘
                                                                          │
                                                                          ▼
                                                                   ┌──────────┐
                                                                   │  Report  │
                                                                   │ + Edge   │
                                                                   │Validation│
                                                                   └──────────┘
```

### Détail par étape :

1. **load_data()** : Charge CSV nettoyés H1/H4/D1/macro depuis `PathConfig.clean/`
2. **build_features()** :
   - Indicateurs techniques (RSI, EMA, ADX, BBands, ATR) sur H1/H4/D1 via `pandas_ta`
   - Features de régime D1 (`calc_regime_features`) → merge sur H1
   - Log-returns macro (XAUUSD, USDCHF) → merge sur H1
   - Target labelling selon `instrument.target_mode` via les fonctions dans `triple_barrier.py`
   - Drop des colonnes listées dans `instrument.features_dropped`
3. **train_model()** : Split temporel strict (train ≤ 2023, val=2024, test=2025), entraîne RF ou GBM selon `target_mode`
4. **evaluate_model()** : Prédit OOS 2024+2025, calcule accuracy/Sharpe/metrics
5. **run_backtest()** : Applique filtres → simule trades stateful → calcule métriques
6. **generate_report()** : Sauvegarde trades CSV + rapport Markdown + edge validation (Bootstrap Sharpe, DSR)

---

## 3. Système de Configuration

### ConfigRegistry — Point d'entrée unique

```python
registry = ConfigRegistry()
entry = registry.get("EURUSD")  # → ConfigEntry
# entry.instrument  → InstrumentConfig (frozen)
# entry.model       → ModelConfig (frozen)
# entry.backtest    → BacktestConfig (frozen)
# entry.paths       → PathConfig (frozen)
```

### InstrumentConfig (champs clés)

| Champ | Type | Défaut | Description |
|---|---|---|---|
| `name` | `str` | — | Identifiant (EURUSD, BTCUSD) |
| `pip_size` | `float` | — | Taille d'un pip (0.0001 forex) |
| `pip_value_eur` | `float` | — | Valeur monétaire d'un pip en EUR |
| `timeframes` | `FrozenSet[str]` | — | TF disponibles (H1, H4, D1) |
| `primary_tf` | `str` | "H1" | TF principal |
| `target_mode` | `TargetMode` | `"triple_barrier"` | Mode de labellisation |
| `target_horizon_hours` | `int` | 24 | Horizon pour forward_return/directional_clean |
| `target_noise_threshold_atr` | `float` | 0.5 | Seuil de bruit pour directional_clean |
| `target_atr_period` | `int` | 14 | Période ATR pour les seuils adaptatifs |
| `target_k_atr` | `float` | 1.0 | Multiplicateur ATR pour cost_aware_v2 |
| `friction_pips` | `float` | 1.5 | Coût total (spread + commission) |
| `features_dropped` | `tuple[str, ...]` | — | Colonnes exclues de l'entraînement |

### TargetMode — Les 4 modes de cible

```python
TargetMode = Literal[
    "triple_barrier",     # Multi-classe -1/0/1, TP/SL fixes
    "forward_return",     # Régression continue, log-return forward sur horizon_hours
    "directional_clean",  # Binaire -1/1, seuil de bruit ATR (0 pour zone grise)
    "cost_aware_v2",      # Binaire -1/0/1, TP adaptatif (k_atr × ATR), SL toujours → -1
]
```

Le routing se fait à 4 niveaux :
1. `features/pipeline.py` : `build_ml_ready()` → choix de la fonction cible
2. `model/training.py` : `train_model()` vs `train_regressor()`
3. `model/prediction.py` : `predict_oos()` vs `predict_oos_regression()`
4. `backtest/simulator.py` : `simulate_trades()` vs `simulate_trades_continuous()`

---

## 4. Design Patterns Clés

### 4a. Dataclasses gelées (`frozen=True`)
Toutes les configs sont immutables. Pour dériver : `dataclasses.replace(config, champ=valeur)`.

### 4b. Extraction du noyau stateful
[`_simulate_stateful_core()`](app/backtest/simulator.py:29) est une fonction pure partagée entre `simulate_trades()` (classifieur) et `simulate_trades_continuous()` (régression). Elle implémente "un seul trade à la fois", gère TP/SL/timeout, retourne un DataFrame de trades.

### 4c. Dispatch par target_mode
Chaque composant route selon `instrument.target_mode` plutôt que via des if/else dispersés. Pattern :
```python
if instrument.target_mode == "forward_return":
    # Chemin régression
else:
    # Chemin classifieur (compatible triple_barrier, directional_clean, cost_aware_v2)
```

### 4d. Protocoles pour l'extensibilité
- `WeightFunction` : `(proba: np.ndarray) -> np.ndarray`
- `SignalFilter` : `apply(df, mask_long, mask_short) -> (mask_long, mask_short, n_rejected)`

### 4e. Split temporel strict (jamais de shuffle)
[`train_test_split_purge()`](app/model/training.py:36) : split chronologique avec purge gap de 48h entre train et test. Les années sont fixes : train ≤ 2023, val=2024, test=2025.

---

## 5. Tests

### Structure
```
tests/
├── unit/                          # 261 tests, 0 failures
│   ├── test_triple_barrier.py     # 16 tests — apply_triple_barrier, cost_aware, label_distribution
│   ├── test_target_regression.py  # 16 tests — forward_return, directional_clean, cost_aware_v2
│   ├── test_cost_aware_labeling.py
│   ├── test_simulator.py          # 8 tests — simulate_trades classifieur
│   ├── test_simulator_continuous.py # 9 tests — simulate_trades_continuous
│   ├── test_regressor_training.py # 7 tests — train_regressor, predict_oos_regression
│   ├── test_training.py           # Split temporel, purge
│   ├── test_prediction.py
│   ├── test_evaluation.py
│   ├── test_filters.py
│   ├── test_metrics.py
│   ├── test_sizing.py
│   ├── test_diagnostics.py
│   ├── test_edge_validation.py
│   ├── test_technical_features.py
│   ├── test_regime.py
│   ├── test_merger.py
│   ├── test_macro.py
│   ├── test_data_validation.py
│   └── test_walk_forward.py
├── integration/
└── acceptance/
    └── test_eurusd_full_pipeline.py
```

### Exécution
```bash
pytest tests/ -v --tb=short          # Tous les tests
pytest tests/unit/ -v                # Unitaires uniquement
pytest -m unit                       # Par marker
```

---

## 6. Roadmap — État Actuel

| Step | Statut | Description |
|---|---|---|
| **01** | ✅ Terminé | Redéfinition cible — 3 nouveaux modes (forward_return, directional_clean, cost_aware_v2) |
| 02 | ⬜ En attente | CPCV + DSR — validation robuste |
| 03 | ⬜ En attente | LightGBM/XGBoost + Optuna (conditionnel à step_01 OU step_02 GO) |
| 04 | ⬜ En attente | Features de session (Tokyo/Londres/NY) |
| 05 | ⬜ En attente | Calendrier économique (NFP, CPI, FOMC, BCE) |
| 06 | ⬜ En attente | Calibration Platt/isotonique + seuil breakeven |
| 07 | ⬜ En attente | Validation multi-actifs (GBPUSD, USDJPY, XAUUSD) |

### Critères go/no-go après Step 01 :
- ✅ GO step_03 si accuracy OOS 2025 > 0.36 (toute variante)
- ❌ Sinon → reconsidérer viabilité EURUSD H1

### Baseline v15 (avant Step 01) :
| Métrique | Valeur |
|---|---|
| Accuracy OOS 2024 | 0.355 |
| Accuracy OOS 2025 | 0.332 |
| Sharpe OOS 2024 | +0.49 |
| Sharpe OOS 2025 | +0.04 |
| DSR 2025 | -1.97 |
| p(Sharpe>0) 2025 | 0.29 |
| WR 2025 | 33.3% |
| Biais directionnel 2025 | 75% SHORT |

---

## 7. Dépendances

```txt
numpy, scipy, pandas>=3.0, pandas-ta, scikit-learn>=1.8
matplotlib, tqdm, numba, colorama
```

**Pas de dépendances lourdes** : pas de LightGBM, XGBoost, PyTorch, TensorFlow. `HistGradientBoostingRegressor` est utilisé (sklearn natif).

---

## 8. Conventions

- **Python 3.12** — `from __future__ import annotations` partout
- **Typage strict** — mypy `--strict`, pas de `Any` sauf ABC
- **Vectorisation pandas** — zéro boucle Python sur les rows, priorité `.shift()`/`.rolling()`
- **Anti-look-ahead** — toute feature à l'instant `t` n'utilise que l'information ≤ `t`
- **Logging structuré** — `get_logger(__name__)` dans chaque module, rotation de fichiers
- **Tests rapides** — < 100ms par test unitaire, fixtures synthétiques, pas d'I/O
- **RTK obligatoire** — toute commande CLI susceptible de produire une sortie longue (>20 lignes) doit être préfixée par `rtk` (`rtk -- pytest`, `rtk -- python run_pipeline.py`, etc.). RTK filtre et résume la sortie avant qu'elle n'atteigne le contexte LLM, réduisant la pollution et le coût en tokens.
- **Langue** : code en anglais, documentation en français

---

## 9. Comment Ajouter un Nouvel Instrument

1. Créer une sous-classe dans [`instruments.py`](app/config/instruments.py:85) :
```python
@dataclass(frozen=True)
class GbpUsdConfig(InstrumentConfig):
    name: str = "GBPUSD"
    pip_size: float = 0.0001
    # ... etc
```
2. Enregistrer dans [`ConfigRegistry._instruments`](app/config/registry.py:39)
3. Créer un pipeline concret dans `pipelines/` héritant de [`BasePipeline`](app/pipelines/base.py:23)

## 10. Comment Ajouter un Nouveau TargetMode

1. Ajouter le literal dans [`TargetMode`](app/config/instruments.py:13)
2. Ajouter la fonction cible dans [`triple_barrier.py`](app/features/triple_barrier.py:296)
3. Router dans [`build_ml_ready()`](app/features/pipeline.py:39)
4. Si nouveau type de modèle → router dans `train_model()`, `evaluate_model()`, `run_backtest()`
5. Ajouter les tests unitaires

---

## 11. Points d'Attention (ce qui a déjà mordu)

- **`sl_pips()` est déjà négatif** — ne pas remettre un `-` devant dans le simulateur
- **`HistGradientBoostingRegressor` n'accepte pas "huber"** — utiliser `"squared_error"`, `"absolute_error"`, `"poisson"`, `"quantile"` ou `"gamma"`
- **Pas de duplicate `__post_init__`** dans les dataclasses — un seul par classe
- **Les targets forward_return/directional_clean/cost_aware_v2 produisent des NaN sur les `horizon_hours` dernières barres** (normal, pas de futur)
- **ATR a besoin de `target_atr_period` barres de warmup** — les premières barres peuvent avoir des NaN
- **Le backtest stateful saute les barres après un trade** — la bougie de sortie est le nouveau point d'entrée
- **`simulate_trades_continuous` utilise `Predicted_Return` et `signal_threshold`** (pas `Prediction_Modele` ni `Proba`)
