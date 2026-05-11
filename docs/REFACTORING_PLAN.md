# Plan de Refactoring — Pipeline ML/Trading Multi-Actif

**Version :** 1.0  
**Date :** 2026-05-11  
**Auteur :** Architect Mode  
**Cible :** EUR/USD (actuel) → BTC/USD (futur)  

---

## 1. État des lieux — Déficit structurel identifié

### 1.1 Anti-patrons dans le codebase actuel

| # | Problème | Localisation | Impact |
|---|----------|-------------|--------|
| P1 | Scripts numérotés monolithiques non réutilisables | [`1_clean_data.py`](../1_clean_data.py), [`2_master_feature_engineering.py`](../2_master_feature_engineering.py), [`3_model_training.py`](../3_model_training.py), [`4_backtest_triple_barrier.py`](../4_backtest_triple_barrier.py) | Impossible de brancher BTC/USD sans duplication |
| P2 | `config.py` plat de 147 lignes, mélange constantes, chemins, commentaires d'audit | [`config.py`](../config.py) | Aucune validation de cohérence entre paramètres |
| P3 | [`backtest_utils.py`](../backtest_utils.py) est un god object (472 lignes, 6 responsabilités) | Chargement, filtrage, simulation, métriques, rapports, logging | Testabilité nulle, couplage fort |
| P4 | Chemins EURUSD hardcodés dans `FILE_EURUSD_H1_CLEAN`, `FILE_ML_READY`, etc. | [`config.py`](../config.py#L141-L146) | Bloque l'extension multi-instrument |
| P5 | Zéro test automatisé | `tests/` inexistant | Régression silencieuse garantie |
| P6 | `print()` au lieu de `logging` | Tous les scripts | Pas de niveaux, pas de redirection, pas d'observabilité |
| P7 | Pas de type hints sur les signatures publiques | Sauf [`_audit_d1_leak.py`](../_audit_d1_leak.py#L10) | `mypy` inutilisable, IDE muet |
| P8 | `optimize_sizing.py` duplique `run_backtest()` de [`4_backtest_triple_barrier.py`](../4_backtest_triple_barrier.py) | Deux fichiers | Divergence garantie à terme |
| P9 | [`5_analyze_losses.py`](../5_analyze_losses.py) script standalone avec `plt.show()` implicite | Pas de `if __name__ == '__main__'` | Inutilisable en librairie |

### 1.2 Flux de données actuel (tel qu'exécuté)

```
data/*.csv ──► 1_clean_data.py ──► cleaned-data/*.csv
                                        │
                    ┌───────────────────┘
                    ▼
          2_master_feature_engineering.py ──► ready-data/EURUSD_Master_ML_Ready.csv
                    │
                    ▼
          3_model_training.py ──► results/Predictions_{YEAR}_TripleBarrier.csv
                    │
                    ├──► 4_backtest_triple_barrier.py ──► predictions/*.md + results/Trades_Detailed_*.csv
                    ├──► optimize_sizing.py
                    └──► 5_analyze_losses.py ──► results/Loss_Analysis_*.png
```

**Problème :** le flux est linéaire et file-based. Chaque étape suppose que la précédente a été exécutée. Aucune API Python importable — tout passe par des CSV intermédiaires.

---

## 2. Architecture cible

### 2.1 Principes directeurs

1. **Séparation des responsabilités** : chaque module fait exactement une chose.
2. **Injection de dépendances** : aucun `import config` global → paramètres passés au constructeur.
3. **Immutabilité** : configuration figée après construction (`dataclass(frozen=True)` ou `@property`).
4. **Typage strict** : `mypy --strict` sur tout le code de production.
5. **Testabilité** : composants mockables via `Protocol` ou ABC.
6. **Logging structuré** : `structlog` ou `logging` JSON → traçabilité complète.

### 2.2 Arborescence des packages

```
learning_machine_learning/           # Package racine (renommé pour importabilité)
├── __init__.py
│
├── config/                          # Configuration par domaine
│   ├── __init__.py                  #   expose ConfigRegistry
│   ├── instruments.py               #   InstrumentConfig (EURUSD, BTCUSD)
│   ├── model.py                     #   ModelConfig (RF_PARAMS, seeds)
│   ├── backtest.py                  #   BacktestConfig (TP/SL, commission, filtres)
│   ├── paths.py                     #   PathConfig (résolu au runtime)
│   └── registry.py                  #   ConfigRegistry : dict[InstrumentName, InstrumentConfig]
│
├── core/                            # Fondations transverses
│   ├── __init__.py
│   ├── types.py                     #   Protocols, TypeAliases, dataclasses
│   ├── logging.py                   #   setup_logging(), get_logger()
│   ├── exceptions.py                #   LookAheadError, DataValidationError, etc.
│   └── io.py                        #   atomic_write(), ensure_dir()
│
├── data/                            # Ingestion + nettoyage
│   ├── __init__.py
│   ├── loader.py                    #   load_mt5_csv(), load_clean_data()
│   ├── cleaning.py                  #   clean_ohlcv() — ex 1_clean_data générique
│   ├── validation.py                #   validate_columns(), validate_no_lookahead()
│   └── schemas.py                   #   OHLCV_SCHEMA, ML_READY_SCHEMA (Pandera)
│
├── features/                        # Feature engineering
│   ├── __init__.py
│   ├── technical.py                 #   rsi, ema_distance, adx, atr_norm, bbands_width
│   ├── temporal.py                  #   Hour_Sin, Hour_Cos, DayOfWeek
│   ├── regime.py                    #   realized_vol, range_atr_ratio, dist_sma200
│   ├── macro.py                     #   cross_asset_return (XAU, CHF, etc.)
│   ├── triple_barrier.py            #   apply_triple_barrier() — labelling
│   ├── merger.py                    #   merge_asof multi-TF avec décalages anti-lookahead
│   └── pipeline.py                  #   FeaturePipeline : orchestrateur par instrument
│
├── model/                           # Entraînement & prédiction
│   ├── __init__.py
│   ├── training.py                  #   train_model(), train_test_split_purge()
│   ├── evaluation.py                #   permutation_importance(), classification_report_df()
│   ├── prediction.py                #   predict_oos(), build_predictions_df()
│   └── persistence.py              #   save_model(), load_model() (joblib)
│
├── backtest/                        # Simulation & métriques
│   ├── __init__.py
│   ├── simulator.py                 #   simulate_trades() — stateful, vectorisé
│   ├── filters.py                   #   TrendFilter, VolFilter, SessionFilter
│   ├── metrics.py                   #   compute_metrics(), sharpe_ratio(), max_drawdown()
│   ├── sizing.py                    #   WeightFunction Protocol + implémentations candidates
│   └── reporting.py                 #   save_report_md(), save_trades_detailed()
│
├── analysis/                        # Diagnostics post-backtest
│   ├── __init__.py
│   ├── loss_analysis.py             #   analyze_losses() — ex 5_analyze_losses
│   ├── direction_diag.py            #   diagnostic_directionnel() — ex _diag_direction
│   └── lookahead_audit.py           #   audit_correlation(), audit_d1_leak()
│
├── pipelines/                       # Orchestrateurs par instrument
│   ├── __init__.py
│   ├── base.py                      #   BasePipeline (ABC)
│   ├── eurusd.py                    #   EurUsdPipeline
│   └── btcusd.py                    #   BtcUsdPipeline (futur)
│
└── tests/                           # Suite de tests (voir §4)
    ├── __init__.py
    ├── conftest.py                  #   Fixtures pandas synthétiques
    ├── fixtures/                    #   Données CSV minimales versionnées (git LFS si >1MB)
    │   ├── eurusd_h1_sample.csv
    │   ├── eurusd_h4_sample.csv
    │   └── eurusd_d1_sample.csv
    ├── unit/                        #   Tests < 100ms, zéro I/O disque
    │   ├── test_triple_barrier.py
    │   ├── test_technical_features.py
    │   ├── test_merger.py
    │   ├── test_filters.py
    │   ├── test_metrics.py
    │   ├── test_sizing.py
    │   └── test_data_validation.py
    ├── integration/                 #   Tests < 5s, I/O limité aux fixtures
    │   ├── test_feature_pipeline.py
    │   ├── test_model_training.py
    │   └── test_backtest_simulator.py
    └── acceptance/                  #   Tests < 5min, pipeline complet
        └── test_eurusd_full_pipeline.py
```

### 2.3 Graphe de dépendances entre packages

```
pipelines ──► features ──► data
   │              │
   ▼              ▼
 model ◄────── features
   │
   ▼
backtest ◄──── model
   │
   ▼
analysis ──► backtest
   │
   ▼
 config ◄──── tout le monde (injection constructeur)
 core   ◄──── tout le monde (types, logging, exceptions)
```

**Règle :** `config` et `core` n'importent rien du projet. Les packages de domaine n'importent que `config` + `core` + les packages situés à leur gauche dans le graphe.

---

## 3. Spécification détaillée des modules clés

### 3.1 `config/instruments.py` — Configuration par actif

```python
from dataclasses import dataclass, field
from typing import FrozenSet

@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration immuable pour un instrument de trading."""
    name: str                          # "EURUSD"
    pip_size: float                    # 0.0001
    pip_value_eur: float               # 1.0
    timeframes: FrozenSet[str]         # {"H1", "H4", "D1"}
    primary_tf: str                    # "H1" — timeframe de signal
    macro_instruments: FrozenSet[str]  # {"XAUUSD", "USDCHF"} ou empty

@dataclass(frozen=True)
class EurUsdConfig(InstrumentConfig):
    name: str = "EURUSD"
    pip_size: float = 0.0001
    pip_value_eur: float = 1.0
    timeframes: FrozenSet[str] = frozenset({"H1", "H4", "D1"})
    primary_tf: str = "H1"
    macro_instruments: FrozenSet[str] = frozenset({"XAUUSD", "USDCHF"})

@dataclass(frozen=True)
class BtcUsdConfig(InstrumentConfig):
    name: str = "BTCUSD"
    pip_size: float = 1.0              # 1$ = 1 "pip" pour BTC
    pip_value_eur: float = 0.92
    timeframes: FrozenSet[str] = frozenset({"H1", "H4", "D1"})
    primary_tf: str = "H1"
    macro_instruments: FrozenSet[str] = frozenset()  # pas de corrélation macro évidente
```

### 3.2 `core/types.py` — Protocols pour injection

```python
from typing import Protocol, runtime_checkable
import pandas as pd
import numpy as np

@runtime_checkable
class WeightFunction(Protocol):
    """Protocole pour une fonction de sizing."""
    def __call__(self, proba: np.ndarray) -> np.ndarray: ...

@runtime_checkable
class SignalFilter(Protocol):
    """Protocole pour un filtre de régime."""
    @property
    def name(self) -> str: ...
    def apply(self, df: pd.DataFrame, mask_long: pd.Series, mask_short: pd.Series) -> tuple[pd.Series, pd.Series, int]: ...
    # Retourne (mask_long filtré, mask_short filtré, n_rejetés)
```

### 3.3 `features/pipeline.py` — Orchestrateur de features

```python
import logging
from dataclasses import dataclass
import pandas as pd

from learning_machine_learning.config.instruments import InstrumentConfig
from learning_machine_learning.features.technical import TechnicalFeatures
from learning_machine_learning.features.temporal import TemporalFeatures
from learning_machine_learning.features.regime import RegimeFeatures
from learning_machine_learning.features.macro import MacroFeatures
from learning_machine_learning.features.triple_barrier import TripleBarrierLabeller
from learning_machine_learning.features.merger import MultiTimeframeMerger

logger = logging.getLogger(__name__)

@dataclass
class FeaturePipeline:
    """Construit le DataFrame ML-ready pour un instrument donné.

    Injection explicite de chaque composant → testable isolément.
    """
    instrument: InstrumentConfig
    technical: TechnicalFeatures
    temporal: TemporalFeatures
    regime: RegimeFeatures
    macro: MacroFeatures
    labeller: TripleBarrierLabeller
    merger: MultiTimeframeMerger

    def build(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Étapes séquentielles avec validation inter-étapes."""
        ...
```

### 3.4 `model/training.py` — Split avec purge

```python
from datetime import timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_test_split_purge(
    df: pd.DataFrame,
    train_end_year: int,
    purge_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporel avec embargo anti-overlap (López de Prado)."""
    ...

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> RandomForestClassifier:
    """Wrapper avec logging + seed control."""
    ...
```

### 3.5 `backtest/filters.py` — Filtres de régime

Chaque filtre est une classe indépendante implémentant `SignalFilter` :

```python
class TrendFilter:
    """Filtre directionnel basé sur Dist_SMA200_D1."""
    ...

class VolFilter:
    """Filtre de volatilité basé sur ATR_Norm vs médiane glissante."""
    ...

class SessionFilter:
    """Filtre de session basse liquidité (22h-01h GMT)."""
    ...

class FilterPipeline:
    """Composite : applique une séquence ordonnée de filtres."""
    def __init__(self, filters: list[SignalFilter]): ...
    def apply(self, df, mask_long, mask_short) -> tuple[...]:
        # Applique chaque filtre, accumule les stats, log chaque étape
```

---

## 4. Stratégie de tests

### 4.1 Pyramide des tests

```
         ╱ acceptance ╲         1-2 tests, pipeline complet, < 5 min
        ╱──────────────╲
       ╱  integration   ╲       10-15 tests, composants chaînés, < 5 sec
      ╱──────────────────╲
     ╱     unitaires       ╲    30-40 tests, fonctions pures, < 100 ms
    ╱────────────────────────╲
```

### 4.2 Tests unitaires — Couverture cible ≥ 85%

| Module testé | Fichier test | Cas critiques |
|---|---|---|
| [`features/triple_barrier.py`](triple_barrier.py) | `test_triple_barrier.py` | TP touché, SL touché, timeout, TP+SL même bougie (ordre), bidirectionnel, fenêtre vide, prix constants |
| [`features/technical.py`](technical.py) | `test_technical_features.py` | RSI bornes [0,100], EMA distance signe cohérent, ATR > 0, BBWidth ≥ 0, ADX bornes |
| [`features/merger.py`](merger.py) | `test_merger.py` | Merge H4/H1 avec shift 4h, D1/H1 avec shift 1j, aucune NaN après merge, pas de futur dans le passé |
| [`backtest/filters.py`](filters.py) | `test_filters.py` | Trend bloque SHORT si Close>SMA200, Vol rejette si ATR>seuil, Session bloque 22h-01h, filtre composite ordre respecté |
| [`backtest/metrics.py`](metrics.py) | `test_metrics.py` | Sharpe=0 si PnL constant, Sharpe>0 si trend haussier, max_dd correct sur scénario connu, B&H cohérent |
| [`backtest/sizing.py`](sizing.py) | `test_sizing.py` | Poids ∈ [borne_inf, borne_sup], monotone avec proba, valeur à proba=seuil |
| [`data/validation.py`](validation.py) | `test_data_validation.py` | Colonnes requises, types, pas de NaN, index monotone, pas de duplicats |

**Stratégie de fixtures :** toutes les données de test sont synthétiques (générées dans `conftest.py` via `numpy.random` avec seed fixe). Aucun fichier CSV externe pour les tests unitaires → exécution < 100ms par test.

### 4.3 Tests d'intégration — Validation des chaînes de composants

| Test | Composants vérifiés | Données |
|---|---|---|
| `test_feature_pipeline.py::test_full_pipeline_no_nan` | Technical → Temporal → Regime → Macro → Labeller → Merger | Fixture CSV réelle (1000 barres EURUSD) |
| `test_feature_pipeline.py::test_no_lookahead_d1` | Merger D1/H1 | Fixture synthétique avec dates précises |
| `test_model_training.py::test_train_accuracy_above_baserate` | train_test_split_purge → train_model | Fixture ML_Ready 1 an |
| `test_backtest_simulator.py::test_stateful_single_trade` | simulate_trades | Prédictions synthétiques |
| `test_backtest_simulator.py::test_no_signal_overlap` | simulate_trades | Deux signaux consécutifs → un seul trade |

### 4.4 Test d'acceptation — Non-régression

```python
# tests/acceptance/test_eurusd_full_pipeline.py
def test_eurusd_pipeline_end_to_end(eurusd_pipeline: EurUsdPipeline):
    """Exécute le pipeline complet et vérifie les invariants."""
    results = eurusd_pipeline.run(train_end_year=2023, eval_years=[2024])

    # Invariant 1 : pas de NaN dans les prédictions
    assert not results.predictions.isna().any().any()

    # Invariant 2 : accuracy > baserate (classe majoritaire)
    baserate = results.y_test.value_counts(normalize=True).max()
    assert results.accuracy > baserate

    # Invariant 3 : Sharpe dans l'intervalle plausible
    assert 0.0 < results.sharpe < 5.0

    # Invariant 4 : WR entre 40% et 75%
    assert 40.0 <= results.win_rate <= 75.0
```

### 4.5 Configuration pytest (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
pythonpath = ["."]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--maxfail=5",
    "--durations=10",
]
markers = [
    "unit: Tests unitaires rapides (pas d'I/O)",
    "integration: Tests d'intégration (I/O fixtures)",
    "acceptance: Tests de bout en bout (lents)",
    "slow: Tests > 1 seconde",
]

[tool.coverage.run]
source = ["learning_machine_learning"]
omit = ["tests/*", "*/__init__.py"]

[tool.coverage.report]
fail_under = 80
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

### 4.6 CI/CD minimal (GitHub Actions)

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/unit/ -x --cov
      - run: pytest tests/integration/ -x --cov-append
      - run: pytest tests/acceptance/ -x --cov-append
      - run: mypy learning_machine_learning/ --strict
```

---

## 5. Gestion de la configuration — Migration `config.py` → `config/*`

### 5.1 Avant/Après

| Avant (`config.py`) | Après (`config/*.py`) |
|---|---|
| 147 lignes, tout mélangé | 5 fichiers, chacun < 60 lignes |
| `TP_PIPS = 20.0` (global mutable) | `BacktestConfig(tp_pips=20.0)` (dataclass frozen) |
| `FILE_EURUSD_H1_CLEAN = './cleaned-data/...'` | `PathConfig` résout les chemins via `pathlib.Path` |
| `FEATURES_DROPPED = [...]` (commentaires inline) | `ModelConfig.features_dropped: tuple[str, ...]` avec raison documentée dans une `FeatureDropRecord` |
| `RF_PARAMS = {...}` | `ModelConfig(rf_params=...)` avec `__post_init__` qui valide `n_estimators > 0` etc. |
| Commentaires d'audit mélangés au code | Supprimés — l'historique est dans git |

### 5.2 Validation au constructeur

```python
@dataclass(frozen=True)
class BacktestConfig:
    tp_pips: float
    sl_pips: float
    window_hours: int
    commission_pips: float
    slippage_pips: float
    confidence_threshold: float

    def __post_init__(self):
        if self.tp_pips <= 0:
            raise ValueError(f"tp_pips must be > 0, got {self.tp_pips}")
        if self.sl_pips <= 0:
            raise ValueError(f"sl_pips must be > 0, got {self.sl_pips}")
        if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}"
            )
```

---

## 6. Plan de migration pas à pas

### Phase 0 — Préparation (1 session)

| Étape | Action | Risque |
|---|---|---|
| 0.1 | Créer `learning_machine_learning/` et `tests/` | Nul |
| 0.2 | Installer `pytest`, `pytest-cov`, `mypy`, `pandera` | Nul |
| 0.3 | Écrire `tests/conftest.py` avec fixtures synthétiques | Nul |
| 0.4 | Geler les résultats de référence (run actuel → `baseline_results.json`) | Moyen : vérifier qu'on capture bien l'état actuel |

### Phase 1 — Extraction des fondations (2 sessions)

| Étape | Action | Fichier source → cible |
|---|---|---|
| 1.1 | Migrer `config.py` → `config/*.py` (dataclasses) | [`config.py`](../config.py) → `config/{instruments,model,backtest,paths}.py` |
| 1.2 | Créer `core/{types,logging,exceptions}.py` | Nouveau |
| 1.3 | Extraire `triple_barrier` → `features/triple_barrier.py` | [`2_master_feature_engineering.py`](../2_master_feature_engineering.py#L28-L59) |
| 1.4 | Extraire `calc_base_features` → `features/technical.py` | [`2_master_feature_engineering.py`](../2_master_feature_engineering.py#L62-L67) |
| 1.5 | **Écrire les tests unitaires correspondants** | `tests/unit/test_triple_barrier.py`, `test_technical_features.py` |

### Phase 2 — Backtest (2 sessions)

| Étape | Action | Fichier source → cible |
|---|---|---|
| 2.1 | Extraire `simulate_trades` → `backtest/simulator.py` | [`backtest_utils.py`](../backtest_utils.py#L84-L264) |
| 2.2 | Extraire filtres → `backtest/filters.py` | [`backtest_utils.py`](../backtest_utils.py#L132-L183) |
| 2.3 | Extraire `compute_metrics` → `backtest/metrics.py` | [`backtest_utils.py`](../backtest_utils.py#L277-L359) |
| 2.4 | Extraire sizing → `backtest/sizing.py` | [`optimize_sizing.py`](../optimize_sizing.py#L32-L59) |
| 2.5 | Extraire reporting → `backtest/reporting.py` | [`backtest_utils.py`](../backtest_utils.py#L362-L471) |
| 2.6 | **Écrire les tests unitaires** | `tests/unit/test_{filters,metrics,sizing}.py` |

### Phase 3 — Data & Features (2 sessions)

| Étape | Action | Fichier source → cible |
|---|---|---|
| 3.1 | Généraliser `clean_data` → `data/cleaning.py` | [`1_clean_data.py`](../1_clean_data.py) |
| 3.2 | Créer `data/loader.py` (abstraction multi-TF) | Nouveau |
| 3.3 | Créer `data/validation.py` + schémas Pandera | Nouveau |
| 3.4 | Extraire features de régime → `features/regime.py` | [`2_master_feature_engineering.py`](../2_master_feature_engineering.py#L104-L108) |
| 3.5 | Extraire features macro → `features/macro.py` | [`2_master_feature_engineering.py`](../2_master_feature_engineering.py#L80-L85) |
| 3.6 | Créer `features/merger.py` (merge_asof générique) | [`2_master_feature_engineering.py`](../2_master_feature_engineering.py#L139-L175) |
| 3.7 | **Tests d'intégration** | `tests/integration/test_feature_pipeline.py` |

### Phase 4 — Modèle (1 session)

| Étape | Action | Fichier source → cible |
|---|---|---|
| 4.1 | Extraire training → `model/training.py` | [`3_model_training.py`](../3_model_training.py#L35-L60) |
| 4.2 | Extraire evaluation → `model/evaluation.py` | [`3_model_training.py`](../3_model_training.py#L62-L93) |
| 4.3 | Extraire prediction → `model/prediction.py` | [`3_model_training.py`](../3_model_training.py#L95-L133) |
| 4.4 | **Tests d'intégration** | `tests/integration/test_model_training.py` |

### Phase 5 — Orchestration & acceptation (1 session)

| Étape | Action |
|---|---|
| 5.1 | Créer `pipelines/base.py` (ABC) |
| 5.2 | Créer `pipelines/eurusd.py` (assemble tous les composants) |
| 5.3 | Migrer `analysis/` (ex `5_analyze_losses.py`, `_diag_direction.py`, `_audit_*.py`) |
| 5.4 | **Test d'acceptation complet** |
| 5.5 | Comparer résultats avec `baseline_results.json` → non-régression |

### Phase 6 — BTC/USD (futur, hors scope immédiat)

| Étape | Action |
|---|---|
| 6.1 | Créer `config/instruments.py::BtcUsdConfig` |
| 6.2 | Créer `pipelines/btcusd.py` (hérite de `BasePipeline`) |
| 6.3 | Ajuster `features/` si BTC nécessite des features spécifiques (on-chain, order book, etc.) |
| 6.4 | Réutiliser `model/`, `backtest/`, `analysis/` sans modification |

---

## 7. Considérations de performance

### 7.1 Vectorisation vs boucle explicite

La [`simulate_trades()`](../backtest_utils.py#L84-L264) actuelle utilise une boucle `while` Python pure. Pour des raisons de clarté et de testabilité, on conserve cette approche dans `backtest/simulator.py`. Une version vectorisée Numba (backlog) pourrait réduire le temps de simulation de ~10s à <1s sur 4 ans de données.

### 7.2 Cache des features

`FeaturePipeline.build()` pourrait implémenter un cache disque (pickle/parquet) avec hash du `InstrumentConfig` + schéma des données d'entrée. Évite de recalculer les features à chaque run de test.

### 7.3 Parallélisme

- [`permutation_importance()`](../3_model_training.py#L77) utilise déjà `n_jobs=-1` → OK.
- Le `GridSearchCV` futur pourra aussi utiliser `n_jobs=-1`.
- Les tests unitaires sont naturellement parallélisables via `pytest-xdist`.

---

## 8. Fichiers supprimés après migration

| Fichier | Raison |
|---|---|
| `1_clean_data.py` | → `data/cleaning.py` |
| `2_master_feature_engineering.py` | → `features/*` |
| `3_model_training.py` | → `model/*` |
| `4_backtest_triple_barrier.py` | → `pipelines/eurusd.py` (orchestrateur) |
| `backtest_utils.py` | → `backtest/*` |
| `optimize_sizing.py` | → `backtest/sizing.py` |
| `5_analyze_losses.py` | → `analysis/loss_analysis.py` |
| `_audit_corr.py` | → `analysis/lookahead_audit.py` |
| `_audit_d1_leak.py` | → `analysis/lookahead_audit.py` |
| `_diag_direction.py` | → `analysis/direction_diag.py` |
| `config.py` | → `config/*` |

Les scripts racines deviennent des points d'entrée minces :

```python
# run_eurusd_pipeline.py
from learning_machine_learning.pipelines.eurusd import EurUsdPipeline
EurUsdPipeline.from_defaults().run()
```

---

## 9. Résumé des bénéfices attendus

| Métrique | Avant | Après |
|---|---|---|
| Couverture de tests | 0% | ≥ 80% |
| Temps d'ajout d'un nouvel instrument | ~2 jours (duplication) | ~30 min (nouveau config + pipeline) |
| Temps d'exécution des tests | ∞ (pas de tests) | < 2 min (unité + intégration), < 5 min (acceptance) |
| Violations `mypy --strict` | ~200+ | 0 |
| Lignes par fonction (médiane) | ~40 | ≤ 25 |
| Nombre de `import config` globaux | 12 | 0 |
| Conformité López de Prado (purge/embargo) | Partielle | Complète (intégrée au splitter) |
