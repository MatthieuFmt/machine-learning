# Step 05 — Spécification d'implémentation : Calendrier Économique Macro

**Date** : 2026-05-13
**Version** : 1
**Dépend de** : Step 04 (terminé, sessions en place), Step 02 (CPCV framework disponible)
**Référence** : [`step_05_economic_calendar_integration.md`](step_05_economic_calendar_integration.md)

---

## 0. Résumé du contexte

Les Steps 01 (cible), 02 (CPCV), et 04 (sessions) sont tous **NO-GO** :
- Sharpe 2025 baseline = +0.04, session-aware = -0.94
- DSR = -5.15, p(Sharpe>0) = 0.009
- Accuracy ≈ aléatoire (0.332)

Le diagnostic consolidé : les features techniques classiques (RSI, ADX, EMA) **ne contiennent pas de signal prédictif exploitable** sur EURUSD H1. Les sessions amplifient la volatilité mais ne prédisent pas la direction. La dernière piste feature exogène avant de conclure à l'absence d'edge sur cet instrument/TF est le **calendrier économique macro**.

### Hypothèses à tester

| # | Hypothèse | Validation |
|---|-----------|------------|
| H₁ | Exclure les trades dans ±2h d'un événement high-impact réduit le drawdown | Sharpe OOS 2025 post-filtre vs baseline |
| H₂ | `surprise_zscore` post-release est prédictif de la direction 24-48h | Permutation importance > 0.005 |

---

## 1. Architecture cible

```
┌─────────────────────────────────────────────────────────────────┐
│                        data/raw/economic_calendar/               │
│  2010.csv ... 2025.csv  (Forex Factory format)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  data/calendar_loader.py           [NOUVEAU]                     │
│  load_calendar(start, end) → DataFrame                           │
│  validate_calendar_schema(df) → None                             │
│  CANONICAL_EVENT_NAMES: dict[str, str]                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  features/calendar.py              [NOUVEAU]                     │
│  compute_minutes_to_next_event(timestamps, events_df, impact)    │
│  compute_minutes_since_last_event(timestamps, events_df, impact) │
│  compute_surprise_zscore(events_df, lookback=50)                 │
│  merge_calendar_features(ohlc, events_df) → DataFrame            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────────┐
│ pipeline.py      │ │ filters.py   │ │ config/backtest.py   │
│ build_ml_ready() │ │ CalendarFilter│ │ +use_calendar_filter │
│ +calendar_df=    │ │ [NOUVEAU]    │ │ +exclude_window      │
│ [MODIFIÉ]        │ │              │ │ +impact_threshold    │
└──────────────────┘ └──────────────┘ └──────────────────────┘
```

### Flux de données

```
EurUsdPipeline.load_data()
  ├─ charge H1/H4/D1 (existant)
  ├─ charge macro XAUUSD/USDCHF (existant)
  └─ charge calendar_df via calendar_loader.load_calendar()  ★ NOUVEAU
       └─ filtre sur [data_start, data_end] de l'OHLC H1

EurUsdPipeline.build_features(data, train_end)
  └─ build_ml_ready(..., calendar_df=data["_calendar"])      ★ MODIFIÉ
       └─ merge_calendar_features(h1, calendar_df)            ★ NOUVEAU
            ├─ minutes_to_next_event (high)
            ├─ minutes_since_last_event (high)
            ├─ surprise_zscore (backward-only)
            └─ near_high_impact_event (flag booléen)

BasePipeline.run_backtest()
  └─ FilterPipeline([MomentumFilter, VolFilter, SessionFilter, CalendarFilter]) ★ MODIFIÉ
       └─ CalendarFilter lit near_high_impact_event OU minutes_to_next_event
```

---

## 2. Fichiers à créer

### 2.1 [`data/calendar_loader.py`](../learning_machine_learning/data/calendar_loader.py)

```python
"""Chargement et validation du calendrier économique macro.

Source : CSV Forex Factory historique.
Format attendu par ligne :
    date,time,currency,event,impact,actual,forecast,previous
    2024-01-05,13:30,USD,Non-Farm Employment Change,High,216K,170K,173K
"""

# ── Constantes ──────────────────────────────────────────────────────────

CANONICAL_EVENT_NAMES: dict[str, str]
# Mapping nom Forex Factory → canonical_name
# Ex: "Non-Farm Employment Change" → "US_NFP"
#     "Consumer Price Index (YoY)" → "US_CPI_YoY"
#     "FOMC Statement"            → "US_FOMC"
#     "Main Refinancing Rate"     → "EU_ECB_Rate"
#     "BOE MPC Official Bank Rate Votes" → "UK_BOE_Rate"
#     "BoJ Interest Rate Decision" → "JP_BOJ_Rate"

# ── Fonctions publiques ─────────────────────────────────────────────────

def load_calendar(
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: str | Path = "data/raw/economic_calendar",
) -> pd.DataFrame:
    """Charge tous les CSV du calendrier entre start et end.
    
    Étapes :
    1. Liste les fichiers data_dir/*.csv
    2. Charge et concatène
    3. Parse timestamps UTC (colonne date + time → datetime UTC)
    4. Normalise les noms d'événements via CANONICAL_EVENT_NAMES
    5. Filtre sur [start, end]
    6. Trie par timestamp
    7. Valide le schéma
    
    Returns:
        DataFrame avec colonnes :
        - timestamp: datetime64[ns, UTC]
        - currency: str (USD, EUR, GBP, JPY, CHF)
        - event_name: str (canonical)
        - impact: str (Low, Medium, High)
        - actual: float
        - forecast: float
        - previous: float
    """

def validate_calendar_schema(df: pd.DataFrame) -> None:
    """Raise DataValidationError si colonnes manquantes ou types incorrects.
    
    Vérifie :
    - Colonnes obligatoires présentes
    - timestamp est datetime64[ns]
    - actual, forecast, previous sont numériques
    - impact ∈ {Low, Medium, High}
    """
```

**Précautions** :
- Forex Factory exporte en `US/Eastern` ou `UTC` selon les paramètres d'export. Détecter automatiquement ou exiger UTC dans les CSV.
- Les `actual` peuvent être des strings comme `"216K"` → parser en float (strip suffix, ×1000 si K, ×1M si M).
- Les événements sans `actual` (pré-release) → `actual = NaN`.

### 2.2 [`features/calendar.py`](../learning_machine_learning/features/calendar.py)

```python
"""Features dérivées du calendrier économique macro.

Toutes les fonctions sont anti-look-ahead : à l'instant t,
seule l'information ≤ t est utilisée.
"""

def compute_minutes_to_next_event(
    timestamps: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    impact: str = "high",
) -> pd.Series:
    """Distance en minutes jusqu'au prochain événement ≥ impact.
    
    Algorithme O(n log m) via searchsorted :
    1. Extrait les timestamps des événements filtrés par impact
    2. Pour chaque timestamp OHLC, searchsorted dans event_times
    3. Δt = event_time - ohlc_time (en minutes)
    4. Si pas d'événement futur → 99999 (valeur sentinelle)
    
    Args:
        timestamps: Index H1 à enrichir.
        events_df: DataFrame calendrier avec colonne 'timestamp'.
        impact: Seuil d'impact ('high' ou 'medium').
    
    Returns:
        pd.Series de float64, même index que timestamps.
    """

def compute_minutes_since_last_event(
    timestamps: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    impact: str = "high",
) -> pd.Series:
    """Distance en minutes depuis le dernier événement ≥ impact.
    
    Symétrique de compute_minutes_to_next_event, utilise searchsorted
    pour trouver l'événement précédent le plus proche.
    
    Si pas d'événement passé → 99999.
    """

def compute_surprise_zscore(
    events_df: pd.DataFrame,
    lookback: int = 50,
) -> pd.DataFrame:
    """Calcule le z-score de surprise (actual - forecast) / rolling_std.
    
    Anti-look-ahead strict :
    - Pour chaque événement à t_i, calcule le rolling std des
      (actual - forecast) sur les `lookback` événements précédents
      du même canonical event_name.
    - Si < 20 occurrences historiques → surprise_zscore = NaN.
    - zscore = (actual_i - forecast_i) / rolling_std.
    
    Args:
        events_df: DataFrame calendrier (doit avoir 'event_name', 'actual',
                   'forecast', 'timestamp').
        lookback: Nombre d'occurrences passées pour rolling std.
    
    Returns:
        DataFrame avec colonnes ajoutées :
        - surprise: float (actual - forecast)
        - surprise_zscore: float (NaN si historique insuffisant)
    """

def merge_calendar_features(
    ohlc: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute les colonnes calendrier au DataFrame OHLC H1.
    
    Utilise merge_asof(direction='backward') pour le surprise_zscore
    (la surprise n'est connue qu'après le release).
    
    Colonnes ajoutées :
    - minutes_to_next_event: float
    - minutes_since_last_event: float
    - surprise_zscore: float (NaN si pas de release récent)
    - near_high_impact_event: int8 (1 si minutes_to_next_event ≤ 120
                               OU minutes_since_last_event ≤ 120)
    
    Args:
        ohlc: DataFrame H1 avec DatetimeIndex.
        events_df: DataFrame calendrier validé.
    
    Returns:
        DataFrame avec les 4 colonnes ajoutées.
    """
```

**Points techniques** :
- `searchsorted` est vectorisé → pas de boucle Python.
- `99999` sentinelle : documenter que c'est un choix conscient (valeur assez grande pour être hors distribution mais pas ∞). Le modèle GBM/RF le traitera comme une valeur extrême sans NaN.
- `near_high_impact_event` : dérivé de `minutes_to_next_event` et `minutes_since_last_event`, donc sans colonne calendrier supplémentaire. C'est la colonne utilisée par le `CalendarFilter`.

### 2.3 [`tests/unit/test_calendar_features.py`](../tests/unit/test_calendar_features.py)

```
Tests à créer (≥ 20) :
├─ test_load_calendar_basic
├─ test_load_calendar_date_range
├─ test_load_calendar_missing_dir_raises
├─ test_validate_schema_ok
├─ test_validate_schema_missing_column_raises
├─ test_validate_schema_bad_type_raises
├─ test_canonical_name_mapping
├─ test_parse_actual_with_K_suffix
├─ test_parse_actual_with_M_suffix
├─ test_minutes_to_next_event_basic
├─ test_minutes_to_next_event_no_future_event
├─ test_minutes_to_next_event_impact_filter
├─ test_minutes_since_last_event_basic
├─ test_minutes_since_last_event_no_past_event
├─ test_surprise_zscore_basic
├─ test_surprise_zscore_insufficient_history
├─ test_merge_calendar_features_columns
├─ test_merge_calendar_features_no_lookahead
├─ test_near_high_impact_flag
└─ test_sentinel_value_99999
```

---

## 3. Fichiers à modifier

### 3.1 [`config/backtest.py`](../learning_machine_learning/config/backtest.py)

Ajouter 3 champs à `BacktestConfig` :

```python
# ── Step 05 — Calendrier économique ──────────────────────────────
use_calendar_filter: bool = True
calendar_exclude_window_minutes: int = 120
calendar_impact_threshold: str = "high"  # "medium" | "high"
```

Validation dans `__post_init__` :
- `calendar_exclude_window_minutes > 0`
- `calendar_impact_threshold in ("medium", "high")`

### 3.2 [`backtest/filters.py`](../learning_machine_learning/backtest/filters.py)

Ajouter la classe `CalendarFilter` après `SessionFilter` (ligne ~135) :

```python
class CalendarFilter:
    """Filtre de calendrier économique — exclut les signaux proches
    d'un événement macro high-impact.
    
    Utilise la colonne 'near_high_impact_event' (int8, 0/1) générée
    par merge_calendar_features(). Si absente, raise ValueError
    explicite.
    """
    
    name = "calendar"
    
    def __init__(
        self,
        exclude_window_minutes: int = 120,
        impact_threshold: str = "high",
    ) -> None:
        self.exclude_window_minutes = exclude_window_minutes
        self.impact_threshold = impact_threshold
    
    def apply(
        self,
        df: pd.DataFrame,
        mask_long: pd.Series,
        mask_short: pd.Series,
    ) -> tuple[pd.Series, pd.Series, int]:
        if "near_high_impact_event" not in df.columns:
            raise ValueError(
                "CalendarFilter nécessite la colonne 'near_high_impact_event'. "
                "Exécuter le feature engineering avec calendar_df fourni."
            )
        
        near_event = df["near_high_impact_event"] == 1
        
        rejected_long = mask_long & near_event
        rejected_short = mask_short & near_event
        n_rejected = int((rejected_long | rejected_short).sum())
        
        mask_long = mask_long & ~near_event
        mask_short = mask_short & ~near_event
        
        logger.debug("CalendarFilter: %d signaux rejetés", n_rejected)
        return mask_long, mask_short, n_rejected
```

### 3.3 [`features/pipeline.py`](../learning_machine_learning/features/pipeline.py)

**Signature de `build_ml_ready()`** : ajouter le paramètre `calendar_df` :

```python
def build_ml_ready(
    instrument: InstrumentConfig,
    data: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame] | None = None,
    calendar_df: pd.DataFrame | None = None,          # ★ NOUVEAU
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    window: int = 24,
    features_dropped: list[str] | None = None,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
```

**Corps** — après l'étape 5 (fusion) et avant l'étape 6 (sélection des colonnes), insérer :

```python
# 5.5 ★ Calendrier économique macro ★
if calendar_df is not None:
    from learning_machine_learning.features.calendar import merge_calendar_features
    n_avant_cal = len(combined)
    cal_features = merge_calendar_features(combined, calendar_df)
    combined = combined.join(cal_features)
    log_row_loss("merge calendar features", n_avant_cal, len(combined))
else:
    logger.info("Aucun calendrier économique fourni — features calendrier ignorées.")
```

**Colonnes finales** — ajouter dans `colonnes_finales` :

```python
colonnes_finales = [
    "Target", "Spread", "Log_Return",
    # ... existant ...
    "session_London", "session_NY", "session_Overlap", "session_LowLiq",
    # ★ Calendrier économique ★
    "minutes_to_next_event", "minutes_since_last_event",
    "surprise_zscore",
]
```

**`FILTER_KEEP`** : ajouter `near_high_impact_event` puisque c'est une colonne de filtre seulement :

```python
FILTER_KEEP: frozenset[str] = frozenset({
    "ATR_Norm", "Dist_SMA200_D1", "Volatilite_Realisee_24h",
    "near_high_impact_event",  # ★ Step 05
})
```

### 3.4 [`model/training.py`](../learning_machine_learning/model/training.py)

**`_FILTER_ONLY_COLS`** : ajouter `near_high_impact_event` :

```python
_FILTER_ONLY_COLS: frozenset[str] = frozenset({
    "ATR_Norm", "Volatilite_Realisee_24h", "near_high_impact_event"
})
```

### 3.5 [`pipelines/eurusd.py`](../learning_machine_learning/pipelines/eurusd.py)

**`load_data()`** — ajouter le chargement du calendrier :

```python
def load_data(self) -> dict[str, pd.DataFrame]:
    # ... existant ...
    
    # ★ Step 05 : Charger le calendrier économique
    from learning_machine_learning.data.calendar_loader import load_calendar
    
    # Déterminer la plage de dates depuis les données H1
    h1_data = data["h1"]
    cal_start = h1_data.index.min()
    cal_end = h1_data.index.max()
    
    try:
        calendar_df = load_calendar(cal_start, cal_end)
        data["_calendar"] = calendar_df
        logger.info("Calendrier économique chargé : %d événements", len(calendar_df))
    except FileNotFoundError:
        logger.warning("Dossier calendrier économique introuvable — ignoré.")
        data["_calendar"] = None
    except Exception as e:
        logger.error("Erreur chargement calendrier : %s — ignoré.", e)
        data["_calendar"] = None
    
    return data
```

**`build_features()`** — passer `calendar_df` :

```python
def build_features(
    self, data: dict[str, Any], train_end: pd.Timestamp | None = None
) -> pd.DataFrame:
    ml = build_ml_ready(
        instrument=self.instrument,
        data={"H1": data["h1"], "H4": data["h4"], "D1": data["d1"]},
        macro_data=data.get("_macro", {}),
        calendar_df=data.get("_calendar"),       # ★ NOUVEAU
        tp_pips=self.backtest_cfg.tp_pips,
        sl_pips=self.backtest_cfg.sl_pips,
        window=self.backtest_cfg.window_hours,
        features_dropped=list(self.instrument.features_dropped),
        train_end=train_end,
    )
    return ml
```

### 3.6 [`pipelines/base.py`](../learning_machine_learning/pipelines/base.py)

**`run_backtest()`** — ajouter `CalendarFilter` dans le pipeline :

```python
from learning_machine_learning.backtest.filters import (
    FilterPipeline,
    MomentumFilter,
    VolFilter,
    SessionFilter,
    CalendarFilter,          # ★ NOUVEAU
)

# ... dans run_backtest() :

if cfg.use_calendar_filter:                          # ★ NOUVEAU
    filters.append(                                   # ★ NOUVEAU
        CalendarFilter(                               # ★ NOUVEAU
            exclude_window_minutes=cfg.calendar_exclude_window_minutes,
            impact_threshold=cfg.calendar_impact_threshold,
        )
    )
```

**`FILTER_COLS`** dans `run_backtest()` — ajouter `near_high_impact_event` :

```python
FILTER_COLS: tuple[str, ...] = (
    "Dist_SMA200_D1", "ATR_Norm", "RSI_D1_delta",
    "near_high_impact_event",  # ★ Step 05
)
```

### 3.7 [`pipelines/base.py`](../learning_machine_learning/pipelines/base.py) — `n_filtres_appliques` dans `simulate_trades()` / `simulate_trades_continuous()`

Ajouter `"calendar": 0` dans le dict par défaut :

```python
n_filtres_appliques: dict[str, int] = {
    "trend": 0, "vol": 0, "session": 0, "momentum": 0, "calendar": 0
}
```

Ceci doit être fait dans **deux** endroits :
- [`simulate_trades()`](../learning_machine_learning/backtest/simulator.py:206)
- [`simulate_trades_continuous()`](../learning_machine_learning/backtest/simulator.py:297)

---

## 4. Ordre d'implémentation

| # | Étape | Fichier(s) | Effort |
|---|---|---|---|
| 1 | Créer `data/calendar_loader.py` + constantes `CANONICAL_EVENT_NAMES` | Nouveau | 1-2h |
| 2 | Créer `features/calendar.py` (4 fonctions) | Nouveau | 2-3h |
| 3 | Créer `tests/unit/test_calendar_features.py` (≥20 tests) | Nouveau | 1-2h |
| 4 | Ajouter `CalendarFilter` dans `backtest/filters.py` | Modification | 15min |
| 5 | Ajouter champs dans `config/backtest.py` | Modification | 10min |
| 6 | Modifier `features/pipeline.py` (signature + merge + FILTER_KEEP) | Modification | 20min |
| 7 | Modifier `model/training.py` (_FILTER_ONLY_COLS) | Modification | 5min |
| 8 | Modifier `pipelines/eurusd.py` (load_data + build_features) | Modification | 15min |
| 9 | Modifier `pipelines/base.py` (CalendarFilter + FILTER_COLS) | Modification | 10min |
| 10 | Modifier `backtest/simulator.py` (dict n_filtres_appliques) | Modification | 5min |
| 11 | Exécuter les tests, corriger | — | 30min |
| 12 | Run pipeline complet, analyser résultats | — | 1h |

**Total estimé** : ~6-8h (une journée pleine)

---

## 5. Anti-look-ahead : checklist de vérification

| # | Règle | Où vérifier |
|---|-------|-------------|
| 1 | `surprise_zscore` utilise `merge_asof(direction='backward')` → la surprise à t n'est visible qu'à t+release | [`features/calendar.py`](../learning_machine_learning/features/calendar.py) — `merge_calendar_features()` |
| 2 | `rolling_std` pour zscore est strictement backward (`lookback` occurrences passées) | [`features/calendar.py`](../learning_machine_learning/features/calendar.py) — `compute_surprise_zscore()` |
| 3 | `minutes_to_next_event` utilise `searchsorted` sur le futur → acceptable car données exogènes connues à l'avance (le calendrier est publié des jours avant) | [`features/calendar.py`](../learning_machine_learning/features/calendar.py) — `compute_minutes_to_next_event()` |
| 4 | `near_high_impact_event` est exclu des features d'entraînement (dans `_FILTER_ONLY_COLS`) → utilisé uniquement par le filtre | [`model/training.py`](../learning_machine_learning/model/training.py) — `_FILTER_ONLY_COLS` |
| 5 | Les données `actual` sont potentiellement révisées post-release → `surprise_zscore` décalé de 2h minimum | [`features/calendar.py`](../learning_machine_learning/features/calendar.py) — `merge_calendar_features()` |

---

## 6. Gestion des erreurs et edge cases

| Cas | Comportement |
|-----|-------------|
| Dossier `data/raw/economic_calendar/` absent | `load_data()` log WARNING, `calendar_df=None`, pipeline continue sans features calendrier |
| CSV calendrier mal formé | `validate_calendar_schema()` raise `DataValidationError`, catché dans `load_data()` → fallback sans calendrier |
| 0 événement high-impact dans la plage | `minutes_to_next_event` = 99999 partout, `near_high_impact_event` = 0 partout → filtre inactif |
| `surprise_zscore` = NaN (historique < 20) | Conservé comme NaN → le modèle ignore naturellement ces lignes (RF/GBM gèrent NaN via split rules) |
| Release révisé (actual change après coup) | Accepté comme bruit de mesure — la révision est généralement marginale |

---

## 7. Critères de succès / arrêt

### GO si :
- Sharpe OOS 2025 post-CalendarFilter > baseline (0.04) **de +0.15 minimum** → Sharpe ≥ 0.19
- **OU** `surprise_zscore` dans le top 5 permutation importance
- **OU** réduction du max drawdown 2025 > 20% (de -688 pips à < -550)

### NO-GO si :
- Le filtre ne change pas le Sharpe de +0.15 ET `surprise_zscore` n'est pas dans le top 5
- → STOP features exogènes. Le problème est plus fondamental (changement d'instrument ou de TF).

---

## 8. Notes de design

### Pourquoi `near_high_impact_event` dans `_FILTER_ONLY_COLS` ?

Cette colonne est dérivée de `minutes_to_next_event` et `minutes_since_last_event` mais avec un seuil fixe (120 min). L'entraîner directement créerait un risque de surapprentissage sur ce seuil arbitraire. Le filtre backtest l'utilise comme règle binaire déterministe, pas comme feature apprise.

### Pourquoi 99999 comme sentinelle ?

- Supérieur à toute valeur réaliste (1 an = 525600 minutes)
- Inférieur à `np.inf` pour éviter les problèmes numériques dans sklearn
- Le modèle le traite comme une valeur extrême sans NaN → pas de drop de lignes

### Pourquoi ne pas utiliser `calendar_df` directement dans le filtre ?

Le filtre ne doit pas dépendre du DataFrame calendrier brut pour rester testable et découplé. `near_high_impact_event` est une colonne booléenne simple dans le DataFrame ML-ready, calculée une fois lors du feature engineering. Le filtre lit juste cette colonne.
