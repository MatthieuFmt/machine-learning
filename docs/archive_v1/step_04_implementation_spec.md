# Step 04 — Spécification d'implémentation : Features Session-Aware

**Date** : 2026-05-13
**Statut** : 🔴 À implémenter (P0 — suite aux NO-GO Step 01 & 02)
**Dépend de** : Aucune (Step 03 GBM bloquée, RF reste le classifieur)
**Fichier stratégique** : [`docs/step_04_session_aware_features.md`](docs/step_04_session_aware_features.md)

---

## 1. Contexte & objectif

### 1.1 Diagnostic racine (Steps 01 & 02)

| Constat | Source |
|----------|--------|
| Accuracy OOS ≈ 0.332 (aléatoire 3-classes) | Step 01 final report |
| DSR = −5.15, 0 split profitable / 200 | Step 02 final report |
| Les 4 target_modes testés → Sharpe < 0 | Step 01 |
| Features actives = 7 indicateurs techniques classiques | [`EurUsdConfig.features_dropped`](../learning_machine_learning/config/instruments.py:95-116) |

**Conclusion** : Les features techniques classiques (RSI, EMA distance, ADX) n'ont **aucun pouvoir prédictif** sur EURUSD H1. Le problème n'est pas la cible, ni le modèle, ni l'overfitting — c'est l'absence de signal dans les features.

### 1.2 Hypothèse de travail

> Les features de **régime microstructurel** (session de trading, volatilité conditionnelle par session, position relative dans la session) capturent des patterns directionnels documentés dans la littérature FX, que les indicateurs techniques classiques ne captent pas.

### 1.3 Objectifs chiffrés

| Métrique | Baseline v15 (RF, triple_barrier) | Cible Step 04 |
|----------|-----------------------------------|---------------|
| Permutation importance `session_id` | N/A | **> 0.005** |
| Permutation importance `ATR_session_zscore` | N/A | **> 0.005** |
| Accuracy OOS 2025 | 0.332 | **> 0.34** |
| Sharpe OOS 2025 | −0.11 | **> 0.30** (ambitieux, cf. critère d'arrêt) |
| Sharpe par session (Overlap) | N/A | **> 1.0** |

**Critère d'arrêt** : Si permutation_importance(session_id) < 0.003 **ET** gain Sharpe OOS < +0.10 → les sessions n'apportent rien → GO Step 05 (calendrier économique) directement.

---

## 2. Architecture des nouvelles features

### 2.1 Mapping des sessions

5 classes discrètes basées sur l'heure UTC de la barre H1 :

| Session ID | Nom | Heures UTC | Durée |
|------------|-----|------------|-------|
| 0 | Tokyo | 01:00–07:00 | 6h |
| 1 | London (pure) | 07:00–12:00 | 5h |
| 2 | NY (pure) | 16:00–21:00 | 5h |
| 3 | **Overlap London-NY** | **12:00–16:00** | **4h** |
| 4 | Low liquidity | 22:00–01:00 | 3h |

**Règles de priorité** (appliquées dans l'ordre) :
1. `hour ∈ {0, 22, 23}` → **low_liq** (4) — priorité maximale, spreads max
2. `hour ∈ [12, 16)` → **LdN_overlap** (3) — prime sur London et NY
3. `hour ∈ [7, 12)` → **London** (1)
4. `hour ∈ [16, 21)` → **NY** (2)
5. `hour ∈ [1, 7) ∪ {21}` → **Tokyo** (0) — catch-all

Le gap 21:00-22:00 est assigné à Tokyo par défaut. Le DST est ignoré (bruit acceptable, cf. R3 dans la spec stratégique).

### 2.2 Features produites

| Feature | Type | Description | Anti-leak |
|---------|------|-------------|-----------|
| `session_id` | Catégorielle (0-4) | Session de trading de la barre | ✅ Pure time feature |
| `ATR_session_zscore` | Continue | ATR_Norm z-scoré par session (stats train-only) | ⚠️ Implémentation avec `train_end` |
| `session_open_range` | Continue | Range cumulé (High-Low) depuis l'ouverture de session | ✅ Regarde le passé local |
| `relative_position_in_session` | Continue ∈ [0,1] | Progression dans la session | ✅ Pure time feature |

Optionnel (si interactions explicites demandées) :
| `session_London`, `session_NY`, `session_Overlap`, `session_LowLiq` | One-hot (4 colonnes) | Encodage one-hot pour RandomForest | ✅ |

### 2.3 Stratégie d'encoding

Puisque Step 03 (GBM) est bloquée et qu'on reste sur **RandomForest**, l'encoding **one-hot** est le choix correct :
- L'ordinal introduit une fausse notion de distance (Tokyo=0, London=1, NY=2…) → RF interprète ça comme un ordre.
- One-hot = 4 colonnes binaires (Tokyo implicite comme baseline), sémantiquement correct pour RF.

```python
# Dans InstrumentConfig (nouveau champ)
session_encoding: Literal["ordinal", "one_hot"] = "one_hot"
```

### 2.4 Diagramme de flux

```
build_ml_ready()
  │
  ├─ 1. Target labelling (existant)
  ├─ 2. Features techniques H1 (existant)
  │     └─ ATR_Norm  ← prérequis pour ATR_session_zscore
  ├─ 3. Regime H1 (existant)
  ├─ 4. Cyclical time (existant)
  │
  ├─ 5. ★ NOUVEAU : Session features ★
  │     ├─ compute_session_id(index) → session_id
  │     ├─ compute_session_open_range(high, low, session_id) → session_open_range
  │     ├─ compute_relative_position_in_session(index, session_id) → rel_pos
  │     └─ SessionVolatilityScaler.fit_transform(atr, session_id, train_end) → ATR_session_zscore
  │
  ├─ 6. Features H4/D1 (existant)
  ├─ 7. Macro (existant)
  ├─ 8. Fusion (existant)
  └─ 9. Colonnes finales (modifié)
```

---

## 3. Plan d'implémentation — Fichier par fichier

### 3.1 [`learning_machine_learning/config/instruments.py`](../learning_machine_learning/config/instruments.py)

**Ajouter le champ `session_encoding`** dans `InstrumentConfig` :

```python
# Après target_k_atr (ligne 45)
session_encoding: Literal["ordinal", "one_hot"] = "one_hot"
```

**Ne pas ajouter les features de session dans `features_dropped`** d'`EurUsdConfig` — elles doivent être actives.

**Impact sur les autres configs** : `BtcUsdConfig` hérite de la valeur par défaut `"one_hot"` — acceptable, les sessions forex ne s'appliquent pas à BTC mais la feature sera simplement non-informative (risque faible).

### 3.2 [`learning_machine_learning/features/regime.py`](../learning_machine_learning/features/regime.py)

Ajouter 4 fonctions pures + 1 classe après les fonctions existantes (après `calc_dist_sma200_d1`, ligne 85).

#### 3.2.1 `compute_session_id`

```python
def compute_session_id(index: pd.DatetimeIndex) -> pd.Series:
    """Détermine la session de trading pour chaque barre H1.
    
    Mapping (heures UTC) :
        0 = Tokyo     01:00-07:00 (+ gap 21:00-22:00)
        1 = London    07:00-12:00
        2 = NY        16:00-21:00
        3 = Overlap   12:00-16:00 (prioritaire sur London et NY)
        4 = Low liq   22:00-01:00 (prioritaire sur tout)
    
    Args:
        index: DatetimeIndex des barres H1.
    
    Returns:
        pd.Series[int8] de session_id, même index.
    """
    import numpy as np
    
    hours = index.hour
    sid = pd.Series(0, index=index, dtype=np.int8)
    
    # Priorité 1 : low liquidity (22h-01h)
    sid[(hours >= 22) | (hours == 0)] = 4
    # Priorité 2 : overlap London-NY (12h-16h)
    sid[(hours >= 12) & (hours < 16)] = 3
    # Priorité 3 : London pure (7h-12h)
    sid[(hours >= 7) & (hours < 12)] = 1
    # Priorité 4 : NY pure (16h-21h)
    sid[(hours >= 16) & (hours < 21)] = 2
    # Reste : Tokyo (1h-7h, 21h-22h) — déjà 0 par défaut
    
    return sid
```

#### 3.2.2 `compute_session_open_range`

```python
def compute_session_open_range(
    high: pd.Series,
    low: pd.Series,
    session_id: pd.Series,
) -> pd.Series:
    """Range cumulé (High - Low) depuis le début de la session courante.
    
    Reset à chaque changement de session_id. À t=0 de session, retourne
    le range de la première barre.
    
    Args:
        high: Série High H1.
        low: Série Low H1.
        session_id: Série session_id (sortie de compute_session_id).
    
    Returns:
        pd.Series du range cumulé en prix natifs (pas normalisé).
    """
    # Détecter les changements de session
    session_change = session_id.diff().fillna(1).ne(0)
    segment_id = session_change.cumsum()
    
    cummax = high.groupby(segment_id, sort=False).cummax()
    cummin = low.groupby(segment_id, sort=False).cummin()
    
    return cummax - cummin
```

#### 3.2.3 `compute_relative_position_in_session`

```python
def compute_relative_position_in_session(
    index: pd.DatetimeIndex,
    session_id: pd.Series,
) -> pd.Series:
    """Position relative dans la session ∈ [0, 1].
    
    0 = début de session, 1 = dernière barre de la session.
    Basé sur l'heure de la barre et les bornes fixes UTC.
    
    Args:
        index: DatetimeIndex des barres H1.
        session_id: Série session_id.
    
    Returns:
        pd.Series float ∈ [0, 1].
    """
    import numpy as np
    
    SESSION_START: dict[int, int] = {0: 23, 1: 7, 2: 16, 3: 12, 4: 22}
    SESSION_DURATION: dict[int, int] = {0: 9, 1: 5, 2: 5, 3: 4, 4: 3}
    
    hours = index.hour
    result = pd.Series(0.0, index=index)
    
    for sid_val in [0, 1, 2, 3, 4]:
        mask = session_id == sid_val
        if not mask.any():
            continue
        start = SESSION_START[sid_val]
        duration = SESSION_DURATION[sid_val]
        elapsed = hours[mask].astype(float) - start
        # Correction midnight wrap (heures 0-7 pour Tokyo)
        elapsed[elapsed < 0] += 24.0
        result[mask] = elapsed / duration
    
    return result.clip(0.0, 1.0)
```

#### 3.2.4 `SessionVolatilityScaler` (classe)

```python
class SessionVolatilityScaler:
    """Scaler de volatilité conditionnel à la session — fit train-only.
    
    Analogue à StandardScaler mais calcule μ et σ par session.
    Le fit() ne doit voir QUE les données d'entraînement.
    
    Usage:
        scaler = SessionVolatilityScaler()
        scaler.fit(atr_norm_train, session_id_train)
        zscored = scaler.transform(atr_norm_all, session_id_all)
    """
    
    def __init__(self) -> None:
        self._stats: dict[int, tuple[float, float]] = {}  # session_id → (μ, σ)
        self._default_mu: float = 0.0
        self._default_sigma: float = 1.0
    
    def fit(self, atr_norm: pd.Series, session_id: pd.Series) -> "SessionVolatilityScaler":
        """Calcule μ et σ par session sur les données d'entraînement uniquement.
        
        Args:
            atr_norm: Série ATR_Norm (déjà normalisée par Close).
            session_id: Série session_id correspondante.
        
        Returns:
            self pour chaînage.
        """
        import numpy as np
        
        valid = atr_norm.notna() & session_id.notna()
        atr_valid = atr_norm[valid]
        sid_valid = session_id[valid]
        
        for sid_val in [0, 1, 2, 3, 4]:
            mask = sid_valid == sid_val
            if mask.sum() < 10:
                # Pas assez de données → utiliser les stats globales
                continue
            self._stats[sid_val] = (
                float(atr_valid[mask].mean()),
                float(atr_valid[mask].std()) + 1e-10,  # éviter /0
            )
        
        # Fallback global si une session manque
        self._default_mu = float(atr_valid.mean())
        self._default_sigma = float(atr_valid.std()) + 1e-10
        
        return self
    
    def transform(self, atr_norm: pd.Series, session_id: pd.Series) -> pd.Series:
        """Applique la standardisation par session.
        
        Args:
            atr_norm: Série ATR_Norm à transformer.
            session_id: Série session_id correspondante.
        
        Returns:
            pd.Series de z-scores (ATR_session_zscore).
        """
        import numpy as np
        
        result = pd.Series(0.0, index=atr_norm.index)
        
        for sid_val in [0, 1, 2, 3, 4]:
            mask = session_id == sid_val
            if not mask.any():
                continue
            mu, sigma = self._stats.get(sid_val, (self._default_mu, self._default_sigma))
            result[mask] = (atr_norm[mask] - mu) / sigma
        
        return result
    
    def fit_transform(
        self, atr_norm: pd.Series, session_id: pd.Series
    ) -> pd.Series:
        """Fit + transform en un appel."""
        self.fit(atr_norm, session_id)
        return self.transform(atr_norm, session_id)
```

### 3.3 [`learning_machine_learning/features/pipeline.py`](../learning_machine_learning/features/pipeline.py)

#### 3.3.1 Imports (ligne 29-34)

Ajouter après les imports `regime` existants :

```python
from learning_machine_learning.features.regime import (
    calc_volatilite_realisee,
    calc_range_atr_ratio,
    calc_rsi_d1_delta,
    calc_dist_sma200_d1,
    compute_session_id,                        # NEW
    compute_session_open_range,                # NEW
    compute_relative_position_in_session,      # NEW
    SessionVolatilityScaler,                   # NEW
)
```

#### 3.3.2 Signature de `build_ml_ready` (ligne 39-47)

Ajouter le paramètre `train_end` :

```python
def build_ml_ready(
    instrument: InstrumentConfig,
    data: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame] | None = None,
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    window: int = 24,
    features_dropped: list[str] | None = None,
    train_end: pd.Timestamp | None = None,  # NEW
) -> pd.DataFrame:
```

#### 3.3.3 Bloc session features (insérer après ligne 172, après `Hour_Cos`)

```python
    # 2.5 ★ Features de session (microstructure FX) ★
    h1["session_id"] = compute_session_id(h1.index)
    h1["session_open_range"] = compute_session_open_range(
        h1["High"], h1["Low"], h1["session_id"]
    )
    h1["relative_position_in_session"] = compute_relative_position_in_session(
        h1.index, h1["session_id"]
    )
    
    # ATR_session_zscore : fit train-only si train_end fourni
    scaler = SessionVolatilityScaler()
    if train_end is not None:
        train_mask = h1.index <= train_end
        scaler.fit(
            h1.loc[train_mask, "ATR_Norm"],
            h1.loc[train_mask, "session_id"],
        )
    else:
        # Mode exploration : fit sur toutes les données
        scaler.fit(h1["ATR_Norm"], h1["session_id"])
    h1["ATR_session_zscore"] = scaler.transform(
        h1["ATR_Norm"], h1["session_id"]
    )
    
    # One-hot encoding si configuré
    if instrument.session_encoding == "one_hot":
        SESSION_LABELS = {
            1: "session_London",
            2: "session_NY",
            3: "session_Overlap",
            4: "session_LowLiq",
        }
        for sid_val, col_name in SESSION_LABELS.items():
            h1[col_name] = (h1["session_id"] == sid_val).astype(np.int8)
        # Tokyo = baseline (toutes les dummies à 0)
```

#### 3.3.4 Colonnes finales (modifier ligne 212-218)

Ajouter les colonnes de session :

```python
    colonnes_finales = [
        "Target", "Spread", "Log_Return",
        "Dist_EMA_50",
        "RSI_14", "ADX_14", "ATR_Norm", "BB_Width",
        "Hour_Sin", "Hour_Cos",
        "Volatilite_Realisee_24h", "Range_ATR_ratio",
        # ★ Session features ★
        "session_id", "ATR_session_zscore",
        "session_open_range", "relative_position_in_session",
    ]
    
    # Ajouter les one-hot si présentes
    if instrument.session_encoding == "one_hot":
        colonnes_finales += [
            "session_London", "session_NY",
            "session_Overlap", "session_LowLiq",
        ]
```

### 3.4 [`learning_machine_learning/pipelines/base.py`](../learning_machine_learning/pipelines/base.py)

Modifier l'appel à `build_ml_ready()` dans la méthode `build_features()` (ou dans `EurUsdPipeline.build_features()`) pour passer `train_end` :

```python
def build_features(self, data: dict[str, Any]) -> pd.DataFrame:
    # Déterminer la date de fin d'entraînement (dernière barre de 2023)
    train_end = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
    
    ml = build_ml_ready(
        instrument=self.instrument,
        data={"H1": data["h1"], "H4": data["h4"], "D1": data["d1"]},
        macro_data=data.get("_macro", {}),
        tp_pips=self.backtest_cfg.tp_pips,
        sl_pips=self.backtest_cfg.sl_pips,
        window=self.backtest_cfg.window_hours,
        features_dropped=list(self.instrument.features_dropped),
        train_end=train_end,  # NEW
    )
    return ml
```

Note : Si le `train_end` est déjà déterminé dynamiquement dans `BasePipeline` (via une config ou un split existant), utiliser cette valeur. Sinon, hardcoder `2023-12-31` (cohérent avec le split temporel strict train ≤ 2023, val=2024, test=2025 documenté dans [`CLAUDE.md`](../CLAUDE.md:165)).

### 3.5 [`learning_machine_learning/config/instruments.py`](../learning_machine_learning/config/instruments.py) — `EurUsdConfig.features_dropped`

**Ne pas ajouter** les features de session dans `features_dropped`. Le tuple existant reste inchangé — les nouvelles colonnes seront automatiquement incluses dans X.

---

## 4. Stratégie de tests

### 4.1 Nouveaux tests unitaires : `tests/unit/test_session_features.py`

| # | Test | Fonction testée | Vérification |
|---|------|----------------|-------------|
| 1 | `test_session_id_mapping` | `compute_session_id` | 24 heures mappées correctement |
| 2 | `test_session_id_priorities` | `compute_session_id` | Overlap > London, LowLiq > Tokyo |
| 3 | `test_session_id_dtype` | `compute_session_id` | dtype = int8 |
| 4 | `test_session_open_range_reset` | `compute_session_open_range` | Reset à chaque changement de session |
| 5 | `test_session_open_range_monotonic` | `compute_session_open_range` | Croissance monotone dans une session |
| 6 | `test_session_open_range_start` | `compute_session_open_range` | t=0 de session → range de la 1ère barre |
| 7 | `test_relative_position_bounds` | `compute_relative_position_in_session` | Toutes les valeurs ∈ [0, 1] |
| 8 | `test_relative_position_monotonic` | `compute_relative_position_in_session` | Croissance dans une session |
| 9 | `test_relative_position_session_end` | `compute_relative_position_in_session` | Dernière barre ≈ 1.0 |
| 10 | `test_scaler_fit_transform` | `SessionVolatilityScaler` | Z-scores centrés (~0) par session |
| 11 | `test_scaler_train_only_no_leak` | `SessionVolatilityScaler` | fit(train) → transform(test modifié) inchangé |
| 12 | `test_scaler_missing_session_fallback` | `SessionVolatilityScaler` | Session absente du train → fallback global |
| 13 | `test_pipeline_adds_session_columns` | `build_ml_ready` | Colonnes session présentes dans la sortie |
| 14 | `test_pipeline_one_hot_columns` | `build_ml_ready` | 4 dummies présentes si encoding='one_hot' |
| 15 | `test_pipeline_ordinal_column` | `build_ml_ready` | Seulement session_id si encoding='ordinal' |
| 16 | `test_pipeline_train_end_separation` | `build_ml_ready` | train_end=None vs train_end='2023-12-31' donne des zscores différents |

### 4.2 Test anti-leak critique (test 11)

```python
def test_scaler_train_only_no_leak(self):
    """Vérifie que le scaler fit sur train est insensible aux données test."""
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    
    # Train : ATR_Norm ~ 0.002, Test : ~ 0.010 (5x plus volatil)
    atr_train = pd.Series(rng.normal(0.002, 0.0005, 100), index=idx[:100])
    atr_test_normal = pd.Series(rng.normal(0.010, 0.002, 100), index=idx[100:])
    atr_test_modified = atr_test_normal * 2.0  # Encore plus extrême
    
    sid_train = compute_session_id(idx[:100])
    sid_test = compute_session_id(idx[100:])
    
    scaler = SessionVolatilityScaler()
    scaler.fit(atr_train, sid_train)
    
    z1 = scaler.transform(atr_test_normal, sid_test)
    z2 = scaler.transform(atr_test_modified, sid_test)
    
    # Les z-scores doivent être différents (la distribution test a changé)
    # mais les stats utilisées sont celles du train → pas de fuite
    assert not np.allclose(z1.values, z2.values, equal_nan=True)
    # Vérifier que les stats internes n'ont pas changé après transform
    stats_after = scaler._stats.copy()
    scaler.transform(atr_test_modified, sid_test)
    assert scaler._stats == stats_after  # Immutable après fit
```

### 4.3 Tests existants à vérifier

- [`test_regime.py`](../tests/unit/test_regime.py) — inchangé
- [`test_filters.py`](../tests/unit/test_filters.py) — `SessionFilter` existant non modifié
- [`test_eurusd_full_pipeline.py`](../tests/acceptance/test_eurusd_full_pipeline.py) — vérifier que le pipeline complet passe avec les nouvelles colonnes

---

## 5. Risques et mitigations

| # | Risque | Prob. | Impact | Mitigation |
|---|--------|-------|--------|------------|
| R1 | `ATR_session_zscore` leak si `train_end` mal passé | Moyen | Critique | Test unitaire 11 + assertion dans `BasePipeline` |
| R2 | Session one-hot × features existantes = explosion dimensionnelle | Faible | Moyen | 4 dummies seulement (+4 colonnes sur ~15) → négligeable |
| R3 | `relative_position_in_session` mal calculée pour Tokyo (midnight wrap) | Moyen | Faible | Test unitaire 9 + `.clip(0, 1)` |
| R4 | `session_open_range` non-normalisé → dominé par le niveau de prix | Élevé | Moyen | Normaliser par `Close` dans la v2 si la feature importance est excessive |
| R5 | DST ignoré → sessions décalées de ±1h en mars/octobre | Élevé | Faible | Accepté comme bruit. Surveiller via Sharpe mensuel. |
| R6 | `session_encoding="ordinal"` avec RF → fausse distance | Faible | Élevé | Défaut = `"one_hot"`. Documenter que ordinal = pour GBM uniquement. |

---

## 6. Plan d'exécution (ordre chronologique)

### Phase A — Nouvelles fonctions pures (20 min)

| Étape | Fichier | Action |
|-------|---------|--------|
| A1 | [`regime.py`](../learning_machine_learning/features/regime.py) | Ajouter `compute_session_id()` |
| A2 | [`regime.py`](../learning_machine_learning/features/regime.py) | Ajouter `compute_session_open_range()` |
| A3 | [`regime.py`](../learning_machine_learning/features/regime.py) | Ajouter `compute_relative_position_in_session()` |
| A4 | [`regime.py`](../learning_machine_learning/features/regime.py) | Ajouter `SessionVolatilityScaler` |

### Phase B — Tests unitaires (30 min)

| Étape | Fichier | Action |
|-------|---------|--------|
| B1 | `tests/unit/test_session_features.py` | Créer le fichier + 16 tests |
| B2 | — | `pytest tests/unit/test_session_features.py -v` → 16/16 |

### Phase C — Intégration pipeline (20 min)

| Étape | Fichier | Action |
|-------|---------|--------|
| C1 | [`instruments.py`](../learning_machine_learning/config/instruments.py) | Ajouter `session_encoding` à `InstrumentConfig` |
| C2 | [`pipeline.py`](../learning_machine_learning/features/pipeline.py) | Mettre à jour les imports |
| C3 | [`pipeline.py`](../learning_machine_learning/features/pipeline.py) | Ajouter le bloc session features (après ligne 172) |
| C4 | [`pipeline.py`](../learning_machine_learning/features/pipeline.py) | Mettre à jour `colonnes_finales` |
| C5 | [`pipeline.py`](../learning_machine_learning/features/pipeline.py) | Ajouter `train_end` au paramètre de signature |

### Phase D — Intégration orchestrateur (10 min)

| Étape | Fichier | Action |
|-------|---------|--------|
| D1 | [`base.py`](../learning_machine_learning/pipelines/base.py) ou [`eurusd.py`](../learning_machine_learning/pipelines/eurusd.py) | Passer `train_end` à `build_ml_ready()` |

### Phase E — Validation complète (20 min)

| Étape | Action |
|-------|--------|
| E1 | `pytest tests/ -v` — vérifier 0 régression |
| E2 | Lancer le pipeline complet → vérifier colonnes session dans le dataset ML |
| E3 | Vérifier permutation importance des nouvelles features |
| E4 | Calculer Sharpe OOS 2025 avec et sans features session |

**Temps total estimé** : ~1h40

---

## 7. Points d'attention spécifiques

1. **`ATR_Norm` est dans `features_dropped`** d'`EurUsdConfig` (ligne 108). Mais on en a besoin pour calculer `ATR_session_zscore`. Heureusement, `build_ml_ready()` calcule `ATR_Norm` à la ligne 162 et l'utilise immédiatement après. Le `features_dropped` est appliqué à la fin (lignes 236-245). Donc `ATR_Norm` est disponible pour le calcul intermédiaire → OK.

2. **`FILTER_KEEP`** (ligne 233) préserve `ATR_Norm` même si dans `features_dropped`. Pas d'impact sur nos nouvelles colonnes.

3. **Le `SessionFilter` existant** ([`filters.py:95`](../learning_machine_learning/backtest/filters.py)) reste inchangé. C'est un filtre backtest, pas une feature. Les deux coexistent indépendamment.

4. **Walk-forward** : `SessionVolatilityScaler` doit être re-fit à chaque fenêtre de walk-forward. Si le walk-forward est dans `BasePipeline`, le `train_end` doit être mis à jour dynamiquement.

5. **Ordinal encoding pour le futur** : Quand Step 03 (GBM/LightGBM) sera débloqué, il suffira de passer `session_encoding="ordinal"` et d'ajouter `categorical_feature=["session_id"]` dans les params LightGBM. Le code est déjà prévu pour.
