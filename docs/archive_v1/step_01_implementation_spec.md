# Step 01 — Spécification d'implémentation détaillée

**Date** : 2026-05-12
**Mode** : Architect (spécification seulement — implémentation en mode Code)
**Fichier stratégique** : [`docs/step_01_target_redefinition.md`](step_01_target_redefinition.md)

---

## Vue d'ensemble du plan d'exécution

8 modifications ordonnées, chaque modification est atomique et testable indépendamment.

| # | Fichier modifié | Type de changement | Impact |
|---|---|---|---|
| 1 | [`triple_barrier.py`](../learning_machine_learning/features/triple_barrier.py) | Ajout 3 fonctions | Nouveau |
| 2 | [`instruments.py`](../learning_machine_learning/config/instruments.py) | Ajout champ `target_mode` | Rétrocompatible |
| 3 | [`pipeline.py`](../learning_machine_learning/features/pipeline.py) | Routing `target_mode` | Rétrocompatible |
| 4 | [`training.py`](../learning_machine_learning/model/training.py) | Support régresseur | Extension |
| 5 | [`prediction.py`](../learning_machine_learning/model/prediction.py) | Sortie continue | Extension |
| 6 | [`simulator.py`](../learning_machine_learning/backtest/simulator.py) | Signaux continus | Extension |
| 7 | [`base.py`](../learning_machine_learning/pipelines/base.py) | Orchestration multi-mode | Extension |
| 8 | Tests unitaires | 3 nouveaux fichiers de test | Nouveau |

---

## Mod 1 — [`triple_barrier.py`](../learning_machine_learning/features/triple_barrier.py) : 3 nouvelles fonctions cible

### 1a. `compute_forward_return_target`

```python
def compute_forward_return_target(
    df: pd.DataFrame,
    horizon_hours: int = 24,
    pip_size: float = 0.0001,
) -> np.ndarray:
    """Cible de régression : log-return forward sur `horizon_hours`.

    Y_i = log(Close_{i+horizon_hours} / Close_i)

    Args:
        df: DataFrame avec colonne 'Close'. Index trié chronologiquement.
        horizon_hours: Horizon de prédiction en nombre de barres (doit être ≥ 1).
        pip_size: Taille d'un pip (utilisé uniquement pour le logging).

    Returns:
        np.ndarray float64 de même longueur que df.
        Les `horizon_hours` dernières barres sont NaN.
    """
```

**Détails d'implémentation** :
- `closes = df["Close"].values`
- `targets[i] = np.log(closes[i + horizon_hours] / closes[i])` pour `i` de `0` à `n - horizon_hours - 1`
- Les `horizon_hours` dernières barres = `np.nan`
- Validation : `horizon_hours >= 1`, colonne `Close` présente
- **Pas de boucle Python sur les barres** — vectorisé avec `np.log(closes[horizon:] / closes[:-horizon])` puis padding de NaN

### 1b. `compute_directional_clean_target`

```python
def compute_directional_clean_target(
    df: pd.DataFrame,
    horizon_hours: int = 24,
    noise_threshold_atr: float = 0.5,
    atr_period: int = 14,
    pip_size: float = 0.0001,
) -> np.ndarray:
    """Cible binaire pure : +1/-1 si |forward_return| > seuil_bruit, sinon 0.

    Y_i = sign(log_return) * 1{|log_return| > noise_threshold_atr * ATR_14(i) / Close_i}

    Args:
        df: DataFrame avec colonnes 'High', 'Low', 'Close'.
        horizon_hours: Horizon de prédiction en barres.
        noise_threshold_atr: Multiplicateur d'ATR pour le seuil de bruit.
        atr_period: Période pour le calcul ATR.
        pip_size: Taille d'un pip.

    Returns:
        np.ndarray float64 ∈ {-1.0, 0.0, 1.0}. Les `horizon_hours` dernières barres sont NaN.
    """
```

**Détails d'implémentation** :
- Calculer `log_return[i] = log(Close[i+horizon] / Close[i])` vectorisé
- Calculer `atr_14[i]` via `pandas_ta.atr(length=atr_period)` puis `atr_norm[i] = atr[i] / Close[i]`
- `threshold[i] = noise_threshold_atr * atr_norm[i]`
- `targets[i] = sign(log_return[i])` si `abs(log_return[i]) > threshold[i]`, sinon `0.0`
- Les `max(horizon_hours, atr_period)` dernières barres = NaN
- Le seuil de bruit utilise l'ATR à la barre `i` (information passée uniquement, pas de look-ahead)

### 1c. `compute_cost_aware_target_v2`

```python
def compute_cost_aware_target_v2(
    df: pd.DataFrame,
    tp_pips: float = 30.0,
    sl_pips: float = 10.0,
    window: int = 24,
    friction_pips: float = 1.5,
    k_atr: float = 1.0,
    pip_size: float = 0.0001,
) -> np.ndarray:
    """Variante cost-aware v2 : seuil de profit minimum adaptatif basé sur l'ATR.

    Un trade n'est labellisé gagnant que si :
        pnl_réel(i) - friction > k_atr * ATR_14(i)

    où pnl_réel est le PnL net de la résolution triple barrière (TP/SL/timeout).

    Args:
        df: DataFrame OHLCV.
        tp_pips, sl_pips, window: Paramètres triple barrière classiques.
        friction_pips: Coût total (commission + slippage) en pips.
        k_atr: Multiplicateur d'ATR pour le profit minimum requis.
        pip_size: Taille d'un pip.

    Returns:
        np.ndarray float64 ∈ {-1.0, 0.0, 1.0}. Les `window` dernières barres sont NaN.
    """
```

**Détails d'implémentation** :
- Réutilise la boucle de résolution de [`apply_triple_barrier`](../learning_machine_learning/features/triple_barrier.py:68) (logique TP/SL/timeout identique)
- Après résolution, calcule le PnL brut (en prix) pour chaque barre
- `min_profit(i) = k_atr * ATR_14(i)` — seuil adaptatif basé sur l'ATR à la barre d'entrée
- Label directionnel seulement si `pnl_net(i) > min_profit(i)`, sinon `0`
- **Différence clé avec `apply_triple_barrier_cost_aware` existante** : le seuil `min_profit` est dynamique (varie avec l'ATR) au lieu d'être un paramètre fixe global

### Notes communes aux 3 fonctions

- Toutes retournent `np.ndarray[float64]` de même longueur que `df`
- Les NaN en fin de série sont gérés par `dropna(subset=["Target"])` dans le pipeline
- Logging structuré via `get_logger(__name__)` comme les fonctions existantes
- Ajouter les 3 fonctions au `__all__` du module si existant

---

## Mod 2 — [`instruments.py`](../learning_machine_learning/config/instruments.py) : champ `target_mode`

### Modification du dataclass `InstrumentConfig`

```python
from typing import Literal

TargetMode = Literal[
    "triple_barrier",    # actuel, valeur par défaut (rétrocompatible)
    "forward_return",    # Ha : régression continue
    "directional_clean", # Hb : binaire pure avec seuil de bruit adaptatif
    "cost_aware_v2",     # Hc : cost-aware avec seuil ATR adaptatif
]

@dataclass(frozen=True)
class InstrumentConfig:
    # ... champs existants ...
    target_mode: TargetMode = "triple_barrier"  # NOUVEAU
    # Paramètres pour forward_return
    target_horizon_hours: int = 24               # NOUVEAU
    # Paramètres pour directional_clean
    target_noise_threshold_atr: float = 0.5      # NOUVEAU
    target_atr_period: int = 14                  # NOUVEAU
    # Paramètres pour cost_aware_v2
    target_k_atr: float = 1.0                    # NOUVEAU
```

**Règle de validation `__post_init__`** :
- `target_horizon_hours >= 1`
- `target_noise_threshold_atr > 0`
- `target_k_atr > 0`
- Si `target_mode == "cost_aware_v2"`, `cost_aware_labeling` devrait être `False` (exclusif)

**Impact sur `EurUsdConfig`** : aucune modification nécessaire, hérite de la valeur par défaut `"triple_barrier"`.

**Rétrocompatibilité** : `target_mode="triple_barrier"` par défaut → comportement existant inchangé.

---

## Mod 3 — [`pipeline.py`](../learning_machine_learning/features/pipeline.py) : routing `target_mode`

### Modification de [`build_ml_ready`](../learning_machine_learning/features/pipeline.py:36)

Remplacer le bloc actuel lignes 80-106 (étape 1 du labelling) par un router.

**Pseudo-code du router** :

```python
# 1. Target labelling selon target_mode
if instrument.target_mode == "forward_return":
    h1["Target"] = compute_forward_return_target(
        h1, horizon_hours=instrument.target_horizon_hours,
        pip_size=instrument.pip_size,
    )
elif instrument.target_mode == "directional_clean":
    h1["Target"] = compute_directional_clean_target(
        h1,
        horizon_hours=instrument.target_horizon_hours,
        noise_threshold_atr=instrument.target_noise_threshold_atr,
        atr_period=instrument.target_atr_period,
        pip_size=instrument.pip_size,
    )
elif instrument.target_mode == "cost_aware_v2":
    h1["Target"] = compute_cost_aware_target_v2(
        h1, tp_pips=tp_pips, sl_pips=sl_pips, window=window,
        friction_pips=instrument.friction_pips,
        k_atr=instrument.target_k_atr,
        pip_size=instrument.pip_size,
    )
elif instrument.cost_aware_labeling:
    # Ancien chemin cost_aware (conservé pour rétrocompatibilité)
    h1["Target"] = apply_triple_barrier_cost_aware(...)
else:
    # Triple barrière classique (défaut)
    h1["Target"] = apply_triple_barrier(...)

h1.dropna(subset=["Target"], inplace=True)
log_row_loss("dropna Target", n_before, len(h1))
```

**Points d'attention** :
- Ajouter l'import des 3 nouvelles fonctions depuis [`triple_barrier.py`](../learning_machine_learning/features/triple_barrier.py)
- Le `cost_aware_labeling` existant reste supporté comme chemin legacy
- Le logging doit indiquer quel mode de target est utilisé

---

## Mod 4 — [`training.py`](../learning_machine_learning/model/training.py) : support régresseur

### 4a. Nouvelle factory pour le régresseur

```python
from sklearn.ensemble import HistGradientBoostingRegressor

def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> HistGradientBoostingRegressor:
    """Entraîne un HistGradientBoostingRegressor avec les paramètres fournis.

    Args:
        X_train: Features d'entraînement.
        y_train: Target continue (log-return forward).
        params: Dict kwargs pour HistGradientBoostingRegressor.

    Returns:
        Régresseur entraîné.
    """
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    logger.info(
        "Régresseur entraîné : max_iter=%d, échantillons=%d",
        params.get("max_iter", "?"), len(X_train),
    )
    return model
```

### 4b. Modification de `train_test_split_purge`

Aucune modification nécessaire — la fonction est déjà agnostique au type de target. Elle extrait `y = df["Target"]` quel que soit le type (classes ou continu).

### 4c. Impact sur `walk_forward_train`

La fonction existante est déjà générique via `ModelFactory`. Pour le mode régression, on passera une factory qui appelle `train_regressor` au lieu de `train_model`.

---

## Mod 5 — [`prediction.py`](../learning_machine_learning/model/prediction.py) : sortie continue

### Nouvelle fonction `predict_oos_regression`

```python
def predict_oos_regression(
    model,  # HistGradientBoostingRegressor | RandomForestRegressor
    df: pd.DataFrame,
    eval_year: int,
    X_cols: list[str],
) -> pd.DataFrame:
    """Prédit sur une année OOS en mode régression.

    Returns:
        DataFrame avec colonnes : Close_Reel_Direction, Predicted_Return,
        Spread.
        Pas de Confiance_*_% (pas de probas en régression).
    """
```

**Sortie du DataFrame** :
- `Close_Reel_Direction` : la target vraie (log-return forward)
- `Predicted_Return` : `model.predict(X_test)` → float continu
- `Spread` : si disponible

### Modification de [`predict_oos`](../learning_machine_learning/model/prediction.py:16) existante

Aucune modification — elle reste pour le mode classifieur. La nouvelle fonction est un chemin parallèle.

---

## Mod 6 — [`simulator.py`](../learning_machine_learning/backtest/simulator.py) : signaux continus

### 6a. Nouvelle fonction `simulate_trades_continuous`

```python
def simulate_trades_continuous(
    df: pd.DataFrame,
    weight_func: Callable[[np.ndarray], np.ndarray],
    tp_pips: float = 30.0,
    sl_pips: float = 10.0,
    window: int = 24,
    pip_size: float = 0.0001,
    signal_threshold: float = 0.0005,  # seuil sur |predicted_return|
    commission_pips: float = 0.5,
    slippage_pips: float = 1.0,
    filter_pipeline: FilterPipeline | None = None,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """Simule la stratégie en mode stateful avec signal continu.

    Le signal est dérivé du predicted_return :
    - mask_long  = predicted_return > +signal_threshold
    - mask_short = predicted_return < -signal_threshold
    - Poids proportionnel à |predicted_return| (via weight_func)

    Args:
        df: DataFrame avec colonne Predicted_Return (au lieu de Prediction_Modele).
        signal_threshold: Seuil minimum sur |predicted_return| pour entrer.

    Returns:
        Identique à simulate_trades.
    """
```

**Différences clés avec [`simulate_trades`](../learning_machine_learning/backtest/simulator.py:27)** :
- `mask_long = df["Predicted_Return"] > signal_threshold` au lieu de `Prediction_Modele == 1`
- `mask_short = df["Predicted_Return"] < -signal_threshold`
- Pas de `Confiance_*_%` → `proba_max` devient `abs(Predicted_Return)` normalisé
- Le reste de la boucle stateful (TP/SL/timeout) est **identique** → extraire en helper `_simulate_stateful_loop()` partagé ?

**Recommandation architecturale** : Extraire la boucle stateful dans `_simulate_stateful_loop(n, dates, highs, lows, closes, signals, weights, spreads, filter_rejected_arr, tp_dist, sl_dist, spread_cost_base, window, pip_size) -> list[dict]` pour éviter la duplication. `simulate_trades` et `simulate_trades_continuous` deviennent des wrappers qui construisent `signals` et `weights` différemment puis appellent la boucle commune.

### 6b. Seuil `signal_threshold`

- Exprimé en log-return (ex: `0.0005` ≈ 5 pips pour EURUSD)
- À calibrer sur val_year : quantile 70% de `|Predicted_Return|` sur l'année de validation
- Stocké dans `BacktestConfig` : `continuous_signal_threshold: float = 0.0005`

---

## Mod 7 — [`base.py`](../learning_machine_learning/pipelines/base.py) : orchestration multi-mode

### 7a. `train_model` → router selon `target_mode`

```python
def train_model(self, ml_data):
    from learning_machine_learning.model.training import (
        _FILTER_ONLY_COLS,
        train_test_split_purge,
        train_model,
        train_regressor,  # NOUVEAU
    )
    X_train, y_train, X_cols = train_test_split_purge(...)

    if self.instrument.target_mode == "forward_return":
        model = train_regressor(X_train, y_train, self.model_cfg.gbm_params)  # NOUVEAU
    else:
        model = train_model(X_train, y_train, self.model_cfg.rf_params)
    return model, X_cols
```

### 7b. `evaluate_model` → router

```python
def evaluate_model(self, model, ml_data, X_cols):
    if self.instrument.target_mode == "forward_return":
        from learning_machine_learning.model.prediction import predict_oos_regression
        for year in self.model_cfg.eval_years:
            preds_df = predict_oos_regression(model, ml_data, year, X_cols)
            ...
    else:
        # chemin classifieur existant
        ...
```

### 7c. `run_backtest` → router

```python
def run_backtest(self, predictions, ml_data, ohlcv_h1):
    if self.instrument.target_mode == "forward_return":
        simulate_func = simulate_trades_continuous
    else:
        simulate_func = simulate_trades

    # ... construction des filtres (inchangée) ...
    for year, preds_df in predictions.items():
        trades_df, n_signaux, n_filtres = simulate_func(
            df=df_backtest,
            signal_threshold=self.backtest_cfg.continuous_signal_threshold,  # NOUVEAU
            ...
        )
```

### 7d. `run_walk_forward` — adaptation

Le walk-forward utilise une factory `model_factory`. Pour le mode régression, la factory doit appeler `train_regressor`. Le code de prédiction dans la boucle fold doit utiliser `model.predict()` (continu) au lieu de `model.predict()` + `model.predict_proba()` (classes).

### 7e. `ModelConfig` — nouveaux hyperparamètres GBM

```python
@dataclass(frozen=True)
class ModelConfig:
    # ... champs RF existants ...
    # GBM regressor params (pour forward_return)
    gbm_max_iter: int = 200
    gbm_max_depth: int = 5
    gbm_min_samples_leaf: int = 50
    gbm_learning_rate: float = 0.05
    gbm_loss: str = "huber"  # "huber" ou "quantile" pour robustesse queues

    @property
    def gbm_params(self) -> dict:
        return {
            "max_iter": self.gbm_max_iter,
            "max_depth": self.gbm_max_depth,
            "min_samples_leaf": self.gbm_min_samples_leaf,
            "learning_rate": self.gbm_learning_rate,
            "loss": self.gbm_loss,
            "random_state": self.random_seed,
        }
```

---

## Mod 8 — Tests unitaires

### 8a. [`test_target_regression.py`](../tests/unit/test_target_regression.py)

| Test | Vérification |
|---|---|
| `test_forward_return_basic` | Hausse de 1% → log return ≈ 0.00995 |
| `test_forward_return_last_nan` | Les `horizon` dernières barres sont NaN |
| `test_forward_return_vectorized` | Cohérence avec calcul boucle Python |
| `test_directional_clean_noise` | Petit mouvement sous seuil → 0 |
| `test_directional_clean_signal` | Grand mouvement au-dessus seuil → ±1 |
| `test_directional_clean_last_nan` | Les `max(horizon, atr_period)` dernières barres NaN |
| `test_cost_aware_v2_atr_threshold` | Vérifie seuil adaptatif vs fixe |
| `test_invalid_params_raises` | `horizon_hours=0`, colonnes manquantes |

### 8b. [`test_simulator_continuous.py`](../tests/unit/test_simulator_continuous.py)

| Test | Vérification |
|---|---|
| `test_long_signal_above_threshold` | `predicted_return > threshold` → trade LONG |
| `test_short_signal_below_neg_threshold` | `predicted_return < -threshold` → trade SHORT |
| `test_no_trade_between_thresholds` | `|predicted_return| < threshold` → pas de trade |
| `test_zero_signal` | Tous les retours sous seuil → 0 trades |
| `test_stateful_same_as_classifier` | Boucle stateful identique au mode classifieur |

### 8c. [`test_regressor_training.py`](../tests/unit/test_regressor_training.py)

| Test | Vérification |
|---|---|
| `test_train_regressor_basic` | `HistGradientBoostingRegressor` s'entraîne sans erreur |
| `test_predict_oos_regression_shape` | Sortie contient `Predicted_Return` |
| `test_regressor_spearman_positive` | Spearman > 0 sur données synthétiques simples |

---

## Décisions architecturales clés

### D1. Extraction de la boucle stateful

**Problème** : `simulate_trades` et `simulate_trades_continuous` partagent 90% de leur code (la boucle stateful TP/SL/timeout). Dupliquer ce code est un risque de divergence.

**Décision** : Extraire la boucle dans `_simulate_stateful_core()` — fonction pure sans dépendance aux colonnes de prédiction.

**Trade-off** : Complexité additionnelle d'une indirection vs garantie de cohérence TP/SL entre les deux modes.

**Verdict** : Extraction justifiée car la boucle stateful est le cœur critique du backtest et ne doit JAMAIS diverger.

### D2. `Predicted_Return` vs `Prediction_Modele`

**Problème** : Le DataFrame de prédictions change de schéma selon le mode.

**Décision** : Deux colonnes mutuellement exclusives :
- Mode classifieur : `Prediction_Modele` + `Confiance_*_%`
- Mode régression : `Predicted_Return`

Le code downstream (reporting, diagnostics) doit check `"Predicted_Return" in df.columns` pour s'adapter.

**Trade-off** : Complexité de branching vs deux fonctions séparées.

**Verdict** : Deux fonctions `predict_oos` / `predict_oos_regression` distinctes, pas de fonction unique avec `if/else`. Plus lisible, plus facile à tester.

### D3. Seuil continu calibré sur val_year

**Problème** : Le seuil `signal_threshold` pour `simulate_trades_continuous` doit être calibré sans look-ahead.

**Décision** :
1. Après entraînement, prédire sur val_year
2. Calculer `threshold = quantile(|predicted_return|, 0.70)` sur val_year
3. Appliquer ce seuil sur test_year
4. Stocker `threshold` dans les résultats pour traçabilité

**Anti-leak** : Le seuil est calculé UNIQUEMENT sur val_year. Le test_year n'est jamais utilisé pour la calibration.

### D4. `HistGradientBoostingRegressor` vs `XGBRegressor`

**Décision** : `HistGradientBoostingRegressor` (sklearn ≥ 1.0)

**Justification** :
- Pas de dépendance supplémentaire (`xgboost` non listé dans `requirements.txt`)
- `loss="huber"` natif pour robustesse aux queues
- Plus rapide que XGBoost sur datasets < 100k lignes
- API cohérente avec le reste du code sklearn

---

## Flux de données complet — mode régression

```
OHLCV H1 brut
  │
  ├─ compute_forward_return_target(h=24)
  │    └─ Target = log(Close_t+24 / Close_t)   [continu, float64]
  │
  ├─ Features H1/H4/D1/Macro (inchangé)
  │
  ├─ Split temporel (train_end_year=2023, purge=48h)
  │
  ├─ HistGradientBoostingRegressor.fit(X_train, y_train)
  │
  ├─ predict_oos_regression → Predicted_Return (float)
  │
  ├─ Calibration seuil sur val_year:
  │    τ = quantile_70%(|Predicted_Return_val|)
  │
  ├─ simulate_trades_continuous:
  │    signal = sign(Predicted_Return) si |Predicted_Return| > τ
  │    weight = |Predicted_Return| normalisé
  │
  └─ Métriques backtest (Sharpe, DSR, WR)
```

---

## Checklist de vérification anti-leak

- [ ] `compute_forward_return_target` n'utilise que `df["Close"]` — pas de High/Low forward
- [ ] `compute_directional_clean_target` : l'ATR est calculé à la barre `i`, pas après
- [ ] Le seuil `τ` pour `simulate_trades_continuous` est calibré sur val_year UNIQUEMENT
- [ ] `train_test_split_purge` applique l'embargo de 48h — les targets dans la zone de purge sont NaN
- [ ] Les `horizon_hours` / `window` dernières barres sont NaN dans TOUTES les fonctions cible
- [ ] `dropna(subset=["Target"])` est appelé APRÈS le labelling dans le pipeline

---

## Ordre d'implémentation recommandé

1. **Mod 1** — Fonctions cible pures (testables indépendamment)
2. **Mod 8a** — Tests unitaires des fonctions cible (TDD)
3. **Mod 2** — `InstrumentConfig.target_mode` (dataclass, simple)
4. **Mod 3** — Router dans `pipeline.py` (intégration légère)
5. **Mod 4 + Mod 7e** — Régresseur + config GBM
6. **Mod 5** — `predict_oos_regression`
7. **Mod 6** — `simulate_trades_continuous` (d'abord extraire `_simulate_stateful_core`)
8. **Mod 8b + 8c** — Tests simulateur + régresseur
9. **Mod 7a-7d** — Orchestration dans `base.py` (intégration finale)
10. **Run complet** — `run_pipeline_v1.py` avec `EurUsdConfig(target_mode="forward_return")`
