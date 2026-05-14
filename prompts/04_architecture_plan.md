# Prompt 04 — Plan d'architecture détaillé

> Document compagnon de [`prompts/04_features_research_harness.md`](04_features_research_harness.md).
> Ce plan est la référence avant de passer en mode Code pour l'implémentation.

---

## 1. Diagramme de composants

```
┌──────────────────────────────────────────────────────────────────────┐
│  scripts/run_feature_research.py  (CLI entry point)                  │
│  --asset US30 --tf D1 --horizon 5 --n-top 20 --train-end 2022-12-31│
└──────────────────────────┬───────────────────────────────────────────┘
                           │ appelle
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  app/features/research.py  ::  rank_features(asset, tf, ...)         │
│                                                                      │
│  1. load_asset(asset, tf)          ──▶ app/data/loader.py            │
│  2. df.loc[:train_end]             ──▶ split temporel strict         │
│  3. compute_all_indicators(df)     ──▶ app/features/indicators.py    │
│  4. target = close.shift(-h) / close - 1                             │
│  5. mutual_info_regression(X, y)   ──▶ sklearn                       │
│  6. X.corrwith(y).abs()            ──▶ pandas                        │
│  7. RF(200, max_depth=4) + perm_imp──▶ sklearn                       │
│  8. composite_rank = mean(rank MI, rank corr, rank perm)             │
│  9. save → predictions/feature_research_{ASSET}_{TF}.json            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────────┐ ┌────────────┐ ┌──────────────────────┐
│ indicators.py    │ │ loader.py  │ │ look_ahead_validator │
│ (18 indicateurs) │ │ (existant) │ │ (existant, réutilisé)│
└──────────────────┘ └────────────┘ └──────────────────────┘
```

**Fichiers à créer (4)** :

| Fichier | Rôle |
|---|---|
| `app/features/indicators.py` | 18 indicateurs vectorisés + `compute_all_indicators()` |
| `app/features/research.py` | Pipeline `rank_features()` |
| `scripts/run_feature_research.py` | CLI argparse |
| `tests/unit/test_indicators.py` | Tests anti-look-ahead + edge cases par indicateur |
| `tests/unit/test_feature_research.py` | Tests d'intégration du pipeline ranking |

**Fichiers modifiés (0)** : aucun existant n'est touché.

**Fichiers réutilisés (sans modification)** :
- [`app/data/loader.py`](../app/data/loader.py) — `load_asset()`
- [`app/testing/look_ahead_validator.py`](../app/testing/look_ahead_validator.py) — `assert_no_look_ahead()`, `@look_ahead_safe`
- [`app/core/exceptions.py`](../app/core/exceptions.py) — `LookAheadError`
- [`app/core/logging.py`](../app/core/logging.py) — `get_logger()`
- [`app/core/seeds.py`](../app/core/seeds.py) — `set_global_seeds()`

---

## 2. [`app/features/indicators.py`](app/features/indicators.py) — Catalogue d'indicateurs

### 2.1 Principe de conception

- **100% vectorisé** : zéro boucle Python row-by-row. Priorité `.shift()`, `.rolling()`, `.ewm()`, `.where()`, `.diff()`, `.cumsum()`.
- **Chaque fonction décorée `@look_ahead_safe`** — vérifiable par `test_all_indicators_are_marked_safe`.
- **Wilder smoothing** pour ATR, ADX, RSI (standard industriel) : `ewm(alpha=1/period, adjust=False)`.
- **Signature uniforme pour les indicateurs uni-série** : `(close: pd.Series, period: int) -> pd.Series`.
- **Signature pour les indicateurs multi-séries** : `(high, low, close, ...) -> pd.Series | pd.DataFrame`.
- **Gestion des NaN** : retourne NaN pendant la période de warmup (pas de forward-fill trompeur).
- **Pas de `inplace=True`** — immutable-style.

### 2.2 Tableau des 18 indicateurs

| # | Catégorie | Fonction | Input | Paramètres | Sortie | Formule / Note |
|---|---|---|---|---|---|---|
| 1 | Trend | `sma` | `close` | `period` | 1 col | `close.rolling(period).mean()` |
| 2 | Trend | `ema` | `close` | `period` | 1 col | `close.ewm(span=period, adjust=False).mean()` |
| 3 | Trend | `dist_sma` | `close` | `period` | 1 col | `(close - sma(close, period)) / sma(close, period)` → distance normalisée |
| 4 | Trend | `slope_sma` | `close` | `period, lookback=5` | 1 col | Pente linéaire de la SMA sur `lookback` barres (linreg slope) |
| 5 | Momentum | `rsi` | `close` | `period=14` | 1 col | Wilder : `avg_gain = gain.ewm(alpha=1/period).mean()`, `rs = avg_gain / avg_loss`, `100 - 100/(1+rs)`. Si `avg_loss=0` → `rs=inf` → RSI=100 |
| 6 | Momentum | `macd` | `close` | `fast=12, slow=26, signal=9` | 3 cols : `macd_line`, `macd_signal`, `macd_histogram` | `macd_line = ema(fast) - ema(slow)`, `signal = ema(macd_line, signal)`, `histogram = macd_line - signal` |
| 7 | Momentum | `stoch` | `high, low, close` | `k=14, d=3` | 2 cols : `stoch_k`, `stoch_d` | `%K = 100*(close - lowest_low) / (highest_high - lowest_low)`. Si dénominateur = 0 → 50. `%D = sma(%K, d)` |
| 8 | Momentum | `williams_r` | `high, low, close` | `period=14` | 1 col | `-100 * (highest_high - close) / (highest_high - lowest_low)`. Si dénominateur = 0 → -50 |
| 9 | Momentum | `cci` | `high, low, close` | `period=20` | 1 col | `typical = (H+L+C)/3`, `cci = (typical - sma(typical)) / (0.015 * mad(typical))` où `mad = mean absolute deviation` |
| 10 | Volatilité | `atr` | `high, low, close` | `period=14` | 1 col | Wilder : `tr = max(H-L, |H-C_prev|, |L-C_prev|)`, `atr = tr.ewm(alpha=1/period, adjust=False).mean()` |
| 11 | Volatilité | `atr_pct` | `close, atr_series` | — | 1 col | `atr / close * 100` — normalisé en % |
| 12 | Volatilité | `bbands_width` | `close` | `period=20, n_std=2` | 1 col | `(upper - lower) / middle` où `middle = sma(period)`, `upper/lower = middle ± n_std * std(period)` |
| 13 | Volatilité | `keltner_width` | `high, low, close` | `period=20, n_atr=2` | 1 col | `(upper - lower) / middle` où `middle = ema(period)`, `upper/lower = middle ± n_atr * atr(period)` |
| 14 | Volume | `obv` | `close, volume` | — | 1 col | `direction = sign(close.diff())`, `obv = (direction * volume).cumsum()` |
| 15 | Volume | `mfi` | `high, low, close, volume` | `period=14` | 1 col | `typical = (H+L+C)/3`, `raw_mf = typical * volume`. `mfi = 100 - 100/(1 + mf_ratio)` où `mf_ratio = sum(pos_mf, period) / sum(neg_mf, period)` |
| 16 | Régime | `adx` | `high, low, close` | `period=14` | 3 cols : `adx_line`, `plus_di`, `minus_di` | Wilder standard : `+DM`, `-DM`, `TR`, `+DI`, `-DI`, `DX = |+DI - -DI|/(+DI + -DI)*100`, `ADX = DX.ewm(alpha=1/period).mean()` |
| 17 | Régime | `efficiency_ratio` | `close` | `period=20` | 1 col | `abs(close - close.shift(period)) / sum(abs(close.diff()), period)` → Kaufman efficiency ratio dans [0,1] |
| 18 | Régime | `realized_vol` | `close` | `period=20` | 1 col | `log_return = log(close/close.shift(1))`, `realized_vol = log_return.rolling(period).std() * sqrt(252)` |

**Total colonnes produites par `compute_all_indicators()`** : 22 colonnes
- 12 uni-série + 5 multi-séries (MACD=3, Stoch=2, ADX=3) = 22

### 2.3 Contrat `compute_all_indicators(df) -> pd.DataFrame`

```python
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule tous les indicateurs sur un DataFrame OHLCV.

    Préconditions :
        - df.index est un DatetimeIndex trié
        - Colonnes requises : [Open, High, Low, Close]
        - Colonne optionnelle : Volume (si absente ou tout à 0, skip OBV/MFI)

    Postconditions :
        - Même index que df
        - Même longueur que df (NaN en warmup, jamais forward-filled)
        - Colonnes nommées exactement comme dans le tableau 2.2
        - Aucun side-effect sur df

    Returns:
        DataFrame avec 20-22 colonnes de features.
    """
```

**Edge cases gérés** :
- Volume absent ou tout à 0 → `logger.warning("Volume absent, skip OBV/MFI")`, retourne 20 colonnes
- Série trop courte pour la période (ex: 5 barres pour `sma(20)`) → colonne entièrement NaN, pas d'erreur
- Division par zéro (Stoch, Williams %R, Keltner) → valeur sentinelle (50, -50, NaN) + `logger.debug`

### 2.4 Dépendances internes de `compute_all_indicators`

```
compute_all_indicators(df)
├── sma(close, 20), sma(close, 50), sma(close, 200)
├── ema(close, 20), ema(close, 50)
├── dist_sma(close, 20), dist_sma(close, 50)
├── slope_sma(close, 20), slope_sma(close, 50)
├── rsi(close, 14)
├── macd(close, 12, 26, 9)          → 3 colonnes
├── stoch(high, low, close, 14, 3)  → 2 colonnes
├── williams_r(high, low, close, 14)
├── cci(high, low, close, 20)
├── atr(high, low, close, 14)
├── atr_pct(close, atr_series)      ← dépend de atr()
├── bbands_width(close, 20, 2)
├── keltner_width(high, low, close, 20, 2)
├── obv(close, volume)              ← conditionnel
├── mfi(high, low, close, volume, 14) ← conditionnel
├── adx(high, low, close, 14)       → 3 colonnes
├── efficiency_ratio(close, 20)
└── realized_vol(close, 20)
```

**Note** : `atr_pct` est le seul indicateur qui dépend d'un autre indicateur (`atr`). `compute_all_indicators` appelle `atr()` une fois et passe le résultat à `atr_pct()` pour éviter un double calcul.

---

## 3. [`app/features/research.py`](app/features/research.py) — Pipeline de ranking

### 3.1 Signature et contrat

```python
def rank_features(
    asset: str,
    tf: str,
    target_horizon: int,
    n_top: int = 20,
    train_end: str = "2022-12-31",
) -> pd.DataFrame:
```

### 3.2 Pipeline détaillé

```
Étape 0 : set_global_seeds()  (règle 12 constitution)
Étape 1 : df = load_asset(asset, tf)            ← app/data/loader.py
Étape 2 : df = df.loc[:train_end]                ← split temporel strict
Étape 3 : features = compute_all_indicators(df)  ← app/features/indicators.py
Étape 4 : target = (df["Close"].shift(-target_horizon) / df["Close"] - 1).rename("y")
Étape 5 : aligned = pd.concat([features, target], axis=1).dropna()
          ├── Vérifier frac_dropped = 1 - len(aligned)/len(df)
          └── Si > 0.10 → logger.warning(">10% des barres droppées")
Étape 6 : X = aligned.drop(columns=["y"]), y = aligned["y"]
Étape 7 : mi = mutual_info_regression(X, y, random_state=42)
Étape 8 : corr = X.corrwith(y).abs()
Étape 9 : rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
          rf.fit(X, y)
          perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
Étape 10: Construire result DataFrame avec colonnes :
          [feature, mutual_info, abs_corr, permutation_importance,
           mutual_info_rank, abs_corr_rank, permutation_importance_rank, composite_rank]
Étape 11: composite_rank = mean(mi_rank, corr_rank, perm_rank)
          Trier par composite_rank ascendant, head(n_top)
Étape 12: Sauvegarder predictions/feature_research_{ASSET}_{TF}.json
Étape 13: Retourner result
```

### 3.3 Edge cases

| Cas | Comportement |
|---|---|
| `train_end` antérieur aux données → `df.loc[:train_end]` vide | `compute_all_indicators` retourne un DF vide → `dropna()` vide → `ValueError` à l'étape 6. **Mitigation** : vérifier `if len(df) == 0: raise DataValidationError(...)` |
| `target_horizon` ≥ longueur des données | Toutes les cibles NaN → `aligned` vide → `ValueError`. **Mitigation** : `if len(aligned) == 0: raise ValueError(f"Horizon {target_horizon} trop grand")` |
| Volume tout à 0 | `compute_all_indicators` skip OBV/MFI → 20 colonnes au lieu de 22. Le pipeline fonctionne normalement |
| `n_top > len(features.columns)` | `head(n_top)` retourne toutes les colonnes, pas d'erreur |
| Actif avec < 1 an de données | `dropna()` peut éliminer > 50% → warning + pipeline continue |

### 3.4 Format de sortie JSON

```json
[
  {
    "feature": "dist_sma_20",
    "mutual_info": 0.0234,
    "abs_corr": 0.0812,
    "permutation_importance": 0.0041,
    "mutual_info_rank": 1.0,
    "abs_corr_rank": 3.0,
    "permutation_importance_rank": 2.0,
    "composite_rank": 2.0
  },
  ...
]
```

**Contrat pour les prompts futurs** (12, 13) : ce fichier JSON est l'input de référence. Toute modification du schéma = breaking change.

---

## 4. [`scripts/run_feature_research.py`](scripts/run_feature_research.py) — CLI

### 4.1 Architecture

```python
from __future__ import annotations

import argparse
import sys

from app.core.seeds import set_global_seeds
from app.core.logging import setup_logging, get_logger
from app.features.research import rank_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank features by predictive power (Prompt 04)"
    )
    parser.add_argument("--asset", required=True, help="Asset name (ex: US30)")
    parser.add_argument("--tf", required=True,
                        choices=["D1", "H4", "H1", "M15", "M5"],
                        help="Timeframe")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward return horizon in bars")
    parser.add_argument("--n-top", type=int, default=20,
                        help="Number of top features to return")
    parser.add_argument("--train-end", default="2022-12-31",
                        help="Train cutoff date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds()
    setup_logging()
    logger = get_logger(__name__)

    try:
        result = rank_features(
            asset=args.asset,
            tf=args.tf,
            target_horizon=args.horizon,
            n_top=args.n_top,
            train_end=args.train_end,
        )
    except Exception as e:
        logger.error("rank_features_failed", extra={"context": {"error": str(e)}})
        sys.exit(1)

    # Affichage console
    print(f"\nTop {len(result)} features for {args.asset} {args.tf} (horizon={args.horizon}):")
    print(result[["feature", "composite_rank", "mutual_info", "abs_corr"]].to_string(index=False))

    logger.info("rank_features_done", extra={"context": {
        "asset": args.asset,
        "tf": args.tf,
        "horizon": args.horizon,
        "n_features_ranked": len(result),
    }})


if __name__ == "__main__":
    main()
```

### 4.2 Usage

```bash
rtk python scripts/run_feature_research.py --asset US30 --tf D1 --horizon 5 --n-top 15
rtk python scripts/run_feature_research.py --asset XAUUSD --tf H4 --horizon 10
```

---

## 5. Tests — Architecture détaillée

### 5.1 [`tests/unit/test_indicators.py`](tests/unit/test_indicators.py)

| Test | Type | Cible | Vérification |
|---|---|---|---|
| `test_non_look_ahead_{name}` | Parametrized × 18 | Chaque indicateur | `assert_no_look_ahead(fn, series, n_samples=30)` passe |
| `test_all_indicators_are_marked_safe` | Méta | Module `indicators.py` | Toute fonction exportée a `_look_ahead_safe = True` |
| `test_rsi_constant_series` | Edge case | `rsi()` | Série constante ne produit pas de division par zéro |
| `test_stoch_flat_series` | Edge case | `stoch()` | Série plate → retourne 50, pas NaN |
| `test_williams_r_flat_series` | Edge case | `williams_r()` | Série plate → retourne -50 |
| `test_sma_short_series` | Edge case | `sma()` | Série plus courte que période → tout NaN |
| `test_atr_wilder_vs_rolling` | Correctness | `atr()` | Vérifie que le premier ATR non-NaN correspond à `tr.rolling(period).mean()` puis diverge (Wilder) |
| `test_compute_all_indicators_shape` | Intégration | `compute_all_indicators()` | Même index que l'entrée, 20-22 colonnes |
| `test_compute_all_indicators_no_volume` | Intégration | `compute_all_indicators()` | DF sans Volume → 20 colonnes, pas d'erreur |
| `test_compute_all_indicators_column_names` | Intégration | `compute_all_indicators()` | Vérifie que les noms de colonnes sont exactement ceux attendus |
| `test_macd_returns_three_columns` | Unitaire | `macd()` | Sortie a 3 colonnes nommées |
| `test_adx_returns_three_columns` | Unitaire | `adx()` | Sortie a 3 colonnes nommées |
| `test_stoch_returns_two_columns` | Unitaire | `stoch()` | Sortie a 2 colonnes nommées |

**Fixture commune** : `_random_walk_ohlcv(n=500)` → DataFrame avec index datetime quotidien et colonnes `[Open, High, Low, Close, Volume]` généré par `np.random.randn().cumsum()`.

### 5.2 [`tests/unit/test_feature_research.py`](tests/unit/test_feature_research.py)

| Test | Type | Vérification |
|---|---|---|
| `test_rank_features_returns_dataframe` | Smoke | `isinstance(result, pd.DataFrame)` |
| `test_rank_features_sorted_by_composite_rank` | Contrat | `result["composite_rank"].is_monotonic_increasing` |
| `test_rank_features_no_duplicate_features` | Intégrité | `len(result["feature"].unique()) == len(result)` |
| `test_rank_features_scores_non_negative` | Validité | `mi ≥ 0`, `abs_corr ≥ 0`, `perm_imp ≥ 0` |
| `test_rank_features_json_output_exists` | I/O | Fichier `predictions/feature_research_TEST_D1.json` créé |
| `test_rank_features_json_valid` | I/O | `json.loads(path.read_text())` → liste de dicts avec les clés attendues |
| `test_rank_features_empty_train_range` | Edge case | `train_end="1990-01-01"` → `DataValidationError` levée |
| `test_rank_features_horizon_too_large` | Edge case | `horizon=9999` → `ValueError` levée |

**Mocking** : `unittest.mock.patch("app.features.research.load_asset")` pour injecter un DF synthétique. Le mock est appliqué dans une fixture `@pytest.fixture(autouse=True)` partagée.

### 5.3 Commande de validation

```bash
rtk pytest tests/unit/test_indicators.py tests/unit/test_feature_research.py -v
```

Objectif : ≥ 30 tests, 0 failures, 0 skips.

---

## 6. Matrice des risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Division par zéro (Stoch, Williams %R, Keltner, RSI) | Moyenne | Faible (NaN localisé) | Valeur sentinelle documentée + test edge case |
| Volume = 0 pour tous les actifs XTB | Élevée | Faible (2 features en moins) | Détection dans `compute_all_indicators` + warning |
| `dropna()` élimine > 50% des barres (actif jeune) | Faible | Moyen (ranking biaisé) | Warning si > 10%, erreur si > 50% |
| `mutual_info_regression` instable sur petit échantillon | Faible | Faible | `random_state=42` fixé, n_samples généralement > 1000 |
| `permutation_importance` lent sur 200 arbres × 10 repeats | Faible | Faible | `n_jobs=-1`, < 30s sur 5000 barres × 22 features |
| `load_asset` échoue (fichier absent) | Moyenne | Bloquant | Déjà géré par `@retry_with_backoff` + `DataValidationError` dans loader |
| Look-ahead dans un indicateur | Faible | Critique (ranking faussé) | Test `assert_no_look_ahead` paramétré sur CHAQUE indicateur |
| CSV à 6/7 colonnes (bug prompt 03) | Faible | Faible | Déjà résolu dans `loader.py` (détection adaptative `n_headers` vs `n_data`) |

---

## 7. Trade-offs architecturels

| Décision | Alternative rejetée | Justification |
|---|---|---|
| **pandas/numpy pur** | pandas-ta | Zéro dépendance lourde, contrôle total des formules, lisibilité |
| **Wilder smoothing (ewm)** | Rolling mean simple | Standard industriel pour ATR/ADX/RSI, cohérent avec littérature López de Prado |
| **RF 200 arbres, max_depth=4** | XGBoost/LightGBM | RF suffisant pour ranking features, pas besoin de gradient boosting ici |
| **Score composite = moyenne des rangs** | Moyenne pondérée des scores bruts | Robuste aux différences d'échelle (MI ∈ [0,∞[, corr ∈ [0,1], perm_imp ∈ [0,∞[) |
| **Pas de pickle du RF** | Sauvegarde du modèle | Le RF n'est qu'un outil de ranking, pas un modèle de trading réutilisable |
| **Colonnes MACD/ADX/Stoch aplaties** | Sortie multi-index ou tuples | Plus simple à consommer pour `research.py` et les prompts futurs |
| **`n_jobs=-1` partout** | `n_jobs=1` | Les machines modernes ont ≥ 4 cœurs, le parallélisme est gratuit |

---

## 8. Checklist de vérification (pour le passage en mode Code)

Avant de marquer le prompt 04 comme `✅ Terminé`, vérifier :

- [ ] Les 18 indicateurs sont dans [`indicators.py`](app/features/indicators.py), tous décorés `@look_ahead_safe`
- [ ] `compute_all_indicators()` gère Volume absent (warning, skip OBV/MFI)
- [ ] `rank_features()` utilise uniquement `df.loc[:train_end]` — pas de fuite OOS
- [ ] `rank_features()` lève une erreur claire si `len(df) == 0` après filtrage
- [ ] `rank_features()` lève une erreur si `target_horizon` trop grand
- [ ] JSON sauvegardé dans `predictions/feature_research_{ASSET}_{TF}.json` avec les 8 clés par feature
- [ ] Script CLI accepte `--asset`, `--tf`, `--horizon`, `--n-top`, `--train-end`
- [ ] `rtk pytest tests/unit/test_indicators.py tests/unit/test_feature_research.py -v` → ≥ 30 tests, 0 failures
- [ ] Aucune modification de fichier hors scope (loader.py, look_ahead_validator.py, etc.)
- [ ] `JOURNAL.md` mis à jour avec l'entrée standard (règle 8 constitution)

---

## 9. Notes pour les prompts futurs

- **Prompt 12 (H11 — features avancées)** : utilisera le top-N de ce ranking comme baseline et ajoutera des features macro-techniques (corrélation, ratio, divergence). Le fichier `feature_research_*.json` est l'input.
- **Prompt 13 (H12 — session features)** : idem, ajoutera des features de session (heure de la journée, jour de la semaine) sur le top-N.
- **Ne pas modifier le schéma JSON** sans mettre à jour les prompts 12 et 13.
