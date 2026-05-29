# Architecture — Prompt 06 : Framework de validation unifié

> Document compagnon de [`prompts/06_validation_framework.md`](06_validation_framework.md).
> Ce plan est la référence avant de passer en mode Code pour l'implémentation.

---

## 1. Audit critique du code v2 existant vs. Prompt 06

Le module [`app/analysis/edge_validation.py`](../app/analysis/edge_validation.py) (402 lignes) et [`app/analysis/cpcv.py`](../app/analysis/cpcv.py) (592 lignes) existent déjà. Voici l'analyse de conformité avec le contrat du Prompt 06.

| # | Fonction requise (Prompt 06) | Existant v2 | Compatible ? | Action |
|---|---|---|---|---|
| 1 | `sharpe_ratio(returns, freq=252) -> float` | `_compute_sharpe_from_returns(returns: np.ndarray) -> float` | ❌ Non annualisé | **Réécrire** — wrapper annualisé + gestion `pd.Series` |
| 2 | `sortino_ratio(returns, freq=252) -> float` | Absent | ❌ | **Créer** — downside deviation uniquement |
| 3 | `max_drawdown(equity: pd.Series) -> float` | Absent (calculé inline) | ❌ | **Créer** — `(equity.cummax() - equity) / equity.cummax()` |
| 4 | `bootstrap_sharpe(returns, n_iter, seed) -> tuple[float, float]` | Inline dans [`validate_edge()`](../app/analysis/edge_validation.py:148) | 🟡 Logique correcte, non extraite | **Extraire** + adapter signature |
| 5 | `deflated_sharpe(sr, n_trials, n_obs, skew, kurtosis) -> tuple[float, float]` | Formule différente dans v2 (lines 158-168) | 🔴 **Formule incompatible** | **Réécrire** selon Bailey & López de Prado (2014) — formule exacte du Prompt 06 |
| 6 | `probabilistic_sharpe(sr, n_obs, skew, kurtosis, sr_benchmark) -> float` | `psr_from_returns(returns, sr_benchmark) -> float` (line 212) | 🟡 Logique correcte, API différente | **Adapter** — prendre SR en paramètre plutôt que returns bruts |
| 7 | `purged_kfold_cv(df, k=5, embargo_pct=0.01)` | `generate_cpcv_splits()` dans [`cpcv.py`](../app/analysis/cpcv.py:43) | 🟡 Sur-ingénieré (CPCV combinatoire), imports `learning_machine_learning.*` | **Créer version simplifiée** — k-fold purgé (López de Prado ch.7), pas CPCV |
| 8 | `walk_forward_split(df, train_months, val_months, step_months)` | `run_walk_forward()` dans [`walk_forward.py`](../app/backtest/walk_forward.py:97) | ❌ C'est un moteur complet, pas un générateur de splits | **Créer** — générateur pur de `(train_idx, val_idx)` |
| 9 | `validate_edge(equity, trades, n_trials) -> EdgeReport` | `validate_edge(trades_df, backtest_cfg, ...) -> dict` | 🔴 **API radicalement différente** | **Réécrire** — basé sur equity curve + critères constitution |
| 10 | `EdgeReport` dataclass | Absent | ❌ | **Créer** |

### Problèmes transverses identifiés

| # | Problème | Gravité | Impact |
|---|---|---|---|
| A | **Imports `learning_machine_learning.*`** dans [`edge_validation.py:21-22`](../app/analysis/edge_validation.py:21) et [`cpcv.py:20-31`](../app/analysis/cpcv.py:20) | 🔴 Bloquant | Le code ne peut pas s'exécuter sans renommer ces imports en `app.*` |
| B | **v2 calcule le Sharpe sur PnL/trade** — [`_compute_sharpe_from_returns`](../app/analysis/edge_validation.py:27) est appelé avec `pnl` (vecteur de PnL par trade), pas des retours quotidiens | 🔴 Constitution Règle 10 | Toute l'infrastructure v2 utilise cette convention erronée. La nouvelle API `validate_edge(equity, ...)` force l'usage de `equity.pct_change()` |
| C | **`BacktestConfig` couplé à `validate_edge`** — la v2 exige `tp_pips`, `sl_pips`, `commission_pips` pour calculer le breakeven WR | 🟡 Architectural | La nouvelle API n'a pas besoin de BacktestConfig. Le breakeven WR n'est plus calculé (remplacé par WR direct > 30%) |
| D | **CPCV v2 trop couplé** — [`cpcv.py`](../app/analysis/cpcv.py) dépend de `backtest.filters`, `backtest.simulator`, `backtest.sizing`, `config.backtest`, `config.instruments` | 🟡 Maintenance | On garde `cpcv.py` tel quel (utilisé par les prompts futurs). Seul `purged_kfold_cv` est une fonction neuve et indépendante |

### Verdict

Le code v2 contient **la logique métier correcte** pour `psr_from_returns` et le bootstrap, mais :
- L'API est incompatible avec le contrat Prompt 06 (equity-based vs trades+config-based)
- La formule DSR est **fondamentalement différente**
- Les imports sont cassés

**Stratégie** : réécrire [`edge_validation.py`](../app/analysis/edge_validation.py) en conservant les helpers internes utiles (`_compute_sharpe_from_returns` → `_annualized_sharpe`, `psr_from_returns` → `probabilistic_sharpe`), et créer les nouvelles fonctions indépendantes. [`cpcv.py`](../app/analysis/cpcv.py) reste intact.

---

## 2. Diagramme des composants

```
┌──────────────────────────────────────────────────────────────────┐
│              app/analysis/edge_validation.py  (REWRITE)           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Fonctions de base (stateless) ────────────────────────────┐ │
│  │  sharpe_ratio(returns, freq) → float                        │ │
│  │  sortino_ratio(returns, freq) → float                       │ │
│  │  max_drawdown(equity) → float  (en %)                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Tests statistiques avancés ───────────────────────────────┐ │
│  │  bootstrap_sharpe(returns, n_iter, seed) → (float, float)   │ │
│  │  deflated_sharpe(sr, n_trials, n_obs, skew, kurt) → (f,f)  │ │
│  │  probabilistic_sharpe(sr, n_obs, skew, kurt, bench) → float │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Générateurs de splits temporels ──────────────────────────┐ │
│  │  purged_kfold_cv(df, k, embargo_pct) → Iterator[(idx,idx)]  │ │
│  │  walk_forward_split(df, train_m, val_m, step_m) → Iterator  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Validation combinée ──────────────────────────────────────┐ │
│  │  EdgeReport (dataclass): go, reasons, metrics                │ │
│  │  validate_edge(equity, trades, n_trials) → EdgeReport        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  [CONSERVÉES telles quelles]                                     │
│  deflated_sharpe_ratio_from_distribution(...) → dict             │
│  validate_edge_distribution(...) → dict                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
┌──────────────────┐    ┌──────────────────────────────────────┐
│ app/analysis/    │    │  app/testing/look_ahead_validator.py │
│ cpcv.py          │    │  (existant, NON modifié)              │
│ (existant,       │    │                                      │
│  NON modifié)    │    │  assert_no_look_ahead(fn, series)     │
│                  │    │  @look_ahead_safe                     │
└──────────────────┘    └──────────────┬───────────────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────────────────┐
                          │  tests/unit/                         │
                          │  ├── test_edge_validation.py (REWRITE)│
                          │  ├── test_walk_forward.py (CREATE)   │
                          │  └── test_indicators_look_ahead.py   │
                          │      (CREATE)                        │
                          └──────────────────────────────────────┘
```

**Fichiers à créer (3)** :

| Fichier | Rôle |
|---|---|
| `tests/unit/test_indicators_look_ahead.py` | Scan de tous les modules `app/features/*.py`, vérification `_look_ahead_safe` + `assert_no_look_ahead` |
| `tests/unit/test_walk_forward.py` | Tests de non-chevauchement des splits walk-forward + embargo |
| `prompts/06_architecture_plan.md` | Ce document |

**Fichiers à réécrire (2)** :

| Fichier | Changement |
|---|---|
| [`app/analysis/edge_validation.py`](../app/analysis/edge_validation.py) | Refonte complète : nouvelles signatures, EdgeReport, imports `app.*` |
| [`tests/unit/test_edge_validation.py`](../tests/unit/test_edge_validation.py) | Tests alignés sur la nouvelle API, ≥ 15 tests dont 5 dégénérés |

**Fichiers conservés sans modification** :
- [`app/analysis/cpcv.py`](../app/analysis/cpcv.py) — utilisé par les prompts 07-19
- [`app/backtest/cpcv.py`](../app/backtest/cpcv.py) — idem
- [`app/testing/look_ahead_validator.py`](../app/testing/look_ahead_validator.py) — idem
- [`app/features/indicators.py`](../app/features/indicators.py) — vérifié (doit déjà avoir `@look_ahead_safe`)
- [`app/features/economic.py`](../app/features/economic.py) — vérifié (doit déjà avoir `@look_ahead_safe`)

---

## 3. Contrats détaillés des fonctions

### 3.1 `sharpe_ratio`

```python
def sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Sharpe ratio annualisé sur retours quotidiens.

    Préconditions :
        - returns est une pd.Series de retours (pas de PnL)
        - freq est le facteur d'annualisation (252 = trading days)

    Edge cases :
        - std(returns) == 0 → retourne 0.0
        - len(returns) < 2 → retourne 0.0
        - returns contient des NaN → dropna() avant calcul

    Formule : mean(returns) / std(returns, ddof=1) * sqrt(freq)
    """
```

### 3.2 `sortino_ratio`

```python
def sortino_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Sortino ratio — pénalise uniquement la volatilité baissière.

    downside_std = std(returns[returns < 0]) si au moins 1 return négatif.
    Retourne 0.0 si aucun return négatif ou len < 2.
    """
```

### 3.3 `max_drawdown`

```python
def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown en pourcentage (valeur positive, ex: 0.15 = 15%).

    Formule : max((cummax - equity) / cummax).
    Edge cases : equity vide → 0.0, equity constant → 0.0.
    Retourne une valeur dans [0, 1].
    """
```

### 3.4 `bootstrap_sharpe`

```python
def bootstrap_sharpe(
    returns: pd.Series,
    n_iter: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap du Sharpe ratio.

    Returns:
        (sharpe_moyen_bootstrap, p_value_sharpe_gt_0)
        p_value = proportion des bootstraps où Sharpe ≤ 0.
    """
```

### 3.5 `deflated_sharpe` (formule Prompt 06)

```python
def deflated_sharpe(
    sr: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Args:
        sr: Sharpe ratio observé (annualisé)
        n_trials: Nombre de stratégies testées (n_trials_cumul)
        n_obs: Nombre d'observations (retours quotidiens, pas trades)
        skew: Skewness des retours (scipy.stats.skew, bias=False)
        kurtosis: Kurtosis EXCESS (scipy.stats.kurtosis, bias=False, fisher=True) + 3 → raw

    Returns:
        (dsr_z, p_value) où p_value = 1 - Φ(dsr_z)

    Formule exacte du Prompt 06 :
        euler = 0.5772156649
        sr_zero = sqrt((1-euler)*Φ⁻¹(1-1/N) + euler*Φ⁻¹(1-1/(N*e)))
        numerator = (sr - sr_zero) * sqrt(n_obs - 1)
        denominator = sqrt(1 - skew*sr + (kurtosis-1)/4 * sr²)
        dsr_z = numerator / denominator

    Edge cases :
        - n_trials < 1 → (NaN, NaN)
        - denominator <= 0 → (NaN, NaN)
        - n_obs < 30 → (NaN, NaN) + logger.warning
    """
```

### 3.6 `probabilistic_sharpe`

```python
def probabilistic_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    PSR(SR*) = Φ( (ŜR − SR*) · √(n−1) / √(1 − skew·ŜR + (kurt−1)/4 · ŜR²) )

    Args:
        sr: Sharpe observé
        sr_benchmark: Sharpe de référence (0 = H₀: SR ≤ 0)

    Returns:
        Probabilité P(SR_vrai > SR_benchmark). NaN si n < 2 ou dénominateur ≤ 0.
    """
```

### 3.7 `purged_kfold_cv`

```python
def purged_kfold_cv(
    df: pd.DataFrame,
    k: int = 5,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purged K-Fold Cross-Validation (López de Prado ch.7).

    Découpe l'index chronologique en k folds. Pour chaque fold i :
    - test = fold i
    - train = tous les folds < i (purgés de embargo_pct % avant test)
    - Les folds > i sont exclus (pas de look-ahead)

    Args:
        df: DataFrame indexé par DatetimeIndex trié.
        k: Nombre de folds.
        embargo_pct: Fraction de l'index à purger avant chaque test (ex: 0.01 = 1%).

    Yields:
        (train_indices, test_indices) — positions entières.

    Raises:
        ValueError: si embargo_pct == 0 (doit être > 0).
        ValueError: si k < 2.

    Invariants :
        - max(train) + embargo < min(test) quand train précède test
        - Aucun chevauchement train/test
    """
```

### 3.8 `walk_forward_split`

```python
def walk_forward_split(
    df: pd.DataFrame,
    train_months: int = 36,
    val_months: int = 6,
    step_months: int = 6,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Générateur de splits walk-forward ancrés (expanding window).

    Pour chaque fenêtre :
    - train = [début, train_end] (taille fixe glissante OU expanding)
    - val = (train_end, train_end + val_months]

    Choix : expanding window (train grossit à chaque step).
    Le train initial = train_months de données.
    La validation avance par pas de step_months.

    Yields:
        (train_indices, val_indices).

    Invariants :
        - train_end < val_start (pas de chevauchement)
        - val_end_monotonic (les fenêtres val ne se chevauchent pas)
    """
```

### 3.9 `EdgeReport` + `validate_edge`

```python
@dataclass
class EdgeReport:
    go: bool
    reasons: list[str]
    metrics: dict


def validate_edge(
    equity: pd.Series,
    trades: pd.DataFrame,
    n_trials: int,
) -> EdgeReport:
    """Validation complète selon les 5 critères de la constitution.

    Args:
        equity: Courbe d'equity (pd.Series, index=datetime).
        trades: DataFrame avec colonne obligatoire 'pnl' (PnL par trade).
        n_trials: Compteur n_trials_cumul du JOURNAL.md.

    Returns:
        EdgeReport avec go=True si TOUS les critères passent.

    5 critères (constitution §2) :
        1. Sharpe walk-forward ≥ 1.0      (sur daily_returns)
        2. DSR > 0 ET p < 0.05            (deflated_sharpe)
        3. Max DD < 15 %                  (max_drawdown)
        4. WR > 30 %                      ((trades['pnl'] > 0).mean())
        5. Trades/an ≥ 30                 (len(trades) / n_years)

    Sharpe calculé sur pct_change().dropna() de l'equity — Règle 10.
    """
```

---

## 4. Stratégie de migration [`edge_validation.py`](../app/analysis/edge_validation.py)

### 4.1 Fonctions conservées (adaptées)

| Fonction v2 | Devient | Changement |
|---|---|---|
| `_compute_sharpe_from_returns` | Intégré dans `sharpe_ratio` | Ajout `sqrt(freq)`, accepte `pd.Series` |
| `_expected_max_sr` | Supprimée (plus utilisée) | La formule Prompt 06 ne l'utilise pas |
| `_var_max_sr` | Supprimée (plus utilisée) | Idem |
| `psr_from_returns` | `probabilistic_sharpe` | Prend SR en paramètre, pas les returns bruts |
| `deflated_sharpe_ratio_from_distribution` | **Conservée telle quelle** | Utilisée par prompts futurs (CPCV) |
| `validate_edge_distribution` | **Conservée telle quelle** | Wrapper combiné, toujours utile |

### 4.2 Fonctions supprimées

- `validate_edge()` v2 — remplacée par la nouvelle API `(equity, trades, n_trials) -> EdgeReport`
- `_expected_max_sr()` — plus utilisée
- `_var_max_sr()` — plus utilisée

### 4.3 Imports à corriger

```python
# AVANT (v2)
from learning_machine_learning.config.backtest import BacktestConfig
from learning_machine_learning.core.logging import get_logger

# APRÈS (v3)
from app.core.logging import get_logger
# BacktestConfig n'est plus nécessaire
```

---

## 5. Architecture [`tests/unit/test_indicators_look_ahead.py`](../tests/unit/test_indicators_look_ahead.py)

### 5.1 Stratégie de scan

```python
import importlib
import inspect
from pathlib import Path

def _discover_feature_modules() -> list[str]:
    """Scanne app/features/ pour tous les .py sauf __init__."""
    features_dir = Path("app/features")
    modules = []
    for f in sorted(features_dir.glob("*.py")):
        if f.stem.startswith("_"):
            continue
        modules.append(f"app.features.{f.stem}")
    return modules

def _public_functions(module_name: str) -> list[Callable]:
    """Retourne toutes les fonctions publiques d'un module."""
    mod = importlib.import_module(module_name)
    return [
        fn for name, fn in inspect.getmembers(mod, inspect.isfunction)
        if not name.startswith("_")
    ]
```

### 5.2 Tests paramétrés

| Test | Cible | Vérification |
|---|---|---|
| `test_module_X_has_no_private_fns_without_decorator` | Chaque module | Aucune fonction privée exposée sans `@look_ahead_safe` |
| `test_all_public_fns_are_marked_safe` | Toutes les `_public_functions()` | `getattr(fn, "_look_ahead_safe", False) is True` |
| `test_no_look_ahead_{module}_{fn_name}` | Chaque fonction publique | `assert_no_look_ahead(lambda x: fn(x), series, n_samples=50)` |

### 5.3 Stratégie d'adaptation de signature

Les fonctions de features ont des signatures hétérogènes :
- `sma(close, period)` — uni-série
- `atr(high, low, close, period)` — multi-séries
- `compute_event_features(price_index, calendar)` — index + DataFrame

Pour le test `assert_no_look_ahead`, on doit adapter :
1. Si la fonction prend un seul paramètre `pd.Series` → appel direct
2. Si elle prend `pd.DataFrame` → on passe un DF avec toutes les colonnes OHLCV
3. Si elle prend un index → on passe un DatetimeIndex

**Heuristique** : on inspecte les noms des paramètres pour deviner le type attendu.

```python
def _build_test_input(fn: Callable) -> pd.Series | pd.DataFrame:
    """Construit un input synthétique adapté à la signature de fn."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    n = 500

    rng = np.random.default_rng(42)
    close = pd.Series(rng.randn(n).cumsum() + 100,
                      index=pd.date_range("2020-01-01", periods=n, freq="D"))

    if len(params) >= 3 and "high" in params and "low" in params:
        return pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + abs(rng.randn(n)),
            "low": close - abs(rng.randn(n)),
            "close": close,
            "volume": np.ones(n),
        }, index=close.index)

    return close
```

---

## 6. Matrice de tests

### 6.1 [`tests/unit/test_edge_validation.py`](../tests/unit/test_edge_validation.py) — ≥ 15 tests

| # | Test | Catégorie | Vérification |
|---|---|---|---|
| 1 | `test_sharpe_ratio_positive` | Nominal | Retours gaussiens μ>0 → Sharpe > 0 |
| 2 | `test_sharpe_ratio_zero_mean` | Nominal | μ=0 → Sharpe ≈ 0 |
| 3 | `test_sharpe_ratio_std_zero` | Dégénéré | Retours constants → Sharpe=0 |
| 4 | `test_sharpe_ratio_single_obs` | Dégénéré | 1 seul return → Sharpe=0 |
| 5 | `test_sharpe_ratio_with_nan` | Edge | NaN internes → dropna, résultat cohérent |
| 6 | `test_sortino_positive` | Nominal | Sortino > Sharpe (pas de pénalité haussière) |
| 7 | `test_sortino_all_positive` | Edge | Tous retours > 0 → Sortino = +∞ → retourne 0.0 |
| 8 | `test_max_drawdown_normal` | Nominal | Série avec creux connu → DD% exact |
| 9 | `test_max_drawdown_monotonic` | Edge | Série croissante → DD=0 |
| 10 | `test_max_drawdown_empty` | Dégénéré | Série vide → 0.0 |
| 11 | `test_bootstrap_distribution` | Nominal | Sur série gaussienne, vérifier CI contient le vrai Sharpe |
| 12 | `test_deflated_sharpe_more_trials_reduces_dsr` | Nominal | n_trials=5 vs 50 → DSR diminue |
| 13 | `test_deflated_sharpe_returns_nan_for_n_trials_zero` | Dégénéré | n_trials=0 → (NaN, NaN) |
| 14 | `test_deflated_sharpe_returns_nan_for_short_series` | Dégénéré | n_obs=10 → (NaN, NaN) + warning |
| 15 | `test_probabilistic_sharpe_strong_edge` | Nominal | SR=2, n=1000 → PSR proche de 1.0 |
| 16 | `test_validate_edge_all_pass` | Intégration | Equity synthétique idéale → go=True |
| 17 | `test_validate_edge_equity_plate` | Dégénéré | Equity constante → go=False, raison "Sharpe" |
| 18 | `test_validate_edge_one_trade` | Dégénéré | 1 seul trade → go=False, raison "Trades/an" |
| 19 | `test_validate_edge_high_drawdown` | Dégénéré | DD > 15% → go=False |
| 20 | `test_validate_edge_low_wr` | Dégénéré | WR < 30% → go=False |
| 21 | `test_purged_kfold_embargo_zero_raises` | Dégénéré | embargo_pct=0 → ValueError |
| 22 | `test_purged_kfold_no_overlap` | Invariant | Intersection train/test = ∅ pour tous les folds |
| 23 | `test_purged_kfold_chronological` | Invariant | max(train) < min(test) pour chaque fold |

### 6.2 [`tests/unit/test_walk_forward.py`](../tests/unit/test_walk_forward.py) — ≥ 5 tests

| # | Test | Vérification |
|---|---|---|
| 1 | `test_splits_no_overlap` | Aucun chevauchement train/val |
| 2 | `test_splits_chronological` | train_end < val_start pour chaque split |
| 3 | `test_splits_val_monotonic` | Les fenêtres val ne se chevauchent pas entre elles |
| 4 | `test_splits_expanding_window` | La taille du train augmente à chaque step |
| 5 | `test_splits_covers_full_range` | L'union des val couvre tout l'espace post-train initial |
| 6 | `test_too_few_bars_returns_empty` | Série trop courte → générateur vide (0 split) |

### 6.3 [`tests/unit/test_indicators_look_ahead.py`](../tests/unit/test_indicators_look_ahead.py)

| # | Test | Vérification |
|---|---|---|
| 1 | `test_all_feature_modules_exist` | Au moins `indicators.py` et `economic.py` |
| 2 | `test_all_public_fns_marked_safe[{module}]` | Paramétré par module — toutes les fns ont `_look_ahead_safe=True` |
| 3 | `test_no_look_ahead[{module}_{fn}]` | Paramétré par fonction — `assert_no_look_ahead` passe |

---

## 7. Matrice des risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| `@look_ahead_safe` absent sur certaines fonctions de features | Élevée (oubli humain) | 🔴 Critique | Test `test_all_public_fns_marked_safe` échoue → CI rouge → correction obligatoire |
| Formule DSR incorrecte (confusion entre formules v2 et Prompt 06) | Faible | 🔴 Critique | La formule est copiée exactement du Prompt 06. Test avec valeurs connues (golden master) |
| `kurtosis` : excess vs raw confusion | Moyenne | 🟡 DSR faussé | `deflated_sharpe` documente que `kurtosis` = raw (excess + 3). Test golden master avec valeurs pré-calculées |
| `validate_edge` calcule Sharpe sur PnL/trade (régression v2) | Faible | 🔴 Critique | La nouvelle API prend `equity: pd.Series`. Le Sharpe est forcément sur `equity.pct_change()`. Test vérifie que le Sharpe n'est pas NaN pour equity monotone. |
| `embargo_pct=0` non détecté | Faible | 🟡 Chevauchement train/test | `purged_kfold_cv` lève `ValueError` si embargo_pct == 0. Test dédié. |
| Module `app/features/regime.py` ou `advanced.py` créé plus tard sans `@look_ahead_safe` | Élevée | 🟡 Détecté au prochain `make test` | Le scan est dynamique — tout nouveau module `.py` dans `app/features/` est automatiquement testé |
| Fonctions avec signature complexe (`compute_event_features`) non testables par `assert_no_look_ahead` | Moyenne | 🟡 Faux négatif | Heuristique de détection de signature + fallback manuel dans le test |
| `cpcv.py` conservé avec imports `learning_machine_learning.*` | Élevée | 🟡 Bloque l'exécution si appelé | Hors scope du Prompt 06. Sera corrigé au prompt 07 (premier à utiliser CPCV). Documenté dans les risques. |

---

## 8. Trade-offs architecturels

| Décision | Alternative rejetée | Justification |
|---|---|---|
| **Nouvelle API `validate_edge(equity, trades, n_trials)`** | Conserver l'API v2 `validate_edge(trades_df, backtest_cfg, ...)` | L'API v2 exige `BacktestConfig` et calcule le breakeven WR. La nouvelle API est découplée du backtest et alignée sur les 5 critères constitution. |
| **`deflated_sharpe` prend SR, skew, kurtosis en paramètres** | Calculer SR/skew/kurtosis en interne | Séparation des responsabilités : le caller décide de la source des moments statistiques (bootstrap, equity curve, CPCV) |
| **`purged_kfold_cv` simplifié (pas CPCV)** | Réutiliser `generate_cpcv_splits` | CPCV = combinatoire (C(N,k_test)), lourd pour un simple k-fold. Le k-fold purgé standard suffit pour la validation. CPCV reste disponible pour les prompts 07+. |
| **Walk-forward expanding window** | Rolling window (taille fixe) | Expanding window = plus de données d'entraînement au fil du temps, standard industriel pour le trading |
| **Scan dynamique des modules de features** | Liste statique de modules | Si un développeur ajoute `app/features/advanced.py` au prompt 12, le test l'attrape automatiquement sans modification |
| **`EdgeReport` dataclass plutôt que dict** | Dict comme en v2 | Typage strict, IDE-friendly, accès attribut sans risque de KeyError |
| **Conserver `deflated_sharpe_ratio_from_distribution` et `validate_edge_distribution`** | Tout supprimer et recréer | Ces fonctions sont utilisées par les prompts futurs (CPCV). Aucune raison de les réécrire. |
| **Tests 100% synthétiques (`tmp_path`)** | Données réelles | Contrôle total des distributions, reproductible, pas de dépendance aux fichiers CSV |

---

## 9. Impact sur les fichiers existants

| Fichier | Action | Justification |
|---|---|---|
| [`app/analysis/edge_validation.py`](../app/analysis/edge_validation.py) | **REWRITE** | API incompatible avec Prompt 06 |
| [`tests/unit/test_edge_validation.py`](../tests/unit/test_edge_validation.py) | **REWRITE** | Tests alignés sur nouvelle API |
| [`app/analysis/cpcv.py`](../app/analysis/cpcv.py) | **AUCUN** | Hors scope, toujours importé par prompts futurs |
| [`app/backtest/cpcv.py`](../app/backtest/cpcv.py) | **AUCUN** | Idem |
| [`app/testing/look_ahead_validator.py`](../app/testing/look_ahead_validator.py) | **AUCUN** | Déjà conforme |
| [`app/features/indicators.py`](../app/features/indicators.py) | **VÉRIFICATION** | S'assurer que chaque fonction a `@look_ahead_safe` |
| [`app/features/economic.py`](../app/features/economic.py) | **VÉRIFICATION** | Idem |
| [`app/features/research.py`](../app/features/research.py) | **AUCUN** | `rank_features` n'est pas un indicateur — pas besoin de `@look_ahead_safe` |

---

## 10. Checklist de vérification (pour le passage en mode Code)

Avant de marquer le prompt 06 comme `✅ Terminé`, vérifier :

- [ ] Les 9 fonctions publiques sont dans [`edge_validation.py`](../app/analysis/edge_validation.py)
- [ ] `validate_edge` utilise `equity.pct_change().dropna()` pour le Sharpe (Règle 10)
- [ ] `deflated_sharpe` implémente exactement la formule du Prompt 06
- [ ] `purged_kfold_cv` lève `ValueError` si `embargo_pct == 0`
- [ ] `walk_forward_split` produit des splits sans chevauchement
- [ ] Aucun import `learning_machine_learning.*` dans les fichiers modifiés
- [ ] `EdgeReport` est une `@dataclass` avec `go: bool`, `reasons: list[str]`, `metrics: dict`
- [ ] `tests/unit/test_indicators_look_ahead.py` scanne tous les modules `app/features/*.py`
- [ ] `tests/unit/test_edge_validation.py` contient ≥ 15 tests, dont les 5 cas dégénérés
- [ ] `tests/unit/test_walk_forward.py` contient ≥ 5 tests
- [ ] `rtk make verify` passe (ruff + mypy + pytest + snooping_check)
- [ ] `JOURNAL.md` mis à jour avec l'entrée standard (Règle 8 constitution)
- [ ] Aucune modification de fichier hors scope

---

## 11. Notes pour les prompts futurs

- **Prompt 07 (H06)** : utilisera `validate_edge` pour valider les résultats Donchian multi-actif. `purged_kfold_cv` et `walk_forward_split` seront les générateurs de splits.
- **Prompt 18 (Validation finale)** : utilisera `n_trials_cumul` du `JOURNAL.md` comme argument `n_trials` de `deflated_sharpe`.
- **Ne pas modifier `EdgeReport`** sans mettre à jour les prompts 07-19 qui l'utiliseront comme contrat.
- **Les fonctions `deflated_sharpe_ratio_from_distribution` et `validate_edge_distribution`** sont conservées pour compatibilité avec CPCV. Elles seront utilisées au prompt 09 (H08).

---

**Fin du plan d'architecture.** Prêt pour implémentation en mode Code.
