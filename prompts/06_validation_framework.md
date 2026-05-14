# Prompt 06 — Framework de validation unifié

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/04_features_research_harness.md`
4. `prompts/05_economic_calendar.md`

## Objectif
Centraliser dans `app/analysis/edge_validation.py` toutes les méthodes de validation statistique (Sharpe corrigé, DSR, PSR, CPCV, bootstrap Sharpe, walk-forward) et fournir une fonction unique `validate_edge(equity_curve, n_trials) -> EdgeReport` qui retourne le verdict GO/NO-GO selon les 5 critères de la constitution.

## Definition of Done (testable)
- [ ] `app/analysis/edge_validation.py` contient :
  - `sharpe_ratio(returns: pd.Series, freq: int = 252) -> float`
  - `sortino_ratio(returns: pd.Series, freq: int = 252) -> float`
  - `max_drawdown(equity: pd.Series) -> float` (en %)
  - `bootstrap_sharpe(returns: pd.Series, n_iter: int = 10_000, seed: int = 42) -> tuple[float, float]` (Sharpe moyen, p(Sharpe > 0))
  - `deflated_sharpe(sr: float, n_trials: int, n_obs: int, skew: float, kurtosis: float) -> tuple[float, float]` (DSR, p-value)
  - `probabilistic_sharpe(sr: float, n_obs: int, skew: float, kurtosis: float, sr_benchmark: float = 0.0) -> float`
  - `purged_kfold_cv(df, k=5, embargo_pct=0.01)` : générateur de splits CPCV
  - `walk_forward_split(df, train_months=36, val_months=6, step_months=6)` : générateur de splits walk-forward
- [ ] `validate_edge(equity: pd.Series, trades: pd.DataFrame, n_trials: int) -> EdgeReport` retourne un dataclass `EdgeReport(go: bool, reasons: list[str], metrics: dict)` qui vérifie :
  1. Sharpe ≥ 1.0
  2. DSR > 0 ET p < 0.05
  3. Max DD < 15 %
  4. WR > 30 %
  5. Trades/an ≥ 30
- [ ] **Anti-look-ahead obligatoire** : tout indicateur de `app/features/indicators.py` (prompt 04) ET toute fonction de `app/features/economic.py` / `app/features/regime.py` / `app/features/advanced.py` doit être décorée par `@look_ahead_safe` de [app/testing/look_ahead_validator.py](../app/testing/look_ahead_validator.py).
- [ ] `tests/unit/test_indicators_look_ahead.py` : un test par indicateur via `assert_no_look_ahead(fn, synthetic_series, n_samples=100)`.
- [ ] `tests/unit/test_edge_validation.py` : ≥ 15 tests, dont **5 cas dégénérés** : equity plate (Sharpe=0), returns constants (DSR=NaN), 1 seul trade (assertions explicites), embargo=0 dans CPCV (doit lever ValueError), série très courte (< warmup).
- [ ] `tests/unit/test_walk_forward.py` : test que les splits ne se chevauchent pas et respectent l'embargo.
- [ ] `rtk make verify` passe.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS implémenter manuellement des choses qui existent dans v2 sans réutiliser. Réutiliser le code de `app/analysis/edge_validation.py` (renommé depuis v2) et de `app/backtest/cpcv.py` si présents.
- Ne PAS oublier l'embargo entre train et test (par défaut 1 % de la longueur).
- Ne PAS calculer le Sharpe sur PnL/trade (cf. Règle 10 constitution).
- Ne PAS tester avec des données réelles (uniquement synthétiques en `tmp_path`).

## Étapes

### Étape 1 — Réutiliser le code v2
Localiser dans `app/` les modules existants depuis v2 :
```bash
rtk grep -rn "deflated_sharpe\|bootstrap_sharpe\|CPCV\|cpcv" app/ --include="*.py"
```
Si déjà présents : refactoriser pour qu'ils soient tous dans `app/analysis/edge_validation.py`. Sinon : implémenter.

### Étape 2 — Implémentation DSR
Référence : Bailey & López de Prado (2014).
```python
import numpy as np
from scipy.stats import norm


def deflated_sharpe(
    sr: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio. Retourne (DSR, p-value)."""
    if n_trials < 1:
        return float("nan"), float("nan")
    euler_mascheroni = 0.5772156649
    sr_zero = np.sqrt(
        (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1 / (n_trials * np.e))
    )
    numerator = (sr - sr_zero) * np.sqrt(n_obs - 1)
    denominator = np.sqrt(1 - skew * sr + (kurtosis - 1) / 4 * sr**2)
    if denominator <= 0:
        return float("nan"), float("nan")
    dsr_z = numerator / denominator
    p_value = 1 - norm.cdf(dsr_z)
    return dsr_z, p_value
```

### Étape 3 — `validate_edge`
```python
from dataclasses import dataclass

@dataclass
class EdgeReport:
    go: bool
    reasons: list[str]
    metrics: dict


def validate_edge(equity: pd.Series, trades: pd.DataFrame, n_trials: int) -> EdgeReport:
    daily_returns = equity.pct_change().dropna()

    sr = sharpe_ratio(daily_returns)
    dd = max_drawdown(equity)
    wr = (trades["pnl"] > 0).mean()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    trades_per_year = len(trades) / n_years if n_years > 0 else 0

    dsr, p = deflated_sharpe(
        sr,
        n_trials=n_trials,
        n_obs=len(daily_returns),
        skew=daily_returns.skew(),
        kurtosis=daily_returns.kurtosis() + 3,  # excess → raw
    )

    reasons = []
    if sr < 1.0:
        reasons.append(f"Sharpe {sr:.2f} < 1.0")
    if not (dsr > 0 and p < 0.05):
        reasons.append(f"DSR={dsr:.2f} (p={p:.3f}) non significatif")
    if dd >= 0.15:
        reasons.append(f"Max DD {dd:.1%} ≥ 15%")
    if wr <= 0.30:
        reasons.append(f"WR {wr:.1%} ≤ 30%")
    if trades_per_year < 30:
        reasons.append(f"Trades/an {trades_per_year:.1f} < 30")

    return EdgeReport(
        go=len(reasons) == 0,
        reasons=reasons,
        metrics={
            "sharpe": sr,
            "dsr": dsr,
            "p_value": p,
            "max_dd": dd,
            "wr": wr,
            "trades_per_year": trades_per_year,
            "n_trades": len(trades),
        },
    )
```

### Étape 4 — Tests
- DSR : sur série gaussienne connue, vérifier que la p-value est proche de la valeur théorique.
- `validate_edge` : créer un equity curve synthétique avec Sharpe = 2, DD = 5%, WR = 50%, 100 trades sur 1 an → `go=True`.
- Cas dégénérés : equity plate → Sharpe = 0, `go=False`.

### Étape 5 — Harness anti-look-ahead
**Le harness est déjà créé au prompt 02b** dans `app/testing/look_ahead_validator.py`. Ce prompt :
1. Vérifie son existence (`from app.testing.look_ahead_validator import assert_no_look_ahead, look_ahead_safe`).
2. Crée `tests/unit/test_indicators_look_ahead.py` qui boucle sur tous les indicateurs de `app/features/indicators.py` et vérifie qu'ils ont l'attribut `_look_ahead_safe` (sinon → fail).
3. Pour chaque indicateur, appelle `assert_no_look_ahead(fn, synthetic_series, n_samples=100)`.

```python
# tests/unit/test_indicators_look_ahead.py
import inspect
import numpy as np
import pandas as pd
import pytest

from app import features
from app.testing.look_ahead_validator import assert_no_look_ahead

def _all_indicator_fns():
    return [
        fn for name, fn in inspect.getmembers(features.indicators, inspect.isfunction)
        if not name.startswith("_")
    ]

@pytest.mark.parametrize("fn", _all_indicator_fns())
def test_indicator_is_marked_safe(fn):
    assert getattr(fn, "_look_ahead_safe", False), f"{fn.__name__} doit être décoré @look_ahead_safe"

@pytest.mark.parametrize("fn", _all_indicator_fns())
def test_indicator_no_look_ahead(fn):
    series = pd.Series(np.random.RandomState(42).randn(500).cumsum() + 100)
    # Adapter selon signature : fn(close) ou fn(df)
    try:
        assert_no_look_ahead(lambda s: fn(s), series, n_samples=100)
    except TypeError:
        df = pd.DataFrame({"open": series, "high": series + 1, "low": series - 1,
                           "close": series, "volume": 1.0})
        assert_no_look_ahead(lambda d: fn(d), df, n_samples=100)
```

### Étape 6 — Cas dégénérés à tester
Dans `tests/unit/test_edge_validation.py` :
- `equity_plate = pd.Series([100.0] * 252)` → `sharpe_ratio = 0`, `validate_edge.go = False`
- `returns_nuls = pd.Series([0.0] * 252)` → DSR retourne NaN (division par 0 contrôlée)
- `equity_avec_1_trade` → `trades_per_year < 30` → `go = False` avec raison explicite
- `embargo_pct = 0` dans `purged_kfold_cv` → `ValueError("embargo must be > 0")`
- `n_obs < 30` → DSR retourne NaN avec warning

## Logging
```markdown
## 2026-MM-DD — Prompt 06 : Validation framework
- **Statut** : ✅ Terminé
- **Fichiers créés** : app/analysis/edge_validation.py (refactoré), app/analysis/look_ahead_check.py, tests/unit/test_edge_validation.py, tests/unit/test_walk_forward.py
- **Tests pytest** : ✅ X tests
- **Couverture validation** : Sharpe, Sortino, DD, DSR, PSR, bootstrap Sharpe, CPCV, walk-forward, anti-look-ahead
```

## Critères go/no-go
- **GO prompt 07** si : `validate_edge` est appelable et retourne `go=True` sur un cas synthétique idéal.
- **NO-GO, revenir à** : ce prompt si DSR ou CPCV ont des bugs (revérifier la formule Bailey & López de Prado).
