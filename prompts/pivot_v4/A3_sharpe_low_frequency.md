# Pivot v4 — A3 : Fix Sharpe pour stratégies à faible fréquence

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. [A2_calibration_costs.md](A2_calibration_costs.md) — **DOIT ÊTRE TERMINÉ AVANT**
3. [../00_constitution.md](../00_constitution.md) — règle 10 (Sharpe sur retours du capital)
4. [app/backtest/metrics.py](../../app/backtest/metrics.py) — fonctions `sharpe_ratio`, `sharpe_daily_from_trades`

## Objectif
Corriger le calcul du Sharpe pour les stratégies à faible fréquence (< 100 trades/an). Le `resample("D").ffill()` actuel écrase artificiellement la variance vers zéro, sous-estimant le Sharpe. La correction route le calcul selon le nombre de trades/an et utilise un Sharpe annualisé approprié.

## Type d'opération
🔧 **Bug fix infrastructure** — 0 n_trial consommé.

## Definition of Done (testable)

- [ ] [app/backtest/metrics.py](../../app/backtest/metrics.py) contient une fonction `sharpe_annualized(equity, trades_df, asset_cfg, capital_eur)` qui :
  - Détecte la fréquence : `trades_per_year = len(trades_df) / years_span`
  - Si `trades_per_year ≥ 100` → Sharpe daily (équivalent v3)
  - Si `30 ≤ trades_per_year < 100` → Sharpe weekly resampling
  - Si `trades_per_year < 30` → Sharpe per-trade × √trades_per_year (méthode v2 H05)
- [ ] La méthode utilisée est **loggée** dans le retour : `metrics["sharpe_method"]: Literal["daily", "weekly", "per_trade"]`
- [ ] `tests/unit/test_sharpe_routing.py` (NOUVEAU) : ≥ 6 tests :
  1. Stratégie 250 trades/an → méthode daily, valeur cohérente avec sklearn/scipy reference
  2. Stratégie 50 trades/an → méthode weekly
  3. Stratégie 15 trades/an → méthode per-trade
  4. Stratégie 1 trade → retourne 0.0 (pas NaN)
  5. Stratégie equity plate → retourne 0.0
  6. Comparaison méthodes sur synthétique : daily ≈ weekly ≈ per-trade pour stratégie hyper-fréquente (cohérence cross-method)
- [ ] `rtk make verify` → 0 erreur
- [ ] `JOURNAL.md` mis à jour

## NE PAS FAIRE

- ❌ Ne PAS supprimer `sharpe_ratio()` ni `sharpe_daily_from_trades()` — les conserver pour rétrocompat.
- ❌ Ne PAS changer la signature de `compute_metrics()` de façon casse-non-rétrocompatible.
- ❌ Ne PAS toucher au test set.
- ❌ Ne PAS commencer A4 avant validation de cette étape.

## Étapes détaillées

### Étape 1 — Refonte de la fonction `sharpe_annualized`

Ajouter dans `app/backtest/metrics.py` :

```python
from typing import Literal


def sharpe_annualized(
    equity: pd.Series,
    trades_df: pd.DataFrame,
    asset_cfg: AssetConfig | None = None,
    capital_eur: float = 10_000.0,
) -> tuple[float, Literal["daily", "weekly", "per_trade"]]:
    """Sharpe annualisé, route selon la fréquence.

    Retourne (sharpe, method).

    Routing :
        ≥ 100 trades/an → daily resample
        30-99 trades/an → weekly resample
        < 30 trades/an  → per-trade × √trades_per_year
    """
    if len(trades_df) < 2 or equity.empty:
        return 0.0, "daily"

    # Calculer la période en années
    if isinstance(trades_df.index, pd.DatetimeIndex):
        span_seconds = (trades_df.index[-1] - trades_df.index[0]).total_seconds()
    else:
        span_seconds = max(1, len(trades_df)) * 86400
    years = max(span_seconds / (365.25 * 86400), 1e-3)
    tpy = len(trades_df) / years

    if tpy >= 100:
        # Méthode daily : résample equity, pct_change
        if isinstance(equity.index, pd.DatetimeIndex):
            daily = equity.resample("D").last().ffill()
        else:
            daily = equity
        returns = daily.pct_change().dropna()
        sr = sharpe_ratio(returns, annual_factor=252.0)
        return sr, "daily"

    if tpy >= 30:
        # Méthode weekly : meilleure pour Donchian D1 et H4
        if isinstance(equity.index, pd.DatetimeIndex):
            weekly = equity.resample("W-FRI").last().ffill()
        else:
            weekly = equity
        returns = weekly.pct_change().dropna()
        sr = sharpe_ratio(returns, annual_factor=52.0)
        return sr, "weekly"

    # Méthode per-trade : pour stratégies < 30 trades/an
    # Returns par trade = (pnl_eur / equity_avant_trade)
    if asset_cfg is None:
        return 0.0, "per_trade"
    if "position_size_lots" not in trades_df.columns:
        # Fallback compat v3 : pas de sizing
        per_trade_returns = trades_df["Pips_Nets"] * asset_cfg.pip_value_eur / capital_eur
    else:
        per_trade_returns = (
            trades_df["Pips_Nets"]
            * trades_df["position_size_lots"]
            * asset_cfg.pip_value_eur
            / capital_eur
        )
    if per_trade_returns.std() == 0 or len(per_trade_returns) < 2:
        return 0.0, "per_trade"
    sr_per_trade = per_trade_returns.mean() / per_trade_returns.std()
    sr_annualized = sr_per_trade * np.sqrt(tpy)
    return float(sr_annualized), "per_trade"
```

### Étape 2 — Intégrer dans `compute_metrics`

Remplacer le bloc Sharpe dans `compute_metrics()` :

```python
# AVANT (à supprimer / remplacer) :
# daily_pips = trades_df["Pips_Nets"].resample("D").sum().dropna()
# daily_returns = _pips_to_return(daily_pips, pip_value_eur, initial_capital)
# sharpe = sharpe_ratio(daily_returns)

# APRÈS :
sharpe, sharpe_method = sharpe_annualized(equity, trades_df, asset_cfg, capital_eur)
```

Et ajouter `sharpe_method` au dict retour :
```python
return {
    ...
    "sharpe": sharpe,
    "sharpe_method": sharpe_method,  # NEW
    ...
}
```

### Étape 3 — Tests `tests/unit/test_sharpe_routing.py`

```python
"""Tests du routing Sharpe selon la fréquence des trades."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import ASSET_CONFIGS
from app.backtest.metrics import sharpe_annualized

US30 = ASSET_CONFIGS["US30"]


def _build_trades(n: int, span_years: float, mean_pip: float, std_pip: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    pips = rng.normal(mean_pip, std_pip, n)
    timestamps = pd.date_range("2024-01-01", periods=n, freq=f"{int(span_years*365.25*24*60/n)}min")
    capital = 10_000.0
    lots = 2.17
    pnl_eur = pips * lots * 0.92
    equity = pd.Series(capital + pnl_eur.cumsum(), index=timestamps)
    trades_df = pd.DataFrame({
        "Pips_Nets": pips,
        "Pips_Bruts": pips,
        "position_size_lots": [lots] * n,
    }, index=timestamps)
    return equity, trades_df


def test_daily_method_high_frequency():
    """250 trades en 1 an → method='daily', Sharpe non nul."""
    equity, trades = _build_trades(n=250, span_years=1.0, mean_pip=1.0, std_pip=10.0)
    sr, method = sharpe_annualized(equity, trades, US30)
    assert method == "daily"
    assert sr != 0.0


def test_weekly_method_mid_frequency():
    """50 trades/an → method='weekly'."""
    equity, trades = _build_trades(n=50, span_years=1.0, mean_pip=1.0, std_pip=10.0)
    sr, method = sharpe_annualized(equity, trades, US30)
    assert method == "weekly"
    assert sr != 0.0


def test_per_trade_method_low_frequency():
    """15 trades/an → method='per_trade'."""
    equity, trades = _build_trades(n=15, span_years=1.0, mean_pip=5.0, std_pip=20.0)
    sr, method = sharpe_annualized(equity, trades, US30)
    assert method == "per_trade"
    assert sr != 0.0


def test_single_trade_returns_zero():
    """1 trade → 0.0 (pas NaN)."""
    equity, trades = _build_trades(n=1, span_years=1.0, mean_pip=1.0, std_pip=1.0)
    sr, method = sharpe_annualized(equity, trades, US30)
    assert sr == 0.0
    assert not np.isnan(sr)


def test_equity_plate_returns_zero():
    """Trades tous à 0 € → Sharpe = 0."""
    timestamps = pd.date_range("2024-01-01", periods=50, freq="D")
    equity = pd.Series([10_000.0] * 50, index=timestamps)
    trades = pd.DataFrame({
        "Pips_Nets": [0.0] * 50,
        "Pips_Bruts": [0.0] * 50,
        "position_size_lots": [1.0] * 50,
    }, index=timestamps)
    sr, _ = sharpe_annualized(equity, trades, US30)
    assert sr == 0.0


def test_methods_cohere_high_frequency():
    """Sur stratégie hyper-fréquente, les 3 méthodes devraient donner des valeurs proches."""
    equity, trades = _build_trades(n=2000, span_years=1.0, mean_pip=0.5, std_pip=5.0)
    sr_daily, _ = sharpe_annualized(equity, trades, US30)
    # Forcer la méthode weekly artificiellement en réduisant n
    equity_w, trades_w = _build_trades(n=80, span_years=1.0, mean_pip=0.5, std_pip=5.0)
    sr_weekly, _ = sharpe_annualized(equity_w, trades_w, US30)
    # Tolerance large : c'est juste un sanity check
    assert abs(sr_daily - sr_weekly) < 5.0
```

### Étape 4 — Mise à jour des scripts utilisateurs

Lister les scripts qui appellent `compute_metrics()` :

```bash
rtk grep -rn "compute_metrics(" --include="*.py"
```

Pour chacun, vérifier qu'il passe `asset_cfg` au lieu de juste `pip_value_eur` (signature étendue par A1). Si non, mettre à jour.

### Étape 5 — Documentation

Créer/étendre `docs/simulator_audit_a1.md` avec une section "A3 — Sharpe routing" :

```markdown
## A3 — Sharpe routing par fréquence

Routing automatique selon `trades_per_year` (tpy) :

| Régime | Méthode | Annual factor |
|---|---|---|
| tpy ≥ 100 | resample daily | √252 |
| 30 ≤ tpy < 100 | resample weekly | √52 |
| tpy < 30 | per-trade | √tpy |

### Justification

- **Daily** : valable si au moins 1 trade tous les 2-3 jours en moyenne. Sinon le ffill écrase la variance.
- **Weekly** : valable pour Donchian D1 (~30-90 trades/an = ~0.6-1.7 trades/semaine).
- **Per-trade** : valable pour Donchian H4 D1 multi-actif où chaque actif fait 10-30 trades/an. La méthode standard suppose i.i.d. ce qui est raisonnable pour des trades indépendants.

### Impact sur résultats v3

Donchian US30 D1 H06 avait 75 trades sur 16 mois = 56 trades/an = méthode weekly avec ce routing. L'ancien Sharpe daily était écrasé. Avec weekly, il devrait remonter de ~0.3.
```

## Tests unitaires associés

Listés en Étape 3. 6 tests dans `tests/unit/test_sharpe_routing.py`.

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 A3 : Sharpe routing par fréquence

- **Statut** : ✅ Terminé
- **Type** : Bug fix infrastructure (0 n_trial consommé)
- **Fichiers créés** : `tests/unit/test_sharpe_routing.py`
- **Fichiers modifiés** : `app/backtest/metrics.py` (sharpe_annualized + compute_metrics)
- **Méthodes ajoutées** : daily / weekly / per_trade selon trades_per_year
- **Tests** : 6/6 passed
- **make verify** : ✅ passé
- **Impact attendu** : Sharpe Donchian D1 (50 trades/an) → routé weekly, valeur non écrasée par ffill
- **Notes** : Aucune lecture test set. Compatibilité v3 maintenue (sharpe_ratio et sharpe_daily_from_trades conservés).
- **Prochaine étape** : A4 — replay H06/H07 train+val avec simulateur corrigé.
```

## Critères go/no-go

- **GO Phase A4** si :
  - 6/6 tests passent
  - Méthode `sharpe_method` ajoutée au retour de `compute_metrics`
  - Aucun script existant cassé (compat v3 maintenue)
- **NO-GO, revenir à** : si tests échouent → vérifier l'index temporel (le routing assume DatetimeIndex).

## Annexes

### A1 — Pourquoi le daily resample biaise le Sharpe

Sur Donchian D1 :
- Trades typiques : entrée lundi, sortie jeudi → 1 trade tous les 3-7 jours.
- `resample("D").last().ffill()` produit une equity quasi-constante 80 % des jours.
- `pct_change()` retourne 0 sur ces jours.
- Donc 80 % des observations sont à 0, écrasant la moyenne et la variance vers zéro.
- Sharpe artificiellement proche de 0 même si l'edge est positif.

**Solution mathématique** : Sharpe per-trade × √n_trades_par_an est correct pour des trades i.i.d. (en première approximation pour stratégies trend-following indépendantes).

### A2 — Sharpe per-trade formule

Pour n trades indépendants avec retours par-trade r_i :

```
sr_per_trade = mean(r_i) / std(r_i)
sr_annualized = sr_per_trade × √trades_per_year
```

Validation : si on a 4 trades/mois (48/an) avec Sharpe per-trade 0.5, Sharpe annualisé = 0.5 × √48 ≈ 3.46. Pour 250 trades/an (daily-like) avec même Sharpe per-trade, on aurait 0.5 × √250 ≈ 7.9. Cohérent avec le fait que plus on a de trades à edge constant, plus le Sharpe annualisé monte (jusqu'à saturation par autocorrélation).

### A3 — Pourquoi 100 trades/an comme seuil daily ?

100 trades/an ≈ 1 trade tous les 2.5 jours ouvrés. À cette fréquence, le ffill ne crée pas plus de 1-2 jours de retour zéro consécutifs → variance daily reste représentative.

Pour < 100 trades/an, on passe en weekly. Le seuil 30 trades/an pour passer à per-trade correspond au minimum statistique pour avoir un Sharpe per-trade significatif (~1 trade tous les 2 semaines).

## Fin du prompt A3.
**Suivant** : [A4_replay_h06_h07.md](A4_replay_h06_h07.md)
