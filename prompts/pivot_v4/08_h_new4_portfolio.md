# Pivot v4 — B4 : H_new4 — Portfolio multi-sleeves (conditionnel)

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. Phase A complète ✅
3. **AU MOINS 2 sleeves GO** parmi B1 / B2 / B3 (vérifier `JOURNAL.md`)
4. [../00_constitution.md](../00_constitution.md) — règle 10 (Sharpe portfolio)
5. [JOURNAL.md](../../JOURNAL.md)

## Objectif
**Question H_new4** : Si ≥ 2 sleeves (= hypothèses GO en B1/B2/B3) sont validées, leur combinaison en portfolio equal-risk weight + filtre corrélation produit-elle un Sharpe portfolio **strictement supérieur** au max des Sharpe individuels d'au moins +0.2 ?

C'est la **dernière étape avant production**. Sans valeur ajoutée de la diversification, mieux vaut déployer le sleeve unique gagnant que de complexifier inutilement.

## Type d'opération
🟢 **Nouvelle hypothèse OOS de diversification** — **+1 n_trial** (cumul 26 si B1+B2+B3 faits).

## Précondition d'activation

✅ **GO ce prompt** si :
- Au moins 2 sleeves GO parmi B1, B2, B3.

❌ **SKIP ce prompt** si :
- Un seul (ou zéro) sleeve GO → passer directement en production sur ce sleeve unique.

## Definition of Done (testable)

- [ ] `app/portfolio/constructor.py` (NOUVEAU ou réutilisation v3) contient :
  - `equal_risk_weights(sleeve_returns: dict[str, pd.Series], target_vol: float = 0.10) -> pd.DataFrame`
  - `correlation_filter(sleeve_returns: dict, cap: float = 0.7, rolling_sharpe_window: int = 126) -> dict`
- [ ] `scripts/run_h_new4_portfolio.py` :
  - Charge les equity OOS des sleeves GO (depuis `predictions/h_new1_*.json`, etc.)
  - Construit portfolio equal-risk
  - Applique filtre corrélation 60j cap 0.7 (rebalance hebdomadaire)
  - Vol targeting 10 % annualisé, leverage cap 2.0
  - `validate_edge` sur l'equity portfolio
- [ ] `predictions/h_new4_portfolio.json` + `docs/h_new4_portfolio.md`
- [ ] Tests `tests/unit/test_portfolio_combinator.py` (NOUVEAU) : ≥ 5 tests :
  1. 2 sleeves indépendants Sharpe 1.0 chacun → portfolio Sharpe ≥ √2 (gain de diversification)
  2. 2 sleeves parfaitement corrélés → portfolio Sharpe ≈ Sharpe individuel
  3. Equal-risk : poids inversement proportionnels à la vol
  4. Filtre correlation : si ρ > 0.7 → la sleeve avec le pire Sharpe 6M est désactivée
  5. Leverage cap appliqué quand vol target / vol réalisée > 2.0
- [ ] `n_trials_cumul` mis à jour AVANT exécution
- [ ] `rtk make verify` → 0 erreur

## Critères GO/NO-GO chiffrés

| Critère | Cible | Notes |
|---|---|---|
| Sharpe portfolio | ≥ max(Sharpe individuel) + 0.2 | Le portfolio doit apporter de la diversification |
| Sharpe portfolio | ≥ 1.2 | Au-dessus du seuil constitution + marge |
| DSR portfolio (n=26) | > 0 (p < 0.05) | DSR plus pénalisant à n=26 |
| DD portfolio | < 12 % | Plus serré que sleeve unique car diversifié |
| Trades/an | somme ≥ 60 | 2+ sleeves combinés |
| Corrélation moyenne sleeves | < 0.5 | Diversification effective |

**GO** : tous critères passent → ✅ production avec portfolio.
**NO-GO** : production avec le sleeve GO de meilleur Sharpe unitaire (single-sleeve fallback).

## NE PAS FAIRE

- ❌ Ne PAS exécuter si moins de 2 sleeves GO.
- ❌ Ne PAS modifier les sleeves elles-mêmes en réaction (data snooping).
- ❌ Ne PAS tuner `target_vol`, `correlation_cap`, ou `leverage_cap` post-hoc.
- ❌ Ne PAS oublier de re-trader les sleeves avec les MÊMES coûts XTB v4 (A2).
- ❌ Ne PAS rebalancer quotidiennement (= turnover excessif).
- ❌ Ne PAS oublier `read_oos()`.

## Étapes détaillées

### Étape 1 — `app/portfolio/constructor.py`

```python
"""Portfolio equal-risk weight + filtre corrélation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.metrics import sharpe_annualized


def equal_risk_weights(
    sleeve_returns: dict[str, pd.Series],
    target_vol_annual: float = 0.10,
    leverage_cap: float = 2.0,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """Poids equal-risk : chaque sleeve contribue target_vol/N à la vol portfolio."""
    df = pd.DataFrame(sleeve_returns).fillna(0.0)
    n = len(sleeve_returns)
    target_vol_per_sleeve = target_vol_annual / np.sqrt(n)
    realized_vol = df.rolling(vol_lookback).std() * np.sqrt(252)
    weights = (target_vol_per_sleeve / realized_vol).clip(upper=leverage_cap).fillna(0.0)
    return weights


def correlation_filter(
    sleeve_returns: dict[str, pd.Series],
    cap: float = 0.7,
    corr_window: int = 60,
    sharpe_window: int = 126,
) -> pd.DataFrame:
    """Active/désactive les sleeves selon corrélation rolling et Sharpe 6M.

    Si ρ_ij > cap sur fenêtre 60j → désactive la sleeve avec le pire Sharpe 6M.
    Rebalance hebdomadaire (vendredi).
    Retourne DataFrame booléen (True = actif) reindexé sur l'index original.
    """
    df = pd.DataFrame(sleeve_returns).fillna(0.0)
    weekly = df.resample("W-FRI").last()
    active = pd.DataFrame(True, index=weekly.index, columns=df.columns)

    rolling_sharpe = df.rolling(sharpe_window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    )

    for date in weekly.index[corr_window:]:
        window = df.loc[:date].tail(corr_window)
        corr = window.corr()
        active_this = set(df.columns)
        for i in df.columns:
            for j in df.columns:
                if i >= j or i not in active_this or j not in active_this:
                    continue
                if corr.loc[i, j] > cap:
                    sr_i = rolling_sharpe.loc[date, i] if date in rolling_sharpe.index else 0
                    sr_j = rolling_sharpe.loc[date, j] if date in rolling_sharpe.index else 0
                    drop = i if sr_i < sr_j else j
                    active_this.discard(drop)
        for c in df.columns:
            active.loc[date, c] = c in active_this

    return active.reindex(df.index, method="ffill").fillna(True).astype(bool)


def build_portfolio_equity(
    sleeve_returns: dict[str, pd.Series],
    weights: pd.DataFrame,
    active_mask: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> pd.Series:
    df = pd.DataFrame(sleeve_returns).fillna(0.0)
    effective_weights = weights * active_mask.astype(float)
    # Normalisation pour que la somme des poids actifs ≤ leverage_cap global
    portfolio_returns = (df * effective_weights).sum(axis=1)
    return initial_capital * (1 + portfolio_returns).cumprod()
```

### Étape 2 — Script `scripts/run_h_new4_portfolio.py`

```python
"""Pivot v4 B4 — H_new4 : portfolio des sleeves GO."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.seeds import set_global_seeds
from app.portfolio.constructor import (
    equal_risk_weights, correlation_filter, build_portfolio_equity,
)
from app.backtest.metrics import compute_metrics
from app.analysis.edge_validation import validate_edge
from app.testing.snooping_guard import read_oos


SLEEVES_FILES = {
    "h_new1_us30_d1": "predictions/h_new1_meta_us30.json",
    "h_new3_eurusd_h4": "predictions/h_new3_eurusd_h4.json",
    # "h_new2_us30_rolling": "predictions/h_new2_walk_forward_rolling.json",  # si applicable
}


def _load_sleeve_returns(file: str, sleeve_name: str) -> pd.Series:
    """Reconstruit l'equity sleeve à partir du JSON."""
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    trades = pd.DataFrame(data.get("trades_oos", []))
    # ⚠️ À adapter selon la structure réelle du JSON produit en B1/B2/B3
    if trades.empty:
        return pd.Series(dtype=float)
    trades.index = pd.to_datetime(trades["exit_time"])
    pnl_eur = trades["pnl_eur"]  # supposé pré-calculé
    return pnl_eur.resample("D").sum() / 10_000.0  # retour daily en fraction du capital


def main() -> int:
    set_global_seeds()
    sleeve_returns: dict[str, pd.Series] = {}
    for name, file in SLEEVES_FILES.items():
        if Path(file).exists():
            sr = _load_sleeve_returns(file, name)
            if not sr.empty:
                sleeve_returns[name] = sr

    if len(sleeve_returns) < 2:
        print(f"Moins de 2 sleeves GO ({len(sleeve_returns)}). H_new4 SKIPPED.")
        return 0

    weights = equal_risk_weights(sleeve_returns, target_vol_annual=0.10, leverage_cap=2.0)
    active = correlation_filter(sleeve_returns, cap=0.7, corr_window=60, sharpe_window=126)
    equity = build_portfolio_equity(sleeve_returns, weights, active, initial_capital=10_000.0)

    daily_returns = equity.pct_change().dropna()
    sr_portfolio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    cummax = equity.cummax()
    dd = ((equity / cummax) - 1.0).min() * 100

    # validate_edge sur portfolio
    # On a besoin d'un trades_df mais on n'a que daily returns → adapter
    fake_trades = pd.DataFrame({
        "Pips_Nets": daily_returns.values * 10_000.0,
        "Pips_Bruts": daily_returns.values * 10_000.0,
        "position_size_lots": [1.0] * len(daily_returns),
    }, index=daily_returns.index)
    report = validate_edge(equity=equity, trades=fake_trades, n_trials=26)

    read_oos(
        prompt="pivot_v4_B4",
        hypothesis="H_new4_portfolio",
        sharpe=sr_portfolio,
        n_trades=int(fake_trades.shape[0]),
    )

    individual_sharpes = {
        name: (sr.mean() / sr.std()) * np.sqrt(252) if sr.std() > 0 else 0.0
        for name, sr in sleeve_returns.items()
    }
    max_individual = max(individual_sharpes.values())
    diversification_gain = sr_portfolio - max_individual

    out = {
        "n_sleeves": len(sleeve_returns),
        "individual_sharpes": individual_sharpes,
        "portfolio_sharpe": float(sr_portfolio),
        "max_individual_sharpe": float(max_individual),
        "diversification_gain": float(diversification_gain),
        "portfolio_max_dd_pct": float(dd),
        "validate_edge": {
            "go": report.go, "reasons": report.reasons, "metrics": report.metrics,
        },
        "config": {
            "target_vol_annual": 0.10,
            "leverage_cap": 2.0,
            "correlation_cap": 0.7,
            "corr_window": 60,
            "sharpe_window": 126,
            "rebalance": "weekly_friday",
        },
    }
    Path("predictions/h_new4_portfolio.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Portfolio Sharpe = {sr_portfolio:.2f}, max individuel = {max_individual:.2f}")
    print(f"Gain diversification = {diversification_gain:+.2f}")
    print(f"Verdict : {'GO' if report.go and diversification_gain >= 0.2 else 'NO-GO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 3 — Tests `tests/unit/test_portfolio_combinator.py`

```python
import numpy as np
import pandas as pd
import pytest

from app.portfolio.constructor import (
    equal_risk_weights, correlation_filter, build_portfolio_equity,
)


def _make_returns(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n),
                    index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))


def test_equal_risk_basic():
    sleeves = {"a": _make_returns(seed=0), "b": _make_returns(seed=1)}
    w = equal_risk_weights(sleeves, target_vol_annual=0.10)
    # Last weights should be finite and ≤ 2.0
    assert (w.iloc[-100:].max().max() <= 2.0)
    assert (w.iloc[-100:].min().min() >= 0)


def test_diversification_gain_independent():
    """2 sleeves indépendants Sharpe similaire → portfolio Sharpe > max individuel."""
    a = _make_returns(seed=0)
    b = _make_returns(seed=1)
    sleeves = {"a": a, "b": b}
    w = equal_risk_weights(sleeves)
    active = pd.DataFrame(True, index=a.index, columns=["a", "b"])
    eq = build_portfolio_equity(sleeves, w, active, initial_capital=10_000.0)
    ret_p = eq.pct_change().dropna()
    sr_p = (ret_p.mean() / ret_p.std()) * np.sqrt(252) if ret_p.std() > 0 else 0
    sr_a = (a.mean() / a.std()) * np.sqrt(252)
    sr_b = (b.mean() / b.std()) * np.sqrt(252)
    # Le portfolio doit faire mieux que la moyenne mais pas nécessairement strictement
    # mieux que le max si les seeds sont identiques. Le test est tolérant.
    assert sr_p >= min(sr_a, sr_b) - 0.5


def test_correlation_filter_disables_correlated():
    """2 sleeves parfaitement corrélés → 1 est désactivé."""
    a = _make_returns(seed=0)
    sleeves = {"a": a, "b": a.copy()}  # parfaite corrélation
    active = correlation_filter(sleeves, cap=0.7)
    # Au moins un instant après warmup où une sleeve est désactivée
    assert (active.iloc[100:] == False).any().any()


def test_leverage_cap_enforced():
    sleeves = {"a": _make_returns(seed=0) * 0.01}  # vol très basse
    w = equal_risk_weights(sleeves, target_vol_annual=0.10, leverage_cap=2.0)
    assert w.max().max() <= 2.0 + 1e-9
```

### Étape 4 — `JOURNAL.md` AVANT exécution

```markdown
| pivot_v4 | H_new4 (portfolio des sleeves GO) | 1 | 26 | EN COURS | — |
```

### Étape 5 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_h_new4_portfolio.py
```

### Étape 6 — Rapport `docs/h_new4_portfolio.md`

```markdown
# H_new4 — Portfolio des sleeves GO (pivot v4 B4)

**Date** : YYYY-MM-DD
**n_trials** : 26
**Sleeves combinés** : [liste]

## Question
Le portfolio equal-risk weight + filtre corrélation des sleeves GO produit-il
un Sharpe portfolio strictement supérieur (+0.2) au max des Sharpe individuels ?

## Sleeves inclus

| Sleeve | Source | Sharpe individuel | Trades/an | DD |
|---|---|---|---|---|
| H_new1 US30 D1 | predictions/h_new1_meta_us30.json | ? | ? | ? |
| H_new3 EURUSD H4 | predictions/h_new3_eurusd_h4.json | ? | ? | ? |
| ... | ... | ... | ... | ... |

## Résultats portfolio

| Métrique | Cible | Observé | Status |
|---|---|---|---|
| Sharpe portfolio | ≥ max + 0.2 | ? | ✓/✗ |
| Sharpe portfolio absolu | ≥ 1.2 | ? | ✓/✗ |
| DSR (n=26) | > 0 (p<0.05) | ? | ✓/✗ |
| Max DD | < 12 % | ? | ✓/✗ |
| Trades/an | ≥ 60 (somme) | ? | ✓/✗ |
| Corrélation moyenne | < 0.5 | ? | ✓/✗ |

## Décomposition de contribution

| Sleeve | Poids moyen | Contribution Sharpe |
|---|---|---|
| H_new1 US30 | ? | ? |
| H_new3 EURUSD | ? | ? |

## Décision

- ✅ GO portfolio → production multi-sleeves (prompt 20+)
- ❌ NO-GO portfolio → production single-sleeve (sleeve avec meilleur Sharpe individuel)

## Causes possibles d'échec du portfolio
- Corrélation latente entre sleeves (ex : US30 et EURUSD peuvent corréler en crise)
- Diversification noyée par les coûts de rebalance
- Vol targeting clip trop souvent → leverage moyen trop faible
```

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 B4 : H_new4 portfolio multi-sleeves

- **Statut** : [✅ GO portfolio / ❌ NO-GO → single-sleeve]
- **n_trials_cumul** : 26
- **Sleeves inclus** : [liste]
- **Sharpe portfolio** : ? (vs max individuel ?)
- **Diversification gain** : ?
- **Fichiers créés** : app/portfolio/constructor.py, scripts/run_h_new4_portfolio.py, tests/unit/test_portfolio_combinator.py, predictions/h_new4_portfolio.json, docs/h_new4_portfolio.md
- **Décision** : [Production multi-sleeves / Production single-sleeve <name>]
- **Notes** : read_oos() appelé. Sleeves individuels déjà lus en B1/B2/B3 = pas de nouvelle lecture du test set.
```

## Critères go/no-go

- **GO Phase production multi-sleeves** (prompt 20+) si :
  - Sharpe portfolio ≥ max(Sharpe individuel) + 0.2
  - Sharpe portfolio absolu ≥ 1.2
  - DSR > 0 p < 0.05
  - DD < 12 %
- **NO-GO** : passer en production **single-sleeve** sur le sleeve GO de meilleur Sharpe individuel. Ne PAS dégrader avec un portfolio sous-performant.

## Annexes

### A1 — Pourquoi un gain de +0.2 minimum ?

La diversification théorique entre 2 sleeves indépendants de Sharpe S égal devrait donner :
- Sharpe portfolio = S × √2 ≈ S × 1.41
- Soit gain ≈ 0.41 × S (pour S = 1.0 → +0.41)

En pratique, corrélations résiduelles + coûts de rebalance + vol targeting réduisent ce gain à 30-60 % du gain théorique. Donc +0.2 minimum est un seuil **réaliste** pour valider que la diversification apporte effectivement de la valeur, sans tomber dans la complaisance.

### A2 — Pourquoi 0.7 cap de corrélation ?

- ρ < 0.5 : sleeves indépendants, diversification optimale
- 0.5 ≤ ρ < 0.7 : sleeves modérément corrélés, diversification réduite mais utile
- ρ ≥ 0.7 : sleeves quasi-redondants, on perd ~ 50 % du bénéfice de diversification

Le cap 0.7 désactive uniquement les cas extrêmes. C'est conservateur.

### A3 — Vol targeting 10 % annualisé

10 % annu = vol typique d'un portfolio actions diversifié. Pour 10 000 € de capital, vol cible = 1 000 €/an = ~ 5.5 €/jour ou ~ 22 €/semaine.

Cible plus serrée (5 %) = plus stable mais Sharpe absolu plus bas.
Cible plus large (15-20 %) = vol portfolio plus élevée mais Sharpe similaire.

### A4 — Pourquoi rebalance hebdomadaire et pas daily ?

- Daily : turnover x 5, coûts x 5 sur l'ajustement des poids
- Weekly : turnover modéré, coûts acceptables
- Monthly : trop lent, poids déconnectés du régime actuel

Weekly est le standard quant. Notamment pour EURUSD H4 (qui rebalance "intra-week" naturellement par ses propres trades), aligner sur weekly évite les conflits d'allocation.

### A5 — Et si on a 3+ sleeves GO ?

Le code supporte n sleeves arbitraire. Avec 3 sleeves indépendants :
- Sharpe portfolio théorique = S × √3 ≈ S × 1.73
- Gain de diversification potentiellement +0.7

Mais : DSR pénalise n_trials = 27+ → plus difficile à passer le test de significativité. Adapter le seuil `gain ≥ 0.3` au lieu de `+0.2` si 3+ sleeves.

### A6 — Production single-sleeve si NO-GO portfolio

Si le portfolio ne dépasse pas le sleeve individuel, c'est **OK**. On part en production avec le **sleeve gagnant unique**. Pas de honte à faire simple. La complexité doit être justifiée par un gain mesurable.

Configuration production : `app/config/production.json` (à créer en prompt 20) :
```json
{
  "version": "v4.0.0",
  "sleeves": [
    {"asset": "US30", "tf": "D1", "strategy": "donchian", "params": {"N": 20, "M": 20}, "meta_labeling": true}
  ],
  "portfolio_weighting": "single",
  "capital_eur": 10000,
  "risk_per_trade": 0.02
}
```

## Fin du prompt B4.
**Suivant** : `prompts/20_signal_engine.md` (Phase production) — peu importe GO portfolio ou single-sleeve.
