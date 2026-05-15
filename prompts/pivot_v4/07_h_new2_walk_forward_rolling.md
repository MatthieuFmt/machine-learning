# Pivot v4 — B3 : H_new2 — Walk-forward rolling adaptatif (conditionnel)

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. Phase A complète ✅
3. [05_h_new1_meta_us30.md](05_h_new1_meta_us30.md) — verdict (GO ou NO-GO)
4. [06_h_new3_eurusd_h4_meanrev.md](06_h_new3_eurusd_h4_meanrev.md) — verdict
5. [JOURNAL.md](../../JOURNAL.md) — note le diagnostic H1 pivot v3 (split mono-classe XAUUSD)
6. [../00_constitution.md](../00_constitution.md)

## Objectif
**Question H_new2** : Un walk-forward avec **fenêtre train rolling de 3 ans** (au lieu du split figé train ≤ 2022) permet-il à des actifs avec distribution temporelle inversée (ex : XAUUSD D1) de produire un edge ?

**Pourquoi conditionnel** : ce prompt n'est exécuté QUE si B1 et B2 sont NO-GO. Sinon, on passe directement en production. C'est l'avant-dernier filet de sécurité.

## Type d'opération
🟢 **Nouvelle hypothèse OOS** — **+1 n_trial** (cumul 25 si B1 et B2 faits).

## Précondition d'activation

✅ **GO ce prompt** si :
- H_new1 (B1) est NO-GO **ET** H_new3 (B2) est NO-GO
- **ET** au moins un actif montrait un Sharpe train+val v4 ≥ 0.8 dans A4 (replay) — sinon, c'est inutile.

❌ **SKIP ce prompt** si :
- H_new1 ou H_new3 est GO → bascule directement Phase production (prompt 20)
- A4 (replay) montre Sharpe train+val v4 < 0.3 partout → l'edge n'existe pas, abandon

## Definition of Done (testable)

- [ ] `app/pipelines/walk_forward_rolling.py` (NOUVEAU) contient :
  - `class WalkForwardRolling` avec `run(df, strat, cfg, train_window_years=3, retrain_months=6)` :
    - Fenêtre train **rolling** (pas expansive comme B1)
    - Retrain tous les 6 mois sur les 3 dernières années
    - Test sur les 6 mois suivants
    - Agrégation OOS continue
- [ ] `scripts/run_h_new2_walk_forward_rolling.py` :
  - Cible : XAUUSD D1 ET US30 D1 (les 2 actifs avec Sharpe train v3 > 1.0)
  - Donchian + méta-labeling RF (mêmes params que B1)
  - Walk-forward rolling 3 ans, retrain 6M
  - Test set lu une seule fois par actif
- [ ] Tests `tests/unit/test_walk_forward_rolling.py` (NOUVEAU) :
  - Fenêtre train respecte train_window_years (≤ 3 × 365 jours)
  - Pas de leak temporel (train_end < oos_start)
  - Embargo respecté
- [ ] `predictions/h_new2_walk_forward_rolling.json` + `docs/h_new2_walk_forward_rolling.md`
- [ ] `n_trials_cumul` mis à jour AVANT exécution (= 25 si B1+B2 faits, 24 si seulement B1)
- [ ] `rtk make verify` → 0 erreur

## Critères GO/NO-GO chiffrés

Identiques à B1 et B2 (Sharpe ≥ 1.0, DSR > 0 p < 0.05, DD < 15 %, WR > 30 %, trades/an ≥ 30 — SUR L'AGRÉGÉ XAUUSD + US30).

Si seulement XAUUSD ou seulement US30 passe → GO sur l'actif qui passe.
Si aucun → NO-GO global → activation du pivot (voir Annexes A5).

## NE PAS FAIRE

- ❌ Ne PAS exécuter ce prompt si B1 ou B2 est GO (gaspillage de n_trial).
- ❌ Ne PAS modifier la stratégie ou les features depuis B1.
- ❌ Ne PAS changer `train_window_years=3` en réaction (ce serait du tuning post-hoc).
- ❌ Ne PAS tester plus de 2 actifs (chaque actif additionnel = +1 n_trial).
- ❌ Ne PAS oublier `read_oos()`.

## Étapes détaillées

### Étape 1 — Implémentation `WalkForwardRolling`

```python
"""Walk-forward avec fenêtre train rolling (pas expansive)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from app.backtest.deterministic import run_deterministic_backtest
from app.config.instruments import AssetConfig
from app.models.meta_labeling import MetaLabelingRF
from app.backtest.metrics import compute_metrics


@dataclass
class RollingSegment:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    n_train: int
    n_oos: int
    sharpe_oos: float


def walk_forward_rolling(
    df: pd.DataFrame,
    strat,
    cfg: AssetConfig,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    target_builder: Callable[[pd.DataFrame, pd.Series], pd.Series],
    train_window_years: int = 3,
    retrain_months: int = 6,
    test_start: str = "2024-01-01",
    capital_eur: float = 10_000.0,
    embargo_days: int = 2,
) -> tuple[pd.DataFrame, list[RollingSegment]]:
    df = df.sort_index()
    retrain_dates = pd.date_range(start=test_start, end=df.index[-1], freq="6MS")

    all_oos_trades: list[pd.DataFrame] = []
    segments: list[RollingSegment] = []

    for i, retrain_dt in enumerate(retrain_dates):
        oos_end = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else df.index[-1]
        train_start = retrain_dt - pd.DateOffset(years=train_window_years)
        train_end = retrain_dt - pd.Timedelta(days=embargo_days)

        df_train = df.loc[train_start:train_end]
        df_oos = df.loc[retrain_dt:oos_end]

        if len(df_train) < 250 or df_oos.empty:
            # < 1 an de train → skip
            continue

        trades_train = run_deterministic_backtest(df_train, strat, cfg)
        if len(trades_train) < 20:
            continue

        X_train = feature_builder(df_train).loc[trades_train.index].dropna()
        y_train = target_builder(df_train, trades_train["Pips_Bruts"]).loc[X_train.index]

        meta = MetaLabelingRF()
        meta.fit(X_train, y_train)

        def _sharpe_for_mask(mask):
            filt = trades_train.loc[mask.values]
            if len(filt) < 5:
                return -np.inf
            return compute_metrics(filt, asset_cfg=cfg, capital_eur=capital_eur)["sharpe"]
        meta.calibrate_threshold(X_train, _sharpe_for_mask)

        trades_oos = run_deterministic_backtest(df_oos, strat, cfg)
        if trades_oos.empty:
            continue
        X_oos = feature_builder(df_oos).loc[trades_oos.index]
        keep_mask = meta.predict(X_oos)
        trades_oos_filt = trades_oos.loc[keep_mask]
        all_oos_trades.append(trades_oos_filt)

        m = compute_metrics(trades_oos_filt, asset_cfg=cfg, capital_eur=capital_eur)
        segments.append(RollingSegment(
            train_start=train_start, train_end=train_end,
            oos_start=retrain_dt, oos_end=oos_end,
            n_train=len(trades_train), n_oos=int(m["trades"]),
            sharpe_oos=float(m["sharpe"]),
        ))

    return (pd.concat(all_oos_trades) if all_oos_trades else pd.DataFrame()), segments
```

### Étape 2 — Script `scripts/run_h_new2_walk_forward_rolling.py`

```python
"""Pivot v4 B3 — H_new2 : walk-forward rolling 3 ans sur XAUUSD + US30."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.config.instruments import ASSET_CONFIGS
from app.strategies.donchian import DonchianBreakout
from app.features.indicators import rsi, adx, atr
from app.pipelines.walk_forward_rolling import walk_forward_rolling
from app.backtest.metrics import compute_metrics
from app.analysis.edge_validation import validate_edge
from app.testing.snooping_guard import read_oos


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr14 = atr(df, 14)
    out = pd.DataFrame({
        "RSI_14": rsi(close, 14),
        "ADX_14": adx(df, 14),
        "Dist_SMA_50": (close - sma50) / atr14,
        "Dist_SMA_200": (close - sma200) / atr14,
        "ATR_Norm_14": atr14 / close,
        "Log_Return_5": np.log(close / close.shift(5)),
    }, index=df.index)
    return out.dropna()


def build_target(df, pnl_brut):
    return (pnl_brut > 0).astype(int)


def run_one_asset(asset: str) -> dict:
    df = load_asset(asset, "D1")
    cfg = ASSET_CONFIGS[asset]
    strat = DonchianBreakout(N=20, M=20)

    trades_oos, segments = walk_forward_rolling(
        df=df, strat=strat, cfg=cfg,
        feature_builder=build_features,
        target_builder=build_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2024-01-01",
        capital_eur=10_000.0,
    )
    metrics = compute_metrics(trades_oos, asset_cfg=cfg, capital_eur=10_000.0)
    equity = 10_000 + (
        trades_oos["Pips_Nets"] * trades_oos["position_size_lots"] * cfg.pip_value_eur
    ).cumsum()
    report = validate_edge(equity=equity, trades=trades_oos, n_trials=25)

    read_oos(
        prompt="pivot_v4_B3",
        hypothesis=f"H_new2_{asset.lower()}_rolling",
        sharpe=metrics["sharpe"],
        n_trades=int(metrics["trades"]),
    )

    return {
        "asset": asset,
        "metrics": metrics,
        "segments": [s.__dict__ for s in segments],
        "validate_edge": {
            "go": report.go, "reasons": report.reasons, "metrics": report.metrics,
        },
    }


def main() -> int:
    set_global_seeds()
    out = {
        "us30": run_one_asset("US30"),
        "xauusd": run_one_asset("XAUUSD"),
    }
    Path("predictions").mkdir(exist_ok=True)
    Path("predictions/h_new2_walk_forward_rolling.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8",
    )
    print(f"H_new2 terminé.")
    print(f"  US30  : verdict {'GO' if out['us30']['validate_edge']['go'] else 'NO-GO'}")
    print(f"  XAUUSD: verdict {'GO' if out['xauusd']['validate_edge']['go'] else 'NO-GO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 3 — Tests `tests/unit/test_walk_forward_rolling.py`

```python
import numpy as np
import pandas as pd
import pytest

from app.config.instruments import ASSET_CONFIGS
from app.strategies.donchian import DonchianBreakout
from app.pipelines.walk_forward_rolling import walk_forward_rolling


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 365 * 6  # 6 years
    idx = pd.date_range("2019-01-01", periods=n, freq="D", tz="UTC")
    close = 40_000 + rng.normal(0, 50, n).cumsum()
    return pd.DataFrame({
        "Open": close, "Close": close,
        "High": close + 100, "Low": close - 100, "Volume": 1.0,
    }, index=idx)


def test_train_window_max_3_years(synthetic_df):
    cfg = ASSET_CONFIGS["US30"]
    strat = DonchianBreakout(N=20, M=20)
    _, segments = walk_forward_rolling(
        df=synthetic_df, strat=strat, cfg=cfg,
        feature_builder=lambda d: pd.DataFrame({"x": d["Close"]}, index=d.index),
        target_builder=lambda d, p: (p > 0).astype(int),
        train_window_years=3, retrain_months=6,
        test_start="2024-01-01",
    )
    for s in segments:
        delta_days = (s.train_end - s.train_start).days
        assert delta_days <= 3 * 365 + 30, f"Train > 3 ans : {delta_days} jours"


def test_no_temporal_leak(synthetic_df):
    cfg = ASSET_CONFIGS["US30"]
    strat = DonchianBreakout(N=20, M=20)
    _, segments = walk_forward_rolling(
        df=synthetic_df, strat=strat, cfg=cfg,
        feature_builder=lambda d: pd.DataFrame({"x": d["Close"]}, index=d.index),
        target_builder=lambda d, p: (p > 0).astype(int),
        train_window_years=3, retrain_months=6,
        test_start="2024-01-01", embargo_days=2,
    )
    for s in segments:
        assert s.train_end < s.oos_start, f"Leak : train_end {s.train_end} >= oos_start {s.oos_start}"
        gap = (s.oos_start - s.train_end).days
        assert gap >= 2, f"Embargo non respecté : {gap} jours"
```

### Étape 4 — Mise à jour `JOURNAL.md` AVANT exécution

```markdown
| pivot_v4 | H_new2 (walk-forward rolling 3y XAUUSD+US30) | 1 | 25 | EN COURS | — |
```

### Étape 5 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_h_new2_walk_forward_rolling.py
```

### Étape 6 — Rapport `docs/h_new2_walk_forward_rolling.md`

```markdown
# H_new2 — Walk-forward rolling adaptatif (pivot v4 B3)

**Date** : YYYY-MM-DD
**n_trials** : 25
**Verdict global** : [GO partiel / GO total / NO-GO]

## Question
La fenêtre train rolling 3 ans résout-elle le problème de split figé qui a tué H1 XAUUSD pivot v3 ?

## Résultats par actif

### US30 D1

| Métrique | Cible | Observé | Status |
|---|---|---|---|
| Sharpe | ≥ 1.0 | ? | ✓/✗ |
| DSR (n=25) | > 0 (p<0.05) | ? | ✓/✗ |
| DD | < 15 % | ? | ✓/✗ |
| WR | > 35 % | ? | ✓/✗ |
| Trades/an | ≥ 30 | ? | ✓/✗ |

### XAUUSD D1

| Métrique | Cible | Observé | Status |
|---|---|---|---|
| Sharpe | ≥ 1.0 | ? | ✓/✗ |
| ... | ... | ... | ... |

## Comparaison vs B1 (expanding window)

| | B1 expanding | B3 rolling 3y |
|---|---|---|
| Train segments | 1 (≤ retrain_date) | 1 (3 ans glissants) |
| Adaptation régime | non | oui |
| Sharpe US30 | ? | ? |
| Sharpe XAUUSD | impossible (split mono-classe) | ? |

## Décision

- 1 actif GO → production sur cet actif uniquement
- 2 actifs GO → portfolio possible (passer à B4)
- 0 GO → abandon / pivot paradigme (cf. Annexe A5)

## Causes possibles d'échec
- Train 3 ans = pas assez d'historique (réduit le pouvoir d'apprentissage RF)
- Régime 2024+ trop différent du régime 2021-2023
- Méta-labeling instable
```

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 B3 : H_new2 walk-forward rolling adaptatif

- **Statut** : [✅ GO X / ❌ NO-GO]
- **n_trials_cumul** : 25
- **Sharpe US30** : ? / **Sharpe XAUUSD** : ?
- **Fichiers créés** : app/pipelines/walk_forward_rolling.py, scripts/run_h_new2_walk_forward_rolling.py, tests/unit/test_walk_forward_rolling.py, predictions/h_new2_walk_forward_rolling.json, docs/h_new2_walk_forward_rolling.md
- **Notes** : test set US30 et XAUUSD 2024+ lus UNE fois, read_oos() x2.
- **Décision** : [Production X / Portfolio / Abandon]
```

## Critères go/no-go

- **GO Phase production** si au moins 1 actif passe les 5 critères (sharpe ≥ 1, DSR > 0 p < 0.05, DD < 15 %, WR > 35 %, trades/an ≥ 30).
- **GO Phase B4 (portfolio)** si 2 actifs passent.
- **NO-GO total** : aucun actif passe → activer le plan d'abandon (Annexe A5).

## Annexes

### A1 — Pourquoi rolling 3 ans (et pas 5 ou 2) ?

- 2 ans = pas assez de données pour RF stable (~ 200-300 samples Donchian).
- 3 ans = sweet spot empirique : ~ 400-600 samples, couvre 1 cycle complet (bull+bear).
- 5 ans = trop ancien, le RF apprend des régimes non pertinents pour 2024+.

### A2 — Pourquoi exactement US30 et XAUUSD ?

Ce sont les 2 actifs qui avaient un Sharpe train v3 ≥ 1.0 en H06 :
- US30 (100, 10) : Sharpe train +0.35 — marginal mais positif
- XAUUSD (100, 20) : Sharpe train +1.13 — fort

Avec correction simulateur (A1-A2), ces Sharpe train devraient remonter de ~+1.5 (cf. estimation de l'audit). Donc cibles probables avec edge réel.

GER30 et US500 avaient Sharpe train v3 < 0.5 → moins probable. À tester séparément en H_new5 si on a encore des n_trials.

### A3 — Et si train_window_years=3 + retrain_months=6 → seulement 2 segments OOS ?

C'est possible : test set 2024-01 → 2026-05 = 28 mois → 5 segments de 6 mois.
- Segment 1 (2024-01 → 2024-06) : train 2021-01 → 2023-12
- Segment 2 (2024-07 → 2024-12) : train 2021-07 → 2024-04 (avec embargo)
- ... etc.

Donc 5 segments × ~10 trades par segment Donchian = 50 trades sur le total. Suffisant pour validate_edge avec trades/an ≥ 30.

### A4 — Pourquoi 1 n_trial et pas 2 (US30 + XAUUSD = 2 actifs) ?

L'hypothèse est : "le walk-forward rolling fonctionne sur les top 2 actifs de H06". C'est UNE hypothèse, pas 2. Le n_trial pénalise les hypothèses statistiques distinctes, pas les exécutions séparées de la même hypothèse.

Cependant : `read_oos()` est appelé **2 fois** (une par actif) pour tracer dans `TEST_SET_LOCK.json`. La traçabilité est plus fine que la pénalisation DSR.

### A5 — Plan d'abandon / pivot si NO-GO total après B1+B2+B3

Si **B1, B2 et B3 sont tous NO-GO**, alors la conclusion est :

> Les CFD XTB D1/H4 (US30, EURUSD, XAUUSD) n'ont pas d'edge exploitable avec :
> - Stratégies déterministes (Donchian, mean-reversion)
> - Méta-labeling RF simple
> - Walk-forward 6M
> - Coûts XTB réels
>
> n_trials cumul ≈ 25. DSR avec si peu d'hypothèses GO → projet statistiquement irrécupérable.

**Options à présenter à l'utilisateur** :

1. **Abandon** : conclure le projet, archiver les leçons, libérer le temps.
2. **Pivot paradigme** :
   - Options (Black-Scholes, volatilité implicite)
   - Market-making (besoin d'un broker FIX/cTrader/MT5 API)
   - Arbitrage statistique multi-actifs (besoin de + d'actifs corrélés)
3. **Pivot broker** :
   - Tester sur IC Markets / Pepperstone (Raw spreads ≈ 0.1 pt + commission). Mais nécessite nouvel audit coûts.
4. **Pivot timeframe ultra-court** :
   - M5/M15 = besoin d'orderbook tick-level data, hors scope CSV daily.
5. **Pivot capital + objectif** :
   - Accepter Sharpe ≥ 0.5 au lieu de 1.0, position sizing 1 % au lieu de 2 %, DD 8 % au lieu de 15 %. Edge plus faible mais robuste.

**Aucune de ces options n'est dans le scope du plan v4**. Elles nécessitent un plan v5.

## Fin du prompt B3.
**Suivant si GO sur ≥ 1 actif** : `prompts/20_signal_engine.md` (Phase production) ou [08_h_new4_portfolio.md](08_h_new4_portfolio.md) si 2 actifs GO.
**Suivant si NO-GO** : Annexe A5 — décision avec l'utilisateur sur le pivot.
