# Pivot v4 — B2 : H_new3 — EURUSD H4 mean-reversion + méta-labeling RF

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. Phase A complète (A1-A4) ✅
3. [05_h_new1_meta_us30.md](05_h_new1_meta_us30.md) — peut être en GO ou NO-GO, mais doit être terminé
4. [../00_constitution.md](../00_constitution.md) — toutes règles
5. [JOURNAL.md](../../JOURNAL.md) — vérifier n_trials_cumul actuel

## Objectif
**Question H_new3** : Une stratégie mean-reversion (RSI(14) extrême + Bollinger reversal) sur EURUSD H4, filtrée par méta-labeling RF et entraînée par walk-forward 6M, produit-elle un edge net après coûts XTB réels ?

**Hypothèse en contre-pied** :
- Toutes les stratégies testées en v3 étaient trend-following → 0 GO.
- Le marché range 70 % du temps → mean-reversion peut capturer le complément.
- EURUSD = paire la plus liquide → spread XTB minimal (0.7 pip).
- H4 = sweet spot bruit/coûts (H1 trop bruyant comme l'a montré v1).
- Méta-labeling RF en surcouche → filtre le bruit.

## Type d'opération
🟢 **Nouvelle hypothèse OOS** — **+1 n_trial** (cumul 24 si après B1, ou 23 si B1 sauté).

> ⚠️ EURUSD H4 n'a JAMAIS été testé en v3. Test set 2024+ vierge pour cet actif/TF.

## Definition of Done (testable)

- [ ] `app/strategies/mean_reversion.py` (NOUVEAU) contient :
  - `class MeanReversionRSIBB(rsi_period=14, rsi_long=30, rsi_short=70, bb_period=20, bb_mult=2.0)` héritant de `BaseStrategy`
  - Signal LONG : `RSI < rsi_long` ET `Close < BB_lower`
  - Signal SHORT : `RSI > rsi_short` ET `Close > BB_upper`
  - Signal sortie : `RSI repasse 50` OU TP/SL touché OU timeout (32 barres H4 = 8 jours)
  - `.shift(1)` anti-look-ahead obligatoire
- [ ] `app/config/instruments.py` contient EURUSD config (déjà ajouté en A2).
- [ ] `data/raw/EURUSD/H4.csv` doit être disponible (cf. inventory). Si absent → erreur clair.
- [ ] `scripts/run_h_new3_eurusd_h4.py` :
  - Charge EURUSD H4
  - Stratégie mean-reversion baseline
  - Méta-labeling RF (réutilise `app/models/meta_labeling.py` de B1)
  - Walk-forward retrain 6M depuis 2024-01-01
  - `read_oos()` appelé une seule fois
- [ ] `tests/unit/test_mean_reversion_rsi_bb.py` (NOUVEAU) : ≥ 5 tests :
  1. Signal LONG produit quand RSI < 30 ET Close < BB_lower
  2. Pas de signal quand seul RSI < 30
  3. `.shift(1)` appliqué (anti-look-ahead via `look_ahead_safe`)
  4. Cas dégénéré : série constante → 0 signal
  5. Symétrie LONG/SHORT
- [ ] `predictions/h_new3_eurusd_h4.json` + `docs/h_new3_eurusd_h4.md`.
- [ ] Tableau `n_trials` mis à jour AVANT exécution.
- [ ] `rtk make verify` → 0 erreur.

## Critères GO/NO-GO chiffrés

| Critère | Cible | Notes |
|---|---|---|
| Sharpe walk-forward test | ≥ 1.0 | Probablement méthode daily (H4 = 100+ trades/an) |
| DSR | > 0 ET p < 0.05 | Avec n_trials_cumul = 24 ou 23 |
| Max DD | < 12 % | Plus serré que B1 car mean-rev intraday |
| WR | > 55 % | R:R 1:1 → besoin WR > 50 % après coûts |
| Trades/an | ≥ 100 | H4 EURUSD doit produire 100-300 trades/an |
| **GO si** | TOUS passent | Production éventuelle |
| **NO-GO si** | UN SEUL échoue | Bascule H_new2 |

## NE PAS FAIRE

- ❌ Ne PAS utiliser H1 sur EURUSD (v1 a échoué dessus, trop bruyant).
- ❌ Ne PAS tester d'autres paires Forex sans hypothèse séparée (= +1 n_trial chacune).
- ❌ Ne PAS optimiser les paramètres RSI/BB en réaction (ce sont des valeurs standards : 14, 30, 70, 20, 2).
- ❌ Ne PAS oublier la session : EURUSD est liquide quasi 24/5, mais le vrai overlap Londres/NY (13-16 UTC) est crucial. Considérer un filtre de session intra-strat.
- ❌ Ne PAS confondre pip EURUSD (0.0001) avec pip indice.
- ❌ Ne PAS oublier `read_oos()` ni incrément `n_trials`.
- ❌ Ne PAS tester pendant H4 weekend (vendredi soir → dimanche soir).

## Étapes détaillées

### Étape 1 — Implémentation `MeanReversionRSIBB`

```python
"""Stratégie mean-reversion : RSI extrême + Bollinger reversal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import BaseStrategy
from app.testing.look_ahead_validator import look_ahead_safe
from app.features.indicators import rsi


@look_ahead_safe
def bb_bands(close: pd.Series, period: int, mult: float) -> tuple[pd.Series, pd.Series]:
    sma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return sma - mult * sd, sma + mult * sd


class MeanReversionRSIBB(BaseStrategy):
    name = "MeanReversionRSIBB"

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_long: float = 30.0,
        rsi_short: float = 70.0,
        bb_period: int = 20,
        bb_mult: float = 2.0,
    ):
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.bb_period = bb_period
        self.bb_mult = bb_mult

    @look_ahead_safe
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        rsi_v = rsi(close, self.rsi_period)
        lower, upper = bb_bands(close, self.bb_period, self.bb_mult)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[(rsi_v < self.rsi_long) & (close < lower)] = 1
        sig[(rsi_v > self.rsi_short) & (close > upper)] = -1
        # Anti-look-ahead : le signal calculé à t s'applique à t+1
        return sig.shift(1).fillna(0).astype(int)

    def __str__(self) -> str:
        return (
            f"MeanReversionRSIBB(rsi={self.rsi_period}/"
            f"{self.rsi_long}/{self.rsi_short}, "
            f"bb={self.bb_period}/{self.bb_mult})"
        )
```

### Étape 2 — Vérifier les données EURUSD H4

```bash
ls data/raw/EURUSD/
```

Doit contenir `H4.csv` (et idéalement `D1.csv` si on veut un overlay). Si absent → demander à l'utilisateur de fournir le CSV au format constitution (timestamp UTC, OHLC + volume optionnel).

### Étape 3 — Script `scripts/run_h_new3_eurusd_h4.py`

```python
"""Pivot v4 B2 — H_new3 : EURUSD H4 mean-reversion + meta-labeling.

⚠️ Consomme 1 n_trial. Lecture OOS test set ≥ 2024 = unique.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.config.instruments import ASSET_CONFIGS
from app.strategies.mean_reversion import MeanReversionRSIBB
from app.features.indicators import rsi, adx, atr
from app.pipelines.walk_forward import walk_forward_meta
from app.backtest.metrics import compute_metrics
from app.analysis.edge_validation import validate_edge
from app.testing.snooping_guard import read_oos


def build_features_h4(df: pd.DataFrame) -> pd.DataFrame:
    """Features méta-labeling pour mean-reversion H4."""
    close = df["Close"]
    atr14 = atr(df, 14)
    sma50 = close.rolling(50).mean()
    out = pd.DataFrame({
        "RSI_14": rsi(close, 14),
        "ADX_14": adx(df, 14),
        "Dist_SMA_50": (close - sma50) / atr14,
        "ATR_Norm_14": atr14 / close,
        "Log_Return_5": np.log(close / close.shift(5)),
        "BB_Width": (close.rolling(20).std() * 4) / close,
        # Features session H4
        "Hour_UTC": pd.Series(df.index.hour, index=df.index).astype(float),
        "Is_London_NY_Overlap": pd.Series(
            ((df.index.hour >= 13) & (df.index.hour < 17)).astype(int),
            index=df.index,
        ).astype(float),
    }, index=df.index)
    return out.dropna()


def build_target_winner(df: pd.DataFrame, pnl_brut: pd.Series) -> pd.Series:
    return (pnl_brut > 0).astype(int)


def main() -> int:
    set_global_seeds()
    df = load_asset("EURUSD", "H4")
    cfg = ASSET_CONFIGS["EURUSD"]

    strat = MeanReversionRSIBB(
        rsi_period=14, rsi_long=30, rsi_short=70,
        bb_period=20, bb_mult=2.0,
    )

    all_trades_oos, segments = walk_forward_meta(
        df=df,
        strat=strat,
        cfg=cfg,
        feature_builder=build_features_h4,
        target_builder=build_target_winner,
        retrain_months=6,
        test_start="2024-01-01",
        capital_eur=10_000.0,
    )

    metrics = compute_metrics(all_trades_oos, asset_cfg=cfg, capital_eur=10_000.0)

    equity = 10_000 + (
        all_trades_oos["Pips_Nets"] * all_trades_oos["position_size_lots"] * cfg.pip_value_eur
    ).cumsum()
    report = validate_edge(
        equity=equity,
        trades=all_trades_oos,
        n_trials=24,  # 23 (avec B1) + B2
    )

    read_oos(
        prompt="pivot_v4_B2",
        hypothesis="H_new3_eurusd_h4_meanrev",
        sharpe=metrics["sharpe"],
        n_trades=int(metrics["trades"]),
    )

    out = {
        "config": {
            "strat": str(strat),
            "asset": "EURUSD",
            "tf": "H4",
            "retrain_months": 6,
            "capital_eur": 10_000.0,
            "risk_per_trade": 0.02,
            "features": list(build_features_h4(df.head(300)).columns),
        },
        "metrics_walk_forward_oos": metrics,
        "segments": [s.__dict__ for s in segments],
        "validate_edge": {
            "go": report.go,
            "reasons": report.reasons,
            "metrics": report.metrics,
        },
    }
    out_path = Path("predictions/h_new3_eurusd_h4.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"H_new3 terminé. Verdict : {'GO' if report.go else 'NO-GO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 4 — Tests unitaires `tests/unit/test_mean_reversion_rsi_bb.py`

```python
"""Tests stratégie MeanReversionRSIBB."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.mean_reversion import MeanReversionRSIBB
from app.testing.look_ahead_validator import assert_no_look_ahead


def _make_df(close_values: list[float]) -> pd.DataFrame:
    n = len(close_values)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "Open": close_values, "Close": close_values,
        "High": [c + 0.001 for c in close_values],
        "Low": [c - 0.001 for c in close_values],
        "Volume": [1000.0] * n,
    }, index=idx)


def test_long_signal_on_oversold_breakdown():
    """Close très bas (RSI < 30 et < BB_lower) → signal LONG."""
    # 80 barres montantes puis chute brutale = bas RSI + close < BB_lower
    closes = list(np.linspace(1.10, 1.15, 80)) + list(np.linspace(1.15, 1.05, 20))
    df = _make_df(closes)
    strat = MeanReversionRSIBB()
    sig = strat.generate_signals(df)
    assert sig.iloc[-2:].max() == 1 or sig.iloc[-5:].max() == 1


def test_short_signal_on_overbought_breakout():
    """Close très haut (RSI > 70 et > BB_upper) → signal SHORT."""
    closes = list(np.linspace(1.10, 1.05, 80)) + list(np.linspace(1.05, 1.20, 20))
    df = _make_df(closes)
    strat = MeanReversionRSIBB()
    sig = strat.generate_signals(df)
    assert sig.iloc[-2:].min() == -1 or sig.iloc[-5:].min() == -1


def test_no_signal_on_flat():
    """Close constant → 0 signal."""
    closes = [1.10] * 100
    df = _make_df(closes)
    strat = MeanReversionRSIBB()
    sig = strat.generate_signals(df)
    assert (sig == 0).all()


def test_anti_look_ahead():
    rng = np.random.default_rng(42)
    closes = (1.10 + rng.normal(0, 0.005, 500).cumsum() * 0.1).tolist()
    df = _make_df(closes)
    strat = MeanReversionRSIBB()
    assert_no_look_ahead(strat.generate_signals, df, n_samples=50)


def test_marker_present():
    strat = MeanReversionRSIBB()
    assert getattr(strat.generate_signals, "_look_ahead_safe", False)
```

### Étape 5 — Mise à jour `JOURNAL.md` AVANT exécution

```markdown
| pivot_v4 | H_new3 (EURUSD H4 mean-rev + meta-labeling) | 1 | 24 | EN COURS | — |
```

### Étape 6 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_h_new3_eurusd_h4.py
```

### Étape 7 — Rapport `docs/h_new3_eurusd_h4.md`

```markdown
# H_new3 — EURUSD H4 mean-reversion + meta-labeling

**Date** : YYYY-MM-DD
**n_trials** : 24 (23 hérités + 1 H_new3)
**Verdict** : [GO / NO-GO]

## Question
Mean-reversion (RSI extrême + BB reversal) sur EURUSD H4, filtré par méta-RF
walk-forward 6M, donne-t-il un Sharpe ≥ 1.0 sur 2024+ après coûts XTB réels ?

## Hypothèses sous-jacentes
1. EURUSD H4 = liquide, spread serré, mean-reversion possible.
2. Session overlap Londres/NY = max liquidité.
3. RSI(14) < 30 + Close < BB_lower(20, 2) = bottom statistique.
4. Méta-labeling RF avec feature de session = filtre le bruit.

## Résultats walk-forward OOS

| Métrique | Cible | Observé | Status |
|---|---|---|---|
| Sharpe walk-forward | ≥ 1.0 | ? | ✓/✗ |
| DSR (n=24) | > 0 (p<0.05) | ? | ✓/✗ |
| Max DD | < 12 % | ? | ✓/✗ |
| WR | > 55 % | ? | ✓/✗ |
| Trades/an | ≥ 100 | ? | ✓/✗ |

## Segments walk-forward

| Période | n_trades | Sharpe | WR | DD |
|---|---|---|---|---|
| 2024 H1 | ? | ? | ? | ? |
| 2024 H2 | ? | ? | ? | ? |
| ... | ... | ... | ... | ... |

## Décision

[GO] → Production
[NO-GO] → Bascule H_new2 (walk-forward rolling adaptatif sur Donchian XAUUSD)

## Causes possibles d'échec
- EURUSD H4 trop efficient sur 2024+ (BCE/Fed convergence)
- Méta-labeling RF rejette tous les signaux (cf. H1 XAUUSD pivot v3)
- Session features insuffisantes
- BB(20, 2) trop large pour H4 EURUSD (vol annuelle ~6 %)
```

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 B2 : H_new3 EURUSD H4 mean-reversion + meta

- **Statut** : [✅ GO / ❌ NO-GO]
- **n_trials_cumul** : 24 (ou 23 si B1 sauté)
- **Sharpe walk-forward OOS** : ?
- **Fichiers créés** : app/strategies/mean_reversion.py, scripts/run_h_new3_eurusd_h4.py, tests/unit/test_mean_reversion_rsi_bb.py, predictions/h_new3_eurusd_h4.json, docs/h_new3_eurusd_h4.md
- **Décision** : [GO Phase production / NO-GO → bascule H_new2]
- **Notes** : test set EURUSD H4 2024+ lu UNE fois, read_oos() appelé.
```

## Critères go/no-go

- **GO Phase production** si TOUS les 5 critères passent.
- **NO-GO, bascule H_new2** ([07_h_new2_walk_forward_rolling.md](07_h_new2_walk_forward_rolling.md)) si :
  - Sharpe < 0.5 ET pas d'amélioration nette vs baseline = pas la peine de continuer mean-reversion
  - WR < 50 % → mean-reversion ne capture pas
  - DD > 20 % → trop volatile

## Annexes

### A1 — Pourquoi EURUSD H4 et pas H1 ?

H1 sur EURUSD (v1) a échoué à cause du bruit microstructurel. H4 = compromis :
- 6 barres/jour (1 460 barres/an vs 8 760 en H1) → moins de bruit
- Encore 200-400 trades/an potentiels (suffisant pour Sharpe daily)
- Spread XTB toujours 0.7 pip (même valeur que H1)

D1 EURUSD donne trop peu de trades (50-80/an, on est limite pour le critère ≥ 30 mais marginal).

### A2 — Pourquoi mean-reversion et pas trend-following ?

Stats marché EURUSD 2018-2023 :
- ~70 % du temps dans un range borné par ±2× ATR(20)
- ~30 % du temps en trend (ADX > 25)
- Trend-following pure → exposé aux 30 %, ignore les 70 %
- Mean-rev → exposé aux 70 %, ignore les 30 %

Au méta-labeling de trier entre vrais setups et faux signaux.

### A3 — RSI(14) seuils 30/70

Standards Wilder (inventeur du RSI). Sont robustes hors-overfit. Tester 25/75 ou 35/65 = +1 n_trial pour pas grand-chose statistiquement.

### A4 — Bollinger(20, 2) seuils

Standards. 2 SD = 95.4 % de la distribution gaussienne théorique → 4.6 % de signaux/jour si distribution normale (en pratique 8-12 % à cause des fat tails forex). Sur H4 EURUSD, ~150 signaux théoriques par an.

### A5 — Filtre session Londres/NY

13-17h UTC = overlap. Liquidité max. Le signal `Is_London_NY_Overlap` permet au RF d'apprendre que les setups hors-session sont moins fiables (slippage élevé, bruit asiatique).

Tu peux aussi essayer un **filtre dur** (ne trader QUE pendant l'overlap) sans méta-labeling. Mais ça = nouvelle hypothèse, +1 n_trial. Pas dans le scope de B2.

### A6 — Timeout 32 barres H4 = 8 jours

Si le trade ne touche ni TP ni SL en 8 jours → close au prix de marché. Évite les trades "zombie" en latérale.

Pour mean-rev, timeout court (vs Donchian 120h = 5 jours sur D1). Calibré pour 1-2 cycles ATR.

## Fin du prompt B2.
**Suivant si GO** : `prompts/20_signal_engine.md` (Phase 4 production)
**Suivant si NO-GO** : [07_h_new2_walk_forward_rolling.md](07_h_new2_walk_forward_rolling.md)
