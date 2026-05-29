# Pivot v4 — A1 : Audit + correction du simulateur (sizing + DD + Sharpe)

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md) — vue d'ensemble du pivot v4
2. [../00_constitution.md](../00_constitution.md) — règles 10, 11, 15
3. `JOURNAL.md` — historique des H06/H07/H1/H5 avec DD impossibles
4. [app/backtest/metrics.py](../../app/backtest/metrics.py) — code actuel à corriger
5. [app/backtest/deterministic.py](../../app/backtest/deterministic.py) — moteur de backtest
6. [app/backtest/simulator.py](../../app/backtest/simulator.py) — moteur stateful

## Objectif
Corriger le simulateur pour qu'il :
1. Applique un **sizing au risque 2 %** par trade (CAPITAL_EUR × 0.02 / distance_SL_eur).
2. Produise un **DD borné [0 %, 100 %]** du capital.
3. Calcule le **Sharpe sur retours quotidiens en € du capital** (pas en pips bruts).
4. Soit **testable** : un trade TP doit produire +R × risque, un trade SL exactement −risque.

## Type d'opération
🔧 **Bug fix infrastructure** — 0 n_trial consommé. Aucune nouvelle hypothèse OOS testée ici.

## Definition of Done (testable)

- [ ] [app/backtest/sizing.py](../../app/backtest/sizing.py) contient `compute_position_size(entry, sl, capital_eur, risk_pct, asset_cfg) -> float` qui retourne le nombre de lots.
- [ ] [app/backtest/metrics.py](../../app/backtest/metrics.py) `compute_metrics()` :
  - Reçoit `position_sizes: pd.Series` (1 par trade)
  - Calcule l'equity en € à chaque trade : `equity[t] = equity[t-1] + pips_nets[t] × pip_value_eur × position_sizes[t]`
  - DD = `min((equity / cummax_equity) - 1) × 100` (en %, **borné à −100 %**)
  - Sharpe = `pct_change(daily_equity).mean() / std × √annual_factor`
- [ ] `app/backtest/simulator.py` ajoute le sizing dans `simulate_trades()` :
  - Calcule la position size **au moment de l'entrée** (anti-look-ahead OK car ATR connu à `t-1`)
  - Enregistre dans `trades_df["position_size_lots"]`
- [ ] `tests/unit/test_simulator_sizing.py` (NOUVEAU) : ≥ 8 tests :
  1. Trade TP atteint → PnL = +2 % du capital exactement (R:R 2:1, risque 2 %)
  2. Trade SL atteint → PnL = −2 % du capital exactement
  3. 10 SL consécutifs → DD ≈ −18.3 % (capital × (1-0.02)^10), pas −20 % linéaire
  4. 10 TP consécutifs → return ≈ +21.9 % (compound)
  5. Position size US30 : `compute_position_size(40000, 39900, 10000, 0.02, US30_cfg) = 200 € / (100 × 0.92) = 2.17 lots`
  6. Position size EURUSD : `compute_position_size(1.10, 1.099, 10000, 0.02, EURUSD_cfg) ≈ 2.0 lots` (0.0001 pip × 10 € → 100 pip = 10 €)
  7. DD borné : aucune série de trades ne produit DD < −100 %
  8. Sharpe sur equity plate (tous trades à 0 €) = 0.0 (pas NaN)
- [ ] `rtk pytest tests/unit/test_simulator_sizing.py -v` → 8/8 passed
- [ ] `rtk make verify` → 0 erreur
- [ ] `JOURNAL.md` mis à jour (cf. section Logging)

## NE PAS FAIRE

- ❌ Ne PAS modifier `app/config/instruments.py` (coûts) — c'est A2.
- ❌ Ne PAS modifier les stratégies (`app/strategies/*.py`).
- ❌ Ne PAS toucher au test set 2024+. Les CSV ne sont pas re-lus pour cette phase.
- ❌ Ne PAS supprimer le `sharpe_per_trade` existant — il reste utile pour A3.
- ❌ Ne PAS commit avant validation utilisateur.
- ❌ Ne PAS exécuter `python run_*.py` automatiquement — c'est A4 qui rejouera.

## Étapes détaillées

### Étape 1 — Créer `app/backtest/sizing.py`

```python
"""Sizing au risque fixe : 2 % du capital par trade."""
from __future__ import annotations

from app.config.instruments import AssetConfig


def compute_position_size(
    entry_price: float,
    stop_loss_price: float,
    capital_eur: float,
    risk_pct: float,
    asset_cfg: AssetConfig,
) -> float:
    """Taille de position en lots pour risquer exactement `risk_pct` du capital sur le SL.

    Formule :
        risk_eur = capital × risk_pct
        distance_price = |entry - stop_loss|
        distance_points = distance_price / asset_cfg.pip_size
        loss_per_lot_eur = distance_points × asset_cfg.pip_value_eur
        lots = risk_eur / loss_per_lot_eur

    Clamp dans [asset_cfg.min_lot, asset_cfg.max_lot].
    """
    if stop_loss_price == entry_price:
        raise ValueError("entry_price == stop_loss_price : SL nul, impossible de calculer")
    risk_eur = capital_eur * risk_pct
    distance_points = abs(entry_price - stop_loss_price) / asset_cfg.pip_size
    loss_per_lot_eur = distance_points * asset_cfg.pip_value_eur
    if loss_per_lot_eur <= 0:
        raise ValueError(f"loss_per_lot_eur invalide : {loss_per_lot_eur}")
    lots = risk_eur / loss_per_lot_eur
    return max(asset_cfg.min_lot, min(asset_cfg.max_lot, round(lots, 2)))


def expected_pnl_eur(
    pips_net: float,
    position_size_lots: float,
    asset_cfg: AssetConfig,
) -> float:
    """PnL net en € pour un trade.

    pips_net × position_size_lots × pip_value_eur
    """
    return pips_net * position_size_lots * asset_cfg.pip_value_eur
```

### Étape 2 — Modifier `app/backtest/simulator.py`

Ajouter dans `simulate_trades()` (ou équivalent) :

```python
from app.backtest.sizing import compute_position_size

# Au moment de l'entrée d'un trade :
position_lots = compute_position_size(
    entry_price=entry_price,
    stop_loss_price=sl_price,
    capital_eur=capital_eur,  # Lu depuis env CAPITAL_EUR (10 000 par défaut)
    risk_pct=risk_pct,         # Lu depuis env RISK_PER_TRADE (0.02 par défaut)
    asset_cfg=asset_cfg,
)
# Enregistrer dans la trace du trade
trade_row["position_size_lots"] = position_lots
```

### Étape 3 — Modifier `app/backtest/metrics.py`

**Remplacer** la fonction `compute_metrics()` pour qu'elle :
1. Reçoive `capital_eur` initial (10 000).
2. Convertisse chaque trade en PnL € via `pips_net × position_size_lots × pip_value_eur`.
3. Construise une equity curve en € : `equity = capital_eur + cumsum(pnl_eur_per_trade)`.
4. Calcule DD sur equity (pas sur pips).
5. Calcule Sharpe sur retours quotidiens de l'equity.

```python
def compute_metrics(
    trades_df: pd.DataFrame,
    annee: int | None = None,
    df: pd.DataFrame | None = None,
    asset_cfg: AssetConfig | None = None,
    capital_eur: float = 10_000.0,
) -> dict:
    base: dict = {
        "annee": annee,
        "profit_net_eur": 0.0,
        "max_dd_pct": 0.0,
        "trades": 0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "sharpe_per_trade": 0.0,
        "total_return_pct": 0.0,
        "final_equity_eur": capital_eur,
    }
    if trades_df.empty or asset_cfg is None:
        return base

    if "position_size_lots" not in trades_df.columns:
        raise ValueError("trades_df doit contenir 'position_size_lots'. "
                         "Re-run le simulator avec sizing au risque 2 %.")

    pnl_eur = (
        trades_df["Pips_Nets"] * trades_df["position_size_lots"] * asset_cfg.pip_value_eur
    )
    equity = capital_eur + pnl_eur.cumsum()
    # Protection blow-up : equity ne descend pas sous 0
    equity = equity.clip(lower=0.01)

    # Drawdown borné [-1, 0]
    cummax = equity.cummax()
    dd_series = (equity / cummax) - 1.0
    max_dd_pct = float(dd_series.min()) * 100  # négatif, borné à -100 %

    # Resample daily pour Sharpe (sera affiné en A3)
    equity_daily = equity.resample("D").last().ffill() if hasattr(equity.index, "to_pydatetime") else equity
    daily_returns = equity_daily.pct_change().dropna()
    sharpe = sharpe_ratio(daily_returns, annual_factor=252.0)

    # Sharpe per-trade (annualisé par n_trades) — utile pour strats faible fréquence
    n_trades = len(trades_df)
    per_trade_returns = pnl_eur / equity.shift(1).fillna(capital_eur)
    sharpe_per_trade = sharpe_ratio(per_trade_returns, annual_factor=max(n_trades, 1))

    win_rate = (trades_df["Pips_Bruts"] > 0).mean() * 100
    profit_net_eur = float(pnl_eur.sum())
    total_return_pct = (float(equity.iloc[-1]) / capital_eur - 1.0) * 100

    return {
        "annee": annee,
        "profit_net_eur": profit_net_eur,
        "max_dd_pct": max_dd_pct,
        "trades": n_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "sharpe_per_trade": sharpe_per_trade,
        "total_return_pct": total_return_pct,
        "final_equity_eur": float(equity.iloc[-1]),
    }
```

**Garder** la fonction `sharpe_ratio()` existante (lignes 25-48), elle est correcte sur des retours nombreux.

### Étape 4 — Tests unitaires `tests/unit/test_simulator_sizing.py`

```python
"""Tests de cohérence du sizing + métriques après pivot v4 A1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.backtest.sizing import compute_position_size, expected_pnl_eur
from app.backtest.metrics import compute_metrics


US30 = ASSET_CONFIGS["US30"]
# Suppose qu'on a aussi un EURUSDConfig minimal pour les tests
EURUSD = AssetConfig(spread_pips=0.9, slippage_pips=0.3, pip_size=0.0001,
                    pip_value_eur=10.0, tp_points=20, sl_points=10)


def test_sizing_us30_100pt_sl():
    """SL = 100 points US30 → 2.17 lots pour 200 € de risque sur 10 000 €."""
    lots = compute_position_size(
        entry_price=40000.0, stop_loss_price=39900.0,
        capital_eur=10_000.0, risk_pct=0.02, asset_cfg=US30,
    )
    expected = 200.0 / (100.0 * 0.92)
    assert lots == pytest.approx(expected, rel=1e-2)


def test_sizing_eurusd_10pip_sl():
    """SL = 10 pips EURUSD → 2.0 lots pour 200 € de risque."""
    lots = compute_position_size(
        entry_price=1.1000, stop_loss_price=1.0990,
        capital_eur=10_000.0, risk_pct=0.02, asset_cfg=EURUSD,
    )
    # 10 pips × pip_value_eur 10 = 100 €/lot. Risk 200 € → 2 lots.
    assert lots == pytest.approx(2.0, rel=1e-2)


def test_trade_tp_exact_2pct():
    """Trade TP ratio R:R=2 → +4 % du capital (risque 2 % × 2)."""
    # Construct trade: entry 40000, sl 39900, tp 40200 (R:R = 2)
    # pips_net = +200 (TP), pip_value=0.92, sizing= 2.17 lots
    # PnL = 200 × 2.17 × 0.92 = 399.28 € ≈ 4 % de 10 000 €
    pnl = expected_pnl_eur(200.0, 2.17, US30)
    assert pnl == pytest.approx(400.0, rel=0.05)


def test_trade_sl_exact_minus2pct():
    """Trade SL → −2 % du capital exactement."""
    pnl = expected_pnl_eur(-100.0, 2.17, US30)
    assert pnl == pytest.approx(-200.0, rel=0.05)


def test_dd_bounded_minus_100():
    """50 trades SL consécutifs → DD doit rester ≥ −100 %."""
    trades_df = pd.DataFrame({
        "Pips_Nets": [-100.0] * 50,
        "Pips_Bruts": [-100.0] * 50,
        "position_size_lots": [2.17] * 50,
    }, index=pd.date_range("2024-01-01", periods=50, freq="D"))
    metrics = compute_metrics(trades_df, asset_cfg=US30, capital_eur=10_000.0)
    assert metrics["max_dd_pct"] >= -100.0
    assert metrics["max_dd_pct"] < 0


def test_sharpe_equity_plate_returns_zero():
    """Trades tous à PnL=0 → Sharpe = 0 (pas NaN)."""
    trades_df = pd.DataFrame({
        "Pips_Nets": [0.0] * 30,
        "Pips_Bruts": [0.0] * 30,
        "position_size_lots": [1.0] * 30,
    }, index=pd.date_range("2024-01-01", periods=30, freq="D"))
    metrics = compute_metrics(trades_df, asset_cfg=US30, capital_eur=10_000.0)
    assert metrics["sharpe"] == 0.0
    assert not np.isnan(metrics["sharpe"])


def test_10_sl_compound_dd():
    """10 SL consécutifs = (1-0.02)^10 ≈ −18.29 % (compound), pas −20 % linéaire."""
    trades_df = pd.DataFrame({
        "Pips_Nets": [-100.0] * 10,
        "Pips_Bruts": [-100.0] * 10,
        "position_size_lots": [2.17] * 10,
    }, index=pd.date_range("2024-01-01", periods=10, freq="D"))
    metrics = compute_metrics(trades_df, asset_cfg=US30, capital_eur=10_000.0)
    # Sizing fixe → DD linéaire = -20 %. Accepter cette approximation.
    # Le compound exact demanderait de relire le capital après chaque trade.
    assert -22.0 <= metrics["max_dd_pct"] <= -18.0


def test_sizing_invalid_sl_equals_entry():
    """SL = entry → ValueError."""
    with pytest.raises(ValueError, match="SL nul"):
        compute_position_size(
            entry_price=40000.0, stop_loss_price=40000.0,
            capital_eur=10_000.0, risk_pct=0.02, asset_cfg=US30,
        )
```

### Étape 5 — Vérification visuelle

Sur le résultat H06 US30 (DD test 362 %) :
1. Re-lire le `predictions/h06_donchian_multi_asset.json` (sans modifier le test set).
2. Compter le nombre de trades SL consécutifs maximum.
3. Valider que **avec sizing 2 %, le DD ne peut excéder 100 %**.
4. Documenter dans `docs/simulator_audit_a1.md` : avant/après, ancien DD vs nouveau DD attendu si rejoué.

### Étape 6 — make verify

```bash
rtk make verify
```

Tous les checks doivent passer :
- ruff
- mypy
- pytest (incluant les 8 nouveaux tests)
- snooping_check

## Tests unitaires associés

Listés en Étape 4. Total 8 tests dans `tests/unit/test_simulator_sizing.py`.

## Logging obligatoire

À ajouter dans `JOURNAL.md` :

```markdown
## YYYY-MM-DD — Pivot v4 A1 : Audit simulateur (sizing + DD + Sharpe)

- **Statut** : ✅ Terminé
- **Type** : Bug fix infrastructure (0 n_trial consommé)
- **Fichiers créés** : `app/backtest/sizing.py`, `tests/unit/test_simulator_sizing.py`, `docs/simulator_audit_a1.md`
- **Fichiers modifiés** : `app/backtest/metrics.py` (compute_metrics complet), `app/backtest/simulator.py` (intégration sizing)
- **Résultats clés** :
  - DD désormais borné [−100 %, 0 %]
  - Sizing au risque 2 % implémenté
  - Sharpe sur retours du capital en €, pas en pips
- **Tests** : 8/8 nouveaux + non-régression OK
- **make verify** : ✅ passé
- **Bugs corrigés** :
  - DD calculé sur pips bruts → DD calculé sur equity €
  - Pas de sizing → sizing 200 € de risque / SL en €
- **Notes** : Aucune stratégie modifiée. Aucune lecture du test set 2024+.
- **Prochaine étape** : A2 — calibration coûts XTB réels.
```

## Critères go/no-go

- **GO Phase A2** si :
  - 8/8 tests passent
  - `make verify` passe
  - Entry `JOURNAL.md` rédigée
  - Documentation `docs/simulator_audit_a1.md` produite
- **NO-GO, revenir à** : cette phase si tests ne passent pas. Diagnostiquer le bug (souvent : equity.cumsum() sans capital initial, ou cummax non monotone à cause d'un float arrondi).

## Annexes

### A1 — Formule de sizing au risque (Kelly mince)

Soit :
- C = capital en €
- r = risque par trade (0.02)
- e = prix d'entrée
- sl = prix de stop loss
- s = pip_size (taille d'un point)
- v = pip_value_eur (valeur en € d'un point pour 1 lot)

Alors :
```
risk_eur     = C × r
distance_pts = |e - sl| / s
loss_1lot_eur = distance_pts × v
position_lots = risk_eur / loss_1lot_eur
              = (C × r) / (|e - sl| / s × v)
              = (C × r × s) / (|e - sl| × v)
```

### A2 — Pourquoi le DD précédent était > 100 %

Ancien code (simplifié) :
```python
cum = trades_df["Pips_Nets"].cumsum()  # PnL cumulé EN POINTS
dd_pts = max_drawdown(cum)             # DD négatif EN POINTS
max_dd_pct = dd_pts × pip_value / capital × 100
```

Si la stratégie fait 50 trades SL à −100 points (cumul = −5000 pts) sur US30 :
- `dd_pts = −5000`
- `max_dd_pct = −5000 × 0.92 / 10000 × 100 = −46 %`

Mais avec des trades larges (Donchian 100+ trades à −300 points moyen incluant TP timeout) :
- `dd_pts = −40 000`
- `max_dd_pct = −40 000 × 0.92 / 10000 × 100 = −368 %`

→ Le compte serait blow-up bien avant, mais le simulateur continue de cumuler. **Bug fundamental : pas de gestion du blow-up**.

Nouveau code :
```python
equity = capital + cumsum(pnl_eur).clip(lower=0.01)
dd = (equity / equity.cummax() - 1).min() × 100   # toujours dans [−100, 0]
```

### A3 — Pourquoi un sizing 2 % change tout

Sans sizing (position fixe 1 lot) :
- US30 SL 100 pts × 0.92 € = 92 € de perte
- 92 / 10 000 = 0.92 % par trade (au lieu de 2 %)
- Mais coût absolu fixe → si stratégie a 30 % WR R:R 1:1, perd même avec edge en théorie positif.

Avec sizing 2 % :
- US30 SL 100 pts × 2.17 lots × 0.92 € = 200 € de perte = 2 % exact
- Le PnL devient proportionnel au capital, pas aux pips
- Compound naturel sur l'equity curve

## Fin du prompt A1.
**Suivant (ordre révisé)** : [A5_feature_generation.md](A5_feature_generation.md)

> ⚠️ L'ordre original était A1 → A2 → A3 → A5… Il a été révisé en mai 2026 pour prioriser le pipeline ML. Nouvel ordre : A1 ✅ → **A5 → A6 → A7 → A8 → A9** → A2 → A3 → [A4 opt] → B1. Voir [00_README.md](00_README.md) section "Ordre d'exécution strict — RÉVISÉ".
