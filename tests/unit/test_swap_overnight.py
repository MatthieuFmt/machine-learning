"""Tests pour la charge swap overnight — Audit v6 Phase F1.

Le swap est une charge appliquée par nuit de détention (carry interbancaire
répercuté par le broker). Convention signée :
    > 0 → crédit  (carry favorable, ex: long AUDJPY)
    < 0 → débit   (carry défavorable, ex: long EURUSD)

Le PnL final = pips_brut + nights_held × swap_per_night.

Couverture :
1. Pas de swap si nights_held = 0 (intraday).
2. Swap appliqué proportionnellement aux nuits.
3. Long et short utilisent les bons champs.
4. Crédit (swap > 0) améliore le PnL.
5. Débit (swap < 0) dégrade le PnL.
6. Rétrocompatibilité : sans swap, résultat identique.
7. Le compteur nights_held est exposé dans le trade dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.deterministic import run_deterministic_backtest
from app.backtest.simulator import _simulate_stateful_core
from app.config.instruments import AssetConfig


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _build_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    return df.set_index("Time")


def _make_asset_cfg(swap_long: float = 0.0, swap_short: float = 0.0) -> AssetConfig:
    """AssetConfig minimal pour tests simulator stateful."""
    return AssetConfig(
        spread_pips=0.0,
        slippage_pips=0.0,
        commission_pips=0.0,
        pip_size=0.0001,
        pip_value_eur=10.0,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        swap_long_pips_per_night=swap_long,
        swap_short_pips_per_night=swap_short,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests pour run_deterministic_backtest (paramètres swap directs)
# ═══════════════════════════════════════════════════════════════════════════


def test_deterministic_no_swap_no_change() -> None:
    """Sans paramètre swap, le PnL reste identique au pré-fix."""
    df = _build_df([
        {"Time": "2024-01-01 00:00", "Open": 0.9995, "High": 1.0000, "Low": 0.9990, "Close": 1.0000},
        {"Time": "2024-01-01 01:00", "Open": 1.0000, "High": 1.0025, "Low": 0.9999, "Close": 1.0020},
    ])
    signals = pd.Series([1, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
        # Pas de paramètres swap (défaut 0.0)
    )
    assert result["trades"][0]["pips_net"] == pytest.approx(20.0)
    # Le trade dure 0 nuit (entry 00:00, exit 01:00 même jour)
    assert result["trades"][0]["nights_held"] == 0


def test_deterministic_intraday_zero_nights() -> None:
    """Trade intraday : nights_held = 0, swap ne s'applique pas même avec swap≠0."""
    df = _build_df([
        {"Time": "2024-01-01 10:00", "Open": 0.9995, "High": 1.0000, "Low": 0.9990, "Close": 1.0000},
        {"Time": "2024-01-01 11:00", "Open": 1.0000, "High": 1.0025, "Low": 0.9999, "Close": 1.0020},
    ])
    signals = pd.Series([1, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
        swap_long_pips_per_night=-0.5,   # débit (même jour → ignoré)
        swap_short_pips_per_night=+0.2,
    )
    trade = result["trades"][0]
    assert trade["nights_held"] == 0
    assert trade["pips_net"] == pytest.approx(20.0)  # pas de charge


def test_deterministic_long_swap_debit_one_night() -> None:
    """Long avec swap débit -0.5 pip/nuit, 1 nuit → PnL diminué de 0.5."""
    # NB fill honnête (entry_on_next_open=True, défaut depuis 2026-08-22) :
    # le signal de la barre i entre à l'OPEN de la barre i+1. Les fixtures
    # portent donc une barre de signal en tête, pour que l'entrée tombe bien
    # sur la barre voulue et que la détention couvre encore la/les nuit(s).
    df = _build_df([
        {"Time": "2024-01-01 22:00", "Open": 0.9995, "High": 0.9998, "Low": 0.9992, "Close": 0.9995},
        {"Time": "2024-01-01 23:00", "Open": 0.9995, "High": 1.0000, "Low": 0.9990, "Close": 1.0000},
        {"Time": "2024-01-02 01:00", "Open": 1.0000, "High": 1.0025, "Low": 0.9999, "Close": 1.0020},
    ])
    signals = pd.Series([1, 0, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
        swap_long_pips_per_night=-0.5,
    )
    trade = result["trades"][0]
    assert trade["nights_held"] == 1
    # PnL = 20 (TP touché) + 1 × (-0.5) = 19.5
    assert trade["pips_net"] == pytest.approx(19.5)


def test_deterministic_long_swap_credit_five_nights() -> None:
    """Long timeout 5 jours, swap crédit +0.3 pip/nuit → PnL amélioré de +1.5."""
    rows = [
        # Open == Close : le PnL brut ne dépend alors pas de la convention de fill.
        {"Time": f"2024-01-{day:02d} 00:00", "Open": 1.0005, "High": 1.0010, "Low": 0.9990, "Close": 1.0005}
        for day in range(1, 9)
    ]
    df = _build_df(rows)
    # Signal long au 01, jamais TP/SL → timeout au bout de 5 barres (window_hours).
    signals = pd.Series([1] + [0] * 7, index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=50, sl_pips=50, window_hours=120,  # 120h = 5 jours D1
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
        swap_long_pips_per_night=+0.3,
    )
    trade = result["trades"][0]
    # Entry 01, exit après 5 jours D1 (timeout window=5 bars) → exit le 06
    assert trade["nights_held"] == 5
    # PnL brut + 5 × 0.3 = brut + 1.5
    # PnL brut = (close[5] - close[0]) / pip_size = (1.0005 - 1.0005) / 0.0001 = 0
    assert trade["pips_net"] == pytest.approx(1.5)


def test_deterministic_short_uses_short_swap() -> None:
    """Short doit utiliser swap_short_pips_per_night, pas swap_long."""
    # NB fill honnête (entry_on_next_open=True, défaut depuis 2026-08-22) :
    # le signal de la barre i entre à l'OPEN de la barre i+1. Les fixtures
    # portent donc une barre de signal en tête, pour que l'entrée tombe bien
    # sur la barre voulue et que la détention couvre encore la/les nuit(s).
    df = _build_df([
        {"Time": "2024-01-01 22:00", "Open": 1.0005, "High": 1.0008, "Low": 1.0002, "Close": 1.0005},
        {"Time": "2024-01-01 23:00", "Open": 1.0005, "High": 1.0010, "Low": 0.9995, "Close": 1.0000},
        {"Time": "2024-01-02 01:00", "Open": 1.0000, "High": 1.0005, "Low": 0.9975, "Close": 0.9980},
    ])
    signals = pd.Series([-1, 0, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
        swap_long_pips_per_night=-99.0,   # ne doit PAS s'appliquer
        swap_short_pips_per_night=+0.4,   # doit s'appliquer (1 nuit)
    )
    trade = result["trades"][0]
    assert trade["signal"] == -1
    assert trade["nights_held"] == 1
    # Short TP touché (-20 pips de prix mouvement profitables) + 1 × +0.4 = 20.4
    assert trade["pips_net"] == pytest.approx(20.4)


def test_deterministic_nights_held_exposed_in_trade() -> None:
    """nights_held doit être présent dans le dict trade pour audit."""
    # NB fill honnête (entry_on_next_open=True, défaut depuis 2026-08-22) :
    # le signal de la barre i entre à l'OPEN de la barre i+1. Les fixtures
    # portent donc une barre de signal en tête, pour que l'entrée tombe bien
    # sur la barre voulue et que la détention couvre encore la/les nuit(s).
    df = _build_df([
        {"Time": "2024-01-01 09:00", "Open": 0.9995, "High": 0.9998, "Low": 0.9992, "Close": 0.9995},
        {"Time": "2024-01-01 10:00", "Open": 0.9995, "High": 1.0000, "Low": 0.9990, "Close": 1.0000},
        {"Time": "2024-01-03 14:00", "Open": 1.0000, "High": 1.0025, "Low": 0.9999, "Close": 1.0020},
    ])
    signals = pd.Series([1, 0, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=240,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
    )
    trade = result["trades"][0]
    assert "nights_held" in trade
    assert trade["nights_held"] == 2  # 01→03 = 2 jours civils


# ═══════════════════════════════════════════════════════════════════════════
# Tests pour _simulate_stateful_core (asset_cfg)
# ═══════════════════════════════════════════════════════════════════════════


def test_stateful_no_asset_cfg_no_swap() -> None:
    """Sans asset_cfg, aucun swap n'est appliqué (rétrocompatibilité)."""
    n = 3
    dates = pd.DatetimeIndex(pd.to_datetime([
        "2024-01-01 00:00", "2024-01-02 00:00", "2024-01-03 00:00"
    ], utc=True))
    closes = np.array([1.0000, 1.0010, 1.0020])
    highs = closes + 0.0005
    lows = closes - 0.0005
    signals = np.array([1, 0, 0])
    weights = np.array([1.0, 1.0, 1.0])
    spreads = np.zeros(n)
    filter_rejected = np.array(["", "", ""])

    trades = _simulate_stateful_core(
        n=n,
        dates=dates,
        highs=highs,
        lows=lows,
        closes=closes,
        signals=signals,
        weights=weights,
        spreads=spreads,
        filter_rejected_arr=filter_rejected,
        tp_dist=0.0020,
        sl_dist=0.0010,
        spread_cost_base=0.0,
        window=2,
        pip_size=0.0001,
        asset_cfg=None,
    )
    assert len(trades) == 1
    assert "nights_held" in trades[0]
    # Le trade timeout après 2 jours, mais asset_cfg=None → pas de modif du PnL
    assert "position_size_lots" not in trades[0]


def test_stateful_long_swap_applied() -> None:
    """Avec asset_cfg, swap appliqué au long."""
    n = 4
    dates = pd.DatetimeIndex(pd.to_datetime([
        "2024-01-01 00:00", "2024-01-02 00:00", "2024-01-03 00:00", "2024-01-04 00:00"
    ], utc=True))
    closes = np.array([1.0000, 1.0005, 1.0008, 1.0010])  # remontée graduelle
    highs = closes + 0.0001  # jamais TP (0.0020 au-dessus)
    lows = closes - 0.0001
    signals = np.array([1, 0, 0, 0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    spreads = np.zeros(n)
    filter_rejected = np.array(["", "", "", ""])

    cfg = _make_asset_cfg(swap_long=-0.5, swap_short=+0.2)

    trades = _simulate_stateful_core(
        n=n,
        dates=dates,
        highs=highs,
        lows=lows,
        closes=closes,
        signals=signals,
        weights=weights,
        spreads=spreads,
        filter_rejected_arr=filter_rejected,
        tp_dist=0.0020,
        sl_dist=0.0010,
        spread_cost_base=0.0,
        window=3,
        pip_size=0.0001,
        asset_cfg=cfg,
    )
    assert len(trades) == 1
    trade = trades[0]
    # 3 nuits (01→04). Swap long = -0.5 → débit total = -1.5 pips.
    assert trade["nights_held"] == 3
    # PnL_brut = (closes[3] - closes[0]) / pip_size = 10 pips
    # PnL_final = 10 + 3*(-0.5) = 8.5
    assert trade["Pips_Bruts"] == pytest.approx(8.5)


def test_stateful_short_uses_short_swap() -> None:
    """Short utilise swap_short_pips_per_night, ignore swap_long."""
    n = 3
    dates = pd.DatetimeIndex(pd.to_datetime([
        "2024-01-01 00:00", "2024-01-02 00:00", "2024-01-03 00:00"
    ], utc=True))
    closes = np.array([1.0000, 1.0001, 1.0002])  # prix monte (perte pour short timeout)
    highs = closes + 0.0001
    lows = closes - 0.0001
    signals = np.array([-1, 0, 0])
    weights = np.array([1.0, 1.0, 1.0])
    spreads = np.zeros(n)
    filter_rejected = np.array(["", "", ""])

    cfg = _make_asset_cfg(swap_long=-99.0, swap_short=+0.4)

    trades = _simulate_stateful_core(
        n=n,
        dates=dates,
        highs=highs,
        lows=lows,
        closes=closes,
        signals=signals,
        weights=weights,
        spreads=spreads,
        filter_rejected_arr=filter_rejected,
        tp_dist=0.0020,
        sl_dist=0.0010,
        spread_cost_base=0.0,
        window=2,
        pip_size=0.0001,
        asset_cfg=cfg,
    )
    trade = trades[0]
    assert trade["nights_held"] == 2
    # PnL_brut short = (entry - exit) / pip_size = (1.0000 - 1.0002) / 0.0001 = -2
    # + 2 nuits × +0.4 (swap short crédit) = -2 + 0.8 = -1.2
    assert trade["Pips_Bruts"] == pytest.approx(-1.2)


def test_stateful_intraday_no_swap_applied() -> None:
    """Trade qui ouvre et ferme le même jour : aucune nuit, swap ignoré."""
    n = 3
    dates = pd.DatetimeIndex(pd.to_datetime([
        "2024-01-01 09:00", "2024-01-01 10:00", "2024-01-01 11:00"
    ], utc=True))
    closes = np.array([1.0000, 1.0010, 1.0025])  # TP touché à barre 2
    highs = np.array([1.0005, 1.0015, 1.0030])
    lows = np.array([0.9995, 1.0005, 1.0020])
    signals = np.array([1, 0, 0])
    weights = np.array([1.0, 1.0, 1.0])
    spreads = np.zeros(n)
    filter_rejected = np.array(["", "", ""])

    cfg = _make_asset_cfg(swap_long=-999.0)  # gros débit, mais 0 nuit

    trades = _simulate_stateful_core(
        n=n,
        dates=dates,
        highs=highs,
        lows=lows,
        closes=closes,
        signals=signals,
        weights=weights,
        spreads=spreads,
        filter_rejected_arr=filter_rejected,
        tp_dist=0.0020,
        sl_dist=0.0010,
        spread_cost_base=0.0,
        window=5,
        pip_size=0.0001,
        asset_cfg=cfg,
    )
    trade = trades[0]
    assert trade["nights_held"] == 0
    # TP = 20 pips, pas de swap car 0 nuit
    assert trade["Pips_Bruts"] == pytest.approx(20.0)
