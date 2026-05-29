"""Tests de cohérence du sizing + métriques après pivot v4 A1.

8 tests : sizing US30/EURUSD, TP/SL exacts, DD borné, Sharpe zéro,
compound 10 SL, ValueError SL==entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.metrics import compute_metrics
from app.backtest.sizing import compute_position_size, expected_pnl_eur
from app.config.instruments import ASSET_CONFIGS, AssetConfig

# ── Fixtures ────────────────────────────────────────────────────────────────

US30_CFG = ASSET_CONFIGS["US30"]
EURUSD_CFG = AssetConfig(
    spread_pips=0.9,
    slippage_pips=0.3,
    pip_size=0.0001,
    pip_value_eur=10.0,
    min_lot=0.01,
    max_lot=10.0,
    tp_points=20,
    sl_points=10,
)


def _make_trades_df(
    pips_nets: list[float],
    pips_bruts: list[float] | None = None,
    position_sizes: list[float] | None = None,
    freq: str = "D",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Fabrique un DataFrame de trades minimal."""
    n = len(pips_nets)
    if pips_bruts is None:
        pips_bruts = pips_nets
    if position_sizes is None:
        position_sizes = [1.0] * n
    index = pd.date_range(start, periods=n, freq=freq, name="Time")
    return pd.DataFrame(
        {
            "Pips_Nets": pips_nets,
            "Pips_Bruts": pips_bruts,
            "position_size_lots": position_sizes,
            "Weight": [1.0] * n,
            "result": ["win" if p > 0 else "loss_sl" for p in pips_nets],
            "filter_rejected": [""] * n,
        },
        index=index,
    )


# ── Tests sizing ────────────────────────────────────────────────────────────


def test_sizing_us30_100pt_sl() -> None:
    """SL = 100 points US30 → ~2.17 lots pour 200 € de risque sur 10 000 €."""
    lots = compute_position_size(
        entry_price=40000.0,
        stop_loss_price=39900.0,
        capital_eur=10_000.0,
        risk_pct=0.02,
        asset_cfg=US30_CFG,
    )
    expected = 200.0 / (100.0 * 0.92)
    assert lots == pytest.approx(expected, rel=1e-2)


def test_sizing_eurusd_10pip_sl() -> None:
    """SL = 10 pips EURUSD → 2.0 lots pour 200 € de risque."""
    lots = compute_position_size(
        entry_price=1.1000,
        stop_loss_price=1.0990,
        capital_eur=10_000.0,
        risk_pct=0.02,
        asset_cfg=EURUSD_CFG,
    )
    assert lots == pytest.approx(2.0, rel=1e-2)


def test_sizing_invalid_sl_equals_entry() -> None:
    """SL = entry → ValueError."""
    with pytest.raises(ValueError, match="SL nul"):
        compute_position_size(
            entry_price=40000.0,
            stop_loss_price=40000.0,
            capital_eur=10_000.0,
            risk_pct=0.02,
            asset_cfg=US30_CFG,
        )


def test_sizing_min_lot_clamp() -> None:
    """SL tellement large que lots < 0.005 → round à 0.00 → clamp à min_lot."""
    lots = compute_position_size(
        entry_price=40000.0,
        stop_loss_price=10000.0,  # 30000 pts SL
        capital_eur=10_000.0,
        risk_pct=0.02,
        asset_cfg=US30_CFG,
    )
    # 200 / (30000 × 0.92) = 0.00725 → round à 0.01 → clamp à 0.01 (min_lot)
    assert lots == pytest.approx(0.01, rel=1e-2)


# ── Tests PnL € ─────────────────────────────────────────────────────────────


def test_trade_sl_exact_minus2pct() -> None:
    """Trade SL → −2 % du capital exactement (approximation)."""
    # Pips nets = -100, position = 2.17 lots, pip_value = 0.92
    pnl = expected_pnl_eur(-100.0, 2.17, US30_CFG)
    assert pnl == pytest.approx(-200.0, rel=0.05)


def test_trade_tp_exact_4pct() -> None:
    """Trade TP ratio R:R=2 → +4 % du capital (risque 2 % × 2)."""
    pnl = expected_pnl_eur(200.0, 2.17, US30_CFG)
    assert pnl == pytest.approx(400.0, rel=0.05)


# ── Tests metrics (mode A1) ─────────────────────────────────────────────────


def test_dd_bounded_minus_100() -> None:
    """60 trades SL consécutifs → DD borné ≥ −100 %, blowup détecté."""
    trades_df = _make_trades_df(
        pips_nets=[-100.0] * 60,
        position_sizes=[2.17] * 60,
    )
    metrics = compute_metrics(trades_df, asset_cfg=US30_CFG, capital_eur=10_000.0)
    assert metrics["max_dd_pct"] >= -100.0
    assert metrics["max_dd_pct"] < 0
    # 60 × −199.64 € = −11,978 € → equity < 0.01 → blowup
    assert metrics["blowup_detected"] is True


def test_sharpe_equity_plate_returns_zero() -> None:
    """Trades tous à PnL=0 → Sharpe = 0 (pas NaN)."""
    trades_df = _make_trades_df(
        pips_nets=[0.0] * 30,
        position_sizes=[1.0] * 30,
    )
    metrics = compute_metrics(trades_df, asset_cfg=US30_CFG, capital_eur=10_000.0)
    assert metrics["sharpe"] == 0.0
    assert not np.isnan(metrics["sharpe"])


def test_10_sl_compound_dd() -> None:
    """10 SL consécutifs = DD ≈ −18 à −22 % (sizing fixe, pas compound parfait)."""
    trades_df = _make_trades_df(
        pips_nets=[-100.0] * 10,
        position_sizes=[2.17] * 10,
    )
    metrics = compute_metrics(trades_df, asset_cfg=US30_CFG, capital_eur=10_000.0)
    assert -22.0 <= metrics["max_dd_pct"] <= -18.0


def test_equity_positive_after_wins() -> None:
    """10 TP → equity finale > capital initial."""
    trades_df = _make_trades_df(
        pips_nets=[200.0] * 10,
        position_sizes=[2.17] * 10,
    )
    metrics = compute_metrics(trades_df, asset_cfg=US30_CFG, capital_eur=10_000.0)
    assert metrics["final_equity_eur"] > 10_000.0
    assert metrics["total_return_pct"] > 0


# ── Test retrocompatibilité legacy ──────────────────────────────────────────


def test_legacy_no_asset_cfg() -> None:
    """Sans asset_cfg, compute_metrics fonctionne en mode legacy."""
    trades_df = _make_trades_df(
        pips_nets=[10.0, -5.0, 15.0],
        position_sizes=None,  # pas de colonne position_size_lots
    )
    # Drop position_size_lots pour simuler l'ancien format
    trades_df = trades_df.drop(columns=["position_size_lots"])
    metrics = compute_metrics(trades_df, pip_value_eur=1.0, initial_capital=10_000.0)
    assert metrics["trades"] == 3
    assert "profit_net_eur" not in metrics  # pas en mode A1
    assert "blowup_detected" not in metrics


# ── Test mode A1 sans position_size_lots → ValueError ───────────────────────


def test_missing_position_size_lots_raises() -> None:
    """Mode A1 sans colonne position_size_lots → ValueError."""
    trades_df = _make_trades_df(
        pips_nets=[10.0, -5.0],
        position_sizes=[1.0, 1.0],
    )
    trades_df = trades_df.drop(columns=["position_size_lots"])
    with pytest.raises(ValueError, match="position_size_lots"):
        compute_metrics(trades_df, asset_cfg=US30_CFG)
