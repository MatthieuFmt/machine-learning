"""Tests trend_pullback — port mécanique de la stratégie manuelle D1/H4.

Couvre : régime D1, anti-fuite (troncature + D1 de la veille), blocage par le
filtre D1, exécution honnête (entrée open suivant, SL prioritaire, gap),
marge de coûts. Fixtures synthétiques, < 100 ms/test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import AssetConfig
from app.strategies.trend_pullback import (
    _margined_costs,
    compute_trend_pullback_signals,
    d1_regime,
    simulate_trend_pullback_trades,
)

_CFG = AssetConfig(
    spread_pips=1.0,
    slippage_pips=0.25,
    commission_pips=0.0,
    pip_size=1.0,
    pip_value_eur=1.0,
    tp_points=20,
    sl_points=10,
    window_hours=120,
    swap_long_pips_per_night=0.0,
    swap_short_pips_per_night=0.0,
)


def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex,
                     spread_frac: float = 0.001) -> pd.DataFrame:
    """OHLC synthétique : Open = close précédent, High/Low engloblent les deux."""
    close = np.asarray(close, dtype=np.float64)
    open_ = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(open_, close) * (1.0 + spread_frac)
    lo = np.minimum(open_, close) * (1.0 - spread_frac)
    return pd.DataFrame(
        {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=index
    )


def _d1_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


def _h4_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")


def _trending_d1(n: int = 300, daily: float = 0.003) -> pd.DataFrame:
    close = 100.0 * np.cumprod(np.full(n, 1.0 + daily))
    return _ohlc_from_close(close, _d1_index(n))


def _oscillating_h4(n: int = 720, trend: float = 0.0006,
                    amp: float = 0.012, period: int = 36) -> pd.DataFrame:
    """Tendance haussière + oscillation : produit replis ET croisements RSI."""
    k = np.arange(n)
    close = 100.0 * np.cumprod(np.full(n, 1.0 + trend)) * (
        1.0 + amp * np.sin(2.0 * np.pi * k / period)
    )
    return _ohlc_from_close(close, _h4_index(n))


def _d1_resampled_from_h4(df_h4: pd.DataFrame) -> pd.DataFrame:
    d1 = df_h4.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    return d1


# ═══════════════════════════════════════════════════════════════════════════
# Régime D1
# ═══════════════════════════════════════════════════════════════════════════


def test_d1_regime_bull_on_uptrend() -> None:
    df = _trending_d1(300, daily=0.003)
    regime = d1_regime(df)
    assert int(regime.iloc[-1]) == 1
    # majorité haussière une fois les EMA installées
    assert (regime.iloc[220:] == 1).mean() > 0.9


def test_d1_regime_bear_on_downtrend() -> None:
    df = _trending_d1(300, daily=-0.003)
    regime = d1_regime(df)
    assert int(regime.iloc[-1]) == -1


def test_d1_regime_neutral_on_flat() -> None:
    close = np.full(300, 100.0)
    df = _ohlc_from_close(close, _d1_index(300), spread_frac=0.0005)
    regime = d1_regime(df)
    # marché plat : jamais haussier ni baissier en fin d'échantillon
    assert int(regime.iloc[-1]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Signaux H4 + filtre D1
# ═══════════════════════════════════════════════════════════════════════════


def test_long_signals_in_bull_regime_no_shorts() -> None:
    df_h4 = _oscillating_h4()
    df_d1 = _d1_resampled_from_h4(df_h4)
    signals = compute_trend_pullback_signals(df_h4, df_d1)
    assert (signals == 1).sum() >= 1, "tendance + replis → au moins un achat"
    assert (signals == -1).sum() == 0, "régime D1 haussier → aucune vente permise"


def test_signals_blocked_when_d1_flat() -> None:
    df_h4 = _oscillating_h4()
    flat_close = np.full(200, 100.0)
    df_d1 = _ohlc_from_close(flat_close, _d1_index(200), spread_frac=0.0005)
    signals = compute_trend_pullback_signals(df_h4, df_d1)
    assert (signals != 0).sum() == 0, "régime D1 neutre → zéro trade"


def test_no_lookahead_truncation_invariance() -> None:
    """Règle 6 (CLAUDE.md §5) : signal(df[:n])[-1] == signal(df)[n-1]."""
    df_h4 = _oscillating_h4(720)
    df_d1 = _d1_resampled_from_h4(df_h4)
    full = compute_trend_pullback_signals(df_h4, df_d1)
    for n in (430, 555, 718):
        cut_ts = df_h4.index[n - 1]
        h4_trunc = df_h4.iloc[:n]
        d1_trunc = df_d1[df_d1.index <= cut_ts]
        trunc = compute_trend_pullback_signals(h4_trunc, d1_trunc)
        assert int(trunc.iloc[-1]) == int(full.iloc[n - 1]), f"fuite détectée à n={n}"


def test_d1_regime_used_is_previous_day() -> None:
    """Le régime du jour J ne doit JAMAIS utiliser la barre D1 de J."""
    df_h4 = _oscillating_h4(720)
    df_d1 = _d1_resampled_from_h4(df_h4)
    full = compute_trend_pullback_signals(df_h4, df_d1)
    # Modifier la barre D1 du DERNIER jour ne change aucun signal de ce jour.
    df_d1_mod = df_d1.copy()
    df_d1_mod.iloc[-1, df_d1_mod.columns.get_loc("Close")] = 1.0  # krach fictif
    mod = compute_trend_pullback_signals(df_h4, df_d1_mod)
    last_day = df_h4.index.normalize() == df_d1.index[-1].normalize()
    pd.testing.assert_series_equal(full[last_day], mod[last_day])


# ═══════════════════════════════════════════════════════════════════════════
# Simulation (exécution honnête)
# ═══════════════════════════════════════════════════════════════════════════


def _flat_frame_with_event(n: int = 60, event_i: int = 50,
                           event_high: float = 110.0,
                           event_low: float = 90.0,
                           event_open: float = 100.0) -> pd.DataFrame:
    """Barres plates (ATR=2 exactement) + une barre 'événement' large en event_i."""
    open_ = np.full(n, 100.0)
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    high[event_i], low[event_i], open_[event_i] = event_high, event_low, event_open
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=_h4_index(n),
    )


def _inject_signal(df: pd.DataFrame, i: int, direction: int) -> pd.Series:
    sig = pd.Series(0, index=df.index, dtype="int64")
    sig.iloc[i] = direction
    return sig


def test_entry_next_open_and_sl_priority_same_bar() -> None:
    # ATR(14) de barres plates H-L=2 → 2.0 ; signal close=100 → SL 97 / TP 106.
    df = _flat_frame_with_event(event_i=50, event_high=107.0, event_low=96.0)
    sig = _inject_signal(df, 48, +1)
    trades = simulate_trend_pullback_trades(df, df, _CFG, signals=sig)
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_price"] == pytest.approx(df["Open"].iloc[49])
    assert t["entry_time"] == df.index[49].isoformat()
    assert t["sl_price"] == pytest.approx(97.0)
    assert t["tp_price"] == pytest.approx(106.0)
    # la barre 50 touche TP (107) ET SL (96) → SL gagne, fill au stop exact
    assert t["exit_reason"] == "sl"
    assert t["exit_price"] == pytest.approx(97.0)


def test_gap_through_sl_fills_at_open() -> None:
    # Gap d'ouverture sous le SL (97) → sortie à l'open 95 (pire), pas au stop.
    df = _flat_frame_with_event(event_i=50, event_high=101.0,
                                event_low=94.0, event_open=95.0)
    sig = _inject_signal(df, 48, +1)
    trades = simulate_trend_pullback_trades(df, df, _CFG, signals=sig)
    t = trades[0]
    assert t["exit_reason"] == "sl"
    assert t["exit_price"] == pytest.approx(95.0)
    assert t["pips_brut"] == pytest.approx(95.0 - 100.0)


def test_tp_exit_when_only_tp_hit() -> None:
    df = _flat_frame_with_event(event_i=50, event_high=106.5, event_low=99.0)
    sig = _inject_signal(df, 48, +1)
    trades = simulate_trend_pullback_trades(df, df, _CFG, signals=sig)
    t = trades[0]
    assert t["exit_reason"] == "tp"
    assert t["exit_price"] == pytest.approx(106.0)  # gap favorable → TP conservateur
    assert t["pips_brut"] == pytest.approx(6.0)


def test_cost_margin_degrades_net() -> None:
    df = _flat_frame_with_event(event_i=50, event_high=106.5, event_low=99.0)
    sig = _inject_signal(df, 48, +1)
    t1 = simulate_trend_pullback_trades(df, df, _CFG, signals=sig)[0]
    t15 = simulate_trend_pullback_trades(
        df, df, _CFG, signals=sig, cost_multiplier=1.5
    )[0]
    # coût a/r : (1 + 2×0.25) = 1.5 pips → ×1.5 = 2.25 pips
    assert t1["pips_net"] == pytest.approx(6.0 - 1.5)
    assert t15["pips_net"] == pytest.approx(6.0 - 2.25)
    assert t15["pips_net"] < t1["pips_net"]


def test_margined_costs_helper() -> None:
    cfg = AssetConfig(
        spread_pips=1.0, slippage_pips=0.25, commission_pips=0.0,
        pip_size=1.0, pip_value_eur=1.0, tp_points=20, sl_points=10,
        window_hours=120,
        swap_long_pips_per_night=-2.0, swap_short_pips_per_night=1.0,
    )
    cost, swl, sws = _margined_costs(cfg, 1.5)
    assert cost == pytest.approx(1.5 * 1.5)
    assert swl == pytest.approx(-3.0)   # débit aggravé
    assert sws == pytest.approx(1.0 / 1.5)  # crédit réduit
    with pytest.raises(ValueError):
        _margined_costs(cfg, 0.9)


def test_one_position_at_a_time() -> None:
    # Deux signaux pendant qu'une position serait ouverte → un seul trade.
    df = _flat_frame_with_event(event_i=55, event_high=107.0, event_low=96.0)
    sig = pd.Series(0, index=df.index, dtype="int64")
    sig.iloc[48] = 1
    sig.iloc[50] = 1  # ignoré : position ouverte
    trades = simulate_trend_pullback_trades(df, df, _CFG, signals=sig)
    assert len(trades) == 1
