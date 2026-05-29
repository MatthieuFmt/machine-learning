"""Pairs Trading — stratégie statistique mean-reversion (Phase H4).

Hypothèse théorique : deux instruments cointégrés (test Engle-Granger)
oscillent autour d'une relation linéaire stable. Quand le spread résiduel
dévie fortement (|z| > 2), on parie sur le retour à la moyenne.

Stratégie V1 (long ET short spread symétriques) :
    - β rolling estimé par OLS sur `beta_lookback` H4 bars (60 = ~10 jours).
    - Spread = `price_a - β × price_b`.
    - Z-score = `(spread - rolling_mean) / rolling_std`, window `zscore_lookback`.
    - Entry : `|z| > z_entry` (défaut 2.0).
        * z > z_entry → spread cher → SHORT spread : sell A, buy B (signal = -1).
        * z < -z_entry → spread bas → LONG spread : buy A, sell B (signal = +1).
    - Exit : `|z| < z_exit` (défaut 0.5) OU `bars_held >= time_stop_bars` (défaut 30).
    - 1 position max simultanée (pas de moyennage).

Sizing V1 : equal-dollar (1 lot par jambe, β n'intervient pas dans le sizing).
PnL en EUR par jambe via `pip_value_eur` de chaque AssetConfig.
Coûts par jambe : `spread + 2 × slippage` × pip_value_eur (round-trip).
Swap : nights_held × signed swap per leg.

⚠️ Requires statsmodels (test cointegration externe). Le module lui-même
n'importe pas statsmodels — le test de cointegration est lancé depuis
le script lanceur pour permettre fallback gracieux.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config.instruments import AssetConfig
from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_rolling_beta(
    series_a: pd.Series,
    series_b: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """OLS rolling β : a = β·b + ε.

    Pour chaque t, β = cov(a, b) / var(b) sur la fenêtre [t-lookback+1, t].

    Args:
        series_a, series_b: Séries de prix indexées tz-aware UTC, même index.
        lookback: Taille de la fenêtre rolling (défaut 60 = ~10 jours H4).

    Returns:
        pd.Series β indexée comme series_a. Premières `lookback-1` valeurs = NaN.
    """
    if not isinstance(series_a.index, pd.DatetimeIndex):
        raise TypeError("series_a.index doit être DatetimeIndex")
    if series_a.index.tz is None:
        raise ValueError("series_a.index doit être tz-aware (UTC)")
    if not series_b.index.equals(series_a.index):
        raise ValueError("series_a et series_b doivent avoir le même index")

    cov_ab = series_a.rolling(window=lookback, min_periods=lookback).cov(series_b)
    var_b = series_b.rolling(window=lookback, min_periods=lookback).var()
    beta = cov_ab / var_b.replace(0, np.nan)
    return beta


def compute_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Spread = a - β·b. Index aligné sur a."""
    return series_a - beta * series_b


def compute_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """Z-score rolling : (spread - mean) / std sur fenêtre `lookback`."""
    mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    std = spread.rolling(window=lookback, min_periods=lookback).std()
    return (spread - mean) / std.replace(0, np.nan)


def simulate_pairs_trades(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    asset_config_a: AssetConfig,
    asset_config_b: AssetConfig,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    time_stop_bars: int = 30,
    beta_lookback: int = 60,
    zscore_lookback: int = 60,
) -> list[dict]:
    """Simule pairs trading mean-reversion.

    Convention spread = a - β·b :
        - z > z_entry  → SHORT spread : SELL a, BUY b (signal = -1)
        - z < -z_entry → LONG spread  : BUY a, SELL b (signal = +1)
        - Exit : |z| < z_exit OU bars_held ≥ time_stop_bars

    Sizing V1 equal-dollar : 1 lot par jambe.
    PnL en EUR via `asset_config.pip_value_eur`.
    Coûts round-trip par jambe + swap signé par nuit holding.

    Args:
        df_a, df_b: OHLCV H4 indexés tz-aware UTC, mêmes timestamps requis.
        asset_config_a, asset_config_b: Configs broker des deux jambes.
        z_entry, z_exit: Seuils d'entrée/sortie en σ.
        time_stop_bars: Time-stop en nombre de bars H4.
        beta_lookback, zscore_lookback: Fenêtres rolling.

    Returns:
        Liste de dicts trades.
    """
    if not isinstance(df_a.index, pd.DatetimeIndex):
        raise TypeError("df_a.index doit être DatetimeIndex")
    if df_a.index.tz is None or df_b.index.tz is None:
        raise ValueError("df_a.index et df_b.index doivent être tz-aware (UTC)")

    common_idx = df_a.index.intersection(df_b.index)
    if len(common_idx) == 0:
        return []
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]

    close_a = df_a["Close"]
    close_b = df_b["Close"]

    beta = compute_rolling_beta(close_a, close_b, lookback=beta_lookback)
    spread = compute_spread(close_a, close_b, beta)
    z = compute_zscore(spread, lookback=zscore_lookback)

    # Coûts round-trip par jambe (en pips)
    cost_a_pips = (
        asset_config_a.spread_pips
        + 2 * asset_config_a.slippage_pips
        + 2 * asset_config_a.commission_pips
    )
    cost_b_pips = (
        asset_config_b.spread_pips
        + 2 * asset_config_b.slippage_pips
        + 2 * asset_config_b.commission_pips
    )
    cost_a_eur = cost_a_pips * asset_config_a.pip_value_eur
    cost_b_eur = cost_b_pips * asset_config_b.pip_value_eur
    cost_total_eur = cost_a_eur + cost_b_eur

    trades: list[dict] = []

    position: int = 0
    entry_idx: int | None = None
    entry_ts: pd.Timestamp | None = None
    entry_z: float | None = None
    entry_beta: float | None = None
    entry_price_a: float | None = None
    entry_price_b: float | None = None

    for i in range(len(common_idx)):
        ts = common_idx[i]
        z_now_raw = z.iloc[i]
        if pd.isna(z_now_raw):
            continue
        z_now = float(z_now_raw)

        # ── Exit logic (si position ouverte) ─────────────────────────
        if position != 0 and entry_idx is not None:
            bars_held = i - entry_idx
            exit_triggered = False
            exit_reason = ""

            if abs(z_now) <= z_exit:
                exit_triggered, exit_reason = True, "mean_reversion"
            elif bars_held >= time_stop_bars:
                exit_triggered, exit_reason = True, "time_stop"

            if exit_triggered:
                exit_price_a = float(close_a.iloc[i])
                exit_price_b = float(close_b.iloc[i])

                # PnL par jambe (en pips, puis EUR)
                if position == 1:
                    # Long spread : LONG a, SHORT b
                    pips_a = (exit_price_a - entry_price_a) / asset_config_a.pip_size
                    pips_b = (entry_price_b - exit_price_b) / asset_config_b.pip_size
                else:
                    # Short spread : SHORT a, LONG b
                    pips_a = (entry_price_a - exit_price_a) / asset_config_a.pip_size
                    pips_b = (exit_price_b - entry_price_b) / asset_config_b.pip_size

                pnl_eur_a = pips_a * asset_config_a.pip_value_eur
                pnl_eur_b = pips_b * asset_config_b.pip_value_eur
                pnl_eur_brut = pnl_eur_a + pnl_eur_b

                # Swap (signed per leg, par nuit civile UTC)
                nights_held = max(0, (ts.normalize() - entry_ts.normalize()).days)
                if position == 1:
                    swap_pips_a = asset_config_a.swap_long_pips_per_night
                    swap_pips_b = asset_config_b.swap_short_pips_per_night
                else:
                    swap_pips_a = asset_config_a.swap_short_pips_per_night
                    swap_pips_b = asset_config_b.swap_long_pips_per_night
                swap_eur_a = nights_held * swap_pips_a * asset_config_a.pip_value_eur
                swap_eur_b = nights_held * swap_pips_b * asset_config_b.pip_value_eur
                swap_eur_total = swap_eur_a + swap_eur_b

                pnl_eur_net = pnl_eur_brut - cost_total_eur + swap_eur_total

                trades.append({
                    "entry_time": entry_ts.isoformat(),
                    "exit_time": ts.isoformat(),
                    "signal": position,
                    "entry_zscore": float(entry_z),
                    "exit_zscore": z_now,
                    "entry_beta": float(entry_beta),
                    "entry_price_a": float(entry_price_a),
                    "exit_price_a": exit_price_a,
                    "entry_price_b": float(entry_price_b),
                    "exit_price_b": exit_price_b,
                    "pips_brut_a": float(pips_a),
                    "pips_brut_b": float(pips_b),
                    "pnl_eur_brut": float(pnl_eur_brut),
                    "pnl_eur_net": float(pnl_eur_net),
                    "cost_eur": float(cost_total_eur),
                    "swap_eur": float(swap_eur_total),
                    "nights_held": int(nights_held),
                    "bars_held": int(bars_held),
                    "exit_reason": exit_reason,
                })

                position = 0
                entry_idx = None
                entry_ts = None

        # ── Entry logic (si flat seulement) ──────────────────────────
        if position == 0:
            beta_now_raw = beta.iloc[i]
            if pd.isna(beta_now_raw):
                continue
            if z_now > z_entry:
                position = -1
            elif z_now < -z_entry:
                position = 1
            else:
                continue
            entry_idx = i
            entry_ts = ts
            entry_z = z_now
            entry_beta = float(beta_now_raw)
            entry_price_a = float(close_a.iloc[i])
            entry_price_b = float(close_b.iloc[i])

    logger.info(
        "pairs_simulated",
        extra={"context": {
            "n_trades": len(trades),
            "n_bars": len(common_idx),
            "z_entry": z_entry, "z_exit": z_exit,
            "time_stop_bars": time_stop_bars,
        }},
    )
    return trades
