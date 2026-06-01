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


def simulate_pairs_honest(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    asset_config_a: AssetConfig,
    asset_config_b: AssetConfig,
    *,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    time_stop_bars: int = 30,
    beta_lookback: int = 60,
    zscore_lookback: int = 60,
    notional_per_leg_eur: float = 10_000.0,
    swap_scale: float = 1.0,
) -> list[dict]:
    """Pairs trading mean-reversion — variante HONNÊTE (Phase 1 recherche d'edge).

    Corrige les deux optimismes de ``simulate_pairs_trades`` :

    1. **Fill honnête** — le signal (z-score) est calculé sur ``Close[i]`` mais
       l'exécution se fait à l'``Open[i+1]`` (on ne peut pas trader au prix de
       clôture qui a généré le signal). Pas de look-ahead.
    2. **Sizing équilibré (dollar-neutral)** — chaque jambe reçoit le MÊME
       notionnel ``notional_per_leg_eur``. Le PnL est calculé en *rendement de
       prix* (``Δprice/entry_price``), donc indépendant des conventions
       pip_size/pip_value_eur : on évite la jambe argent qui écrase l'or ×10.

    Convention spread = a − β·b, signal identique au module legacy :
        - z > z_entry  → SHORT spread : short A, long  B (signal = −1)
        - z < −z_entry → LONG  spread : long  A, short B (signal = +1)
        - Exit : |z| < z_exit (mean_reversion) OU bars_held ≥ time_stop_bars.

    β/z servent UNIQUEMENT de déclencheur ; le sizing reste dollar-neutral
    (plus robuste qu'un sizing β qui dépendrait du β rolling bruité).

    Coûts (par jambe, en fraction de prix) :
        ``total_cost_pips × pip_size / entry_price`` (aller-retour) ; le swap
        signé s'applique par nuit civile UTC sur les DEUX jambes (sur CFD on
        paie le financement des deux côtés).

    Args:
        df_a, df_b: OHLCV indexés tz-aware UTC (colonne ``Open`` requise).
        asset_config_a, asset_config_b: Configs broker des deux jambes.
        z_entry, z_exit: Seuils d'entrée/sortie en σ.
        time_stop_bars: Time-stop (nombre de bars depuis l'exécution d'entrée).
        beta_lookback, zscore_lookback: Fenêtres rolling.
        notional_per_leg_eur: Notionnel € par jambe (gross = 2× ; net ≈ 0).
        swap_scale: Multiplicateur appliqué au swap (1.0 = swap réel ; 0.0 =
            aucun swap, pour tester la sensibilité du verdict au coût de nuit).

    Returns:
        Liste de dicts trades. Chaque trade porte ``pnl_eur_net`` ET un alias
        ``pips_net`` (= pnl_eur_net) pour ``sharpe_daily_from_trades``.
        Une position encore ouverte en fin de données n'est PAS enregistrée.
    """
    if not isinstance(df_a.index, pd.DatetimeIndex):
        raise TypeError("df_a.index doit être DatetimeIndex")
    if df_a.index.tz is None or df_b.index.tz is None:
        raise ValueError("df_a.index et df_b.index doivent être tz-aware (UTC)")
    if "Open" not in df_a.columns or "Open" not in df_b.columns:
        raise KeyError("df_a et df_b doivent contenir une colonne 'Open'")

    common_idx = df_a.index.intersection(df_b.index)
    if len(common_idx) < 2:
        return []
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]

    close_a = df_a["Close"]
    close_b = df_b["Close"]
    beta = compute_rolling_beta(close_a, close_b, lookback=beta_lookback)
    spread = compute_spread(close_a, close_b, beta)
    z = compute_zscore(spread, lookback=zscore_lookback)

    # Arrays pour un accès rapide (décision sur Close[i], exécution sur Open[i+1]).
    open_a = df_a["Open"].to_numpy(dtype=float)
    open_b = df_b["Open"].to_numpy(dtype=float)
    z_arr = z.to_numpy(dtype=float)
    beta_arr = beta.to_numpy(dtype=float)
    ps_a, ps_b = asset_config_a.pip_size, asset_config_b.pip_size
    cost_pips_a = asset_config_a.total_cost_pips
    cost_pips_b = asset_config_b.total_cost_pips
    n = len(common_idx)

    trades: list[dict] = []
    position = 0
    entry_exec_bar = -1
    entry_ts: pd.Timestamp | None = None
    entry_a = entry_b = entry_z = entry_beta = 0.0

    for i in range(n - 1):  # i = barre de DÉCISION ; exécution à i+1
        zi = z_arr[i]

        # ── Sortie (si position ouverte) ─────────────────────────────
        if position != 0:
            bars_held = i - entry_exec_bar
            do_exit, reason = False, ""
            if np.isfinite(zi) and abs(zi) <= z_exit:
                do_exit, reason = True, "mean_reversion"
            elif bars_held >= time_stop_bars:
                do_exit, reason = True, "time_stop"

            if do_exit:
                exit_ts = common_idx[i + 1]
                exit_a = open_a[i + 1]
                exit_b = open_b[i + 1]
                ret_a = (exit_a - entry_a) / entry_a
                ret_b = (exit_b - entry_b) / entry_b
                gross_eur = position * (ret_a - ret_b) * notional_per_leg_eur

                cost_eur = (
                    cost_pips_a * ps_a / entry_a + cost_pips_b * ps_b / entry_b
                ) * notional_per_leg_eur

                nights = max(0, (exit_ts.normalize() - entry_ts.normalize()).days)
                if position == 1:  # long A, short B
                    swap_pips_a = asset_config_a.swap_long_pips_per_night
                    swap_pips_b = asset_config_b.swap_short_pips_per_night
                else:              # short A, long B
                    swap_pips_a = asset_config_a.swap_short_pips_per_night
                    swap_pips_b = asset_config_b.swap_long_pips_per_night
                swap_eur = nights * (
                    swap_pips_a * ps_a / entry_a + swap_pips_b * ps_b / entry_b
                ) * notional_per_leg_eur * swap_scale

                net_eur = gross_eur - cost_eur + swap_eur
                trades.append({
                    "entry_time": entry_ts.isoformat(),
                    "exit_time": exit_ts.isoformat(),
                    "signal": position,
                    "entry_zscore": float(entry_z),
                    "exit_zscore": float(zi) if np.isfinite(zi) else float("nan"),
                    "entry_beta": float(entry_beta),
                    "entry_price_a": float(entry_a),
                    "exit_price_a": float(exit_a),
                    "entry_price_b": float(entry_b),
                    "exit_price_b": float(exit_b),
                    "ret_a": float(ret_a),
                    "ret_b": float(ret_b),
                    "pnl_eur_brut": float(gross_eur),
                    "pnl_eur_net": float(net_eur),
                    "pips_net": float(net_eur),  # alias pour sharpe_daily_from_trades
                    "cost_eur": float(cost_eur),
                    "swap_eur": float(swap_eur),
                    "nights_held": int(nights),
                    "bars_held": int(bars_held),
                    "exit_reason": reason,
                })
                position = 0

        # ── Entrée (si flat) ─────────────────────────────────────────
        if position == 0:
            bi = beta_arr[i]
            if np.isfinite(zi) and np.isfinite(bi):
                if zi > z_entry:
                    position = -1
                elif zi < -z_entry:
                    position = 1
                if position != 0:
                    entry_exec_bar = i + 1
                    entry_ts = common_idx[i + 1]
                    entry_a = open_a[i + 1]
                    entry_b = open_b[i + 1]
                    entry_z = zi
                    entry_beta = bi

    logger.info(
        "pairs_simulated_honest",
        extra={"context": {
            "n_trades": len(trades),
            "n_bars": n,
            "z_entry": z_entry, "z_exit": z_exit,
            "time_stop_bars": time_stop_bars,
        }},
    )
    return trades
