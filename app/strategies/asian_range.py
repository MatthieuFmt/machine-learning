"""Asian Range Breakout — stratégie intraday haute fréquence (Phase H2).

Hypothèse théorique : sur les paires forex liées à USD/JPY/CHF/EUR/GBP,
la session asiatique pré-Europe (00:00-06:00 UTC) consolide le prix dans
un range étroit. Le breakout confirmé à la close 07:00 UTC (juste avant
l'ouverture européenne) matérialise un mouvement directionnel qui se
poursuit souvent jusqu'à la fin de la session New York (~22:00 UTC).

Stratégie V1 (long ET short symétriques) :
    - Tokyo range = [High, Low] des 7 barres H1 [00:00, 06:00] UTC.
      Note : on exclut volontairement la barre 07:00 du range, car son
      Close sert au breakout — l'inclure rendrait Close(07:00) ≤ tokyo_high
      par construction OHLC (et donc breakout long impossible).
    - À la close de la barre 07:00 (= 8ème barre, hors range) :
        * Long si Close(07:00) > High(Tokyo).
        * Short si Close(07:00) < Low(Tokyo).
        * Sinon : pas de trade ce jour-là.
    - Entry à l'Open de la barre 08:00.
    - TP = entry ± 1.5 × range, SL = entry ∓ 0.5 × range (R:R = 3:1).
    - Time-stop : Close de la barre 22:00 UTC si ni TP ni SL touché.
    - 1 signal max par jour (premier breakout détecté à 07:00).

Conventions :
    - Path-dependent : pour chaque barre [entry, time_stop], on check
      d'abord SL (conservateur), puis TP. Si les deux touchent la même
      barre, SL prioritaire.
    - Stratégie intraday stricte : nights_held = 0 attendu → swap nul,
      mais le calcul reste générique au cas où.
    - Coûts : cost_total = spread + 2 × (slippage + commission).
"""
from __future__ import annotations

import pandas as pd

from app.config.instruments import AssetConfig
from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_tokyo_range(
    df_h1: pd.DataFrame,
    start_hour_utc: int = 0,
    end_hour_utc: int = 6,
) -> pd.DataFrame:
    """Calcule le High/Low/range Tokyo par jour UTC.

    Args:
        df_h1: OHLCV H1 indexé DatetimeIndex tz-aware UTC.
        start_hour_utc: Heure de début incluse (défaut 0).
        end_hour_utc: Heure de fin incluse (défaut 6 → 7 barres 00:00-06:00).
            ⚠️ Doit être strictement < signal_hour_utc dans simulate, sinon
            le breakout par close serait impossible (cf. note module).

    Returns:
        DataFrame indexé par date (datetime.date), colonnes :
        tokyo_high, tokyo_low, tokyo_range. Jours avec moins de
        `(end - start + 1)` barres sont skippés.
    """
    if not isinstance(df_h1.index, pd.DatetimeIndex):
        raise TypeError("df_h1.index doit être DatetimeIndex")
    if df_h1.index.tz is None:
        raise ValueError("df_h1.index doit être tz-aware (UTC)")

    n_expected = end_hour_utc - start_hour_utc + 1
    hours = df_h1.index.hour
    mask = (hours >= start_hour_utc) & (hours <= end_hour_utc)
    tokyo = df_h1.loc[mask]
    if tokyo.empty:
        return pd.DataFrame(columns=["tokyo_high", "tokyo_low", "tokyo_range"])

    grouped = tokyo.groupby(tokyo.index.date).agg(
        tokyo_high=("High", "max"),
        tokyo_low=("Low", "min"),
        n_bars=("Close", "count"),
    )
    valid = grouped[grouped["n_bars"] == n_expected].copy()
    valid["tokyo_range"] = valid["tokyo_high"] - valid["tokyo_low"]
    return valid[["tokyo_high", "tokyo_low", "tokyo_range"]]


def simulate_asian_range_trades(
    df_h1: pd.DataFrame,
    asset_config: AssetConfig,
    tp_mult: float = 1.5,
    sl_mult: float = 0.5,
    time_stop_hour_utc: int = 22,
    signal_hour_utc: int = 7,
    entry_hour_utc: int = 8,
    tokyo_start_hour_utc: int = 0,
    tokyo_end_hour_utc: int = 6,
) -> list[dict]:
    """Simule la stratégie Asian Range Breakout (long + short).

    Args:
        df_h1: OHLCV H1 indexé tz-aware UTC, colonnes [Open, High, Low, Close].
        asset_config: Coûts et paramètres broker (spread, slippage, swap, pip_size).
        tp_mult: Multiplicateur TP appliqué au range Tokyo (défaut 1.5).
        sl_mult: Multiplicateur SL appliqué au range Tokyo (défaut 0.5).
        time_stop_hour_utc: Heure UTC du time-stop (défaut 22 → close 22:00).
        signal_hour_utc: Heure UTC où on évalue le signal (défaut 7,
            doit être > tokyo_end_hour_utc).
        entry_hour_utc: Heure UTC d'entrée (défaut 8 → Open 08:00).
        tokyo_start_hour_utc, tokyo_end_hour_utc: Fenêtre Tokyo (défaut 0-6,
            7 barres exclusives de la barre signal_hour_utc).

    Returns:
        Liste de dicts trades, clés :
        date, signal (+1/-1), entry_time, exit_time, entry_price, exit_price,
        tokyo_high, tokyo_low, tokyo_range, tp_price, sl_price,
        pips_brut, pips_net, nights_held, exit_reason ∈ {"tp", "sl", "time_stop"}.
    """
    if not isinstance(df_h1.index, pd.DatetimeIndex):
        raise TypeError("df_h1.index doit être DatetimeIndex")
    if df_h1.index.tz is None:
        raise ValueError("df_h1.index doit être tz-aware (UTC)")
    if signal_hour_utc <= tokyo_end_hour_utc:
        raise ValueError(
            f"signal_hour_utc ({signal_hour_utc}) doit être > "
            f"tokyo_end_hour_utc ({tokyo_end_hour_utc}) — sinon Close(signal) "
            f"≤ tokyo_high par construction OHLC, rendant le breakout long impossible."
        )
    if entry_hour_utc < signal_hour_utc:
        raise ValueError(
            f"entry_hour_utc ({entry_hour_utc}) doit être ≥ signal_hour_utc ({signal_hour_utc})"
        )

    ranges = compute_tokyo_range(df_h1, tokyo_start_hour_utc, tokyo_end_hour_utc)
    if ranges.empty:
        return []

    cost_per_side = asset_config.commission_pips + asset_config.slippage_pips
    cost_total = 2 * cost_per_side + asset_config.spread_pips
    pip_size = asset_config.pip_size

    trades: list[dict] = []
    hours = df_h1.index.hour
    dates = df_h1.index.date

    for date, row in ranges.iterrows():
        tokyo_high = float(row["tokyo_high"])
        tokyo_low = float(row["tokyo_low"])
        tokyo_range = float(row["tokyo_range"])

        day_mask = dates == date

        # ── Signal bar (close 07:00 UTC ce jour) ─────────────────────
        signal_mask = day_mask & (hours == signal_hour_utc)
        signal_bars = df_h1[signal_mask]
        if signal_bars.empty:
            continue
        close_signal = float(signal_bars.iloc[-1]["Close"])

        if close_signal > tokyo_high:
            signal = 1
        elif close_signal < tokyo_low:
            signal = -1
        else:
            continue  # Pas de breakout strict

        # ── Entry bar (Open 08:00 UTC ce jour) ───────────────────────
        entry_mask = day_mask & (hours == entry_hour_utc)
        entry_bars = df_h1[entry_mask]
        if entry_bars.empty:
            continue
        entry_ts = entry_bars.index[0]
        entry_price = float(entry_bars.iloc[0]["Open"])

        # ── TP / SL en prix absolus ──────────────────────────────────
        if signal == 1:
            tp_price = entry_price + tp_mult * tokyo_range
            sl_price = entry_price - sl_mult * tokyo_range
        else:
            tp_price = entry_price - tp_mult * tokyo_range
            sl_price = entry_price + sl_mult * tokyo_range

        # ── Boucle path-dependent [entry_hour, time_stop_hour] ───────
        in_window_mask = day_mask & (hours >= entry_hour_utc) & (hours <= time_stop_hour_utc)
        bars_in_window = df_h1[in_window_mask]

        exit_ts: pd.Timestamp | None = None
        exit_price: float | None = None
        exit_reason: str | None = None

        for ts, bar in bars_in_window.iterrows():
            hi = float(bar["High"])
            lo = float(bar["Low"])

            if signal == 1:
                # Convention conservatrice : SL d'abord
                if lo <= sl_price:
                    exit_ts, exit_price, exit_reason = ts, sl_price, "sl"
                    break
                if hi >= tp_price:
                    exit_ts, exit_price, exit_reason = ts, tp_price, "tp"
                    break
            else:
                if hi >= sl_price:
                    exit_ts, exit_price, exit_reason = ts, sl_price, "sl"
                    break
                if lo <= tp_price:
                    exit_ts, exit_price, exit_reason = ts, tp_price, "tp"
                    break

        # ── Time-stop : Close de la barre time_stop_hour ─────────────
        if exit_ts is None:
            time_stop_mask = day_mask & (hours == time_stop_hour_utc)
            time_stop_bars = df_h1[time_stop_mask]
            if time_stop_bars.empty:
                continue  # Pas de barre 22:00 → on skip ce trade
            exit_ts = time_stop_bars.index[-1]
            exit_price = float(time_stop_bars.iloc[-1]["Close"])
            exit_reason = "time_stop"

        # ── PnL ──────────────────────────────────────────────────────
        if signal == 1:
            pips_brut = (exit_price - entry_price) / pip_size
        else:
            pips_brut = (entry_price - exit_price) / pip_size

        pips_net = pips_brut - cost_total
        nights_held = max(0, (exit_ts.normalize() - entry_ts.normalize()).days)
        if nights_held > 0:
            swap = (
                asset_config.swap_long_pips_per_night
                if signal == 1
                else asset_config.swap_short_pips_per_night
            )
            pips_net += nights_held * swap

        trades.append({
            "date": str(date),
            "signal": signal,
            "entry_time": entry_ts.isoformat(),
            "exit_time": exit_ts.isoformat(),
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "tokyo_high": tokyo_high,
            "tokyo_low": tokyo_low,
            "tokyo_range": tokyo_range,
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "pips_brut": float(pips_brut),
            "pips_net": float(pips_net),
            "nights_held": nights_held,
            "exit_reason": exit_reason,
        })

    logger.info(
        "asian_range_simulated",
        extra={"context": {
            "n_trades": len(trades),
            "n_days_with_range": len(ranges),
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
        }},
    )
    return trades
