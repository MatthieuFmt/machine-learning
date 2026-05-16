"""Superset de features pour méta-labeling ML (pivot v4 A5).

Toutes les fonctions sont vectorisées, anti-look-ahead via @look_ahead_safe.
Ce module GÉNÈRE le superset, il ne SÉLECTIONNE PAS (cf. A6).

Categories:
    1. Trend                 (~12 cols)
    2. Momentum              (~6 cols)
    3. Oscillators           (~5 cols)
    4. Volatility            (~4 cols)
    5. Price action          (~10 cols)
    6. Statistical rolling   (~11 cols)
    7. Market regime         (~8 cols)
    8. Economic              (~9 cols)
    9. Sessions              (~8 cols)
    10. Cross-asset          (~3 cols, optional)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.indicators import (
    adx,
    atr,
    bbands_width,
    cci,
    efficiency_ratio,
    keltner_width,
    macd,
    mfi,
    rsi,
    stoch,
    williams_r,
)
from app.testing.look_ahead_validator import look_ahead_safe

# ═══════════════════════════════════════════════════════════════════════
# Category 1 — Trend (~12 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """SMA, EMA, distance prix-MA (normalisée ATR), slope MA."""
    close = df["Close"]
    h, lo = df["High"], df["Low"]
    atr14 = atr(h, lo, close, 14).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)

    for period in [20, 50, 200]:
        sma_vals = close.rolling(period, min_periods=max(1, period // 2)).mean()
        out[f"sma_{period}"] = sma_vals
        out[f"dist_sma_{period}"] = (close - sma_vals) / atr14

    for period in [12, 26]:
        ema_vals = close.ewm(span=period, adjust=False, min_periods=max(1, period // 2)).mean()
        out[f"ema_{period}"] = ema_vals
        out[f"dist_ema_{period}"] = (close - ema_vals) / atr14

    # Slope: SMA diff sur 5 barres normalisé par ATR
    sma20 = close.rolling(20, min_periods=10).mean()
    sma50 = close.rolling(50, min_periods=25).mean()
    out["slope_sma_20"] = sma20.diff(5) / (5 * atr14)
    out["slope_sma_50"] = sma50.diff(5) / (5 * atr14)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Category 2 — Momentum (~6 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """RSI (7, 14, 21) + MACD."""
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    for period in [7, 14, 21]:
        out[f"rsi_{period}"] = rsi(close, period)
    macd_df = macd(close, 12, 26, 9)
    out["macd"] = macd_df["macd_line"]
    out["macd_signal"] = macd_df["macd_signal"]
    out["macd_hist"] = macd_df["macd_histogram"]
    return out


# ═══════════════════════════════════════════════════════════════════════
# Category 3 — Oscillators (~5 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stoch K/D, Williams %R, CCI, MFI (si Volume présent)."""
    h, lo, c = df["High"], df["Low"], df["Close"]
    out = pd.DataFrame(index=df.index)
    stoch_df = stoch(h, lo, c, 14, 3)
    out["stoch_k_14"] = stoch_df["stoch_k"]
    out["stoch_d_14"] = stoch_df["stoch_d"]
    out["williams_r_14"] = williams_r(h, lo, c, 14)
    out["cci_20"] = cci(h, lo, c, 20)
    if "Volume" in df.columns and (df["Volume"] > 0).any():
        out["mfi_14"] = mfi(h, lo, c, df["Volume"], 14)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Category 4 — Volatility (~4 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """ATR, ATR%, BB width, KC width."""
    close, h, lo = df["Close"], df["High"], df["Low"]
    atr14 = atr(h, lo, close, 14)
    out = pd.DataFrame(index=df.index)
    out["atr_14"] = atr14
    out["atr_pct_14"] = atr14 / close.replace(0, np.nan)
    out["bb_width_20"] = bbands_width(close, 20, 2.0)
    out["kc_width_20"] = keltner_width(h, lo, close, 20, 2.0)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Category 5 — Price action (~10 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candlestick ratios, gaps, consecutive bars, inside/outside/doji."""
    o, h, lo, c = df["Open"], df["High"], df["Low"], df["Close"]
    range_ = (h - lo).replace(0, np.nan)
    atr14 = atr(h, lo, c, 14).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out["body_to_range_ratio"] = (c - o).abs() / range_

    # Shadows
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - lo
    out["upper_shadow_ratio"] = upper_shadow / range_
    out["lower_shadow_ratio"] = lower_shadow / range_

    # Overnight gap
    out["gap_overnight"] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)

    # Consecutive up/down (vectorized)
    up = (c.diff() > 0).astype(int)
    down = (c.diff() < 0).astype(int)
    out["consecutive_up"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    out["consecutive_down"] = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)

    # Range vs ATR
    out["range_atr_ratio"] = range_ / atr14

    # Candlestick patterns
    out["inside_bar"] = ((h < h.shift(1)) & (lo > lo.shift(1))).astype(int)
    out["outside_bar"] = ((h > h.shift(1)) & (lo < lo.shift(1))).astype(int)
    out["doji"] = ((c - o).abs() < 0.1 * range_).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Category 6 — Statistical rolling (~11 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-scores, percentiles, skew, kurtosis, autocorrelation."""
    close = df["Close"]
    h, lo = df["High"], df["Low"]
    log_ret = np.log(close / close.shift(1))
    atr14 = atr(h, lo, close, 14)

    out = pd.DataFrame(index=df.index)

    # Z-scores
    for window in [20, 60]:
        mean = close.rolling(window, min_periods=max(1, window // 2)).mean()
        std = close.rolling(window, min_periods=max(1, window // 2)).std().replace(0, np.nan)
        out[f"close_zscore_{window}"] = (close - mean) / std

    atr_mean = atr14.rolling(60, min_periods=20).mean()
    atr_std = atr14.rolling(60, min_periods=20).std().replace(0, np.nan)
    out["atr_zscore_60"] = (atr14 - atr_mean) / atr_std

    if "Volume" in df.columns and (df["Volume"] > 0).any():
        vol_mean = df["Volume"].rolling(20, min_periods=10).mean()
        vol_std = df["Volume"].rolling(20, min_periods=10).std().replace(0, np.nan)
        out["volume_zscore_20"] = (df["Volume"] - vol_mean) / vol_std

    # Return percentiles
    for window in [20, 60]:
        out[f"return_percentile_{window}"] = log_ret.rolling(window, min_periods=max(1, window // 2)).rank(pct=True)

    vol_realized = log_ret.rolling(20, min_periods=10).std()
    out["vol_percentile_60"] = vol_realized.rolling(60, min_periods=30).rank(pct=True)

    out["skew_returns_20"] = log_ret.rolling(20, min_periods=10).skew()
    out["kurt_returns_20"] = log_ret.rolling(20, min_periods=10).kurt()

    # Autocorrelation lag 1 over 20-bar window
    out["autocorr_returns_lag1_20"] = (
        log_ret.rolling(20, min_periods=10)
        .apply(lambda x: x.autocorr(1) if x.std() > 0 else 0.0, raw=False)
    )

    return out


# ═══════════════════════════════════════════════════════════════════════
# Category 7 — Market regime (~8 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Efficiency ratio, ADX trend strength, vol regime one-hot."""
    close = df["Close"]
    h, lo = df["High"], df["Low"]
    adx_df = adx(h, lo, close, 14)
    adx14 = adx_df["adx_line"]
    atr14 = atr(h, lo, close, 14).replace(0, np.nan)
    sma200 = close.rolling(200, min_periods=50).mean()

    out = pd.DataFrame(index=df.index)
    out["efficiency_ratio_20"] = efficiency_ratio(close, 20)
    out["trend_strength"] = (adx14 > 25).astype(int)
    dist_abs = ((close - sma200) / atr14).abs()
    out["dist_sma_200_abs_atr"] = dist_abs
    out["regime_trending_binary"] = ((adx14 > 25) & (dist_abs > 2.0)).astype(int)

    # Vol regime: terciles ATR rolling 60 → one-hot
    atr_rolling = atr(h, lo, close, 14).rolling(60, min_periods=20)
    p33 = atr_rolling.quantile(0.33)
    p66 = atr_rolling.quantile(0.66)
    atr14_raw = atr(h, lo, close, 14)
    out["vol_regime_low"] = (atr14_raw <= p33).astype(int)
    out["vol_regime_mid"] = ((atr14_raw > p33) & (atr14_raw <= p66)).astype(int)
    out["vol_regime_high"] = (atr14_raw > p66).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Category 8 — Economic (~9 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def economic_features_for_index(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Délègue à app/features/economic.py si calendrier dispo. Sinon fallback -1."""
    try:
        from app.features.economic import compute_event_features, load_calendar

        years = sorted({ts.year for ts in price_index})
        cal = load_calendar(years)
        return compute_event_features(price_index, cal)
    except Exception:
        return pd.DataFrame(
            -1.0,
            index=price_index,
            columns=[
                "event_high_within_1h_USD",
                "event_high_within_4h_USD",
                "event_high_within_24h_USD",
                "event_high_within_1h_EUR",
                "event_high_within_4h_EUR",
                "event_high_within_24h_EUR",
                "hours_since_last_nfp",
                "hours_since_last_fomc",
                "hours_to_next_event_high",
            ],
        )


# ═══════════════════════════════════════════════════════════════════════
# Category 9 — Sessions (~8 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def session_features(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sessions UTC + encodage cyclique du temps."""
    idx = pd.DatetimeIndex(price_index)
    hour = idx.hour
    weekday = idx.weekday
    month = idx.month

    out = pd.DataFrame(index=idx)
    out["session_tokyo"] = ((hour >= 0) & (hour < 9)).astype(int)
    out["session_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    out["session_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    out["session_overlap_london_ny"] = ((hour >= 13) & (hour < 16)).astype(int)
    out["day_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["day_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Category 10 — Cross-asset (optional, ~3 features)
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def cross_asset_features(
    price_index: pd.DatetimeIndex,
    asset: str | None = None,
) -> pd.DataFrame:
    """Features cross-asset si données disponibles. Sinon colonnes NaN (droppées en sortie)."""
    out = pd.DataFrame(index=price_index)

    try:
        from app.data.loader import load_asset
        from app.data.registry import discover_assets

        available = discover_assets()
    except Exception:
        return out

    tf_target = "D1"
    macro_map: dict[str, tuple[str, str]] = {
        "usdchf_return_5": ("USDCHF", tf_target),
        "xauusd_return_5": ("XAUUSD", tf_target),
        "btcusd_return_5": ("BTCUSD", tf_target),
    }

    for name, (sym, tf) in macro_map.items():
        tfs = available.get(sym, [])
        if sym in available and tf in tfs and sym != asset:
            try:
                df_macro = load_asset(sym, tf)
                ret = np.log(
                    df_macro["Close"] / df_macro["Close"].shift(5).replace(0, np.nan)
                )
                out[name] = ret.reindex(price_index, method="ffill")
            except Exception:
                out[name] = np.nan
        else:
            out[name] = np.nan

    return out


# ═══════════════════════════════════════════════════════════════════════
# Main aggregator
# ═══════════════════════════════════════════════════════════════════════


@look_ahead_safe
def build_superset(df: pd.DataFrame, asset: str | None = None) -> pd.DataFrame:
    """Construit le superset de ~70 features pour méta-labeling.

    Args:
        df: DataFrame OHLCV (PascalCase: Open, High, Low, Close, Volume).
        asset: Nom de l'actif (pour exclure du cross-asset).

    Returns:
        DataFrame indexé comme df, 60-75 colonnes float64. NaN possibles
        sur les premières ~200 lignes (warmup SMA200).
    """
    if df.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = [
        trend_features(df),
        momentum_features(df),
        oscillator_features(df),
        volatility_features(df),
        price_action_features(df),
        statistical_features(df),
        regime_features(df),
        economic_features_for_index(df.index),
        session_features(df.index),
        cross_asset_features(df.index, asset),
    ]

    out = pd.concat(parts, axis=1)
    # Drop columns that are entirely NaN (unavailable cross-asset)
    out = out.dropna(axis=1, how="all")
    out = out.astype(np.float64)
    out = out.loc[:, ~out.columns.duplicated()]

    return out
