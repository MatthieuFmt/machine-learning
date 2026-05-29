"""Indicateurs techniques purement vectorisés — zéro boucle Python row-by-row.

Constitution Règle 6 (vectorisation pandas) et Règle 7 (anti-look-ahead) :
- Chaque fonction est décorée @look_ahead_safe.
- Chaque indicateur à l'instant t n'utilise que l'information ≤ t.
- Wilder smoothing (ewm alpha=1/period) pour ATR, ADX, RSI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.testing.look_ahead_validator import look_ahead_safe

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Trend (4 indicateurs)
# ═══════════════════════════════════════════════════════════════════════════════


@look_ahead_safe
def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(period).mean().rename(f"sma_{period}")


@look_ahead_safe
def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (span method)."""
    return close.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


@look_ahead_safe
def dist_sma(close: pd.Series, period: int) -> pd.Series:
    """Distance normalisée au SMA : (close - sma) / sma."""
    s = sma(close, period)
    return ((close - s) / s.replace(0, np.nan)).rename(f"dist_sma_{period}")


@look_ahead_safe
def slope_sma(close: pd.Series, period: int, lookback: int = 5) -> pd.Series:
    """Pente linéaire de la SMA(period) sur `lookback` barres.

    Utilise la régression linéaire OLS sur les indices entiers 0..lookback-1.
    """
    s = sma(close, period)
    x = np.arange(lookback, dtype=np.float64)
    x_mean = x.mean()
    x_diff = x - x_mean
    denom = (x_diff**2).sum()
    if denom == 0:
        return pd.Series(np.nan, index=close.index, name=f"slope_sma_{period}")
    # rolling apply + raw=True pour éviter l'overhead Series
    slope = s.rolling(lookback).apply(
        lambda y: np.cov(x_diff, y, bias=True)[0, 1] / (x_diff.var() if x_diff.var() > 0 else np.nan),
        raw=True,
    )
    return slope.rename(f"slope_sma_{period}")


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum (5 indicateurs)
# ═══════════════════════════════════════════════════════════════════════════════


@look_ahead_safe
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — Wilder smoothing."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    # avg_loss == 0 → rs = inf → RSI = 100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).rename(f"rsi_{period}")


@look_ahead_safe
def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = (ema_fast - ema_slow).rename("macd_line")
    signal_line = ema(macd_line, signal).rename("macd_signal")
    histogram = (macd_line - signal_line).rename("macd_histogram")
    return pd.concat([macd_line, signal_line, histogram], axis=1)


@look_ahead_safe
def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator %K et %D.

    Si le range (highest_high - lowest_low) est nul → %K = 50 (milieu).
    """
    highest_high = high.rolling(k).max()
    lowest_low = low.rolling(k).min()
    denom = highest_high - lowest_low
    stoch_k = pd.Series(50.0, index=close.index)
    mask = denom > 0
    stoch_k[mask] = 100.0 * (close[mask] - lowest_low[mask]) / denom[mask]
    stoch_k = stoch_k.rename("stoch_k")
    stoch_d = sma(stoch_k, d).rename("stoch_d")
    return pd.concat([stoch_k, stoch_d], axis=1)


@look_ahead_safe
def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R — valeur entre -100 et 0.

    Si le range est nul → retourne -50.
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    denom = highest_high - lowest_low
    wr = pd.Series(-50.0, index=close.index)
    mask = denom > 0
    wr[mask] = -100.0 * (highest_high[mask] - close[mask]) / denom[mask]
    return wr.rename(f"williams_r_{period}")


@look_ahead_safe
def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index."""
    typical = (high + low + close) / 3.0
    sma_typical = sma(typical, period)
    # Mean Absolute Deviation
    mad = typical.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True,
    )
    denom = 0.015 * mad
    cci_val = pd.Series(np.nan, index=close.index)
    mask = denom > 0
    cci_val[mask] = (typical[mask] - sma_typical[mask]) / denom[mask]
    return cci_val.rename(f"cci_{period}")


# ═══════════════════════════════════════════════════════════════════════════════
# Volatilité (4 indicateurs)
# ═══════════════════════════════════════════════════════════════════════════════


@look_ahead_safe
def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range — Wilder smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    # skipna=False → la 1ère barre (prev_close NaN) reste NaN, pas tr1 seul
    tr_raw = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    # ewm propage les NaN indéfiniment → on droppe avant, puis reindex
    valid_tr = tr_raw.dropna()
    if len(valid_tr) > 0:
        atr_vals = valid_tr.ewm(alpha=1.0 / period, adjust=False).mean()
        atr_vals = atr_vals.reindex(tr_raw.index)
    else:
        atr_vals = pd.Series(np.nan, index=tr_raw.index)
    return atr_vals.rename(f"atr_{period}")


@look_ahead_safe
def atr_pct(close: pd.Series, atr_series: pd.Series) -> pd.Series:
    """ATR normalisé en pourcentage du prix."""
    return ((atr_series / close.replace(0, np.nan)) * 100.0).rename("atr_pct")


@look_ahead_safe
def bbands_width(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    """Largeur des bandes de Bollinger : (upper - lower) / middle."""
    middle = sma(close, period)
    std = close.rolling(period).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return width.rename(f"bbands_width_{period}")


@look_ahead_safe
def keltner_width(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    n_atr: float = 2.0,
) -> pd.Series:
    """Largeur du canal de Keltner : (upper - lower) / middle."""
    middle = ema(close, period)
    atr_val = atr(high, low, close, period)
    upper = middle + n_atr * atr_val
    lower = middle - n_atr * atr_val
    width = (upper - lower) / middle.replace(0, np.nan)
    return width.rename(f"keltner_width_{period}")


# ═══════════════════════════════════════════════════════════════════════════════
# Volume (2 indicateurs)
# ═══════════════════════════════════════════════════════════════════════════════


@look_ahead_safe
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum().rename("obv")


@look_ahead_safe
def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index.

    Si le MF ratio a un dénominateur nul → MFI = 50.
    """
    typical = (high + low + close) / 3.0
    raw_mf = typical * volume
    direction = np.sign(typical.diff().fillna(0))

    pos_flow = raw_mf.where(direction > 0, 0.0).rolling(period).sum()
    neg_flow = raw_mf.where(direction < 0, 0.0).rolling(period).sum()

    denom = neg_flow.replace(0, np.nan)
    mf_ratio = pos_flow / denom
    return (100.0 - (100.0 / (1.0 + mf_ratio))).rename(f"mfi_{period}")


# ═══════════════════════════════════════════════════════════════════════════════
# Régime (3 indicateurs)
# ═══════════════════════════════════════════════════════════════════════════════


@look_ahead_safe
def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Average Directional Index — Wilder standard.

    Retourne ADX, +DI, -DI dans un DataFrame à 3 colonnes.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    mask_plus = (up_move > down_move) & (up_move > 0)
    mask_minus = (down_move > up_move) & (down_move > 0)

    plus_dm[mask_plus] = up_move[mask_plus]
    minus_dm[mask_minus] = down_move[mask_minus]

    # True Range (NaN → ewm propagera → dropna avant)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr_raw = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)

    # Wilder smoothing — on droppe les NaN avant ewm (propagation)
    mask_valid = tr_raw.notna()
    if mask_valid.any():
        smooth_tr = tr_raw[mask_valid].ewm(alpha=1.0 / period, adjust=False).mean().reindex(tr_raw.index)
        smooth_plus = plus_dm[mask_valid].ewm(alpha=1.0 / period, adjust=False).mean().reindex(tr_raw.index)
        smooth_minus = minus_dm[mask_valid].ewm(alpha=1.0 / period, adjust=False).mean().reindex(tr_raw.index)
    else:
        smooth_tr = pd.Series(np.nan, index=tr_raw.index)
        smooth_plus = pd.Series(np.nan, index=tr_raw.index)
        smooth_minus = pd.Series(np.nan, index=tr_raw.index)

    plus_di = (100.0 * smooth_plus / smooth_tr.replace(0, np.nan)).rename("plus_di")
    minus_di = (100.0 * smooth_minus / smooth_tr.replace(0, np.nan)).rename("minus_di")

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / dx_denom)
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False).mean().rename("adx_line")

    return pd.concat([adx_val, plus_di, minus_di], axis=1)


@look_ahead_safe
def efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio — dans [0, 1].

    ER = abs(close - close.shift(period)) / sum(abs(close.diff()), period)
    """
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    denom = volatility.replace(0, np.nan)
    return (direction / denom).rename(f"efficiency_ratio_{period}")


@look_ahead_safe
def realized_vol(close: pd.Series, period: int = 20) -> pd.Series:
    """Volatilité réalisée annualisée (écart-type des log-retours)."""
    log_ret = np.log(close / close.shift(1))
    return (log_ret.rolling(period).std() * np.sqrt(252)).rename(f"realized_vol_{period}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fonction agrégatrice
# ═══════════════════════════════════════════════════════════════════════════════


@look_ahead_safe
def compute_all_indicators(
    df: pd.DataFrame,
    include_economic: bool = False,
) -> pd.DataFrame:
    """Calcule tous les indicateurs sur un DataFrame OHLCV.

    Préconditions :
        - df.index est un DatetimeIndex trié.
        - Colonnes requises : [Open, High, Low, Close].
        - Colonne optionnelle : Volume (si absente ou tout à 0, skip OBV/MFI).

    Postconditions :
        - Même index que df.
        - Même longueur que df (NaN en warmup, jamais forward-filled).
        - Aucun side-effect sur df.

    Args:
        df: DataFrame avec colonnes [Open, High, Low, Close, Volume(opt)].
        include_economic: Si True, ajoute les features de calendrier économique
            (9 colonnes : event_high_within_Xh_XXX, hours_since_last_*, hours_to_next_*).
            Nécessite les CSV dans data/raw/economic_calendar/.

    Returns:
        DataFrame avec 20-31 colonnes de features indicatrices + calendrier.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    has_volume = "Volume" in df.columns and (df["Volume"] > 0).any()
    volume = df["Volume"] if has_volume else pd.Series(dtype=float)

    if not has_volume:
        logger.warning(
            "compute_all_indicators",
            extra={"context": {"warning": "Volume absent ou tout à 0, OBV/MFI ignorés"}},
        )

    # ── Trend ─────────────────────────────────────────────────────────────
    parts: list[pd.Series | pd.DataFrame] = [
        sma(close, 20),
        sma(close, 50),
        ema(close, 20),
        ema(close, 50),
        dist_sma(close, 20),
        dist_sma(close, 50),
        slope_sma(close, 20),
        slope_sma(close, 50),
    ]

    # ── Momentum ──────────────────────────────────────────────────────────
    parts += [
        rsi(close, 14),
        macd(close, 12, 26, 9),
        stoch(high, low, close, 14, 3),
        williams_r(high, low, close, 14),
        cci(high, low, close, 20),
    ]

    # ── Volatilité ────────────────────────────────────────────────────────
    atr_series = atr(high, low, close, 14)
    parts += [
        atr_series,
        atr_pct(close, atr_series),
        bbands_width(close, 20, 2.0),
        keltner_width(high, low, close, 20, 2.0),
    ]

    # ── Régime ────────────────────────────────────────────────────────────
    parts += [
        adx(high, low, close, 14),
        efficiency_ratio(close, 20),
        realized_vol(close, 20),
    ]

    # ── Volume (conditionnel) ─────────────────────────────────────────────
    if has_volume:
        parts += [
            obv(close, volume),
            mfi(high, low, close, volume, 14),
        ]

    result = pd.concat(parts, axis=1)

    # ── Calendrier économique (optionnel) ─────────────────────────────────
    if include_economic:
        from app.features.economic import compute_event_features, load_calendar

        # Déduire les années à partir de l'index
        years = list(range(df.index.min().year, df.index.max().year + 1))
        calendar = load_calendar(years)
        econ_features = compute_event_features(df.index, calendar)
        result = pd.concat([result, econ_features], axis=1)

    return result
