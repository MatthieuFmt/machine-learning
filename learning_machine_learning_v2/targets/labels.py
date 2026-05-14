"""Fonctions cibles pour labelling — zéro dépendance à features.pipeline.

Implémente :
- compute_directional_clean : cible binaire symétrique avec seuil de bruit ATR
- compute_triple_barrier : labels directionnels selon méthodologie López de Prado

Utilise uniquement pd et numpy — pas de dépendance interne à learning_machine_learning_v2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from learning_machine_learning_v2.core.logging import get_logger

logger = get_logger(__name__)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range vectorisé — zéro boucle Python.

    Args:
        high, low, close: Arrays OHLC (même longueur).
        period: Période ATR (défaut 14).

    Returns:
        np.ndarray de même longueur — les 'period' premières valeurs sont NaN.
    """
    n = len(close)
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    # EMA du TR via lissage de Wilder
    atr_arr = np.full(n, np.nan)
    atr_arr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, n):
        atr_arr[i] = (1 - alpha) * atr_arr[i - 1] + alpha * tr[i]
    return atr_arr


def compute_directional_clean(
    ohlc_data: pd.DataFrame,
    horizon_hours: int = 120,
    noise_atr: float = 0.5,
    atr_period: int = 14,
) -> pd.Series:
    """Cible binaire symétrique : HAUSSE (+1) / BAISSE (-1) / NEUTRE (0).

    Principe :
    - Calcule le return forward sur horizon_hours.
    - Si |return| < noise_atr × ATR(period) → label=0 (bruit, pas de conviction).
    - Sinon → label=+1 si hausse, -1 si baisse.

    La cible est SYMÉTRIQUE : sous H₀ (brownien sans drift), P(+1) ≈ P(-1) ≈ 0.5
    avant seuillage. Contrairement à triple_barrier avec TP=3×SL qui a un biais
    structurel de 25% pour le TP.

    Args:
        ohlc_data: DataFrame avec colonnes Open, High, Low, Close, index datetime.
        horizon_hours: Horizon forward en heures (120 = 5 jours en D1).
        noise_atr: Multiplicateur ATR pour le seuil de bruit (0.5 = bruit modéré).
        atr_period: Période ATR (défaut 14).

    Returns:
        pd.Series[int8] indexée comme ohlc_data, valeurs ∈ {−1, 0, 1}.
        Les 'horizon_bars' dernières valeurs sont NaN (pas assez de forward data).
    """
    if ohlc_data.empty:
        raise ValueError("ohlc_data est vide")

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(ohlc_data.columns)
    if missing:
        raise ValueError(f"Colonnes OHLC manquantes : {sorted(missing)}")

    df = ohlc_data.sort_index()
    closes = df["Close"].values.astype(np.float64)
    highs = df["High"].values.astype(np.float64)
    lows = df["Low"].values.astype(np.float64)

    # Détermination du nombre de barres forward
    # En D1 : 1 barre = 24h → horizon_hours / 24
    # On estime le delta temporel médian
    if len(df) >= 2:
        median_delta = df.index.to_series().diff().median()
        if pd.notna(median_delta):
            hours_per_bar = median_delta.total_seconds() / 3600.0
        else:
            hours_per_bar = 24.0
    else:
        hours_per_bar = 24.0

    horizon_bars = max(1, int(round(horizon_hours / hours_per_bar)))
    logger.info(
        "directional_clean : horizon_hours=%d, hours_per_bar=%.1f, horizon_bars=%d",
        horizon_hours, hours_per_bar, horizon_bars,
    )

    n = len(closes)
    atr_arr = _atr(highs, lows, closes, atr_period)

    labels = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - horizon_bars):
        future_close = closes[i + horizon_bars]
        fwd_return = (future_close - closes[i]) / closes[i]

        atr_i = atr_arr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        atr_pct = atr_i / closes[i]
        threshold = noise_atr * atr_pct

        if abs(fwd_return) < threshold:
            labels[i] = 0.0
        elif fwd_return > 0:
            labels[i] = 1.0
        else:
            labels[i] = -1.0

    result = pd.Series(labels, index=df.index, name="Target", dtype="Int8")

    # Log distribution
    valid = result.dropna()
    if len(valid) > 0:
        dist = valid.value_counts(normalize=True).sort_index()
        logger.info(
            "directional_clean distribution : -1=%.1f%%, 0=%.1f%%, 1=%.1f%% (n=%d)",
            dist.get(-1, 0) * 100,
            dist.get(0, 0) * 100,
            dist.get(1, 0) * 100,
            len(valid),
        )

    return result


def compute_triple_barrier(
    ohlc_data: pd.DataFrame,
    tp_pips: float,
    sl_pips: float,
    window_hours: int,
    pip_size: float = 1.0,
) -> pd.Series:
    """Labels directionnels selon méthodologie Triple Barrière (López de Prado).

    Pour chaque barre i :
    - LONG gagnant (=1)  : TP touché AVANT SL (ou rien touché mais Close > entry).
    - SHORT gagnant (=-1) : TP touché AVANT SL (ou rien touché mais Close < entry).
    - Neutre (=0)         : rien touché (timeout) OU ambigu (deux directions gagnent)
                            OU SL touché avant TP dans les deux directions.

    Règle cardinale : la target pour la barre i est calculée UNIQUEMENT
    à partir des barres i+1 à i+window. Jamais d'information de la barre i elle-même.

    Args:
        ohlc_data: DataFrame avec colonnes High, Low, Close, index datetime.
        tp_pips: Take-profit en pips (US30 → 1 point = 1 pip).
        sl_pips: Stop-loss en pips.
        window_hours: Horizon max en heures.
        pip_size: Taille d'un pip (1.0 pour US30).

    Returns:
        pd.Series[int8] indexée comme ohlc_data. Les 'window' dernières valeurs sont NaN.
    """
    if ohlc_data.empty:
        raise ValueError("ohlc_data est vide")

    required = {"High", "Low", "Close"}
    missing = required - set(ohlc_data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    if tp_pips <= sl_pips:
        raise ValueError(f"tp_pips ({tp_pips}) doit être > sl_pips ({sl_pips})")

    df = ohlc_data.sort_index()

    # Estimation du nombre de barres par heure
    if len(df) >= 2:
        median_delta = df.index.to_series().diff().median()
        if pd.notna(median_delta):
            hours_per_bar = median_delta.total_seconds() / 3600.0
        else:
            hours_per_bar = 24.0
    else:
        hours_per_bar = 24.0

    window = max(1, int(round(window_hours / hours_per_bar)))
    logger.info(
        "triple_barrier : window_hours=%d, hours_per_bar=%.1f, window_bars=%d",
        window_hours, hours_per_bar, window,
    )

    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(closes)

    labels = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - 1):
        entry_price = closes[i]
        tp_long = entry_price + tp_dist
        sl_long = entry_price - sl_dist
        tp_short = entry_price - tp_dist
        sl_short = entry_price + sl_dist

        long_win = False
        long_dead = False
        short_win = False
        short_dead = False

        max_j = min(window, n - i - 1)

        for j in range(1, max_j + 1):
            idx = i + j
            curr_high, curr_low = highs[idx], lows[idx]

            if not long_win and not long_dead:
                if curr_low <= sl_long:
                    long_dead = True
                elif curr_high >= tp_long:
                    long_win = True

            if not short_win and not short_dead:
                if curr_high >= sl_short:
                    short_dead = True
                elif curr_low <= tp_short:
                    short_win = True

            # Sortie anticipée si les deux directions sont résolues
            if (long_win or long_dead) and (short_win or short_dead):
                break

        # Timeout : on regarde le Close final
        if not (long_win or long_dead) and not (short_win or short_dead):
            # Rien touché → timeout
            labels[i] = 0.0
            continue

        # Cas ambigu : les deux directions gagnent
        if long_win and short_win:
            labels[i] = 0.0
            continue

        # LONG gagnant
        if long_win and not short_win:
            labels[i] = 1.0
            continue

        # SHORT gagnant
        if short_win and not long_win:
            labels[i] = -1.0
            continue

        # Les deux perdent ou un perd / l'autre rien
        labels[i] = 0.0

    result = pd.Series(labels, index=df.index, name="Target", dtype="Int8")

    valid = result.dropna()
    if len(valid) > 0:
        dist = valid.value_counts(normalize=True).sort_index()
        logger.info(
            "triple_barrier distribution : -1=%.1f%%, 0=%.1f%%, 1=%.1f%% "
            "(tp=%.1f pips, sl=%.1f pips, window=%d bars, n=%d)",
            dist.get(-1, 0) * 100,
            dist.get(0, 0) * 100,
            dist.get(1, 0) * 100,
            tp_pips, sl_pips, window, len(valid),
        )

    return result


def label_distribution(labels: pd.Series) -> dict[str, float]:
    """Distribution en pourcentage des classes dans une Series de labels.

    Args:
        labels: pd.Series de valeurs {-1, 0, 1} ou NaN.

    Returns:
        Dict avec clés '-1', '0', '1' → pourcentage (float).
    """
    valid = labels.dropna()
    if len(valid) == 0:
        return {"-1": 0.0, "0": 0.0, "1": 0.0}
    counts = valid.value_counts(normalize=True) * 100
    return {
        "-1": float(counts.get(-1, 0)),
        "0": float(counts.get(0, 0)),
        "1": float(counts.get(1, 0)),
    }
