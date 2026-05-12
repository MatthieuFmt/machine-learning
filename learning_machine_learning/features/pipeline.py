"""Pipeline de feature engineering pour un instrument.

Orchestre les etapes : target labelling, features techniques, features de regime,
features macro, fusion multi-timeframe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from learning_machine_learning.config.instruments import InstrumentConfig
from learning_machine_learning.core.logging import get_logger
from learning_machine_learning.features.triple_barrier import apply_triple_barrier
from learning_machine_learning.features.merger import merge_features, log_row_loss
from learning_machine_learning.features.macro import calc_macro_return
from learning_machine_learning.features.technical import (
    calc_base_features,
    calc_bb_width,
    calc_ema_distance,
)
from learning_machine_learning.features.regime import (
    calc_volatilite_realisee,
    calc_range_atr_ratio,
    calc_rsi_d1_delta,
    calc_dist_sma200_d1,
)

logger = get_logger(__name__)


def build_ml_ready(
    instrument: InstrumentConfig,
    data: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame] | None = None,
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    window: int = 24,
    features_dropped: list[str] | None = None,
) -> pd.DataFrame:
    """Construit le DataFrame ML-ready pour un instrument.

    Etapes :
    1. Triple barrier labelling sur le timeframe primaire (H1).
    2. Features techniques H1.
    3. Features techniques H4 et D1.
    4. Features de regime (D1).
    5. Features macro (si macro_data fourni).
    6. Fusion multi-timeframe via merge_asof.
    7. Selection des colonnes finales (hors features_dropped si specifie).

    Args:
        instrument: Configuration de l'instrument.
        data: Dict {timeframe: DataFrame OHLCV}. Doit contenir le primary_tf.
        macro_data: Dict {instrument_name: DataFrame OHLCV H1} pour les macro.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window: Horizon max en barres.
        features_dropped: Liste de colonnes a exclure (stockees mais pas dans X).

    Returns:
        DataFrame ML-ready indexe par Time.
    """
    if features_dropped is None:
        features_dropped = list(instrument.features_dropped)

    primary_tf = instrument.primary_tf
    if primary_tf not in data:
        raise KeyError(
            f"Timeframe primaire '{primary_tf}' absent de data. "
            f"Timeframes disponibles: {list(data.keys())}"
        )

    h1 = data[primary_tf].copy()

    # 1. Triple barrier
    logger.info("Application de la triple barriere (TP=%.1f, SL=%.1f, Window=%dh)...",
                tp_pips, sl_pips, window)
    n_before = len(h1)
    h1["Target"] = apply_triple_barrier(
        h1, tp_pips=tp_pips, sl_pips=sl_pips, window=window,
        pip_size=instrument.pip_size,
    )
    h1.dropna(subset=["Target"], inplace=True)
    log_row_loss("dropna Target (triple barrier)", n_before, len(h1))

    # 2. Features techniques H1
    h1["Log_Return"] = np.log(h1["Close"] / h1["Close"].shift(1))
    ema_dists = calc_ema_distance(h1, periods=(9, 21, 50))
    h1["Dist_EMA_9"] = ema_dists["Dist_EMA_9"]
    h1["Dist_EMA_21"] = ema_dists["Dist_EMA_21"]
    h1["Dist_EMA_50"] = ema_dists["Dist_EMA_50"]
    h1["RSI_14"] = ta.rsi(h1["Close"], length=14)
    h1["ADX_14"] = ta.adx(h1["High"], h1["Low"], h1["Close"], length=14)["ADX_14"]
    h1["ATR_Norm"] = ta.atr(h1["High"], h1["Low"], h1["Close"], length=14) / h1["Close"]

    # Regime H1
    h1["Volatilite_Realisee_24h"] = calc_volatilite_realisee(h1["Log_Return"], window=24)
    h1["Range_ATR_ratio"] = calc_range_atr_ratio(h1["High"], h1["Low"], h1["Close"])

    h1["BB_Width"] = calc_bb_width(h1)

    # Cyclical time
    h1["Hour_Sin"] = np.sin(h1.index.hour * (2.0 * np.pi / 24))
    h1["Hour_Cos"] = np.cos(h1.index.hour * (2.0 * np.pi / 24))

    # 3. Features H4 et D1
    feat_h4 = None
    feat_d1 = None
    for tf in instrument.timeframes:
        if tf == primary_tf:
            continue
        if tf not in data:
            logger.warning("Timeframe %s absent, ignore", tf)
            continue
        tf_df = data[tf]
        tf_features = calc_base_features(tf_df, f"_{tf}")

        if tf == "D1":
            # Regime D1
            d1_rsi = ta.rsi(tf_df["Close"], length=14)
            tf_features["RSI_D1_delta"] = calc_rsi_d1_delta(d1_rsi, diff_periods=3)
            tf_features["Dist_SMA200_D1"] = calc_dist_sma200_d1(tf_df["Close"])
            feat_d1 = tf_features
        elif tf == "H4":
            feat_h4 = tf_features

    # 4. Macro
    macro_frames = []
    if macro_data:
        for name, macro_df in macro_data.items():
            suffix = name.replace("USD", "")[:3]  # XAUUSD -> XAU, USDCHF -> CHF
            mf = calc_macro_return(macro_df["Close"], f"{suffix}_Return")
            macro_frames.append(mf)

    # 5. Fusion
    combined = merge_features(
        h1,
        feat_h4 if feat_h4 is not None else pd.DataFrame(),
        feat_d1 if feat_d1 is not None else pd.DataFrame(),
        macro_frames,
    )

    # 6. Selection des colonnes finales
    colonnes_finales = [
        "Target", "Spread", "Log_Return",
        "Dist_EMA_9", "Dist_EMA_21", "Dist_EMA_50",
        "RSI_14", "ADX_14", "ATR_Norm", "BB_Width",
        "Hour_Sin", "Hour_Cos",
        "Volatilite_Realisee_24h", "Range_ATR_ratio",
    ]

    if feat_h4 is not None:
        colonnes_finales += ["RSI_14_H4", "Dist_EMA_20_H4", "Dist_EMA_50_H4"]
    if feat_d1 is not None:
        colonnes_finales += ["RSI_14_D1", "Dist_EMA_20_D1", "Dist_EMA_50_D1",
                             "RSI_D1_delta", "Dist_SMA200_D1"]
    if macro_frames:
        for mf in macro_frames:
            colonnes_finales += [c for c in mf.columns if c != "Time"]

    # Ne garder que les colonnes presentes
    colonnes_finales = [c for c in colonnes_finales if c in combined.columns]

    # Colonnes preservees meme si dans features_dropped (necessaires aux filtres backtest)
    FILTER_KEEP: frozenset[str] = frozenset({"ATR_Norm", "Dist_SMA200_D1"})

    # Appliquer le filtrage features_dropped (R4 fix — etait defini mais jamais applique)
    n_avant_drop = len(colonnes_finales)
    colonnes_finales = [
        c for c in colonnes_finales
        if c not in features_dropped or c in FILTER_KEEP
    ]
    logger.info(
        "Filtrage features_dropped : %d -> %d colonnes (%d exclues, %d preservees filtres)",
        n_avant_drop, len(colonnes_finales), n_avant_drop - len(colonnes_finales),
        len([c for c in colonnes_finales if c in FILTER_KEEP]),
    )

    dataset_ml = combined[colonnes_finales]

    logger.info(
        "Dataset ML genere : %d lignes, %d colonnes (features_dropped: %d)",
        len(dataset_ml), len(colonnes_finales), len(features_dropped),
    )
    return dataset_ml
