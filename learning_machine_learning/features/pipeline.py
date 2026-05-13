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
from learning_machine_learning.features.triple_barrier import (
    apply_triple_barrier,
    apply_triple_barrier_cost_aware,
    compute_forward_return_target,
    compute_directional_clean_target,
    compute_cost_aware_target_v2,
)
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
    compute_session_id,
    compute_session_open_range,
    compute_relative_position_in_session,
    SessionVolatilityScaler,
)

logger = get_logger(__name__)


def build_ml_ready(
    instrument: InstrumentConfig,
    data: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame] | None = None,
    calendar_df: pd.DataFrame | None = None,
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    window: int = 24,
    features_dropped: list[str] | None = None,
    train_end: pd.Timestamp | None = None,
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
        calendar_df: DataFrame calendrier économique (optionnel).
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window: Horizon max en barres.
        features_dropped: Liste de colonnes a exclure (stockees mais pas dans X).
        train_end: Si fourni, le SessionVolatilityScaler est fit uniquement
            sur les données ≤ train_end (anti-look-ahead).

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

    # 1. Target labelling selon target_mode
    n_before = len(h1)
    mode = instrument.target_mode

    if mode == "forward_return":
        logger.info(
            "Target FORWARD_RETURN (régression): horizon=%dh...",
            instrument.target_horizon_hours,
        )
        h1["Target"] = compute_forward_return_target(
            h1,
            horizon_hours=instrument.target_horizon_hours,
            pip_size=instrument.pip_size,
        )
    elif mode == "directional_clean":
        logger.info(
            "Target DIRECTIONAL_CLEAN (binaire): horizon=%dh, "
            "noise=%.1fx ATR_%d...",
            instrument.target_horizon_hours,
            instrument.target_noise_threshold_atr,
            instrument.target_atr_period,
        )
        h1["Target"] = compute_directional_clean_target(
            h1,
            horizon_hours=instrument.target_horizon_hours,
            noise_threshold_atr=instrument.target_noise_threshold_atr,
            atr_period=instrument.target_atr_period,
            pip_size=instrument.pip_size,
        )
    elif mode == "cost_aware_v2":
        logger.info(
            "Target COST_AWARE_V2: TP=%.1f, SL=%.1f, Window=%dh, "
            "Friction=%.1fp, k_ATR=%.1f...",
            tp_pips, sl_pips, window,
            instrument.friction_pips, instrument.target_k_atr,
        )
        h1["Target"] = compute_cost_aware_target_v2(
            h1,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            window=window,
            friction_pips=instrument.friction_pips,
            k_atr=instrument.target_k_atr,
            pip_size=instrument.pip_size,
        )
    elif instrument.cost_aware_labeling:
        # Legacy : cost_aware_labeling (conservé pour rétrocompatibilité)
        logger.info(
            "Application de la triple barriere COST-AWARE "
            "(TP=%.1f, SL=%.1f, Window=%dh, Friction=%.1fp, MinProfit=%.1fp)...",
            tp_pips, sl_pips, window,
            instrument.friction_pips, instrument.min_profit_pips_cost_aware,
        )
        h1["Target"] = apply_triple_barrier_cost_aware(
            h1, tp_pips=tp_pips, sl_pips=sl_pips, window=window,
            pip_size=instrument.pip_size,
            friction_pips=instrument.friction_pips,
            min_profit_pips=instrument.min_profit_pips_cost_aware,
        )
    else:
        # Triple barrière classique (défaut)
        logger.info(
            "Application de la triple barriere classique "
            "(TP=%.1f, SL=%.1f, Window=%dh)...",
            tp_pips, sl_pips, window,
        )
        h1["Target"] = apply_triple_barrier(
            h1, tp_pips=tp_pips, sl_pips=sl_pips, window=window,
            pip_size=instrument.pip_size,
        )
    h1.dropna(subset=["Target"], inplace=True)
    log_row_loss("dropna Target (triple barrier)", n_before, len(h1))

    # 2. Features techniques H1
    h1["Log_Return"] = np.log(h1["Close"] / h1["Close"].shift(1))
    ema_dists = calc_ema_distance(h1, periods=(50,))
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

    # 2.5 ★ Features de session (microstructure FX) ★
    h1["session_id"] = compute_session_id(h1.index)
    h1["session_open_range"] = compute_session_open_range(
        h1["High"], h1["Low"], h1["session_id"]
    )
    h1["relative_position_in_session"] = compute_relative_position_in_session(
        h1.index, h1["session_id"]
    )

    # ATR_session_zscore : fit train-only si train_end fourni
    scaler = SessionVolatilityScaler()
    if train_end is not None:
        train_mask = h1.index <= train_end
        scaler.fit(
            h1.loc[train_mask, "ATR_Norm"],
            h1.loc[train_mask, "session_id"],
        )
    else:
        # Mode exploration : fit sur toutes les données
        scaler.fit(h1["ATR_Norm"], h1["session_id"])
    h1["ATR_session_zscore"] = scaler.transform(
        h1["ATR_Norm"], h1["session_id"]
    )

    # One-hot encoding si configuré
    if instrument.session_encoding == "one_hot":
        SESSION_LABELS = {
            1: "session_London",
            2: "session_NY",
            3: "session_Overlap",
            4: "session_LowLiq",
        }
        for sid_val, col_name in SESSION_LABELS.items():
            h1[col_name] = (h1["session_id"] == sid_val).astype(np.int8)
        # Tokyo = baseline (toutes les dummies à 0)

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

    # 5.5 ★ Calendrier économique macro ★
    if calendar_df is not None:
        from learning_machine_learning.features.calendar import merge_calendar_features
        n_avant_cal = len(combined)
        cal_features = merge_calendar_features(combined, calendar_df)
        combined = combined.join(cal_features)
        log_row_loss("merge calendar features", n_avant_cal, len(combined))
    else:
        logger.info("Aucun calendrier économique fourni — features calendrier ignorées.")

    # 6. Selection des colonnes finales
    colonnes_finales = [
        "Target", "Spread", "Log_Return",
        "Dist_EMA_50",
        "RSI_14", "ADX_14", "ATR_Norm", "BB_Width",
        "Hour_Sin", "Hour_Cos",
        "Volatilite_Realisee_24h", "Range_ATR_ratio",
        # ★ Session features ★
        "session_id", "ATR_session_zscore",
        "session_open_range", "relative_position_in_session",
        # ★ Calendar features ★
        "near_high_impact_event", "minutes_to_next_event",
        "minutes_since_last_event", "surprise_zscore",
    ]

    # Ajouter les one-hot si présentes
    if instrument.session_encoding == "one_hot":
        colonnes_finales += [
            "session_London", "session_NY",
            "session_Overlap", "session_LowLiq",
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
    FILTER_KEEP: frozenset[str] = frozenset({
        "ATR_Norm", "Dist_SMA200_D1", "Volatilite_Realisee_24h",
        "near_high_impact_event",
    })

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
