"""Simulation de backtest stateful — un seul trade ouvert à la fois.

Extrait de backtest_utils.py original, réarchitecturé avec :
- Injection des filtres (FilterPipeline)
- Injection du weight_func (WeightFunction Protocol)
- Paramètres passés explicitement (plus de `from config import *`)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from learning_machine_learning.backtest.filters import FilterPipeline
from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def _normalize_seuil(seuil: float) -> float:
    """Normalise un seuil qui pourrait être en pourcentage (45 → 0.45)."""
    return seuil if seuil < 1.0 else seuil / 100.0


def simulate_trades(
    df: pd.DataFrame,
    weight_func: Callable[[np.ndarray], np.ndarray],
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    window: int = 24,
    pip_size: float = 0.0001,
    seuil_confiance: float = 0.35,
    commission_pips: float = 0.5,
    slippage_pips: float = 1.0,
    filter_pipeline: FilterPipeline | None = None,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """Simule la stratégie en mode stateful (un trade à la fois).

    Args:
        df: DataFrame avec colonnes Prediction_Modele, Confiance_Hausse_%,
            Confiance_Neutre_%, Confiance_Baisse_%, High, Low, Close, Spread,
            et colonnes nécessaires aux filtres.
        weight_func: Fonction (proba: np.ndarray) -> np.ndarray de poids.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window: Horizon max en nombre de barres.
        pip_size: Taille d'un pip.
        seuil_confiance: Seuil minimum de probabilité pour entrer.
        commission_pips: Commission broker aller-retour.
        slippage_pips: Slippage estimé.
        filter_pipeline: Optionnel — pipeline de filtres de régime.

    Returns:
        Tuple (trades_df, n_signaux, n_filtres_appliques).
        trades_df: indexé par Time d'entrée, colonnes Pips_Nets, Pips_Bruts,
                   Weight, result, filter_rejected.
        n_signaux: Nombre total de barres avec signal (avant stateful).
        n_filtres_appliques: dict {nom_filtre: n_rejetés}.
    """
    df = df.copy()

    # Calcul proba_max
    df["proba_max"] = (
        df[["Confiance_Hausse_%", "Confiance_Neutre_%", "Confiance_Baisse_%"]]
        .max(axis=1)
        / 100.0
    )

    seuil = _normalize_seuil(seuil_confiance)

    # Masques de base
    mask_long = (df["Prediction_Modele"] == 1) & (
        df["Confiance_Hausse_%"] / 100.0 >= seuil
    )
    mask_short = (df["Prediction_Modele"] == -1) & (
        df["Confiance_Baisse_%"] / 100.0 >= seuil
    )

    # Application des filtres
    n_filtres_appliques: dict[str, int] = {"trend": 0, "vol": 0, "session": 0}
    df["Filter_Rejected"] = ""
    if filter_pipeline is not None:
        mask_long, mask_short, n_filtres_appliques, rejection_reason = filter_pipeline.apply(
            df, mask_long, mask_short
        )
        df["Filter_Rejected"] = rejection_reason

    # Signaux finaux
    df["Signal"] = 0
    df.loc[mask_long, "Signal"] = 1
    df.loc[mask_short, "Signal"] = -1

    signal_mask = df["Signal"] != 0
    df.loc[signal_mask, "Weight"] = weight_func(df.loc[signal_mask, "proba_max"].values)
    n_signaux = int(signal_mask.sum())

    # Extraction NumPy pour la boucle rapide
    dates = df.index
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    signals = df["Signal"].values
    weights = df["Weight"].values
    spreads = df["Spread"].values
    filter_rejected_arr = df["Filter_Rejected"].values

    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size
    spread_cost_base = commission_pips + slippage_pips
    n = len(df)

    trades: list[dict] = []
    i = 0

    while i < n:
        if signals[i] != 0:
            entry_time = dates[i]
            entry_price = closes[i]
            signal = signals[i]
            spread_cost = spreads[i] / 10.0 + spread_cost_base
            weight = weights[i]
            entry_filter_rejected = filter_rejected_arr[i]

            if signal == 1:
                tp = entry_price + tp_dist
                sl = entry_price - sl_dist
            else:
                tp = entry_price - tp_dist
                sl = entry_price + sl_dist

            pips_brut = 0.0
            result_type = "loss_timeout"

            for j in range(1, window + 1):
                idx = i + j
                if idx >= n:
                    i = n
                    break
                curr_high, curr_low = highs[idx], lows[idx]

                if signal == 1:
                    if curr_low <= sl:
                        pips_brut = -sl_pips - spread_cost
                        result_type = "loss_sl"
                        i = idx
                        break
                    elif curr_high >= tp:
                        pips_brut = tp_pips - spread_cost
                        result_type = "win"
                        i = idx
                        break
                else:
                    if curr_high >= sl:
                        pips_brut = -sl_pips - spread_cost
                        result_type = "loss_sl"
                        i = idx
                        break
                    elif curr_low <= tp:
                        pips_brut = tp_pips - spread_cost
                        result_type = "win"
                        i = idx
                        break
            else:
                # Timeout : PnL réel basé sur le Close final (B2 fix)
                exit_idx = min(i + window, n - 1)
                exit_price = closes[exit_idx]
                if signal == 1:
                    pips_brut = (exit_price - entry_price) / pip_size - spread_cost
                else:
                    pips_brut = (entry_price - exit_price) / pip_size - spread_cost
                i += window
                result_type = "loss_timeout"

            trades.append({
                "Time": entry_time,
                "Pips_Nets": pips_brut * weight,
                "Pips_Bruts": pips_brut,
                "Weight": weight,
                "result": result_type,
                "filter_rejected": entry_filter_rejected,
            })
            continue
        i += 1

    if not trades:
        empty = pd.DataFrame(
            columns=["Time", "Pips_Nets", "Pips_Bruts", "Weight", "result", "filter_rejected"]
        )
        return empty.set_index("Time"), n_signaux, n_filtres_appliques

    return pd.DataFrame(trades).set_index("Time"), n_signaux, n_filtres_appliques
