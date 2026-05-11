"""Module d'analyse post-backtest — diagnostic des pertes, directionnel, corrélations.

Fonctions pures, sans I/O fichier — les DataFrames sont passés en paramètres.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def analyze_losses(
    trades_df: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """Analyse les trades perdants vs gagnants.

    Args:
        trades_df: DataFrame de trades (avec colonnes Pips_Nets, result).
        top_n: Nombre de features à reporter.

    Returns:
        Dict avec 'n_total', 'n_losses', 'n_wins', 'win_rate',
        'n_loss_sl', 'n_loss_timeout', 'top_features'.
    """
    losses = trades_df[trades_df["Pips_Nets"] < 0]
    wins = trades_df[trades_df["Pips_Nets"] > 0]
    n_total = len(trades_df)

    if n_total == 0:
        return {"n_total": 0, "n_losses": 0, "n_wins": 0, "win_rate": 0.0}

    n_losses = len(losses)
    n_wins = len(wins)
    win_rate = n_wins / n_total * 100

    n_loss_sl = int((losses["result"] == "loss_sl").sum()) if n_losses > 0 else 0
    n_loss_timeout = int((losses["result"] == "loss_timeout").sum()) if n_losses > 0 else 0

    # Top features où les moyennes diffèrent le plus
    exclude = {"Signal", "Pips_Nets", "Pips_Bruts", "Weight", "result",
               "proba_hausse", "proba_neutre", "proba_baisse", "filter_rejected"}
    feature_cols = [c for c in trades_df.columns if c not in exclude]

    top_features = []
    if n_wins > 0 and n_losses > 0 and feature_cols:
        diffs = {}
        for f in feature_cols:
            try:
                diffs[f] = abs(float(wins[f].mean()) - float(losses[f].mean()))
            except (TypeError, KeyError):
                continue
        sorted_features = sorted(diffs, key=diffs.get, reverse=True)[:top_n]
        for f in sorted_features:
            top_features.append({
                "feature": f,
                "mean_wins": round(float(wins[f].mean()), 4),
                "mean_losses": round(float(losses[f].mean()), 4),
                "abs_diff": round(float(diffs[f]), 4),
            })

    return {
        "n_total": n_total,
        "n_losses": n_losses,
        "n_wins": n_wins,
        "win_rate": round(win_rate, 1),
        "n_loss_sl": n_loss_sl,
        "n_loss_timeout": n_loss_timeout,
        "top_features": top_features,
    }


def diagnostic_direction(
    trades_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    ml_data: pd.DataFrame | None = None,
) -> dict:
    """Diagnostic directionnel : ventile PnL par direction LONG/SHORT.

    Args:
        trades_df: DataFrame de trades.
        predictions_df: DataFrame de prédictions (avec Prediction_Modele).
        ml_data: Optionnel, pour vérifier le respect du filtre tendance
                 (colonne Dist_SMA200_D1).

    Returns:
        Dict avec 'total_pnl', 'long' et 'short' (chacun avec n, wr, pnl, esp).
    """
    merged = trades_df.join(predictions_df[["Prediction_Modele"]], how="left")
    merged["direction"] = merged["Prediction_Modele"].map({1.0: "LONG", -1.0: "SHORT"})

    result: dict[str, Any] = {"total_trades": len(trades_df),
                               "total_pnl": round(float(trades_df["Pips_Nets"].sum()), 1)}

    for direction in ["LONG", "SHORT"]:
        sub = merged[merged["direction"] == direction]
        if sub.empty:
            result[direction.lower()] = {"n": 0, "wr": 0.0, "pnl": 0.0, "esp": 0.0}
            continue

        n = len(sub)
        n_wins = int((sub["result"] == "win").sum())
        wr = round(n_wins / n * 100, 1)
        pnl = round(float(sub["Pips_Nets"].sum()), 1)
        esp = round(float(sub["Pips_Nets"].mean()), 2)

        result[direction.lower()] = {"n": n, "wr": wr, "pnl": pnl, "esp": esp}

    # Vérification filtre tendance
    if ml_data is not None and "Dist_SMA200_D1" in ml_data.columns:
        if "Dist_SMA200_D1" not in merged.columns:
            merged = merged.join(ml_data[["Dist_SMA200_D1"]], how="left")
        merged["above_sma"] = merged["Dist_SMA200_D1"] > 0

        for direction, expected_above in [("LONG", True), ("SHORT", False)]:
            sub = merged[merged["direction"] == direction]
            if len(sub) > 0:
                pct = (sub["above_sma"] == expected_above).mean() * 100
                result[f"{direction.lower()}_trend_respected_pct"] = round(float(pct), 0)

    return result
