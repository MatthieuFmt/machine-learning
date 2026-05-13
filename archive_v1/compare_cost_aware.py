"""Comparaison des labellisations classique vs cost-aware — v15.

Genere les distributions de labels et statistiques comparatives
pour les deux methodes sur EURUSD H1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from learning_machine_learning.config.instruments import EurUsdConfig
from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def _load_ohlcv(path: str) -> pd.DataFrame:
    """Charge un CSV OHLCV avec index datetime."""
    df = pd.read_csv(path, sep="\t", parse_dates=["Time"], index_col="Time")
    df.sort_index(inplace=True)
    return df


def main() -> None:
    """Compare les deux methodes de labelling et sauvegarde le rapport."""
    from learning_machine_learning.features.triple_barrier import (
        apply_triple_barrier,
        apply_triple_barrier_cost_aware,
        label_distribution,
    )

    instrument = EurUsdConfig()
    data_path = Path("data/EURUSD_H1.csv")
    if not data_path.exists():
        logger.error("Fichier de donnees introuvable : %s", data_path)
        return

    logger.info("Chargement des donnees EURUSD H1...")
    h1 = _load_ohlcv(str(data_path))
    logger.info("Donnees chargees : %d lignes, %s → %s", len(h1),
                h1.index[0].strftime("%Y-%m-%d"), h1.index[-1].strftime("%Y-%m-%d"))

    tp, sl, window = 20.0, 10.0, 24
    friction = instrument.friction_pips
    min_profit = instrument.min_profit_pips_cost_aware

    logger.info("Calcul labellisation CLASSIQUE (TP=%.1f, SL=%.1f, W=%d)...", tp, sl, window)
    targets_classic = apply_triple_barrier(
        h1, tp_pips=tp, sl_pips=sl, window=window,
        pip_size=instrument.pip_size,
    )

    logger.info("Calcul labellisation COST-AWARE (Friction=%.1fp, MinProfit=%.1fp)...",
                friction, min_profit)
    targets_cost_aware = apply_triple_barrier_cost_aware(
        h1, tp_pips=tp, sl_pips=sl, window=window,
        pip_size=instrument.pip_size,
        friction_pips=friction, min_profit_pips=min_profit,
    )

    # Distributions
    dist_classic = label_distribution(targets_classic)
    dist_cost_aware = label_distribution(targets_cost_aware)

    # Comptages
    valid_classic = targets_classic[~np.isnan(targets_classic)]
    valid_cost_aware = targets_cost_aware[~np.isnan(targets_cost_aware)]

    n_classic = len(valid_classic)
    n_cost_aware = len(valid_cost_aware)

    classic_long = int(np.sum(valid_classic == 1.0))
    classic_short = int(np.sum(valid_classic == -1.0))
    classic_neutral = int(np.sum(valid_classic == 0.0))

    cost_long = int(np.sum(valid_cost_aware == 1.0))
    cost_short = int(np.sum(valid_cost_aware == -1.0))
    cost_neutral = int(np.sum(valid_cost_aware == 0.0))

    # Trades elimines par le cost-aware (etaient TP dans classique mais pas rentables)
    tp_classic_mask = (valid_classic != 0.0)
    tp_cost_mask = (valid_cost_aware != 0.0)
    trades_eliminated = int(tp_classic_mask.sum()) - int(tp_cost_mask.sum())

    rapport = {
        "v15_cost_aware_comparison": {
            "parameters": {
                "tp_pips": tp,
                "sl_pips": sl,
                "window": window,
                "friction_pips": friction,
                "min_profit_pips": min_profit,
                "pip_size": instrument.pip_size,
            },
            "classic": {
                "total_valid": n_classic,
                "long_pct": dist_classic["1"],
                "short_pct": dist_classic["-1"],
                "neutral_pct": dist_classic["0"],
                "long_count": classic_long,
                "short_count": classic_short,
                "neutral_count": classic_neutral,
            },
            "cost_aware": {
                "total_valid": n_cost_aware,
                "long_pct": dist_cost_aware["1"],
                "short_pct": dist_cost_aware["-1"],
                "neutral_pct": dist_cost_aware["0"],
                "long_count": cost_long,
                "short_count": cost_short,
                "neutral_count": cost_neutral,
            },
            "impact": {
                "trades_eliminated": trades_eliminated,
                "elimination_pct": round(trades_eliminated / max(n_classic, 1) * 100, 2),
                "long_change_pct": round(dist_cost_aware["1"] - dist_classic["1"], 2),
                "short_change_pct": round(dist_cost_aware["-1"] - dist_classic["-1"], 2),
                "neutral_increase_pct": round(dist_cost_aware["0"] - dist_classic["0"], 2),
            },
        }
    }

    output_path = Path("predictions/cost_aware_comparison_v15.json")
    output_path.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Rapport sauvegarde : %s", output_path)

    # Resume console
    print("\n=== Comparaison Labellisation v15 ===")
    print(f"Classique  : {classic_long:>6d} LONG, {classic_short:>6d} SHORT, "
          f"{classic_neutral:>6d} NEUTRE ({n_classic} total)")
    print(f"Cost-Aware : {cost_long:>6d} LONG, {cost_short:>6d} SHORT, "
          f"{cost_neutral:>6d} NEUTRE ({n_cost_aware} total)")
    print(f"Trades elimines par cout : {trades_eliminated} "
          f"({trades_eliminated / max(n_classic, 1) * 100:.1f}%)")
    print(f"Neutre : {dist_classic['0']:.1f}% -> {dist_cost_aware['0']:.1f}% "
          f"({dist_cost_aware['0'] - dist_classic['0']:+.1f}pp)")


if __name__ == "__main__":
    main()
