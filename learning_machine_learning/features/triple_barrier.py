"""Labelling triple barrière (López de Prado) — bidirectionnel.

Applique la méthode triple barrier sur un DataFrame OHLCV : pour chaque barre
d'entrée, définit un label (1=LONG gagnant, -1=SHORT gagnant, 0=neutre) selon
le premier événement atteint parmi TP, SL ou timeout.

La fonction est pure : pas d'état global, pas d'I/O, pas de side effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def apply_triple_barrier(
    df: pd.DataFrame,
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    window: int = 24,
    pip_size: float = 0.0001,
) -> np.ndarray:
    """Applique la triple barrière bidirectionnelle sur un DataFrame OHLCV.

    Pour chaque barre i :
    - LONG  : TP = entry + tp_dist, SL = entry - sl_dist
    - SHORT : TP = entry - tp_dist, SL = entry + sl_dist
    Si TP est touché avant SL → label = 1 (LONG gagnant) ou -1 (SHORT gagnant).
    Si SL est touché avant TP → pas de label pour cette direction.
    Si les deux directions gagnent → 0 (ambigu).
    Si rien dans la fenêtre → 0 (timeout).

    Args:
        df: DataFrame avec colonnes 'High', 'Low', 'Close'. Index trié chronologiquement.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window: Horizon max en nombre de barres.
        pip_size: Taille d'un pip pour l'instrument (0.0001 pour EURUSD).

    Returns:
        np.ndarray de même longueur que df, valeurs ∈ {-1.0, 0.0, 1.0}.
        Les `window` dernières barres sont NaN (pas assez de forward bars).

    Raises:
        ValueError: Si les colonnes requises sont absentes.
    """
    required = {"High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes requises manquantes : {missing}")

    n = len(df)
    targets = np.full(n, np.nan, dtype=np.float64)
    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    # Le nombre effectif de barres labellisables
    limit = n - window

    for i in range(limit):
        entry_price = closes[i]

        # Bornes LONG
        long_tp = entry_price + tp_dist
        long_sl = entry_price - sl_dist

        # Bornes SHORT
        short_tp = entry_price - tp_dist
        short_sl = entry_price + sl_dist

        long_win = False
        long_dead = False
        short_win = False
        short_dead = False

        # Parcours forward jusqu'à window barres
        for j in range(1, window + 1):
            idx = i + j
            curr_high = highs[idx]
            curr_low = lows[idx]

            # Vérification LONG (abandonnée si déjà gagné ou SL touché)
            if not long_win and not long_dead:
                if curr_low <= long_sl:
                    long_dead = True  # SL touché avant TP → LONG perd définitivement
                elif curr_high >= long_tp:
                    long_win = True

            # Vérification SHORT (abandonnée si déjà gagné ou SL touché)
            if not short_win and not short_dead:
                if curr_high >= short_sl:
                    short_dead = True  # SL touché avant TP → SHORT perd définitivement
                elif curr_low <= short_tp:
                    short_win = True

            # Optimisation : si les deux directions sont résolues, sortir
            if (long_win or long_dead) and (short_win or short_dead):
                break

        # Label final : une seule direction gagnante → label directionnel
        if long_win and not short_win:
            targets[i] = 1.0
        elif short_win and not long_win:
            targets[i] = -1.0
        else:
            targets[i] = 0.0

    return targets


def compute_target_series(
    df: pd.DataFrame,
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    window: int = 24,
    pip_size: float = 0.0001,
) -> pd.Series:
    """Version de apply_triple_barrier qui retourne une Series pandas.

    Les NaN (barres non labellisables) sont automatiquement supprimées.

    Args:
        df: DataFrame OHLCV.
        tp_pips, sl_pips, window, pip_size: Voir apply_triple_barrier.

    Returns:
        pd.Series avec mêmes index que df, sans les NaN.
    """
    targets = apply_triple_barrier(df, tp_pips, sl_pips, window, pip_size)
    series = pd.Series(targets, index=df.index, name="Target", dtype="float64")
    n_before = len(series)
    series = series.dropna()
    n_after = len(series)
    logger.info(
        "Triple barrière : %d → %d barres labellisées (%.1f%% loss)",
        n_before,
        n_after,
        (n_before - n_after) / n_before * 100 if n_before else 0,
    )
    return series


def label_distribution(targets: np.ndarray | pd.Series) -> dict[str, float]:
    """Retourne la distribution des labels en pourcentage.

    Args:
        targets: Array ou Series de labels (-1, 0, 1).

    Returns:
        Dict avec clés '-1', '0', '1' et leur pourcentage.
    """
    values = targets.values if isinstance(targets, pd.Series) else targets
    valid = values[~np.isnan(values)]
    total = len(valid)
    if total == 0:
        return {"-1": 0.0, "0": 0.0, "1": 0.0}
    unique, counts = np.unique(valid, return_counts=True)
    dist = {str(int(k)): 0.0 for k in [-1.0, 0.0, 1.0]}
    for k, c in zip(unique, counts):
        dist[str(int(k))] = round(c / total * 100, 2)
    return dist
