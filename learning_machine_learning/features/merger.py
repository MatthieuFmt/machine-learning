"""Fusion multi-timeframe et macro — merge_asof avec correction look-ahead.

Le merge_asof projette les features des timeframes superieurs (H4, D1)
et les rendements macro sur chaque barre H1, en respectant la contrainte
de non-anticipation :

- H4 : decale de +4h (une barre H4 04:00 n'est close qu'a 04:00)
- D1 : decale de +1j (une barre D1 du jour J n'est close qu'a J+1 00:00)
"""

from __future__ import annotations

import pandas as pd

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def _log_merge_nan(
    label: str, combined: pd.DataFrame, probe_col: str, threshold_pct: float = 5.0
) -> None:
    """Log un avertissement si le merge produit > threshold_pct% de NaN."""
    n_total = len(combined)
    n_nan = int(combined[probe_col].isna().sum())
    pct = n_nan / n_total * 100 if n_total else 0
    if pct > threshold_pct:
        logger.warning(
            "merge %s: %d lignes, %d NaN dans %s (%.2f%%)",
            label, n_total, n_nan, probe_col, pct,
        )
    else:
        logger.info(
            "merge %s: %d lignes, %d NaN dans %s (%.2f%%)",
            label, n_total, n_nan, probe_col, pct,
        )


def log_row_loss(
    label: str, before: int, after: int, threshold_pct: float = 5.0
) -> None:
    """Log la perte de lignes apres une operation de dropna/filtrage."""
    loss = before - after
    pct = loss / before * 100 if before else 0
    if pct > threshold_pct:
        logger.warning(
            "%s: %d -> %d lignes (perte: %d, %.2f%%)",
            label, before, after, loss, pct,
        )
    else:
        logger.info(
            "%s: %d -> %d lignes (perte: %d, %.2f%%)",
            label, before, after, loss, pct,
        )


def merge_features(
    h1: pd.DataFrame,
    feat_h4: pd.DataFrame,
    feat_d1: pd.DataFrame,
    macro_frames: list[pd.DataFrame] | None = None,
    h4_offset: pd.Timedelta | None = None,
    d1_offset: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Fusionne les features H4, D1 et macro sur l'index H1 via merge_asof.

    Args:
        h1: DataFrame H1 avec index Time (deja reset_index ou non).
        feat_h4: Features H4 avec colonne 'Time'.
        feat_d1: Features D1 avec colonne 'Time'.
        macro_frames: Liste de DataFrames macro avec colonne 'Time'.
        h4_offset: Decalage anti-lookahead H4 (defaut: 4h).
        d1_offset: Decalage anti-lookahead D1 (defaut: 1j).

    Returns:
        DataFrame fusionne, indexe par Time.
    """
    if h4_offset is None:
        h4_offset = pd.Timedelta(hours=4)
    if d1_offset is None:
        d1_offset = pd.Timedelta(days=1)
    if macro_frames is None:
        macro_frames = []

    # Preparation : reset_index pour merge_asof
    h1 = h1.sort_index().reset_index()
    feat_h4 = feat_h4.sort_index().reset_index()
    feat_d1 = feat_d1.sort_index().reset_index()

    # Normalisation resolution datetime (pandas >= 3.0 peut produire us/ns mix)
    h1["Time"] = h1["Time"].astype("datetime64[ns]")
    feat_h4["Time"] = feat_h4["Time"].astype("datetime64[ns]")
    feat_d1["Time"] = feat_d1["Time"].astype("datetime64[ns]")

    # Correction look-ahead
    feat_h4["Time"] = feat_h4["Time"] + h4_offset
    feat_d1["Time"] = feat_d1["Time"] + d1_offset

    # Merge H4
    combined = pd.merge_asof(h1, feat_h4, on="Time", direction="backward")
    probe_h4 = [c for c in feat_h4.columns if c != "Time"]
    if probe_h4:
        _log_merge_nan("feat_h4", combined, probe_h4[0])

    # Merge D1
    combined = pd.merge_asof(combined, feat_d1, on="Time", direction="backward")
    probe_d1 = [c for c in feat_d1.columns if c != "Time"]
    if probe_d1:
        _log_merge_nan("feat_d1", combined, probe_d1[0])

    # Merge macro
    for i, macro_df in enumerate(macro_frames):
        macro_df = macro_df.sort_index().reset_index()
        macro_df["Time"] = macro_df["Time"].astype("datetime64[ns]")
        combined = pd.merge_asof(combined, macro_df, on="Time", direction="backward")
        probe = [c for c in macro_df.columns if c != "Time"]
        if probe:
            _log_merge_nan(f"macro_{i}", combined, probe[0])

    combined.set_index("Time", inplace=True)

    n_before = len(combined)
    combined.dropna(inplace=True)
    log_row_loss("dropna final (toutes features)", n_before, len(combined))

    return combined
