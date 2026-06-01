"""Time-Series Momentum (tendance) crypto — rendements quotidiens (Phase 1).

Hypothèse PRÉ-ENREGISTRÉE (momentum, Moskowitz-Ooi-Pedersen ; la crypto est
réputée l'un des marchés où le trend est le plus robuste) : le signe du
rendement passé prédit le rendement futur. On suit la tendance : long si le
rendement des `lookback` derniers jours est positif, short sinon.

Particularité crypto : les moves sont énormes → ils peuvent **dépasser le swap**
(le tueur des holds multi-jours sur indices/forex). C'est le seul « multi-jours »
où ça vaut la peine d'essayer.

Travaille en RENDEMENTS QUOTIDIENS (position continue, sign-flipping), avec :
  - signal SANS look-ahead : ``sign(close_{t-1}/close_{t-1-lookback} − 1)`` →
    position appliquée le jour t (décidée à l'ouverture, sur l'info d'hier).
  - swap signé chaque nuit selon la position (long/short).
  - coût aller-retour à chaque RETOURNEMENT de position.
Le vol-scaling éventuel s'applique ENSUITE (multiplier les rendements par le poids
vol-target), à l'appelant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config.instruments import AssetConfig
from app.core.logging import get_logger

logger = get_logger(__name__)


def tsmom_daily_returns(
    df: pd.DataFrame,
    asset_config: AssetConfig,
    *,
    lookback: int = 100,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rendements quotidiens d'une stratégie TSMOM long/short (coûts + swap).

    Args:
        df: OHLCV D1 indexé tz-aware UTC (colonne ``Close`` requise).
        asset_config: Coûts/paramètres broker (pip_size, swaps, total_cost_pips).
        lookback: Fenêtre du momentum (jours). Signe du rendement sur cette
            fenêtre = direction de la position.

    Returns:
        (net_ret, gross_ret, position) — Series alignées sur ``df.index``.
        ``net_ret`` = gross + swap (négatif) − coût de retournement.
        Les premières lignes (warmup) sont NaN/0 (position 0).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index doit être DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("df.index doit être tz-aware (UTC)")
    if "Close" not in df.columns:
        raise KeyError("df doit contenir la colonne 'Close'")
    if lookback < 2:
        raise ValueError(f"lookback doit être >= 2, reçu {lookback}")

    close = df["Close"].astype(float)
    r = close.pct_change()

    # Signal SANS look-ahead : la position du jour t utilise les prix jusqu'à t-1.
    trailing = close / close.shift(lookback) - 1.0
    position = np.sign(trailing).shift(1).fillna(0.0)

    gross = position * r

    # Swap signé par nuit détenue (en fraction du notionnel).
    nights = close.index.to_series().diff().dt.days.fillna(1).clip(lower=0)
    nights.index = close.index
    swap_pips = pd.Series(
        np.where(
            position > 0, asset_config.swap_long_pips_per_night,
            np.where(position < 0, asset_config.swap_short_pips_per_night, 0.0),
        ),
        index=close.index,
    )
    swap_frac = swap_pips * asset_config.pip_size / close * nights

    # Coût de retournement : |Δposition| crossings × (coût a/r / 2) one-way.
    # Première barre : coût d'ouverture de la position initiale (|position|).
    pos_change = position.diff().abs()
    pos_change.iloc[0] = abs(float(position.iloc[0]))
    flip_cost_frac = (
        pos_change * (asset_config.total_cost_pips / 2.0) * asset_config.pip_size / close
    )

    net = gross + swap_frac.fillna(0.0) - flip_cost_frac.fillna(0.0)

    logger.info(
        "tsmom_crypto_simulated",
        extra={"context": {
            "n_days": int(close.notna().sum()),
            "lookback": lookback,
            "n_flips": int((pos_change > 0).sum()),
        }},
    )
    return net.rename("net"), gross.rename("gross"), position.rename("position")
