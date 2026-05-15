"""Sizing au risque fixe : 2 % du capital par trade.

Modules utilisés uniquement par le simulateur et les métriques.
Zéro dépendance circulaire.
"""

from __future__ import annotations

import numpy as np

from app.config.instruments import AssetConfig


def compute_position_size(
    entry_price: float,
    stop_loss_price: float,
    capital_eur: float,
    risk_pct: float,
    asset_cfg: AssetConfig,
) -> float:
    """Taille de position en lots pour risquer exactement `risk_pct` du capital sur le SL.

    Formule :
        risk_eur = capital × risk_pct
        distance_price = |entry - stop_loss|
        distance_points = distance_price / asset_cfg.pip_size
        loss_per_lot_eur = distance_points × asset_cfg.pip_value_eur
        lots = risk_eur / loss_per_lot_eur

    Clamp dans [asset_cfg.min_lot, asset_cfg.max_lot].

    Args:
        entry_price: Prix d'entrée.
        stop_loss_price: Prix du stop-loss.
        capital_eur: Capital actuel en euros.
        risk_pct: Fraction du capital risquée (ex: 0.02 pour 2 %).
        asset_cfg: Configuration de l'actif (pip_size, pip_value_eur, min/max lots).

    Returns:
        Nombre de lots (arrondi à 2 décimales, clampé).

    Raises:
        ValueError: Si entry_price == stop_loss_price (SL nul).
    """
    if stop_loss_price == entry_price:
        raise ValueError("entry_price == stop_loss_price : SL nul, impossible de calculer")
    risk_eur = capital_eur * risk_pct
    distance_points = abs(entry_price - stop_loss_price) / asset_cfg.pip_size
    loss_per_lot_eur = distance_points * asset_cfg.pip_value_eur
    if loss_per_lot_eur <= 0:
        raise ValueError(f"loss_per_lot_eur invalide : {loss_per_lot_eur}")
    lots = risk_eur / loss_per_lot_eur
    return max(asset_cfg.min_lot, min(asset_cfg.max_lot, round(lots, 2)))


def expected_pnl_eur(
    pips_net: float | np.ndarray,
    position_size_lots: float | np.ndarray,
    asset_cfg: AssetConfig,
) -> float | np.ndarray:
    """PnL net en € pour un trade (ou plusieurs, vectorisé).

    Args:
        pips_net: Pips nets du trade (après coûts, après weight).
            Accepte float ou np.ndarray.
        position_size_lots: Nombre de lots (float ou np.ndarray).
        asset_cfg: Configuration de l'actif.

    Returns:
        PnL en euros (float si scalaire, np.ndarray si vectoriel).
    """
    return pips_net * position_size_lots * asset_cfg.pip_value_eur
