"""Sizing au risque fixe : 2 % du capital par trade.

Modules utilisés uniquement par le simulateur et les métriques.
Zéro dépendance circulaire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def weight_centered(x: np.ndarray) -> np.ndarray:
    """Poids égal pour chaque trade (fallback quand aucun sizing spécifique n'est défini)."""
    return np.ones_like(x)


def volatility_target_weights(
    returns: pd.Series,
    target_vol_annual: float = 0.10,
    lookback: int = 60,
    max_leverage: float = 3.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Poids de *volatility targeting*, SANS look-ahead.

    Mise à l'échelle de la position pour viser une volatilité constante : on
    augmente l'exposition quand le marché est calme, on la réduit (voire ~0)
    quand il est agité. Pour la date `t`, le poids appliqué au rendement de `t`
    est calculé à partir de la volatilité réalisée sur les `lookback` rendements
    JUSQU'À `t-1` (``.shift(1)``) — donc connu en début de période, pas de triche.

    Formule : ``weight_t = clip(target_vol_annual / vol_réalisée_annualisée_t,
    0, max_leverage)``.

    Args:
        returns: rendements périodiques (quotidiens recommandés), index temporel.
        target_vol_annual: volatilité annualisée cible (0.10 = 10 %/an).
        lookback: fenêtre (en périodes) de la volatilité réalisée.
        max_leverage: plafond du poids (évite un levier absurde en très basse vol).
        periods_per_year: facteur d'annualisation (252 pour daily).

    Returns:
        pd.Series de poids alignée sur `returns`. Les `lookback` premières valeurs
        sont NaN (warmup) ; l'appelant les traite (drop ou 0).
    """
    if lookback < 2:
        raise ValueError(f"lookback doit être >= 2, reçu {lookback}")
    realized = returns.rolling(window=lookback, min_periods=lookback).std().shift(1)
    realized_ann = (realized * np.sqrt(periods_per_year)).replace(0.0, np.nan)
    weights = target_vol_annual / realized_ann
    return weights.clip(lower=0.0, upper=max_leverage)
