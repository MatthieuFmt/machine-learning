"""Pipeline de ranking de features par pouvoir prédictif (Prompt 04).

Train ≤ 2022 uniquement. Combine mutual information, corrélation Pearson,
et permutation importance d'un Random Forest pour classer les indicateurs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

from app.core.exceptions import DataValidationError
from app.core.logging import get_logger
from app.data.loader import load_asset
from app.features.indicators import compute_all_indicators
from app.testing.look_ahead_validator import look_ahead_safe

logger = get_logger(__name__)


@look_ahead_safe
def rank_features(
    asset: str,
    tf: str,
    target_horizon: int,
    n_top: int = 20,
    train_end: str = "2022-12-31",
) -> pd.DataFrame:
    """Classe les indicateurs techniques par pouvoir prédictif.

    Utilise strictement les données ≤ train_end. Combine 3 métriques :
    1. Mutual information (dépendance non-linéaire)
    2. Corrélation Pearson absolue (dépendance linéaire)
    3. Permutation importance d'un RF 200 arbres (importance contextuelle)

    Args:
        asset: Nom du dossier dans data/raw/ (ex: "US30", "XAUUSD").
        tf: Timeframe ("D1", "H4", "H1", "M15", "M5").
        target_horizon: Horizon du forward return en nombre de barres.
        n_top: Nombre de features à retourner dans le top.
        train_end: Date de fin d'entraînement (inclusive), format "YYYY-MM-DD".

    Returns:
        DataFrame trié par composite_rank ascendant, avec colonnes :
        [feature, mutual_info, abs_corr, permutation_importance,
         mutual_info_rank, abs_corr_rank, permutation_importance_rank,
         composite_rank].

    Raises:
        DataValidationError: Si train_end est antérieur aux données disponibles.
        ValueError: Si target_horizon est trop grand (0 barres après dropna).
    """
    logger.info("rank_features_start", extra={"context": {
        "asset": asset,
        "tf": tf,
        "target_horizon": target_horizon,
        "n_top": n_top,
        "train_end": train_end,
    }})

    # ── Chargement et split temporel strict ──────────────────────────────
    df = load_asset(asset, tf)
    df = df.loc[:train_end]

    if len(df) == 0:
        raise DataValidationError(
            f"Aucune donnée ≤ {train_end} pour {asset} {tf}. "
            f"Plage disponible : {load_asset(asset, tf).index[0]} – {load_asset(asset, tf).index[-1]}"
        )

    # ── Features et cible ─────────────────────────────────────────────────
    features = compute_all_indicators(df)
    target = (df["Close"].shift(-target_horizon) / df["Close"] - 1).rename("y")

    aligned = pd.concat([features, target], axis=1).dropna()

    frac_dropped = 1.0 - len(aligned) / len(df)
    if frac_dropped > 0.50:
        raise ValueError(
            f"Plus de 50% des barres éliminées après dropna ({frac_dropped:.1%}). "
            f"Horizon={target_horizon} probablement trop grand pour {len(df)} barres."
        )
    if frac_dropped > 0.10:
        logger.warning("high_dropna_rate", extra={"context": {
            "frac_dropped": round(frac_dropped, 3),
            "n_before": len(df),
            "n_after": len(aligned),
        }})

    if len(aligned) == 0:
        raise ValueError(
            f"Aucune barre alignée après dropna. target_horizon={target_horizon} "
            f"trop grand pour {asset} {tf} (train_end={train_end})."
        )

    x = aligned.drop(columns=["y"])
    y = aligned["y"]

    logger.info("features_computed", extra={"context": {
        "n_features": len(x.columns),
        "n_samples": len(x),
    }})

    # ── Métrique 1 : Mutual Information ───────────────────────────────────
    mi = mutual_info_regression(x, y, random_state=42)
    logger.info("mutual_info_done", extra={"context": {"max_mi": round(float(mi.max()), 6)}})

    # ── Métrique 2 : Corrélation Pearson absolue ──────────────────────────
    corr = x.corrwith(y).abs()

    # ── Métrique 3 : Permutation importance (RF 200 arbres) ───────────────
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=4, random_state=42, n_jobs=-1,
    )
    rf.fit(x, y)
    perm = permutation_importance(
        rf, x, y, n_repeats=10, random_state=42, n_jobs=-1,
    )

    # ── Assemblage des résultats ──────────────────────────────────────────
    result = pd.DataFrame(
        {
            "feature": x.columns,
            "mutual_info": mi,
            "abs_corr": corr.values,
            "permutation_importance": perm.importances_mean,
        }
    )

    # Rangs (1 = meilleur)
    for col in ["mutual_info", "abs_corr", "permutation_importance"]:
        result[f"{col}_rank"] = result[col].rank(ascending=False)

    # Score composite = moyenne des rangs
    rank_cols = ["mutual_info_rank", "abs_corr_rank", "permutation_importance_rank"]
    result["composite_rank"] = result[rank_cols].mean(axis=1)

    result = result.sort_values("composite_rank").reset_index(drop=True)
    result = result.head(n_top)

    # ── Sauvegarde JSON ───────────────────────────────────────────────────
    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"feature_research_{asset}_{tf}.json"
    out_path.write_text(
        json.dumps(result.to_dict(orient="records"), indent=2, default=_json_default),
        encoding="utf-8",
    )

    logger.info("rank_features_done", extra={"context": {
        "output": str(out_path),
        "n_features_ranked": len(result),
        "top_feature": result.iloc[0]["feature"],
        "top_composite_rank": round(float(result.iloc[0]["composite_rank"]), 2),
    }})

    return result


def _json_default(obj: object) -> float:
    """Sérialise les types numpy en float pour JSON."""
    if isinstance(obj, (np.integer,)):
        return float(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return float(obj)
    raise TypeError(f"Type non sérialisable : {type(obj)}")
