"""Pipeline ML v4 — FROZEN après A9. Tout pipeline = lookup ici.

CE FICHIER NE DOIT PAS ÊTRE MODIFIÉ APRÈS A9.
Toute modification = data snooping → invalide la statistique Phase B.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config.features_selected import FEATURES_SELECTED
from app.config.hyperparams_tuned import HYPERPARAMS_TUNED
from app.config.model_selected import MODEL_SELECTED

PIPELINE_VERSION: str = "v4.0.0-locked"


@dataclass(frozen=True)
class MLPipelineConfig:
    """Configuration complète du pipeline ML pour un (asset, tf) donné."""

    asset: str
    tf: str
    features: tuple[str, ...]
    model_name: str
    model_params: dict
    threshold: float
    expected_sharpe_outer: float
    expected_wr: float
    version: str = PIPELINE_VERSION

    def __post_init__(self) -> None:
        if not 0.50 <= self.threshold <= 0.80:
            raise ValueError(f"Seuil hors plage [0.50, 0.80]: {self.threshold}")
        if not self.features:
            raise ValueError("Aucune feature sélectionnée")
        if self.model_name not in ("rf", "hgbm", "stacking"):
            raise ValueError(f"Modèle inconnu : {self.model_name}")


def get_pipeline(asset: str, tf: str) -> MLPipelineConfig:
    """Récupère le pipeline gelé pour (asset, tf). Raise KeyError si non configuré."""
    key = (asset, tf)
    if key not in FEATURES_SELECTED:
        raise KeyError(f"Pas de features sélectionnées pour {asset} {tf}")
    if key not in MODEL_SELECTED:
        raise KeyError(f"Pas de modèle sélectionné pour {asset} {tf}")
    if key not in HYPERPARAMS_TUNED:
        raise KeyError(f"Pas d'hyperparams tunés pour {asset} {tf}")

    h = HYPERPARAMS_TUNED[key]
    return MLPipelineConfig(
        asset=asset,
        tf=tf,
        features=FEATURES_SELECTED[key],
        model_name=h["model"],
        model_params=h["params"],
        threshold=h["threshold"],
        expected_sharpe_outer=h.get("expected_sharpe_outer", 0.0),
        expected_wr=h.get("expected_wr", 0.0),
    )


def all_configured_pairs() -> list[tuple[str, str]]:
    """Retourne tous les (asset, tf) avec pipeline complet."""
    return [
        k
        for k in FEATURES_SELECTED
        if k in MODEL_SELECTED and k in HYPERPARAMS_TUNED
    ]
