"""Découverte dynamique des actifs/timeframes disponibles dans ``data/raw/``.

Ne lit JAMAIS le contenu des CSV — se base uniquement sur l'arborescence et les
noms de fichiers ``<*>_<TF>.csv``.
"""
from __future__ import annotations

from pathlib import Path

from app.data.loader import KNOWN_TIMEFRAMES


def discover_assets(data_root: Path = Path("data/raw")) -> dict[str, list[str]]:
    """Scanne ``data_root`` et retourne ``{asset: [tf, ...]}``.

    Un dossier ``data/raw/<ASSET>/`` contenant ``EURUSD_D1.csv`` et ``EURUSD_H4.csv``
    produit ``{"EURUSD": ["D1", "H4"]}``. Les timeframes inconnus sont ignorés.
    """
    root = Path(data_root)
    if not root.is_dir():
        return {}

    out: dict[str, list[str]] = {}
    for asset_dir in sorted(root.iterdir()):
        if not asset_dir.is_dir():
            continue
        tfs = sorted(
            {
                tf
                for csv in asset_dir.glob("*.csv")
                for tf in KNOWN_TIMEFRAMES
                if csv.stem.endswith(f"_{tf}")
            }
        )
        if tfs:
            out[asset_dir.name] = tfs
    return out
