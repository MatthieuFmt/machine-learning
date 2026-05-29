"""Couche de données : chargement, validation et découverte des CSV OHLCV.

Modules :
- ``loader``   : ``load_asset(asset, tf)`` — lecture + validation stricte d'un CSV.
- ``registry`` : ``discover_assets()`` — découverte des actifs/timeframes disponibles.

Cette couche ne fait AUCUN feature engineering (cf. ``app/features``) et ne dépend
d'aucune lib externe au-delà de pandas. Le contrat CSV est documenté dans
``app/data/loader.py``.
"""
from __future__ import annotations
