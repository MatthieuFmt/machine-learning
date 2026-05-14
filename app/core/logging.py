"""Configuration centralisée du logging structuré.

Utilise le module `logging` standard avec un format JSON-like lisible.
Tous les modules du projet obtiennent leur logger via `get_logger(__name__)`.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, TextIO


def setup_logging(
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> None:
    """Configure le logger racine une fois au démarrage.

    Args:
        level: Niveau de log (logging.INFO par défaut).
        fmt: Format string. Défaut : '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        stream: Flux de sortie (stderr par défaut).
    """
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)
    # Évite les doublons si setup_logging() est appelé plusieurs fois
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé (usage : `get_logger(__name__)`).

    Args:
        name: Nom du logger (convention : __name__ du module appelant).

    Returns:
        logging.Logger configuré.
    """
    return logging.getLogger(name)
