"""Recherche d'edge — harnais de validation honnête (Phase 1).

Point d'entrée unique qui câble TOUS les correctifs de fiabilité dans le bon
ordre, pour qu'une stratégie déclarée GO le soit pour de vrai :
- fill honnête (entrée à l'ouverture de la barre suivante, pas de look-ahead),
- coûts XTB réels (spread + slippage + commission) + swap overnight,
- split IS/OOS gelé (sélection sur IS, UN seul regard sur l'OOS),
- Sharpe annualisé routé par fréquence,
- DSR avec n_trials dérivé automatiquement du registre anti-snooping.
"""
from __future__ import annotations

from app.research.edge_harness import (
    EdgeResult,
    evaluate_oos,
    run_honest_backtest,
    screen_candidates,
)

__all__ = [
    "EdgeResult",
    "evaluate_oos",
    "run_honest_backtest",
    "screen_candidates",
]
