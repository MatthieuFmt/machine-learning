"""Stratégies déterministes pour backtest H03/H07.

Exporte ALL_STRATEGIES (v2) et ALL_STRATEGIES_V3 (H07 — trend-following additionnelles).
"""

from __future__ import annotations

from learning_machine_learning_v2.strategies.base import BaseStrategy
from learning_machine_learning_v2.strategies.sma_crossover import SmaCrossover
from learning_machine_learning_v2.strategies.donchian import DonchianBreakout
from learning_machine_learning_v2.strategies.rsi_contrarian import RsiContrarian
from learning_machine_learning_v2.strategies.bollinger import BollingerBands
from learning_machine_learning_v2.strategies.ts_momentum import TsMomentum
from learning_machine_learning_v2.strategies.dual_ma import DualMovingAverage
from learning_machine_learning_v2.strategies.keltner import KeltnerChannel
from learning_machine_learning_v2.strategies.chandelier import ChandelierExit
from learning_machine_learning_v2.strategies.parabolic import ParabolicSAR

ALL_STRATEGIES: list[tuple[type[BaseStrategy], dict[str, list]]] = [
    (SmaCrossover, {"fast": [5, 10, 20], "slow": [20, 50, 100]}),
    (DonchianBreakout, {"N": [20, 50, 100], "M": [10, 20, 50]}),
    (RsiContrarian, {"N": [7, 14, 21], "oversold": [25, 30, 35]}),
    (BollingerBands, {"N": [14, 20, 50], "K": [1.5, 2.0, 2.5]}),
    (TsMomentum, {"T": [5, 10, 20, 50, 100]}),
]

# H07 — Stratégies trend-following additionnelles pour v3
ALL_STRATEGIES_V3: list[tuple[type[BaseStrategy], dict[str, list]]] = [
    (DualMovingAverage, {"fast": [5, 10, 20], "slow": [50, 100, 200]}),
    (KeltnerChannel, {"period": [10, 20, 50], "mult": [1.5, 2.0, 2.5]}),
    (ChandelierExit, {"period": [11, 22, 44], "k_atr": [2.0, 3.0, 4.0]}),
    (ParabolicSAR, {"step": [0.01, 0.02, 0.03], "af_max": [0.1, 0.2, 0.3]}),
]

__all__ = [
    "ALL_STRATEGIES",
    "ALL_STRATEGIES_V3",
    "BaseStrategy",
    "SmaCrossover",
    "DonchianBreakout",
    "RsiContrarian",
    "BollingerBands",
    "TsMomentum",
    "DualMovingAverage",
    "KeltnerChannel",
    "ChandelierExit",
    "ParabolicSAR",
]
