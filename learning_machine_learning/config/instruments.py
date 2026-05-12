"""Configuration par instrument — dataclasses immuables.

Un InstrumentConfig définit toutes les propriétés spécifiques à un actif :
taille du pip, timeframes, instruments macro corrélés, etc.
Ajouter un nouvel actif = créer une nouvelle sous-classe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Literal

TargetMode = Literal[
    "triple_barrier",
    "forward_return",
    "directional_clean",
    "cost_aware_v2",
]


@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration immuable pour un instrument de trading.

    Tous les champs sont gelés après construction. Pour dériver une config,
    utiliser `dataclasses.replace(config, name='BTCUSD')`.
    """

    name: str
    pip_size: float
    pip_value_eur: float
    timeframes: FrozenSet[str]
    primary_tf: str
    macro_instruments: FrozenSet[str]
    features_dropped: tuple[str, ...] = ()
    cost_aware_labeling: bool = False
    friction_pips: float = 1.5
    min_profit_pips_cost_aware: float = 3.0

    # ── Step 01 — Redéfinition de la cible ─────────────────────────────
    target_mode: TargetMode = "triple_barrier"
    target_horizon_hours: int = 24
    target_noise_threshold_atr: float = 0.5
    target_atr_period: int = 14
    target_k_atr: float = 1.0

    def __post_init__(self) -> None:
        if self.pip_size <= 0:
            raise ValueError(f"pip_size doit être > 0, reçu {self.pip_size}")
        if self.pip_value_eur <= 0:
            raise ValueError(f"pip_value_eur doit être > 0, reçu {self.pip_value_eur}")
        if self.primary_tf not in self.timeframes:
            raise ValueError(
                f"primary_tf '{self.primary_tf}' doit être dans timeframes {self.timeframes}"
            )
        if not self.timeframes:
            raise ValueError("timeframes ne peut pas être vide")
        if self.target_horizon_hours < 1:
            raise ValueError(
                f"target_horizon_hours doit être >= 1, reçu {self.target_horizon_hours}"
            )
        if self.target_noise_threshold_atr <= 0:
            raise ValueError(
                f"target_noise_threshold_atr doit être > 0, "
                f"reçu {self.target_noise_threshold_atr}"
            )
        if self.target_atr_period < 1:
            raise ValueError(
                f"target_atr_period doit être >= 1, reçu {self.target_atr_period}"
            )
        if self.target_k_atr <= 0:
            raise ValueError(
                f"target_k_atr doit être > 0, reçu {self.target_k_atr}"
            )

    def path_suffix(self, timeframe: str) -> str:
        """Retourne le suffixe de chemin pour un timeframe donné.

        >>> EurUsdConfig().path_suffix('H1')
        'EURUSD_H1_cleaned.csv'
        """
        return f"{self.name}_{timeframe}_cleaned.csv"


@dataclass(frozen=True)
class EurUsdConfig(InstrumentConfig):
    """Configuration EUR/USD — l'actif principal actuel."""

    name: str = "EURUSD"
    pip_size: float = 0.0001
    pip_value_eur: float = 1.0
    timeframes: FrozenSet[str] = frozenset({"H1", "H4", "D1"})
    primary_tf: str = "H1"
    macro_instruments: FrozenSet[str] = frozenset({"XAUUSD", "USDCHF"})
    features_dropped: tuple[str, ...] = (
        "Dist_EMA_9",
        "Dist_EMA_21",
        "Dist_EMA_20",
        "Log_Return",
        "CHF_Return",
        "Dist_EMA_50_D1",
        "BB_Width",
        "Hour_Cos",
        "Hour_Sin",
        "RSI_14_H4",
        "Dist_EMA_20_H4",
        "Dist_EMA_50_H4",
        "ATR_Norm",
        "Volatilite_Realisee_24h",
        "Range_ATR_ratio",
        "Momentum_5",
        "Momentum_10",
        "Momentum_20",
        "EMA_20_50_cross",
        "Volatility_Ratio",
    )


@dataclass(frozen=True)
class BtcUsdConfig(InstrumentConfig):
    """Configuration BTC/USD — futur actif cible.

    BTC n'a pas de pip au sens forex : 1$ = 1 unité.
    pip_size=1.0, pip_value_eur=0.92 (taux EUR/USD ≈ 1.08).
    Pas d'instruments macro corrélés connus.
    """

    name: str = "BTCUSD"
    pip_size: float = 1.0
    pip_value_eur: float = 0.92
    timeframes: FrozenSet[str] = frozenset({"H1", "H4", "D1"})
    primary_tf: str = "H1"
    macro_instruments: FrozenSet[str] = frozenset()
    features_dropped: tuple[str, ...] = ()
