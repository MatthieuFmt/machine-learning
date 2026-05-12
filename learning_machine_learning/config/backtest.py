"""Configuration du backtest (TP/SL, commission, filtres)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BacktestConfig:
    """Paramètres de simulation de backtest.

    Validation stricte dans __post_init__ pour éviter les configurations
    absurdes (TP négatif, seuil > 1, etc.).
    """

    tp_pips: float = 30.0
    sl_pips: float = 10.0
    window_hours: int = 24
    commission_pips: float = 0.5
    slippage_pips: float = 1.0
    confidence_threshold: float = 0.33
    initial_capital: float = 10_000.0
    pip_value_eur: float = 1.0

    # Filtres de régime
    use_trend_filter: bool = False
    use_momentum_filter: bool = True
    use_vol_filter: bool = True
    use_session_filter: bool = True

    # Paramètres des filtres
    vol_filter_window: int = 168
    vol_filter_multiplier: float = 2.0
    session_exclude_start: int = 22
    session_exclude_end: int = 1
    momentum_filter_threshold: float = 3.0

    # Variantes TP/SL
    tp_sl_variants: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "baseline": (30.0, 10.0),
        "ratio_1_1": (20.0, 20.0),
        "ratio_3_1": (30.0, 10.0),
    })

    # Seuils alternatifs
    seuils_alternatifs: list[float] = field(default_factory=lambda: [0.36, 0.38, 0.40, 0.42])

    def __post_init__(self) -> None:
        if self.tp_pips <= 0:
            raise ValueError(f"tp_pips doit être > 0, reçu {self.tp_pips}")
        if self.sl_pips <= 0:
            raise ValueError(f"sl_pips doit être > 0, reçu {self.sl_pips}")
        if self.window_hours <= 0:
            raise ValueError(f"window_hours doit être > 0, reçu {self.window_hours}")
        if self.commission_pips < 0:
            raise ValueError(f"commission_pips doit être >= 0, reçu {self.commission_pips}")
        if self.slippage_pips < 0:
            raise ValueError(f"slippage_pips doit être >= 0, reçu {self.slippage_pips}")
        if not (0 < self.confidence_threshold <= 1):
            raise ValueError(
                f"confidence_threshold doit être dans (0, 1], reçu {self.confidence_threshold}"
            )
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital doit être > 0, reçu {self.initial_capital}")
        if self.momentum_filter_threshold <= 0:
            raise ValueError(
                f"momentum_filter_threshold doit être > 0, reçu {self.momentum_filter_threshold}"
            )

    def pips_to_return(self, pips: float) -> float:
        """Convertit des pips en fraction de capital.

        >>> BacktestConfig().pips_to_return(100)
        0.01
        """
        return pips * self.pip_value_eur / self.initial_capital
