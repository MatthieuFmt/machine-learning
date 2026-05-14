"""Configs typées immutables. Évite les bugs silencieux."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

TFLiteral = Literal["D1", "H4", "H1"]


@dataclass(frozen=True)
class Sleeve:
    asset: str
    tf: TFLiteral
    strategy_name: str
    strategy_params: Mapping[str, int | float | str]
    regime_filter: bool = False
    meta_labeling: bool = False
    meta_threshold: float = 0.50

    def __post_init__(self) -> None:
        if not 0.40 <= self.meta_threshold <= 0.80:
            raise ValueError(f"meta_threshold hors plage : {self.meta_threshold}")


@dataclass(frozen=True)
class ProductionConfig:
    version: str
    sleeves: tuple[Sleeve, ...]
    portfolio_weighting: Literal["equal_risk", "correlation_aware"]
    vol_targeting: bool
    target_vol_annual: float
    leverage_cap: float
    retrain_months: int

    def __post_init__(self) -> None:
        if not 0 < self.target_vol_annual < 1.0:
            raise ValueError(f"target_vol_annual hors (0,1) : {self.target_vol_annual}")
        if not 1.0 <= self.leverage_cap <= 3.0:
            raise ValueError(f"leverage_cap hors [1,3] : {self.leverage_cap}")
        if len(self.sleeves) == 0:
            raise ValueError("ProductionConfig sans sleeve")
        if self.retrain_months not in (3, 6, 12):
            raise ValueError(f"retrain_months ∉ {{3,6,12}} : {self.retrain_months}")
