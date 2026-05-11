"""Configuration par domaine — dataclasses immuables (frozen=True)."""

from learning_machine_learning.config.instruments import (
    EurUsdConfig,
    BtcUsdConfig,
    InstrumentConfig,
)
from learning_machine_learning.config.model import ModelConfig
from learning_machine_learning.config.backtest import BacktestConfig
from learning_machine_learning.config.paths import PathConfig
from learning_machine_learning.config.registry import ConfigEntry, ConfigRegistry

__all__ = [
    "ConfigEntry",
    "ConfigRegistry",
    "EurUsdConfig",
    "BtcUsdConfig",
    "InstrumentConfig",
    "ModelConfig",
    "BacktestConfig",
    "PathConfig",
]
