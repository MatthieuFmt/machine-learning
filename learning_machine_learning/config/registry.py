"""Registre des configurations par instrument.

Usage :
    registry = ConfigRegistry()
    entry = registry.get("EURUSD")  # ConfigEntry composite
    print(entry.instrument.name, entry.model.rf_params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from learning_machine_learning.config.instruments import (
    BtcUsdConfig,
    EurUsdConfig,
    InstrumentConfig,
)
from learning_machine_learning.config.model import ModelConfig
from learning_machine_learning.config.backtest import BacktestConfig
from learning_machine_learning.config.paths import PathConfig


@dataclass(frozen=True)
class ConfigEntry:
    """Regroupe les 4 facettes de config pour un instrument."""
    instrument: InstrumentConfig
    model: ModelConfig
    backtest: BacktestConfig
    paths: PathConfig


class ConfigRegistry:
    """Registre centralisé des configurations par instrument.

    Découple totalement le code métier des noms d'instruments hardcodés.
    """

    _instruments: Dict[str, InstrumentConfig] = {
        "EURUSD": EurUsdConfig(),
        "BTCUSD": BtcUsdConfig(),
    }

    def __init__(
        self,
        model: ModelConfig | None = None,
        backtest: BacktestConfig | None = None,
        paths: PathConfig | None = None,
    ) -> None:
        self._model = model or ModelConfig()
        self._backtest = backtest or BacktestConfig()
        self._paths = paths or PathConfig()

    def get(self, name: str) -> ConfigEntry:
        """Retourne une ConfigEntry composite (KeyError si instrument inconnu)."""
        if name not in self._instruments:
            raise KeyError(
                f"Instrument '{name}' inconnu. Disponibles : {list(self._instruments.keys())}"
            )
        return ConfigEntry(
            instrument=self._instruments[name],
            model=self._model,
            backtest=self._backtest,
            paths=self._paths,
        )

    def list_instruments(self) -> list[str]:
        """Liste tous les instruments enregistrés."""
        return list(self._instruments.keys())

    def register(self, config: InstrumentConfig) -> None:
        """Enregistre une nouvelle configuration d'instrument."""
        self._instruments[config.name] = config
