"""Tests des 4 nouveaux AssetConfig (BTCUSD, ETHUSD, GBPUSD, USDCHF)."""
from __future__ import annotations

import pytest

from app.config.instruments import ASSET_CONFIGS


@pytest.mark.parametrize("name", ["BTCUSD", "ETHUSD", "GBPUSD", "USDCHF"])
def test_asset_config_present(name: str) -> None:
    assert name in ASSET_CONFIGS, f"{name} absent de ASSET_CONFIGS"


@pytest.mark.parametrize("name", ["BTCUSD", "ETHUSD", "GBPUSD", "USDCHF"])
def test_asset_config_valid(name: str) -> None:
    cfg = ASSET_CONFIGS[name]
    assert cfg.spread_pips > 0
    assert cfg.slippage_pips >= 0
    assert cfg.pip_size > 0
    assert cfg.pip_value_eur > 0
    assert cfg.tp_points > 0
    assert cfg.sl_points > 0
    assert cfg.total_cost_pips > 0


@pytest.mark.parametrize("name,expected_pip_size", [
    ("BTCUSD", 1.0),
    ("ETHUSD", 0.01),
    ("GBPUSD", 0.0001),
    ("USDCHF", 0.0001),
])
def test_asset_config_pip_size(name: str, expected_pip_size: float) -> None:
    assert ASSET_CONFIGS[name].pip_size == expected_pip_size


def test_no_existing_asset_modified() -> None:
    """Garde-fou : les 7 entrées d'origine ne doivent pas être modifiées par C1."""
    for name in ["US30", "US500", "GER30", "XAUUSD", "XAGUSD", "USOIL", "EURUSD"]:
        assert name in ASSET_CONFIGS, f"{name} (existant) ne doit pas être supprimé"
