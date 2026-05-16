"""Tests des coûts post pivot v4 A2 — calibration XTB Standard Account.

Valide que chaque AssetConfig a des coûts réalistes (spread + slippage)
et que le ratio coût/SL reste ≤ 10 % (sinon stratégie mathématiquement
impossible). Les assertions portent sur la propriété `total_cost_pips`
qui somme spread + slippage + commission.

Source : docs/cost_audit_v2.md, valeurs XTB Standard Account 2025.
"""

from __future__ import annotations

import pytest

from app.config.instruments import ASSET_CONFIGS

# ── Tests fixes (4) ────────────────────────────────────────────────────────


def test_us30_spread_realistic() -> None:
    """US30 : spread ≤ 2.0 pts, total_cost ≤ 2.5 pts (v3 → v4 ÷ 4.4)."""
    cfg = ASSET_CONFIGS["US30"]
    assert cfg.spread_pips <= 2.0, f"US30 spread {cfg.spread_pips} > 2.0"
    assert cfg.total_cost_pips <= 2.5, f"US30 total_cost {cfg.total_cost_pips} > 2.5"


def test_xauusd_costs_realistic() -> None:
    """XAUUSD : spread en USD ≤ 1.0 (v3 35 USD → v4 0.35 USD, ÷ 100)."""
    cfg = ASSET_CONFIGS["XAUUSD"]
    spread_usd = cfg.spread_pips * cfg.pip_size
    assert spread_usd <= 1.0, f"XAUUSD spread USD {spread_usd} > 1.0"


def test_xagusd_costs_realistic() -> None:
    """XAGUSD : spread en USD ≤ 0.05 (v3 45 USD → v4 0.025 USD × 0.001 → réaliste)."""
    cfg = ASSET_CONFIGS["XAGUSD"]
    spread_usd = cfg.spread_pips * cfg.pip_size
    assert spread_usd <= 0.05, f"XAGUSD spread USD {spread_usd} > 0.05"


def test_eurusd_present_and_correct() -> None:
    """EURUSD ajouté en v4, pip_size forex standard 0.0001."""
    assert "EURUSD" in ASSET_CONFIGS, "EURUSD manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["EURUSD"]
    assert cfg.pip_size == 0.0001, (
        f"EURUSD pip_size {cfg.pip_size} != 0.0001 (standard forex)"
    )
    assert cfg.spread_pips <= 1.5, (
        f"EURUSD spread {cfg.spread_pips} > 1.5 pips (anormal pour majeur forex)"
    )


# ── Tests paramétrés (7 actifs → 7 tests) ──────────────────────────────────


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_asset_total_cost_positive(asset: str) -> None:
    """Le coût total doit être strictement positif pour chaque actif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.total_cost_pips > 0, (
        f"{asset}: total_cost_pips={cfg.total_cost_pips} doit être > 0"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_asset_spread_nonnegative(asset: str) -> None:
    """Le spread ne doit jamais être négatif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.spread_pips >= 0, (
        f"{asset}: spread_pips={cfg.spread_pips} < 0"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_asset_slippage_nonnegative(asset: str) -> None:
    """Le slippage ne doit jamais être négatif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.slippage_pips >= 0, (
        f"{asset}: slippage_pips={cfg.slippage_pips} < 0"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_cost_vs_sl_ratio(asset: str) -> None:
    """Le coût total ne doit pas dépasser 10 % du SL (sinon stratégie impossible).

    Ratio = total_cost_pips / sl_points.
    Si > 10 %, le coût d'entrée/sortie absorbe trop de la marge de sécurité
    du stop-loss, rendant l'espérance mathématique négative même avec
    un win-rate de 50 %.
    """
    cfg = ASSET_CONFIGS[asset]
    ratio = cfg.total_cost_pips / cfg.sl_points
    assert ratio <= 0.10, (
        f"{asset}: coût {cfg.total_cost_pips:.4f} > 10% du SL {cfg.sl_points}. "
        f"Ratio={ratio:.3f}. Stratégie mathématiquement impossible."
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_tp_gt_sl(asset: str) -> None:
    """Take-profit strictement supérieur au stop-loss (ratio risque/récompense)."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.tp_points > cfg.sl_points, (
        f"{asset}: tp_points={cfg.tp_points} ≤ sl_points={cfg.sl_points}"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_pip_size_positive(asset: str) -> None:
    """pip_size doit être strictement positif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.pip_size > 0, (
        f"{asset}: pip_size={cfg.pip_size} ≤ 0"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_pip_value_eur_positive(asset: str) -> None:
    """pip_value_eur doit être strictement positif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.pip_value_eur > 0, (
        f"{asset}: pip_value_eur={cfg.pip_value_eur} ≤ 0"
    )
