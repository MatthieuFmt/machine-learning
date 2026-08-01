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


def test_gbpusd_present_and_correct() -> None:
    """GBPUSD ajouté en v4, pip_size forex standard 0.0001, spread ≤ 1.5 pips."""
    assert "GBPUSD" in ASSET_CONFIGS, "GBPUSD manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["GBPUSD"]
    assert cfg.pip_size == 0.0001, (
        f"GBPUSD pip_size {cfg.pip_size} != 0.0001 (standard forex)"
    )
    assert cfg.spread_pips <= 1.5, (
        f"GBPUSD spread {cfg.spread_pips} > 1.5 pips (anormal pour majeur forex)"
    )


def test_usdchf_present_and_correct() -> None:
    """USDCHF ajouté en v4, pip_size forex standard 0.0001, spread ≤ 1.5 pips."""
    assert "USDCHF" in ASSET_CONFIGS, "USDCHF manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["USDCHF"]
    assert cfg.pip_size == 0.0001, (
        f"USDCHF pip_size {cfg.pip_size} != 0.0001 (standard forex)"
    )
    assert cfg.spread_pips <= 1.5, (
        f"USDCHF spread {cfg.spread_pips} > 1.5 pips (anormal pour majeur forex)"
    )


def test_btcusd_costs_realistic() -> None:
    """BTCUSD : spread dans [60, 600] USD, slippage ≥ spread × 0.3.

    ⚠️ FIX 2026-08-01 (double correction) :
    1. La fourchette [10, 60] USD était une ESTIMATION jamais mesurée. Relevé
       réel sur l'app XTB (marché ouvert) : bid 62 849.7 / ask 63 039.2 →
       **189.5 USD**, soit 0.30 % du prix. Nouvelle fourchette ancrée sur la
       mesure : 0.1 % à 1 % d'un BTC autour de 60 k$ → [60, 600] USD.
    2. Le test comparait `spread_pips` (en PIPS) à une borne en USD, sans
       multiplier par `pip_size` — même confusion d'unité que celle qui a rendu
       tous les spreads du projet ×6 à ×15 trop bas. On compare désormais des USD
       à des USD.
    """
    assert "BTCUSD" in ASSET_CONFIGS, "BTCUSD manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["BTCUSD"]
    spread_usd = cfg.spread_pips * cfg.pip_size
    assert 60.0 <= spread_usd <= 600.0, (
        f"BTCUSD spread {spread_usd} USD hors fourchette réaliste [60, 600] "
        f"(relevé XTB 2026-08-01 : 189.5 USD)"
    )
    assert cfg.slippage_pips >= 0.3 * cfg.spread_pips, (
        f"BTCUSD slippage {cfg.slippage_pips} < 0.3 × spread {cfg.spread_pips}"
    )
    assert cfg.pip_size == 1.0, (
        f"BTCUSD pip_size {cfg.pip_size} != 1.0 (1 USD par défaut crypto)"
    )


def test_ethusd_costs_realistic() -> None:
    """ETHUSD : spread dans [1, 10] USD, slippage ≥ spread × 0.3.

    ⚠️ FIX 2026-08-01 : le test comparait `spread_pips` (300 pips) à une borne en
    USD, alors que `pip_size=0.01` → le vrai spread vaut 300 × 0.01 = 3.00 USD,
    bien DANS la fourchette. L'échec était un bug d'unité du test, pas de la
    config. Même classe d'erreur que le « pip interne vs pip broker » qui a
    faussé tous les spreads du projet.
    ⚠️ Valeurs ETHUSD encore NON RELEVÉES sur l'app XTB (contrairement à BTCUSD).
    """
    assert "ETHUSD" in ASSET_CONFIGS, "ETHUSD manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["ETHUSD"]
    spread_usd = cfg.spread_pips * cfg.pip_size
    assert 1.0 <= spread_usd <= 10.0, (
        f"ETHUSD spread {spread_usd} USD hors fourchette réaliste [1, 10]"
    )
    assert cfg.slippage_pips >= 0.3 * cfg.spread_pips, (
        f"ETHUSD slippage {cfg.slippage_pips} < 0.3 × spread {cfg.spread_pips}"
    )
    assert cfg.pip_size == 0.01, (
        f"ETHUSD pip_size {cfg.pip_size} != 0.01 (cotation au centime)"
    )


# ── Tests paramétrés (11 actifs → 11 tests) ─────────────────────────────────


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


# Seuils de ratio coût/SL par classe d'actif
_COST_SL_THRESHOLDS: dict[str, float] = {
    "BTCUSD": 0.25,
    "ETHUSD": 0.25,
    "GBPUSD": 0.12,
    "USDCHF": 0.15,
}


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_cost_vs_sl_ratio(asset: str) -> None:
    """Le coût total ne doit pas dépasser N % du SL (sinon stratégie impossible).

    Ratio = total_cost_pips / sl_points.
    Si > seuil, le coût d'entrée/sortie absorbe trop de la marge de sécurité
    du stop-loss, rendant l'espérance mathématique négative même avec
    un win-rate de 50 %.

    Seuils :
      - Standard : 10 % (forex, indices, matières premières)
      - Crypto   : 25 % (volatilité plus élevée, SL plus large)
    """
    cfg = ASSET_CONFIGS[asset]
    ratio = cfg.total_cost_pips / cfg.sl_points
    threshold = _COST_SL_THRESHOLDS.get(asset, 0.10)
    assert ratio <= threshold, (
        f"{asset}: coût {cfg.total_cost_pips:.4f} > {threshold:.0%} du SL {cfg.sl_points}. "
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
